"""RAG 系统命令行入口。

用法：
    # 索引单个文件
    python -m rag index path/to/file.md

    # 索引整个目录
    python -m rag index path/to/docs/

    # 查询
    python -m rag query "如何使用 Linear 层"

    # 查询并指定返回数量
    python -m rag query "optimizer 初始化" --top-n 10

    # 指定配置文件
    python -m rag --config rag_config.yaml index docs/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import ConfigManager, RAGConfig
from .rag_system import RAGSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG 系统命令行工具：索引文档和执行检索查询",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（可选，默认从环境变量加载）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # index 子命令
    index_parser = subparsers.add_parser("index", help="索引文件或目录")
    index_parser.add_argument(
        "path",
        type=str,
        help="Markdown 文件路径或包含 .md 文件的目录路径",
    )
    index_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="目录索引时的最大并发数（默认 5）",
    )

    # query 子命令
    query_parser = subparsers.add_parser("query", help="执行检索查询")
    query_parser.add_argument(
        "query",
        type=str,
        help="查询文本",
    )
    query_parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="返回的最大结果数量（默认使用配置中的 rerank_top_n）",
    )
    query_parser.add_argument(
        "--top-docs",
        type=int,
        default=None,
        help="返回最相关的文档数量（默认使用配置中的 top_docs，未配置则为 5）",
    )

    return parser


def load_config(config_path: str | None) -> RAGConfig:
    """加载配置。"""
    manager = ConfigManager()
    return manager.load_config(config_path)


async def cmd_index(config: RAGConfig, path: str, concurrency: int) -> None:
    """执行索引命令。"""
    target = Path(path)

    if not target.exists():
        print(f"错误：路径不存在: {path}", file=sys.stderr)
        sys.exit(1)

    async with RAGSystem(config) as system:
        if target.is_file():
            if not target.suffix == ".md":
                print(f"警告：{path} 不是 .md 文件，仍尝试索引", file=sys.stderr)
            count = await system.index_document(str(target))
            print(json.dumps({
                "status": "ok",
                "file": str(target),
                "chunks": count,
            }, ensure_ascii=False, indent=2))
        elif target.is_dir():
            def _progress(file_path: str, done: int, total: int) -> None:
                pct = done * 100 // total
                bar_len = 30
                filled = bar_len * done // total
                bar = "█" * filled + "░" * (bar_len - filled)
                name = Path(file_path).name
                # \033[K 清除行尾残留字符
                print(
                    f"\r  [{bar}] {done}/{total} ({pct}%) {name}\033[K",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            print(f"索引目录: {target}", file=sys.stderr)
            results = await system.index_directory(
                str(target), concurrency=concurrency, on_progress=_progress
            )
            print("", file=sys.stderr)  # 换行

            total_chunks = sum(results.values())
            print(json.dumps({
                "status": "ok",
                "directory": str(target),
                "files_indexed": len(results),
                "total_chunks": total_chunks,
                "details": {k: v for k, v in results.items()},
            }, ensure_ascii=False, indent=2))
        else:
            print(f"错误：{path} 既不是文件也不是目录", file=sys.stderr)
            sys.exit(1)


async def cmd_query(config: RAGConfig, query: str, top_n: int | None, top_docs: int | None) -> None:
    """执行查询命令，输出 JSON 格式结果。"""
    # --top-docs 命令行参数优先，否则用配置文件值
    effective_top_docs = top_docs if top_docs is not None else config.top_docs
    async with RAGSystem(config) as system:
        # 从磁盘加载 BM25 索引（索引时已持久化），加载失败则从 DB 重建
        system.bm25_search.ensure_index()

        results = await system.retrieve(query, top_n=top_n)

        chunks_output = []
        for chunk, score in results:
            content_with_context = (
                f"{chunk.context}\n\n{chunk.content}" if chunk.context else chunk.content
            )
            chunks_output.append({
                "chunk_id": chunk.chunk_id,
                "content_with_context": content_with_context,
                "source_file": chunk.source_file,
                "chunk_type": chunk.chunk_type.value,
                "heading_text": chunk.heading_text,
                "start_line": chunk.start_line,
                "score": round(score, 6),
            })

        # 聚合最相关文档：按 source_file 取最高 chunk 分数排序
        doc_scores: dict[str, float] = {}
        doc_chunks: dict[str, int] = {}
        for chunk, score in results:
            sf = chunk.source_file
            if sf not in doc_scores or score > doc_scores[sf]:
                doc_scores[sf] = score
            doc_chunks[sf] = doc_chunks.get(sf, 0) + 1

        related_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        docs_output = [
            {
                "source_file": sf,
                "score": round(score, 6),
                "matched_chunks": doc_chunks[sf],
            }
            for sf, score in related_docs[:effective_top_docs]
        ]

        output = {
            "chunks": chunks_output,
            "related_docs": docs_output,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 配置日志
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 执行子命令
    if args.command == "index":
        asyncio.run(cmd_index(config, args.path, args.concurrency))
    elif args.command == "query":
        asyncio.run(cmd_query(config, args.query, args.top_n, args.top_docs))


if __name__ == "__main__":
    main()
