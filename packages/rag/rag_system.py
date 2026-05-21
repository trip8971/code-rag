"""RAG system main entry point.

Coordinates all components to provide document indexing and retrieval functionality.
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from .bm25_search import BM25Search
from .chunker import Chunk, Chunker
from .config import RAGConfig
from .database import LocalDatabase
from .dense_search import DenseSearch
from .embedding import CHROMADB_DEFAULT_DIM, create_embedding_engine
from .evaluator import EvalRecord, EvalReport, Evaluator
from .interfaces import EmbeddingBase, QueryRewriterBase, RerankerBase
from .query_rewriter import create_query_rewriter
from .reranker import create_reranker

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG 系统主入口，协调各组件完成文档索引与检索。

    支持 async context manager 模式，确保资源正确释放：

        async with RAGSystem(config) as system:
            await system.index_document("docs/api.md")
            results = await system.retrieve("how to use Linear")
    """

    def __init__(self, config: RAGConfig):
        """初始化 RAG 系统所有组件。

        Args:
            config: RAG 系统全局配置
        """
        self.config = config

        # 初始化分块器
        self.chunker = Chunker(
            max_chunk_size=config.max_chunk_size, overlap=config.chunk_overlap
        )

        # 初始化嵌入引擎（通过工厂函数，根据配置自动选择 Provider）
        effective_embedding_dim = config.embedding_dim
        if not config.embedding.api_key:
            effective_embedding_dim = CHROMADB_DEFAULT_DIM

        self.embedding_engine: EmbeddingBase = create_embedding_engine(
            config=config.embedding,
            embedding_dim=effective_embedding_dim,
            batch_size=config.embedding_batch_size,
        )

        # 初始化本地数据库（ChromaDB）
        self.database = LocalDatabase(
            db_path=config.chroma_persist_dir, embedding_dim=effective_embedding_dim
        )
        self.database.initialize()

        # 初始化查询改写器（通过工厂函数，根据配置自动选择 Provider）
        self.query_rewriter: QueryRewriterBase = create_query_rewriter(
            config=config.query_rewriter
        )

        # 初始化稠密向量检索
        self.dense_search = DenseSearch(
            embedding_engine=self.embedding_engine, database=self.database
        )

        # 初始化 BM25 稀疏检索
        self.bm25_search = BM25Search(database=self.database)

        # 初始化重排序器（通过工厂函数，根据配置自动选择 Provider）
        self.reranker: RerankerBase = create_reranker(
            config=config.reranker,
            dense_weight=config.dense_weight,
            bm25_weight=config.bm25_weight,
        )

        # 初始化评估器
        self.evaluator = Evaluator(retriever=self._retrieve_ids, k=config.rerank_top_n)

    async def __aenter__(self) -> "RAGSystem":
        """进入 async context manager。"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出 async context manager，释放资源。"""
        await self.close()

    async def close(self) -> None:
        """关闭所有资源（HTTP 客户端、数据库连接等）。"""
        await self.embedding_engine.close()
        self.database.close()

    async def index_document(self, file_path: str, _skip_bm25: bool = False) -> int:
        """索引单个 Markdown 文档，返回生成的 Chunk 数量。

        流程：读取文件 → 分块 → 删除旧 Chunk → 向量化（分批）→ 存储 → 更新 BM25 索引

        Args:
            file_path: Markdown 文件路径
            _skip_bm25: 内部参数，批量索引时跳过 BM25 重建

        Returns:
            生成的 Chunk 数量

        Raises:
            FileNotFoundError: 文件不存在时抛出
            EmbeddingError: 向量化失败时抛出
        """
        # 1. 读取文件
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        # 2. 分块
        chunks = self.chunker.chunk_document(content, file_path)

        if not chunks:
            return 0

        # 3. 删除旧 Chunk（支持重新索引）
        self.database.delete_by_source(file_path)

        # 4. 向量化（EmbeddingEngine 内部自动分批）
        # 使用 embedding_text 属性：代码块会包含上下文以增强语义检索
        texts = [chunk.embedding_text for chunk in chunks]
        logger.info(
            "Embedding %s: %d chunks, %d chars total",
            file_path, len(texts), sum(len(t) for t in texts),
        )
        embeddings = await self.embedding_engine.embed_texts(texts)

        # 5. 存储 Chunk 和向量
        self.database.store_chunks(chunks, embeddings)

        # 6. 更新 BM25 索引（批量索引时跳过，最后统一重建）
        if not _skip_bm25:
            self.bm25_search.update_index(chunks)

        return len(chunks)

    async def index_directory(
        self, dir_path: str, concurrency: int = 5,
        on_progress: "Callable[[str, int, int], None] | None" = None,
    ) -> dict[str, int]:
        """索引目录下所有 Markdown 文档（并发执行）。

        使用 asyncio.Semaphore 控制并发数，避免同时发起过多 API 请求。
        索引完成后统一重建 BM25 索引确保一致性。

        Args:
            dir_path: 目录路径
            concurrency: 最大并发索引文件数，默认 5
            on_progress: 进度回调函数 (file_path, completed_count, total_count)

        Returns:
            字典 {file_path: chunk_count}，记录每个文件生成的 Chunk 数量
        """
        directory = Path(dir_path)
        md_files = sorted(directory.rglob("*.md"))

        if not md_files:
            return {}

        total = len(md_files)
        results: dict[str, int] = {}
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0

        async def _index_one(md_file: Path) -> tuple[str, int]:
            nonlocal completed_count
            file_path = str(md_file)
            async with semaphore:
                chunk_count = await self.index_document(file_path, _skip_bm25=True)
                completed_count += 1
                logger.info("Indexed %s: %d chunks", file_path, chunk_count)
                if on_progress:
                    on_progress(file_path, completed_count, total)
                return file_path, chunk_count

        tasks = [_index_one(f) for f in md_files]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.error("Failed to index a file: %s", str(result))
                raise result
            file_path, chunk_count = result
            results[file_path] = chunk_count

        # 批量索引完成后，从数据库重建 BM25 索引确保完全一致
        self.bm25_search.rebuild_index()

        return results

    async def retrieve(
        self, query: str, top_n: Optional[int] = None
    ) -> list[tuple[Chunk, float]]:
        """执行完整检索管道，返回排序后的结果。

        流程：查询改写 → 并行执行 DenseSearch 和 BM25Search → Reranker 重排序

        Args:
            query: 用户查询文本
            top_n: 返回的最大结果数量，默认使用配置中的 rerank_top_n

        Returns:
            按相关性排序的 (Chunk, score) 列表
        """
        detailed = await self.retrieve_detailed(query, top_n=top_n)
        return [(r["chunk"], r["score"]) for r in detailed["results"]]

    async def retrieve_detailed(
        self, query: str, top_n: Optional[int] = None
    ) -> dict:
        """执行完整检索管道，返回包含各阶段分数的详细结果。

        Returns:
            dict:
                rewritten_query: 改写后的查询
                results: 列表，每项为 dict:
                    chunk: Chunk 对象
                    score: 最终重排序分数
                    dense_score: 稠密检索分数（未命中则为 None）
                    bm25_score: BM25 检索分数（未命中则为 None）
        """
        if top_n is None:
            top_n = self.config.rerank_top_n

        # 1. 查询改写
        rewritten_query = await self.query_rewriter.rewrite(query)

        # 2. 并行执行 DenseSearch 和 BM25Search
        async def _bm25_search() -> list[tuple[Chunk, float]]:
            return self.bm25_search.search(
                rewritten_query, top_k=self.config.bm25_top_k
            )

        dense_results, bm25_results = await asyncio.gather(
            self.dense_search.search(rewritten_query, top_k=self.config.dense_top_k),
            _bm25_search(),
        )

        # Build score lookup maps
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_results}
        bm25_scores = {chunk.chunk_id: score for chunk, score in bm25_results}

        # 3. Reranker 重排序
        results = await self.reranker.rerank(
            query=rewritten_query,
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_n=top_n,
        )

        detailed = []
        for chunk, score in results:
            detailed.append({
                "chunk": chunk,
                "score": score,
                "dense_score": dense_scores.get(chunk.chunk_id),
                "bm25_score": bm25_scores.get(chunk.chunk_id),
            })

        return {
            "rewritten_query": rewritten_query,
            "results": detailed,
        }

    def evaluate(
        self,
        dataset: list[EvalRecord],
        previous_report: Optional[EvalReport] = None,
    ) -> EvalReport:
        """评估检索质量。

        Args:
            dataset: 评估数据集
            previous_report: 上一次评估报告（用于对比）

        Returns:
            EvalReport 评估报告

        Raises:
            EvalDatasetError: 数据集格式无效时抛出
        """
        return self.evaluator.evaluate(dataset, previous_report=previous_report)

    def _retrieve_ids(self, query: str) -> list[str]:
        """Helper for evaluator: 同步包装器，返回 chunk_ids。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.retrieve(query))
                results = future.result()
        else:
            results = asyncio.run(self.retrieve(query))

        return [chunk.chunk_id for chunk, _score in results]
