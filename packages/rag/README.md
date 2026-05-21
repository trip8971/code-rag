# Code RAG System

基于 Context7 架构启发的文档检索增强生成系统。支持 Markdown 文档索引、混合检索（稠密向量 + BM25）、结果重排序和检索质量评估。

## 安装

```bash
# 安装主依赖
uv sync

# 安装开发依赖（测试）
uv sync --all-extras
```

## 配置

系统需要配置外部服务的 API。其中 Query Rewriter 为必需，Embedding 和 Reranker 为可选（未配置时使用本地替代方案）。

### 方式一：环境变量（推荐）

```bash
# Embedding 服务（可选，不配置则使用 ChromaDB 内置模型）
export RAG_EMBEDDING_URL="https://api.openai.com/v1/embeddings"
export RAG_EMBEDDING_API_KEY="sk-your-embedding-key"
export RAG_EMBEDDING_TIMEOUT=30        # 可选，默认 30 秒
export RAG_EMBEDDING_MAX_RETRIES=3     # 可选，默认 3 次

# Reranker 服务（可选，不配置则使用 RRF 融合排序）
export RAG_RERANKER_URL="https://api.cohere.ai/v1/rerank"
export RAG_RERANKER_API_KEY="your-cohere-key"
export RAG_RERANKER_TIMEOUT=30
export RAG_RERANKER_MAX_RETRIES=3

# Query Rewriter 服务（必需）
export RAG_QUERY_REWRITER_URL="https://api.openai.com/v1/chat/completions"
export RAG_QUERY_REWRITER_API_KEY="sk-your-openai-key"
export RAG_QUERY_REWRITER_TIMEOUT=10
export RAG_QUERY_REWRITER_MAX_RETRIES=3
```

### 方式二：YAML 配置文件

创建 `rag_config.yaml`：

```yaml
embedding:
  url: "https://api.openai.com/v1/embeddings"
  api_key: "sk-your-embedding-key"
  timeout: 30
  max_retries: 3

reranker:
  url: "https://api.cohere.ai/v1/rerank"
  api_key: "your-cohere-key"
  timeout: 30
  max_retries: 3

query_rewriter:
  url: "https://api.openai.com/v1/chat/completions"
  api_key: "sk-your-openai-key"
  timeout: 10
  max_retries: 3

chunker:
  max_chunk_size: 1500
  overlap: 200

retrieval:
  dense_top_k: 20
  bm25_top_k: 10
  rerank_top_n: 5
  dense_weight: 0.7
  bm25_weight: 0.3

database:
  path: "rag_index.db"
  embedding_dim: 1536
```

> **优先级**：环境变量 > 配置文件。两者同时存在时，环境变量的值会覆盖配置文件。

### 兼容的 API 服务

| 组件 | 兼容服务 | 说明 |
|------|----------|------|
| Embedding | OpenAI、Azure OpenAI、任何兼容 `/v1/embeddings` 接口的服务 | 需返回 `{"data": [{"embedding": [...]}]}` 格式 |
| Reranker | Cohere Rerank、Jina Rerank | 需返回 `{"results": [{"index": 0, "relevance_score": 0.9}]}` 格式 |
| Query Rewriter | OpenAI Chat、任何兼容 Chat Completions 接口的服务 | 需返回 `{"choices": [{"message": {"content": "..."}}]}` 格式 |

## 命令行使用

### 索引文档

```bash
# 索引单个 Markdown 文件
uv run python -m rag index docs/api_reference.md

# 索引整个目录（递归查找所有 .md 文件）
uv run python -m rag index docs/

# 指定并发数
uv run python -m rag index docs/ --concurrency 10

# 使用配置文件
uv run python -m rag --config rag_config.yaml index docs/
```

### 检索查询

```bash
# 执行查询，返回 JSON 格式结果
uv run python -m rag query "如何使用 Linear 层"

# 指定返回数量
uv run python -m rag query "optimizer 初始化" --top-n 10

# 开启详细日志
uv run python -m rag -v query "attention mechanism"
```

查询输出示例：

```json
[
  {
    "chunk_id": "a3f2...",
    "content": "## torch.nn.Linear\n线性变换层...",
    "source_file": "docs/api.md",
    "chunk_type": "text",
    "heading_text": "## torch.nn.Linear",
    "start_line": 42,
    "score": 0.847231
  }
]
```

## 使用方法

### 基本用法

```python
import asyncio
from rag import RAGSystem, ConfigManager

async def main():
    # 从配置文件 + 环境变量加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config("rag_config.yaml")  # 或不传参数，仅用环境变量

    # 初始化系统
    system = RAGSystem(config)

    # 索引单个文档
    chunk_count = await system.index_document("docs/api_reference.md")
    print(f"索引完成，生成 {chunk_count} 个文档片段")

    # 索引整个目录
    results = await system.index_directory("docs/")
    for file_path, count in results.items():
        print(f"  {file_path}: {count} chunks")

    # 检索
    results = await system.retrieve("如何使用 torch.nn.Linear")
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk.content[:100]}...")

asyncio.run(main())
```

### 评估检索质量

```python
from rag import RAGSystem, ConfigManager, EvalRecord

async def evaluate():
    config = ConfigManager().load_config("rag_config.yaml")
    system = RAGSystem(config)

    # 准备评估数据集
    dataset = [
        EvalRecord(
            query="如何创建线性层",
            relevant_chunk_ids=["chunk_id_1", "chunk_id_2"]
        ),
        EvalRecord(
            query="Adam 优化器参数",
            relevant_chunk_ids=["chunk_id_3"]
        ),
    ]

    report = system.evaluate(dataset)
    print(f"Precision@K: {report.precision_at_k:.3f}")
    print(f"Recall@K:    {report.recall_at_k:.3f}")
    print(f"MRR@K:       {report.mrr_at_k:.3f}")

asyncio.run(evaluate())
```

## 运行测试

```bash
# 运行全部测试
uv run pytest rag/tests/ -v

# 运行特定模块测试
uv run pytest rag/tests/test_chunker_props.py -v

# 运行集成测试
uv run pytest rag/tests/test_integration.py -v
```

## 项目结构

```
rag/
├── __init__.py          # 公共 API 导出
├── __main__.py          # python -m rag 入口
├── cli.py               # 命令行工具（index / query）
├── config.py            # 配置管理（YAML + 环境变量）
├── chunker.py           # Markdown 文档分块
├── embedding.py         # 嵌入引擎（外部 API / ChromaDB 本地模型）
├── database.py          # ChromaDB 本地存储 + 向量搜索
├── query_rewriter.py    # 查询改写（调用外部 LLM）
├── dense_search.py      # 稠密向量检索
├── bm25_search.py       # BM25 稀疏关键词检索（支持持久化）
├── reranker.py          # 结果重排序（Reranker API / RRF 融合）
├── evaluator.py         # 检索质量评估
├── rag_system.py        # 系统主入口
├── exceptions.py        # 自定义异常
└── tests/               # 测试套件（225 个测试）
    ├── test_*_props.py  # 属性测试（hypothesis）
    ├── test_*.py        # 单元测试
    └── test_integration.py  # 集成测试
```

## 架构概览

```
索引流程: Markdown → Chunker → EmbeddingEngine → LocalDatabase + BM25Index
检索流程: Query → QueryRewriter → DenseSearch ∥ BM25Search → Reranker → Results
```

### 优雅降级

- **Embedding 未配置 API Key** → 使用 ChromaDB 内置模型（all-MiniLM-L6-v2，384 维，本地推理）
- **Reranker 未配置** → 使用 RRF（Reciprocal Rank Fusion）融合排序
- **Reranker API 不可用** → 回退到 RRF 融合排序
- **Query Rewriter 不可用** → 使用原始查询继续检索
- **代码 Chunk** → 跳过 Reranker，直接用 RRF 算分（避免 Reranker 误判代码相关性）
