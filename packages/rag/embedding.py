"""Embedding engine factory and re-exports.

实现类位于 rag.providers.embedding_* 模块中。
本模块提供工厂函数和向后兼容的导入路径。
"""

import asyncio  # noqa: F401 — kept for backward-compat monkeypatch in tests
import logging

from .config import ServiceConfig
from .interfaces import EmbeddingBase
from .providers.embedding_local import CHROMADB_DEFAULT_DIM, LocalEmbeddingEngine
from .providers.embedding_openai import OpenAIEmbeddingEngine
from .providers.embedding_voyage import VoyageEmbeddingEngine

logger = logging.getLogger(__name__)


def create_embedding_engine(
    config: ServiceConfig, embedding_dim: int, batch_size: int = 20
) -> EmbeddingBase:
    """工厂函数：根据配置创建对应的 EmbeddingEngine 实例。

    优先级：
    - URL 以 "local://bge" 开头 → BGEEmbeddingEngine（懒加载，需 `pip install code-rag[bge]`）
    - api_key 为空 → LocalEmbeddingEngine（ChromaDB 默认 MiniLM）
    - URL 包含 "voyage" → VoyageEmbeddingEngine
    - 其他 → OpenAIEmbeddingEngine
    """
    url_lower = config.url.lower() if config.url else ""

    if url_lower.startswith("local://bge"):
        # 懒加载：BGE provider 依赖 sentence-transformers / torch，是可选依赖。
        # 仅当用户配置了 local://bge URL 时才 import，避免给纯 API 用户增加重型依赖。
        try:
            from rag.providers.embedding_bge import BGEEmbeddingEngine
        except ImportError as e:
            raise RuntimeError(
                "BGE embedding requires sentence-transformers. "
                "Install with: pip install 'code-rag[bge]'"
            ) from e
        model = config.model or "BAAI/bge-m3"
        return BGEEmbeddingEngine(
            model_name=model,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )

    if not config.api_key:
        return LocalEmbeddingEngine()

    if "voyage" in url_lower:
        return VoyageEmbeddingEngine(config, embedding_dim, batch_size)

    return OpenAIEmbeddingEngine(config, embedding_dim, batch_size)


# Backward-compatible alias
EmbeddingEngine = OpenAIEmbeddingEngine

__all__ = [
    "EmbeddingBase",
    "OpenAIEmbeddingEngine",
    "VoyageEmbeddingEngine",
    "LocalEmbeddingEngine",
    "EmbeddingEngine",
    "CHROMADB_DEFAULT_DIM",
    "create_embedding_engine",
]
