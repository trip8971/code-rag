"""Local ChromaDB default embedding (all-MiniLM-L6-v2) implementation."""

import logging

from ..interfaces import EmbeddingBase

logger = logging.getLogger(__name__)

# ChromaDB default embedding dimension (all-MiniLM-L6-v2)
CHROMADB_DEFAULT_DIM = 384


class LocalEmbeddingEngine(EmbeddingBase):
    """基于 ChromaDB 内置 all-MiniLM-L6-v2 模型的本地嵌入引擎。"""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._local_ef = DefaultEmbeddingFunction()
        logger.info(
            "LocalEmbeddingEngine initialized: using ChromaDB default "
            "embedding (all-MiniLM-L6-v2, dim=%d)",
            CHROMADB_DEFAULT_DIM,
        )

    async def embed_texts(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._local_ef(texts)
        return [[float(x) for x in e] for e in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query], input_type="query")
        return results[0]

    async def close(self) -> None:
        pass  # 本地模型无需释放资源
