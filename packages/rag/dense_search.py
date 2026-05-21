"""Dense vector search using cosine similarity."""

from .chunker import Chunk
from .database import LocalDatabase
from .exceptions import EmbeddingError
from .interfaces import EmbeddingBase


class DenseSearch:
    """基于向量相似度的稠密语义检索。

    将查询文本通过 EmbeddingBase 转换为向量，然后在 LocalDatabase 中
    执行余弦相似度搜索，返回最相似的 Top-K 结果。

    当 EmbeddingEngine 不可用时，EmbeddingError 会传播给调用方，
    指明向量化失败原因。
    """

    def __init__(self, embedding_engine: EmbeddingBase, database: LocalDatabase):
        """初始化稠密向量检索。

        Args:
            embedding_engine: 嵌入引擎，用于将查询文本转换为向量。
            database: 本地数据库，用于执行余弦相似度搜索。
        """
        self._embedding_engine = embedding_engine
        self._database = database

    async def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """执行稠密向量检索，返回 (Chunk, score) 列表。

        将查询文本通过 EmbeddingEngine 向量化，然后在数据库中执行
        余弦相似度搜索，返回相似度最高的前 top_k 个结果。

        Args:
            query: 用户查询文本。
            top_k: 返回的最大结果数量，默认 20，有效范围 1-200。

        Returns:
            按相似度降序排列的 (Chunk, score) 列表。
            如果数据库中符合条件的 Chunk 数量少于 top_k，
            则返回所有可用的 Chunk。

        Raises:
            EmbeddingError: 当 EmbeddingEngine 服务不可用或请求超时时抛出，
                包含向量化失败原因信息。
        """
        # Step 1: 将查询文本向量化
        # 如果 EmbeddingEngine 不可用，EmbeddingError 会自动传播给调用方
        query_vector = await self._embedding_engine.embed_query(query)

        # Step 2: 在数据库中执行余弦相似度搜索
        results = self._database.search_by_cosine(query_vector, top_k)

        return results
