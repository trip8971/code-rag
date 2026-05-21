"""Abstract base classes for external service integrations.

定义 QueryRewriter、Embedding、Reranker 三个外部 API 的抽象接口，
使 RAG 系统核心逻辑与具体 Provider 解耦。
"""

from abc import ABC, abstractmethod

from .chunker import Chunk


class QueryRewriterBase(ABC):
    """查询改写器抽象接口。"""

    @abstractmethod
    async def rewrite(self, query: str) -> str:
        """改写查询以提升检索召回率。

        Args:
            query: 用户原始查询文本。

        Returns:
            改写后的查询文本。实现应保证在失败时返回原始查询。
        """
        ...


class EmbeddingBase(ABC):
    """文本向量化抽象接口。"""

    @abstractmethod
    async def embed_texts(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """批量将文本转换为向量。

        Args:
            texts: 待向量化的文本列表。
            input_type: 输入类型提示，"document" 或 "query"。

        Returns:
            向量列表，每个向量为 float 列表。

        Raises:
            EmbeddingError: 向量化失败时抛出。
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """将单个查询文本转换为向量。

        Args:
            query: 待向量化的查询文本。

        Returns:
            查询的向量表示。

        Raises:
            EmbeddingError: 向量化失败时抛出。
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """释放资源（HTTP 客户端等）。"""
        ...


class RerankerBase(ABC):
    """重排序器抽象接口。"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_n: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """合并、去重、重排序检索结果。

        Args:
            query: 用户查询文本。
            dense_results: 稠密检索结果列表 [(Chunk, score), ...]。
            bm25_results: BM25 检索结果列表 [(Chunk, score), ...]。
            top_n: 返回的最大结果数量。

        Returns:
            重排序后的 Top-N 结果列表 [(Chunk, score), ...]。
        """
        ...
