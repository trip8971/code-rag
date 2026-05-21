"""Reciprocal Rank Fusion (RRF) reranker implementation."""

import logging

from ..chunker import Chunk
from ..interfaces import RerankerBase

logger = logging.getLogger(__name__)


class RRFReranker(RerankerBase):
    """基于 Reciprocal Rank Fusion (RRF) 的纯本地重排序器。

    不依赖外部服务，直接使用 RRF 公式融合稠密检索和 BM25 检索结果。
    """

    def __init__(
        self,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    async def rerank(
        self,
        query: str,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_n: int = 5,
    ) -> list[tuple[Chunk, float]]:
        logger.info("Using RRF fusion for reranking")
        rrf_results = self._rrf_fusion(dense_results, bm25_results)
        return rrf_results[:top_n]

    def _rrf_fusion(
        self,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Reciprocal Rank Fusion (RRF) 融合排序。

        RRF 公式：score(d) = Σ weight_i / (k + rank_i(d))
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0)
                + self.dense_weight / (self.rrf_k + rank)
            )
            chunk_map.setdefault(chunk.chunk_id, chunk)

        for rank, (chunk, _score) in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0)
                + self.bm25_weight / (self.rrf_k + rank)
            )
            chunk_map.setdefault(chunk.chunk_id, chunk)

        results: list[tuple[Chunk, float]] = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in rrf_scores.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
