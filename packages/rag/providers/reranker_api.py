"""External Reranker API implementation."""

import logging

import httpx

from ..chunker import Chunk, ChunkType
from ..config import ServiceConfig
from ..interfaces import RerankerBase
from .reranker_rrf import RRFReranker

logger = logging.getLogger(__name__)


class APIReranker(RerankerBase):
    """基于外部 Reranker API 的重排序器。

    排序策略：
    - 文本 chunk 调用 API 重排序
    - 代码 chunk 跳过 reranker，使用 RRF 分数（避免 reranker 误判代码相关性）
    - 两组结果合并后按分数排序
    - API 失败时回退到加权融合（_fallback_score）
    """

    def __init__(
        self,
        config: ServiceConfig,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self._config = config
        self.config = config  # backward compat for tests
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._rrf = RRFReranker(dense_weight, bm25_weight, rrf_k)

    async def rerank(
        self,
        query: str,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_n: int = 5,
    ) -> list[tuple[Chunk, float]]:
        # 分离代码 chunk 和文本 chunk
        dense_code = [(c, s) for c, s in dense_results if c.chunk_type == ChunkType.CODE]
        dense_text = [(c, s) for c, s in dense_results if c.chunk_type != ChunkType.CODE]
        bm25_code = [(c, s) for c, s in bm25_results if c.chunk_type == ChunkType.CODE]
        bm25_text = [(c, s) for c, s in bm25_results if c.chunk_type != ChunkType.CODE]

        # 代码 chunk 直接用 RRF
        code_results = self._rrf._rrf_fusion(dense_code, bm25_code)

        # 文本 chunk 走 reranker API
        text_results = await self._rerank_text(query, dense_text, bm25_text, top_n)

        # 如果只有一组结果，直接返回（无需归一化）
        if not code_results:
            text_results.sort(key=lambda x: x[1], reverse=True)
            return text_results[:top_n]
        if not text_results:
            code_results.sort(key=lambda x: x[1], reverse=True)
            return code_results[:top_n]

        # 合并两组结果，归一化后统一排序
        merged = self._merge_code_and_text(code_results, text_results)
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_n]

    async def _rerank_text(
        self,
        query: str,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_n: int,
    ) -> list[tuple[Chunk, float]]:
        """对文本 chunk 调用外部 Reranker API。失败时回退到加权融合。"""
        merged_chunks = self._merge_and_deduplicate(dense_results, bm25_results)

        if not merged_chunks:
            return []

        try:
            documents = [chunk.embedding_text for chunk in merged_chunks]
            payload: dict = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
            if self._config.model:
                payload["model"] = self._config.model
            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                response = await client.post(
                    self._config.url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

            data = response.json()
            results: list[tuple[Chunk, float]] = []
            for item in data["results"]:
                index = item["index"]
                score = item["relevance_score"]
                results.append((merged_chunks[index], score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            logger.warning(
                "Reranker API failed, falling back to weighted fusion for text chunks: %s",
                str(e),
            )
            return self._fallback_score(dense_results, bm25_results)

    def _fallback_score(
        self,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """加权融合排序：score = dense_weight × dense_score + bm25_weight × bm25_score。"""
        dense_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for chunk, score in dense_results:
            dense_score_map.setdefault(chunk.chunk_id, score)
            chunk_map.setdefault(chunk.chunk_id, chunk)

        for chunk, score in bm25_results:
            bm25_score_map.setdefault(chunk.chunk_id, score)
            chunk_map.setdefault(chunk.chunk_id, chunk)

        results: list[tuple[Chunk, float]] = []
        for chunk_id, chunk in chunk_map.items():
            d_score = dense_score_map.get(chunk_id, 0.0)
            b_score = bm25_score_map.get(chunk_id, 0.0)
            fused = self.dense_weight * d_score + self.bm25_weight * b_score
            results.append((chunk, fused))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _merge_code_and_text(
        self,
        code_results: list[tuple[Chunk, float]],
        text_results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """合并代码和文本结果，归一化分数到 [0, 1] 区间。"""
        normalized: list[tuple[Chunk, float]] = []

        if code_results:
            code_max = max(s for _, s in code_results)
            if code_max > 0:
                for chunk, score in code_results:
                    normalized.append((chunk, score / code_max))
            else:
                normalized.extend(code_results)

        if text_results:
            text_max = max(s for _, s in text_results)
            if text_max > 0:
                for chunk, score in text_results:
                    normalized.append((chunk, score / text_max))
            else:
                normalized.extend(text_results)

        return normalized

    def _merge_and_deduplicate(
        self,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
    ) -> list[Chunk]:
        """合并两组结果并按 Chunk ID 去重。"""
        seen: set[str] = set()
        merged: list[Chunk] = []

        for chunk, _score in dense_results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)

        for chunk, _score in bm25_results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)

        return merged
