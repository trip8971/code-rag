"""OpenAI-compatible embedding API implementation."""

import asyncio
import logging
import os

import httpx

from ..config import ServiceConfig
from ..exceptions import EmbeddingError
from ..interfaces import EmbeddingBase

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# ChromaDB default embedding dimension (all-MiniLM-L6-v2)
CHROMADB_DEFAULT_DIM = 384


class OpenAIEmbeddingEngine(EmbeddingBase):
    """基于 OpenAI 兼容 API 的嵌入引擎。

    支持 OpenAI、Gemini 等兼容 /embeddings 端点的服务。
    """

    def __init__(self, config: ServiceConfig, embedding_dim: int, batch_size: int = 20):
        self._config = config
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size
        self._model = config.model or DEFAULT_EMBEDDING_MODEL

        self._proxy = (
            os.environ.get("RAG_EMBEDDING_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or None
        )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=config.timeout),
            proxy=self._proxy,
        )
        logger.info(
            "OpenAIEmbeddingEngine initialized: model=%s, url=%s, dim=%d, proxy=%s",
            self._model,
            config.url,
            self._embedding_dim,
            self._proxy or "(none)",
        )

    async def embed_texts(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response_data = await self._call_api(batch, input_type=input_type)

            for item in response_data["data"]:
                vector = item["embedding"]
                if not self._validate_dimension(vector):
                    raise EmbeddingError(
                        f"Dimension mismatch: expected {self._embedding_dim}, "
                        f"got {len(vector)}",
                        retries_attempted=0,
                    )
                all_embeddings.append(vector)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query], input_type="query")
        return results[0]

    async def close(self) -> None:
        await self._client.aclose()

    def _validate_dimension(self, vector: list[float]) -> bool:
        """验证向量维度是否与配置一致。"""
        return len(vector) == self._embedding_dim

    async def _call_api(self, texts: list[str], input_type: str = "document") -> dict:
        """调用外部嵌入 API，带指数退避重试。"""
        max_retries = self._config.max_retries
        initial_interval = 1.0
        backoff_factor = 2.0
        last_error: Exception | None = None

        payload: dict = {
            "input": texts,
            "model": self._model,
        }

        # OpenAI 兼容路径：仅在维度非默认时请求降维
        if self._embedding_dim and self._embedding_dim not in (CHROMADB_DEFAULT_DIM,):
            payload["dimensions"] = self._embedding_dim

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    self._config.url,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < max_retries and (
                    e.response.status_code == 429 or e.response.status_code >= 500
                ):
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = initial_interval * (backoff_factor**attempt)
                    else:
                        base = 5.0 if e.response.status_code == 429 else initial_interval
                        wait_time = base * (backoff_factor**attempt)
                    logger.warning(
                        "Embedding API %d, retry %d/%d after %.1fs",
                        e.response.status_code, attempt + 1, max_retries, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    if e.response.status_code != 429 and e.response.status_code < 500:
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(initial_interval * (backoff_factor**attempt))
            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = initial_interval * (backoff_factor**attempt)
                    await asyncio.sleep(wait_time)

        raise EmbeddingError(
            f"Embedding API request failed after {max_retries} retries: {last_error}",
            retries_attempted=max_retries,
        )
