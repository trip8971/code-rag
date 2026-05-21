"""Voyage AI embedding API implementation."""

import asyncio
import logging
import os

import httpx

from ..config import ServiceConfig
from ..exceptions import EmbeddingError
from ..interfaces import EmbeddingBase

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


class VoyageEmbeddingEngine(EmbeddingBase):
    """基于 Voyage AI API 的嵌入引擎。"""

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
            "VoyageEmbeddingEngine initialized: model=%s, url=%s, dim=%d, proxy=%s",
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

        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        all_embeddings: list[list[float]] = []
        for batch_idx, i in enumerate(range(0, len(texts), self._batch_size), start=1):
            batch = texts[i : i + self._batch_size]
            logger.info(
                "Voyage embed batch %d/%d: size=%d chars=%d",
                batch_idx, total_batches, len(batch), sum(len(t) for t in batch),
            )
            response_data = await self._call_api(batch, input_type=input_type)
            usage = response_data.get("usage", {})
            logger.info(
                "Voyage embed batch %d/%d done: tokens=%s",
                batch_idx, total_batches, usage.get("total_tokens", "?"),
            )

            for item in response_data["data"]:
                vector = item["embedding"]
                if len(vector) != self._embedding_dim:
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

    async def _call_api(self, texts: list[str], input_type: str = "document") -> dict:
        """调用 Voyage AI API，带指数退避重试。"""
        max_retries = self._config.max_retries
        initial_interval = 1.0
        backoff_factor = 2.0
        last_error: Exception | None = None

        payload: dict = {
            "input": texts,
            "model": self._model,
            "input_type": input_type,
        }
        if self._embedding_dim:
            payload["output_dimension"] = self._embedding_dim

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
