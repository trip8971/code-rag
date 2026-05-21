"""Local BGE-M3 embedding engine using sentence-transformers."""

import asyncio
import logging
from typing import Optional

from ..interfaces import EmbeddingBase

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


class BGEEmbeddingEngine(EmbeddingBase):
    """基于 sentence-transformers 的本地 BGE 嵌入引擎。

    首次使用会从 HuggingFace 下载模型（bge-m3 约 2.3GB）。
    Apple Silicon 上自动使用 MPS，否则回退 CPU。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_dim: int = DEFAULT_DIM,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info(
            "Loading BGE model %s on device=%s (first run downloads ~2.3GB)",
            model_name, device,
        )
        self._model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size
        self._device = device
        logger.info(
            "BGEEmbeddingEngine initialized: model=%s dim=%d device=%s",
            model_name, embedding_dim, device,
        )

    async def embed_texts(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        # sentence-transformers 是同步的，扔到线程池避免阻塞事件循环
        return await asyncio.to_thread(self._encode_batch, texts)

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query], input_type="query")
        return results[0]

    async def close(self) -> None:
        pass

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        logger.info(
            "BGE encoding %d texts (batch_size=%d, device=%s)",
            len(texts), self._batch_size, self._device,
        )
        # normalize_embeddings=True 用于 cosine 相似度场景
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) >= 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vectors]
