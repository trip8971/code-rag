"""Google Gemini generateContent API query rewriter implementation."""

import logging
import os

import httpx

from ..config import ServiceConfig
from ..interfaces import QueryRewriterBase
from ..prompts import REWRITER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GeminiQueryRewriter(QueryRewriterBase):
    """基于 Google Gemini generateContent API 的查询改写器。"""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, config: ServiceConfig):
        self._config = config
        self._model = config.model or self.DEFAULT_MODEL
        self._proxy = self._resolve_proxy()

        logger.info(
            "GeminiQueryRewriter initialized: model=%s, url=%s, proxy=%s",
            self._model,
            config.url,
            self._proxy or "(none)",
        )

    async def rewrite(self, query: str) -> str:
        if not self._is_config_valid():
            logger.warning(
                "GeminiQueryRewriter config invalid, skipping rewrite."
            )
            return query

        try:
            url = self._build_url()
            async with httpx.AsyncClient(
                timeout=self._config.timeout, proxy=self._proxy
            ) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    params={"key": self._config.api_key},
                    json={
                        "system_instruction": {
                            "parts": [{"text": REWRITER_SYSTEM_PROMPT}],
                        },
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": query}],
                            }
                        ],
                    },
                )
                response.raise_for_status()
                data = response.json()
                rewritten = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = self._truncate_result(rewritten)
                logger.info("Query rewritten: '%s' -> '%s'", query, result)
                return result
        except Exception as e:
            logger.warning("GeminiQueryRewriter failed: %s, using original query.", e)
            return query

    def _build_url(self) -> str:
        url = self._config.url.rstrip("/")
        if ":generateContent" in url:
            return url
        return f"{url}/v1beta/models/{self._model}:generateContent"

    def _is_config_valid(self) -> bool:
        url = self._config.url
        if not self._config.api_key or not self._config.api_key.strip():
            return False
        if not url or not url.strip():
            return False
        url = url.strip()
        return url.startswith("http://") or url.startswith("https://")

    @staticmethod
    def _resolve_proxy() -> str | None:
        return (
            os.environ.get("RAG_QUERY_REWRITER_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or None
        )

    @staticmethod
    def _truncate_result(text: str, max_length: int = 500) -> str:
        return text[:max_length] if len(text) > max_length else text
