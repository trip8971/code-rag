"""OpenAI Chat Completions API query rewriter implementation."""

import logging
import os

import httpx

from ..config import ServiceConfig
from ..interfaces import QueryRewriterBase
from ..prompts import REWRITER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAIQueryRewriter(QueryRewriterBase):
    """基于 OpenAI Chat Completions API 的查询改写器。"""

    DEFAULT_MODEL = "gpt-4"

    def __init__(self, config: ServiceConfig):
        self._config = config
        self._model = config.model or self.DEFAULT_MODEL
        self._proxy = self._resolve_proxy()

        logger.info(
            "OpenAIQueryRewriter initialized: model=%s, url=%s, proxy=%s",
            self._model,
            config.url,
            self._proxy or "(none)",
        )

    async def rewrite(self, query: str) -> str:
        if not self._is_config_valid():
            logger.warning(
                "OpenAIQueryRewriter config invalid, skipping rewrite."
            )
            return query

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout, proxy=self._proxy
            ) as client:
                response = await client.post(
                    self._config.url,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                            {"role": "user", "content": query},
                        ],
                        # 固定输出 → 同一查询每次得到相同改写，
                        # 避免下游召回结果抖动
                        "temperature": 0,
                        "seed": 42,
                    },
                )
                response.raise_for_status()
                data = response.json()
                rewritten = data["choices"][0]["message"]["content"].strip()
                result = self._truncate_result(rewritten)
                logger.info("Query rewritten: '%s' -> '%s'", query, result)
                return result
        except Exception as e:
            logger.warning("OpenAIQueryRewriter failed: %s, using original query.", e)
            return query

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
