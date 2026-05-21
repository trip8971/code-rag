"""Query rewriter factory and re-exports.

实现类位于 rag.providers.query_rewriter_* 模块中。
本模块提供工厂函数和向后兼容的导入路径。
"""

import logging

from .config import ServiceConfig
from .interfaces import QueryRewriterBase
from .providers.query_rewriter_gemini import GeminiQueryRewriter
from .providers.query_rewriter_noop import NoopQueryRewriter
from .providers.query_rewriter_openai import OpenAIQueryRewriter

logger = logging.getLogger(__name__)

# Gemini API URL 特征
_GEMINI_URL_MARKERS = ("generativelanguage.googleapis.com",)


def create_query_rewriter(config: ServiceConfig) -> QueryRewriterBase:
    """工厂函数：根据配置创建对应的 QueryRewriter 实例。

    - URL 包含 generativelanguage.googleapis.com → GeminiQueryRewriter
    - URL 非空且不匹配 Gemini → OpenAIQueryRewriter
    - URL 为空 → NoopQueryRewriter
    """
    if not config.url:
        logger.info("No query rewriter URL configured, using NoopQueryRewriter.")
        return NoopQueryRewriter()

    if any(marker in config.url for marker in _GEMINI_URL_MARKERS):
        return GeminiQueryRewriter(config)

    return OpenAIQueryRewriter(config)


# Backward-compatible alias
QueryRewriter = OpenAIQueryRewriter

__all__ = [
    "QueryRewriterBase",
    "OpenAIQueryRewriter",
    "GeminiQueryRewriter",
    "NoopQueryRewriter",
    "QueryRewriter",
    "create_query_rewriter",
]
