"""No-op query rewriter (passthrough)."""

from ..interfaces import QueryRewriterBase


class NoopQueryRewriter(QueryRewriterBase):
    """空实现，直接返回原始查询（未配置时使用）。"""

    async def rewrite(self, query: str) -> str:
        return query
