"""Reranker factory and re-exports.

实现类位于 rag.providers.reranker_* 模块中。
本模块提供工厂函数和向后兼容的导入路径。
"""

import logging

from .config import ServiceConfig
from .interfaces import RerankerBase
from .providers.reranker_api import APIReranker
from .providers.reranker_rrf import RRFReranker

logger = logging.getLogger(__name__)


def create_reranker(
    config: ServiceConfig,
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> RerankerBase:
    """工厂函数：根据配置创建对应的 Reranker 实例。

    - URL 非空 → APIReranker（调用外部服务，代码 chunk 回退 RRF）
    - URL 为空 → RRFReranker（纯本地 RRF 融合）
    """
    if config.url:
        return APIReranker(config, dense_weight, bm25_weight)

    return RRFReranker(dense_weight, bm25_weight)


# Backward-compatible alias
Reranker = APIReranker

__all__ = [
    "RerankerBase",
    "APIReranker",
    "RRFReranker",
    "Reranker",
    "create_reranker",
]
