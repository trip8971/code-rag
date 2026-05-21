"""Code RAG (Retrieval-Augmented Generation) System.

Public API exports for the RAG package.
"""

from .chunker import Chunk
from .config import ConfigManager, RAGConfig
from .evaluator import EvalRecord, EvalReport
from .exceptions import (
    ConfigError,
    EmbeddingError,
    EvalDatasetError,
    RAGError,
)
from .rag_system import RAGSystem

__all__ = [
    # Core system
    "RAGSystem",
    "RAGConfig",
    "ConfigManager",
    # Data models
    "Chunk",
    "EvalRecord",
    "EvalReport",
    # Exceptions
    "RAGError",
    "ConfigError",
    "EmbeddingError",
    "EvalDatasetError",
]
