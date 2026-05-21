"""Custom exception hierarchy for the RAG system."""


class RAGError(Exception):
    """RAG 系统基础异常"""

    pass


class ConfigError(RAGError):
    """配置相关错误

    Raised when configuration is missing required keys or contains invalid values.
    """

    def __init__(
        self,
        missing_keys: list[str] | None = None,
        invalid_items: dict[str, str] | None = None,
    ):
        self.missing_keys = missing_keys or []
        self.invalid_items = invalid_items or {}

        # Build an informative error message
        parts: list[str] = []
        if self.missing_keys:
            keys_str = ", ".join(self.missing_keys)
            parts.append(f"Missing required configuration keys: [{keys_str}]")
        if self.invalid_items:
            items_str = "; ".join(
                f"{key}: {reason}" for key, reason in self.invalid_items.items()
            )
            parts.append(f"Invalid configuration values: {{{items_str}}}")

        message = ". ".join(parts) if parts else "Configuration error"
        super().__init__(message)


class EmbeddingError(RAGError):
    """嵌入服务错误

    Raised when the embedding service fails after exhausting retries.
    """

    def __init__(self, message: str, retries_attempted: int):
        super().__init__(message)
        self.retries_attempted = retries_attempted


class EvalDatasetError(RAGError):
    """评估数据集格式错误

    Raised when the evaluation dataset has invalid format or content.
    """

    def __init__(self, message: str, record_index: int | None = None):
        super().__init__(message)
        self.record_index = record_index
