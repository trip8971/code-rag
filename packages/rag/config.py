"""Configuration management for the RAG system."""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .exceptions import ConfigError


@dataclass
class ServiceConfig:
    """单个外部服务的配置"""

    url: str = ""
    api_key: str = ""
    timeout: int = 30  # 秒，有效范围 1-300
    max_retries: int = 3  # 有效范围 0-10
    model: str = ""  # 模型名称（可选，各服务有默认值）


@dataclass
class RAGConfig:
    """RAG 系统全局配置"""

    embedding: ServiceConfig = field(default_factory=ServiceConfig)
    reranker: ServiceConfig = field(default_factory=ServiceConfig)
    query_rewriter: ServiceConfig = field(default_factory=ServiceConfig)

    # Chunker 配置
    max_chunk_size: int = 1500
    chunk_overlap: int = 200

    # 检索配置
    dense_top_k: int = 20
    bm25_top_k: int = 10
    rerank_top_n: int = 5
    top_docs: int = 5  # 返回最相关文档数量

    # 回退权重配置
    dense_weight: float = 0.7
    bm25_weight: float = 0.3

    # 数据库配置
    db_path: str = "rag_index.db"
    embedding_dim: int = 1536

    # ChromaDB 配置
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "rag_chunks"

    # Embedding 批量配置
    embedding_batch_size: int = 20


class ConfigManager:
    """配置管理器，支持配置文件和环境变量"""

    # Mapping of service names to their environment variable prefixes
    _SERVICE_NAMES = ["embedding", "reranker", "query_rewriter"]

    # Environment variable prefix
    _ENV_PREFIX = "RAG"

    def load_config(self, config_path: Optional[str] = None) -> RAGConfig:
        """加载配置，环境变量优先于配置文件。

        Args:
            config_path: YAML 配置文件路径，为 None 时仅从环境变量加载。

        Returns:
            RAGConfig 实例。

        Raises:
            ConfigError: 当配置验证失败时抛出。
        """
        # Start with defaults
        config = RAGConfig()

        # Load from YAML file if provided
        if config_path is not None:
            file_config = self._load_from_yaml(config_path)
            config = self._apply_yaml_config(config, file_config)

        # Override with environment variables (env vars take priority)
        config = self._apply_env_vars(config)

        # Validate and raise if invalid
        self.validate_config(config)

        return config

    def validate_config(self, config: RAGConfig) -> list[str]:
        """验证配置项，返回错误信息列表。

        Raises:
            ConfigError: 当存在缺失项或无效项时抛出，包含所有错误信息。
        """
        missing_keys: list[str] = []
        invalid_items: dict[str, str] = {}

        # Check all services for required fields and valid values
        # All services are optional at config level:
        # - embedding: if not configured, ChromaDB default embedding is used
        # - reranker: if not configured, RRF fusion is used
        # - query_rewriter: if not configured, original query is used as-is
        services: dict[str, object] = {}
        optional_services = {
            "embedding": config.embedding,
            "reranker": config.reranker,
            "query_rewriter": config.query_rewriter,
        }

        for service_name, service_config in services.items():
            # Check required: url must be non-empty
            if not service_config.url:
                missing_keys.append(f"{service_name}.url")

            # Check required: api_key must be non-empty
            if not service_config.api_key:
                missing_keys.append(f"{service_name}.api_key")

        # Merge all services for range validation (including optional ones)
        all_services = {**services, **optional_services}

        for service_name, service_config in all_services.items():
            # Validate URL format (only if non-empty)
            # local:// 是本地模型 sentinel，不走 HTTP，跳过 URL 格式校验
            if service_config.url and not service_config.url.startswith("local://"):
                if not (
                    service_config.url.startswith("http://")
                    or service_config.url.startswith("https://")
                ):
                    invalid_items[f"{service_name}.url"] = (
                        f"must start with http://, https://, or local://, got '{service_config.url}'"
                    )

            # Validate timeout range 1-300
            if not (1 <= service_config.timeout <= 300):
                invalid_items[f"{service_name}.timeout"] = (
                    f"must be in range 1-300, got {service_config.timeout}"
                )

            # Validate max_retries range 0-10
            if not (0 <= service_config.max_retries <= 10):
                invalid_items[f"{service_name}.max_retries"] = (
                    f"must be in range 0-10, got {service_config.max_retries}"
                )

        # Collect all error messages for return value
        errors: list[str] = []
        for key in missing_keys:
            errors.append(f"Missing required key: {key}")
        for key, reason in invalid_items.items():
            errors.append(f"Invalid value for {key}: {reason}")

        # Raise ConfigError if any issues found
        if missing_keys or invalid_items:
            raise ConfigError(missing_keys=missing_keys, invalid_items=invalid_items)

        return errors

    def _load_from_yaml(self, config_path: str) -> dict:
        """从 YAML 文件加载配置。"""
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data is not None else {}

    def _apply_yaml_config(self, config: RAGConfig, yaml_data: dict) -> RAGConfig:
        """将 YAML 配置应用到 RAGConfig。"""
        for service_name in self._SERVICE_NAMES:
            if service_name in yaml_data:
                service_data = yaml_data[service_name]
                service_config = getattr(config, service_name)

                if "url" in service_data:
                    service_config.url = str(service_data["url"])
                if "api_key" in service_data:
                    service_config.api_key = str(service_data["api_key"])
                if "timeout" in service_data:
                    service_config.timeout = int(service_data["timeout"])
                if "max_retries" in service_data:
                    service_config.max_retries = int(service_data["max_retries"])
                if "model" in service_data:
                    service_config.model = str(service_data["model"])

        # Apply chunker config
        if "chunker" in yaml_data:
            chunker_data = yaml_data["chunker"]
            if "max_chunk_size" in chunker_data:
                config.max_chunk_size = int(chunker_data["max_chunk_size"])
            if "overlap" in chunker_data:
                config.chunk_overlap = int(chunker_data["overlap"])

        # Apply retrieval config
        if "retrieval" in yaml_data:
            retrieval_data = yaml_data["retrieval"]
            if "dense_top_k" in retrieval_data:
                config.dense_top_k = int(retrieval_data["dense_top_k"])
            if "bm25_top_k" in retrieval_data:
                config.bm25_top_k = int(retrieval_data["bm25_top_k"])
            if "rerank_top_n" in retrieval_data:
                config.rerank_top_n = int(retrieval_data["rerank_top_n"])
            if "dense_weight" in retrieval_data:
                config.dense_weight = float(retrieval_data["dense_weight"])
            if "bm25_weight" in retrieval_data:
                config.bm25_weight = float(retrieval_data["bm25_weight"])
            if "top_docs" in retrieval_data:
                config.top_docs = int(retrieval_data["top_docs"])

        # Apply database config
        if "database" in yaml_data:
            db_data = yaml_data["database"]
            if "path" in db_data:
                config.db_path = str(db_data["path"])
            if "embedding_dim" in db_data:
                config.embedding_dim = int(db_data["embedding_dim"])

        # Top-level overrides
        if "embedding_batch_size" in yaml_data:
            config.embedding_batch_size = int(yaml_data["embedding_batch_size"])

        return config

    def _apply_env_vars(self, config: RAGConfig) -> RAGConfig:
        """应用环境变量覆盖，环境变量优先于配置文件。

        环境变量格式：
        - RAG_EMBEDDING_URL, RAG_EMBEDDING_API_KEY, RAG_EMBEDDING_TIMEOUT, RAG_EMBEDDING_MAX_RETRIES
        - RAG_RERANKER_URL, RAG_RERANKER_API_KEY, RAG_RERANKER_TIMEOUT, RAG_RERANKER_MAX_RETRIES
        - RAG_QUERY_REWRITER_URL, RAG_QUERY_REWRITER_API_KEY, RAG_QUERY_REWRITER_TIMEOUT, RAG_QUERY_REWRITER_MAX_RETRIES
        """
        for service_name in self._SERVICE_NAMES:
            service_config = getattr(config, service_name)
            env_prefix = f"{self._ENV_PREFIX}_{service_name.upper()}"

            # URL
            env_url = os.environ.get(f"{env_prefix}_URL")
            if env_url is not None:
                service_config.url = env_url

            # API Key
            env_api_key = os.environ.get(f"{env_prefix}_API_KEY")
            if env_api_key is not None:
                service_config.api_key = env_api_key

            # Timeout
            env_timeout = os.environ.get(f"{env_prefix}_TIMEOUT")
            if env_timeout is not None:
                try:
                    service_config.timeout = int(env_timeout)
                except ValueError:
                    # Keep the existing value; validation will catch invalid values
                    pass

            # Max retries
            env_max_retries = os.environ.get(f"{env_prefix}_MAX_RETRIES")
            if env_max_retries is not None:
                try:
                    service_config.max_retries = int(env_max_retries)
                except ValueError:
                    # Keep the existing value; validation will catch invalid values
                    pass

            # Model
            env_model = os.environ.get(f"{env_prefix}_MODEL")
            if env_model is not None:
                service_config.model = env_model

        return config
