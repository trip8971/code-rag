"""Unit tests for ConfigManager, ServiceConfig, and RAGConfig."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from rag.config import ConfigManager, RAGConfig, ServiceConfig
from rag.exceptions import ConfigError


class TestServiceConfig:
    """Tests for ServiceConfig dataclass."""

    def test_default_values(self):
        config = ServiceConfig()
        assert config.url == ""
        assert config.api_key == ""
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_custom_values(self):
        config = ServiceConfig(
            url="https://api.example.com",
            api_key="test-key",
            timeout=60,
            max_retries=5,
        )
        assert config.url == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.timeout == 60
        assert config.max_retries == 5


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_default_values(self):
        config = RAGConfig()
        assert isinstance(config.embedding, ServiceConfig)
        assert isinstance(config.reranker, ServiceConfig)
        assert isinstance(config.query_rewriter, ServiceConfig)
        assert config.max_chunk_size == 1500
        assert config.chunk_overlap == 200
        assert config.dense_top_k == 20
        assert config.bm25_top_k == 10
        assert config.rerank_top_n == 5
        assert config.dense_weight == 0.7
        assert config.bm25_weight == 0.3
        assert config.db_path == "rag_index.db"
        assert config.embedding_dim == 1536


class TestConfigManagerLoadConfig:
    """Tests for ConfigManager.load_config()."""

    def _make_valid_yaml(self, tmp_path: Path) -> Path:
        """Create a valid YAML config file."""
        config_data = {
            "embedding": {
                "url": "https://api.openai.com/v1/embeddings",
                "api_key": "sk-embed-key",
                "timeout": 30,
                "max_retries": 3,
            },
            "reranker": {
                "url": "https://api.cohere.ai/v1/rerank",
                "api_key": "cohere-key",
                "timeout": 30,
                "max_retries": 3,
            },
            "query_rewriter": {
                "url": "https://api.openai.com/v1/chat/completions",
                "api_key": "sk-rewriter-key",
                "timeout": 10,
                "max_retries": 3,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        return config_file

    def test_load_from_yaml(self, tmp_path):
        config_file = self._make_valid_yaml(tmp_path)
        manager = ConfigManager()
        config = manager.load_config(str(config_file))

        assert config.embedding.url == "https://api.openai.com/v1/embeddings"
        assert config.embedding.api_key == "sk-embed-key"
        assert config.reranker.url == "https://api.cohere.ai/v1/rerank"
        assert config.query_rewriter.timeout == 10

    def test_env_vars_override_yaml(self, tmp_path, monkeypatch):
        config_file = self._make_valid_yaml(tmp_path)
        monkeypatch.setenv("RAG_EMBEDDING_URL", "https://override.example.com/embed")
        monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "env-key-override")

        manager = ConfigManager()
        config = manager.load_config(str(config_file))

        # Env vars should override YAML values
        assert config.embedding.url == "https://override.example.com/embed"
        assert config.embedding.api_key == "env-key-override"
        # Non-overridden values should remain from YAML
        assert config.reranker.url == "https://api.cohere.ai/v1/rerank"

    def test_load_from_env_vars_only(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_URL", "https://embed.example.com")
        monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "embed-key")
        monkeypatch.setenv("RAG_RERANKER_URL", "https://rerank.example.com")
        monkeypatch.setenv("RAG_RERANKER_API_KEY", "rerank-key")
        monkeypatch.setenv("RAG_QUERY_REWRITER_URL", "https://rewrite.example.com")
        monkeypatch.setenv("RAG_QUERY_REWRITER_API_KEY", "rewrite-key")

        manager = ConfigManager()
        config = manager.load_config()

        assert config.embedding.url == "https://embed.example.com"
        assert config.embedding.api_key == "embed-key"
        assert config.reranker.url == "https://rerank.example.com"
        assert config.query_rewriter.url == "https://rewrite.example.com"

    def test_env_var_timeout_override(self, tmp_path, monkeypatch):
        config_file = self._make_valid_yaml(tmp_path)
        monkeypatch.setenv("RAG_EMBEDDING_TIMEOUT", "120")

        manager = ConfigManager()
        config = manager.load_config(str(config_file))

        assert config.embedding.timeout == 120

    def test_env_var_max_retries_override(self, tmp_path, monkeypatch):
        config_file = self._make_valid_yaml(tmp_path)
        monkeypatch.setenv("RAG_RERANKER_MAX_RETRIES", "7")

        manager = ConfigManager()
        config = manager.load_config(str(config_file))

        assert config.reranker.max_retries == 7

    def test_yaml_with_chunker_and_retrieval_config(self, tmp_path):
        config_data = {
            "embedding": {
                "url": "https://api.openai.com/v1/embeddings",
                "api_key": "sk-key",
            },
            "reranker": {
                "url": "https://api.cohere.ai/v1/rerank",
                "api_key": "co-key",
            },
            "query_rewriter": {
                "url": "https://api.openai.com/v1/chat",
                "api_key": "sk-key2",
            },
            "chunker": {
                "max_chunk_size": 2000,
                "overlap": 300,
            },
            "retrieval": {
                "dense_top_k": 30,
                "bm25_top_k": 15,
                "rerank_top_n": 8,
                "dense_weight": 0.6,
                "bm25_weight": 0.4,
            },
            "database": {
                "path": "custom.db",
                "embedding_dim": 768,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        manager = ConfigManager()
        config = manager.load_config(str(config_file))

        assert config.max_chunk_size == 2000
        assert config.chunk_overlap == 300
        assert config.dense_top_k == 30
        assert config.bm25_top_k == 15
        assert config.rerank_top_n == 8
        assert config.dense_weight == 0.6
        assert config.bm25_weight == 0.4
        assert config.db_path == "custom.db"
        assert config.embedding_dim == 768


class TestConfigManagerValidation:
    """Tests for ConfigManager.validate_config()."""

    def _make_valid_config(self) -> RAGConfig:
        return RAGConfig(
            embedding=ServiceConfig(
                url="https://api.openai.com/v1/embeddings",
                api_key="sk-key",
                timeout=30,
                max_retries=3,
            ),
            reranker=ServiceConfig(
                url="https://api.cohere.ai/v1/rerank",
                api_key="co-key",
                timeout=30,
                max_retries=3,
            ),
            query_rewriter=ServiceConfig(
                url="https://api.openai.com/v1/chat",
                api_key="sk-key2",
                timeout=10,
                max_retries=3,
            ),
        )

    def test_valid_config_passes(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        errors = manager.validate_config(config)
        assert errors == []

    def test_missing_url_reported(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.url = ""

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "embedding.url" in exc_info.value.missing_keys

    def test_missing_api_key_reported(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.reranker.api_key = ""

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "reranker.api_key" in exc_info.value.missing_keys

    def test_all_missing_keys_reported_at_once(self):
        manager = ConfigManager()
        config = RAGConfig()  # All URLs and API keys are empty

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        # All 6 required fields should be reported
        assert len(error.missing_keys) == 6
        assert "embedding.url" in error.missing_keys
        assert "embedding.api_key" in error.missing_keys
        assert "reranker.url" in error.missing_keys
        assert "reranker.api_key" in error.missing_keys
        assert "query_rewriter.url" in error.missing_keys
        assert "query_rewriter.api_key" in error.missing_keys

    def test_invalid_url_format(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.url = "ftp://invalid-protocol.com"

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        assert "embedding.url" in error.invalid_items
        assert "ftp://invalid-protocol.com" in error.invalid_items["embedding.url"]

    def test_timeout_below_range(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.timeout = 0

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "embedding.timeout" in exc_info.value.invalid_items

    def test_timeout_above_range(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.reranker.timeout = 301

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "reranker.timeout" in exc_info.value.invalid_items

    def test_max_retries_below_range(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.query_rewriter.max_retries = -1

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "query_rewriter.max_retries" in exc_info.value.invalid_items

    def test_max_retries_above_range(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.max_retries = 11

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        assert "embedding.max_retries" in exc_info.value.invalid_items

    def test_multiple_invalid_items_reported(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.url = "not-a-url"
        config.reranker.timeout = 500
        config.query_rewriter.max_retries = -5

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        assert "embedding.url" in error.invalid_items
        assert "reranker.timeout" in error.invalid_items
        assert "query_rewriter.max_retries" in error.invalid_items

    def test_missing_and_invalid_reported_together(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.url = ""  # missing
        config.reranker.timeout = 0  # invalid

        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        assert "embedding.url" in error.missing_keys
        assert "reranker.timeout" in error.invalid_items

    def test_http_url_is_valid(self):
        manager = ConfigManager()
        config = self._make_valid_config()
        config.embedding.url = "http://localhost:8080/embed"

        # Should not raise
        errors = manager.validate_config(config)
        assert errors == []

    def test_boundary_timeout_values(self):
        manager = ConfigManager()
        config = self._make_valid_config()

        # Minimum valid timeout
        config.embedding.timeout = 1
        config.reranker.timeout = 300  # Maximum valid timeout
        errors = manager.validate_config(config)
        assert errors == []

    def test_boundary_max_retries_values(self):
        manager = ConfigManager()
        config = self._make_valid_config()

        # Minimum valid retries
        config.embedding.max_retries = 0
        config.reranker.max_retries = 10  # Maximum valid retries
        errors = manager.validate_config(config)
        assert errors == []
