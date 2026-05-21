"""Property-based tests for ConfigManager."""

import os

import pytest
import yaml
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from rag.config import ConfigManager, RAGConfig, ServiceConfig
from rag.exceptions import ConfigError


# --- Strategies ---

# Generate invalid URLs: strings that do NOT start with http:// or https://
invalid_url_strategy = st.text(min_size=1, max_size=200).filter(
    lambda s: not s.startswith("http://") and not s.startswith("https://")
)

# Generate invalid timeouts: integers outside range 1-300
invalid_timeout_strategy = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=301),
)

# Generate invalid max_retries: integers outside range 0-10
invalid_max_retries_strategy = st.one_of(
    st.integers(max_value=-1),
    st.integers(min_value=11),
)

# Service names to test against
service_names = ["embedding", "reranker", "query_rewriter"]


def _make_valid_config() -> RAGConfig:
    """Create a fully valid RAGConfig for baseline."""
    return RAGConfig(
        embedding=ServiceConfig(
            url="https://api.openai.com/v1/embeddings",
            api_key="sk-embed-key",
            timeout=30,
            max_retries=3,
        ),
        reranker=ServiceConfig(
            url="https://api.cohere.ai/v1/rerank",
            api_key="co-rerank-key",
            timeout=30,
            max_retries=3,
        ),
        query_rewriter=ServiceConfig(
            url="https://api.openai.com/v1/chat/completions",
            api_key="sk-rewriter-key",
            timeout=10,
            max_retries=3,
        ),
    )


class TestProperty16InvalidConfigValuesReportNameAndValue:
    """Property 16: 无效配置值报告名称和当前值

    For any 格式无效的配置项（URL 格式不合法、超时时间超出 1-300 范围、
    重试次数超出 0-10 范围），验证错误信息应包含该配置项的名称及其当前无效值。

    **Validates: Requirements 8.6**
    """

    @settings(max_examples=100)
    @given(
        invalid_url=invalid_url_strategy,
        service_idx=st.integers(min_value=0, max_value=2),
    )
    def test_invalid_url_reports_name_and_value(self, invalid_url, service_idx):
        """Invalid URL values are reported with config key name and current value."""
        service_name = service_names[service_idx]
        config = _make_valid_config()
        service_config = getattr(config, service_name)
        service_config.url = invalid_url

        manager = ConfigManager()
        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        key = f"{service_name}.url"
        assert key in error.invalid_items, (
            f"Expected '{key}' in invalid_items, got: {error.invalid_items}"
        )
        assert invalid_url in error.invalid_items[key], (
            f"Expected invalid value '{invalid_url}' in error message for '{key}', "
            f"got: '{error.invalid_items[key]}'"
        )

    @settings(max_examples=100)
    @given(
        invalid_timeout=invalid_timeout_strategy,
        service_idx=st.integers(min_value=0, max_value=2),
    )
    def test_invalid_timeout_reports_name_and_value(self, invalid_timeout, service_idx):
        """Invalid timeout values are reported with config key name and current value."""
        service_name = service_names[service_idx]
        config = _make_valid_config()
        service_config = getattr(config, service_name)
        service_config.timeout = invalid_timeout

        manager = ConfigManager()
        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        key = f"{service_name}.timeout"
        assert key in error.invalid_items, (
            f"Expected '{key}' in invalid_items, got: {error.invalid_items}"
        )
        assert str(invalid_timeout) in error.invalid_items[key], (
            f"Expected invalid value '{invalid_timeout}' in error message for '{key}', "
            f"got: '{error.invalid_items[key]}'"
        )

    @settings(max_examples=100)
    @given(
        invalid_retries=invalid_max_retries_strategy,
        service_idx=st.integers(min_value=0, max_value=2),
    )
    def test_invalid_max_retries_reports_name_and_value(
        self, invalid_retries, service_idx
    ):
        """Invalid max_retries values are reported with config key name and current value."""
        service_name = service_names[service_idx]
        config = _make_valid_config()
        service_config = getattr(config, service_name)
        service_config.max_retries = invalid_retries

        manager = ConfigManager()
        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value
        key = f"{service_name}.max_retries"
        assert key in error.invalid_items, (
            f"Expected '{key}' in invalid_items, got: {error.invalid_items}"
        )
        assert str(invalid_retries) in error.invalid_items[key], (
            f"Expected invalid value '{invalid_retries}' in error message for '{key}', "
            f"got: '{error.invalid_items[key]}'"
        )


# All required config keys that must be non-empty
ALL_REQUIRED_KEYS = [
    "embedding.url",
    "embedding.api_key",
    "reranker.url",
    "reranker.api_key",
    "query_rewriter.url",
    "query_rewriter.api_key",
]


def _set_key_empty(config: RAGConfig, key: str) -> None:
    """Set a specific config key to empty string to simulate it being missing."""
    service_name, field_name = key.split(".")
    service_config = getattr(config, service_name)
    setattr(service_config, field_name, "")


class TestProperty15MissingKeysAllReported:
    """Property 15: 缺失配置项全部报告

    For any non-empty subset of required config keys that are left empty,
    the ConfigError raised by validate_config() must contain ALL of those
    keys in its missing_keys attribute.

    **Validates: Requirements 8.4**
    """

    @settings(max_examples=100)
    @given(
        missing_indices=st.lists(
            st.integers(min_value=0, max_value=len(ALL_REQUIRED_KEYS) - 1),
            min_size=1,
            max_size=len(ALL_REQUIRED_KEYS),
            unique=True,
        )
    )
    def test_all_missing_keys_reported_in_error(self, missing_indices: list[int]):
        """Validates: Requirements 8.4

        Given a random non-empty subset of required config keys left empty,
        validate_config() should raise ConfigError with missing_keys containing
        ALL of those keys.
        """
        # Determine which keys to leave empty
        missing_keys = [ALL_REQUIRED_KEYS[i] for i in missing_indices]

        # Start with a valid config and clear the selected keys
        config = _make_valid_config()
        for key in missing_keys:
            _set_key_empty(config, key)

        # validate_config should raise ConfigError
        manager = ConfigManager()
        with pytest.raises(ConfigError) as exc_info:
            manager.validate_config(config)

        error = exc_info.value

        # ALL missing keys must be reported - none should be omitted
        for key in missing_keys:
            assert key in error.missing_keys, (
                f"Expected missing key '{key}' to be reported in ConfigError.missing_keys, "
                f"but got: {error.missing_keys}"
            )

        # The reported missing keys should be a superset of (or equal to) our missing keys
        assert set(missing_keys).issubset(set(error.missing_keys))



# --- Property 14 Strategies ---

# Valid URL strategy: must start with http:// or https://
_valid_url_strategy = st.one_of(
    st.from_regex(
        r"https://[a-z][a-z0-9]{1,20}\.[a-z]{2,4}(/[a-z0-9]{1,10}){0,3}",
        fullmatch=True,
    ),
    st.from_regex(
        r"http://[a-z][a-z0-9]{1,20}\.[a-z]{2,4}(/[a-z0-9]{1,10}){0,3}",
        fullmatch=True,
    ),
)

# Valid API key strategy: non-empty alphanumeric strings
_valid_api_key_strategy = st.from_regex(r"[a-zA-Z0-9\-_]{5,40}", fullmatch=True)

# Valid timeout strategy: 1-300
_valid_timeout_strategy = st.integers(min_value=1, max_value=300)

# Valid max_retries strategy: 0-10
_valid_max_retries_strategy = st.integers(min_value=0, max_value=10)


@st.composite
def _service_config_dict(draw):
    """Generate a valid service config dictionary."""
    return {
        "url": draw(_valid_url_strategy),
        "api_key": draw(_valid_api_key_strategy),
        "timeout": draw(_valid_timeout_strategy),
        "max_retries": draw(_valid_max_retries_strategy),
    }


@st.composite
def _service_config_pair(draw):
    """Generate two different valid service configs (file vs env var)."""
    file_config = draw(_service_config_dict())
    env_config = draw(_service_config_dict())

    # Ensure at least one field differs between file and env
    assume(
        file_config["url"] != env_config["url"]
        or file_config["api_key"] != env_config["api_key"]
        or file_config["timeout"] != env_config["timeout"]
        or file_config["max_retries"] != env_config["max_retries"]
    )

    return file_config, env_config


class TestProperty14EnvVarsOverrideConfigFile:
    """Property 14: 环境变量优先于配置文件

    For any configuration key, when both an environment variable and a config
    file provide a value for that key, the final effective value should equal
    the environment variable's value.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """

    @settings(max_examples=100)
    @given(
        embedding_pair=_service_config_pair(),
        reranker_pair=_service_config_pair(),
        query_rewriter_pair=_service_config_pair(),
    )
    def test_env_vars_always_override_config_file(
        self, embedding_pair, reranker_pair, query_rewriter_pair, tmp_path_factory
    ):
        """Environment variable values always take priority over YAML config file values.

        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        embedding_file, embedding_env = embedding_pair
        reranker_file, reranker_env = reranker_pair
        qr_file, qr_env = query_rewriter_pair

        # Create a YAML config file with the "file" values
        yaml_data = {
            "embedding": embedding_file,
            "reranker": reranker_file,
            "query_rewriter": qr_file,
        }

        tmp_path = tmp_path_factory.mktemp("config")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data), encoding="utf-8")

        # Set environment variables with the "env" values
        env_vars = {
            "RAG_EMBEDDING_URL": embedding_env["url"],
            "RAG_EMBEDDING_API_KEY": embedding_env["api_key"],
            "RAG_EMBEDDING_TIMEOUT": str(embedding_env["timeout"]),
            "RAG_EMBEDDING_MAX_RETRIES": str(embedding_env["max_retries"]),
            "RAG_RERANKER_URL": reranker_env["url"],
            "RAG_RERANKER_API_KEY": reranker_env["api_key"],
            "RAG_RERANKER_TIMEOUT": str(reranker_env["timeout"]),
            "RAG_RERANKER_MAX_RETRIES": str(reranker_env["max_retries"]),
            "RAG_QUERY_REWRITER_URL": qr_env["url"],
            "RAG_QUERY_REWRITER_API_KEY": qr_env["api_key"],
            "RAG_QUERY_REWRITER_TIMEOUT": str(qr_env["timeout"]),
            "RAG_QUERY_REWRITER_MAX_RETRIES": str(qr_env["max_retries"]),
        }

        # Save original env state and set new values
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            manager = ConfigManager()
            config = manager.load_config(str(config_file))

            # Verify: environment variable values override config file values
            # Embedding service
            assert config.embedding.url == embedding_env["url"], (
                f"embedding.url: expected env '{embedding_env['url']}', "
                f"got '{config.embedding.url}' (file: '{embedding_file['url']}')"
            )
            assert config.embedding.api_key == embedding_env["api_key"], (
                f"embedding.api_key: expected env '{embedding_env['api_key']}', "
                f"got '{config.embedding.api_key}'"
            )
            assert config.embedding.timeout == embedding_env["timeout"], (
                f"embedding.timeout: expected env {embedding_env['timeout']}, "
                f"got {config.embedding.timeout}"
            )
            assert config.embedding.max_retries == embedding_env["max_retries"], (
                f"embedding.max_retries: expected env {embedding_env['max_retries']}, "
                f"got {config.embedding.max_retries}"
            )

            # Reranker service
            assert config.reranker.url == reranker_env["url"], (
                f"reranker.url: expected env '{reranker_env['url']}', "
                f"got '{config.reranker.url}'"
            )
            assert config.reranker.api_key == reranker_env["api_key"], (
                f"reranker.api_key: expected env '{reranker_env['api_key']}', "
                f"got '{config.reranker.api_key}'"
            )
            assert config.reranker.timeout == reranker_env["timeout"], (
                f"reranker.timeout: expected env {reranker_env['timeout']}, "
                f"got {config.reranker.timeout}"
            )
            assert config.reranker.max_retries == reranker_env["max_retries"], (
                f"reranker.max_retries: expected env {reranker_env['max_retries']}, "
                f"got {config.reranker.max_retries}"
            )

            # Query rewriter service
            assert config.query_rewriter.url == qr_env["url"], (
                f"query_rewriter.url: expected env '{qr_env['url']}', "
                f"got '{config.query_rewriter.url}'"
            )
            assert config.query_rewriter.api_key == qr_env["api_key"], (
                f"query_rewriter.api_key: expected env '{qr_env['api_key']}', "
                f"got '{config.query_rewriter.api_key}'"
            )
            assert config.query_rewriter.timeout == qr_env["timeout"], (
                f"query_rewriter.timeout: expected env {qr_env['timeout']}, "
                f"got {config.query_rewriter.timeout}"
            )
            assert config.query_rewriter.max_retries == qr_env["max_retries"], (
                f"query_rewriter.max_retries: expected env {qr_env['max_retries']}, "
                f"got {config.query_rewriter.max_retries}"
            )

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
