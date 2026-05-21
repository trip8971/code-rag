"""Tests for the custom exception hierarchy."""

import pytest

from rag.exceptions import (
    ConfigError,
    EmbeddingError,
    EvalDatasetError,
    RAGError,
)


class TestRAGError:
    """Tests for the base RAGError exception."""

    def test_is_exception(self):
        assert issubclass(RAGError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RAGError):
            raise RAGError("something went wrong")

    def test_message_preserved(self):
        err = RAGError("test message")
        assert str(err) == "test message"


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_inherits_from_rag_error(self):
        assert issubclass(ConfigError, RAGError)

    def test_missing_keys_default_empty(self):
        err = ConfigError()
        assert err.missing_keys == []
        assert err.invalid_items == {}

    def test_missing_keys_stored(self):
        err = ConfigError(missing_keys=["embedding.url", "reranker.api_key"])
        assert err.missing_keys == ["embedding.url", "reranker.api_key"]

    def test_invalid_items_stored(self):
        err = ConfigError(invalid_items={"timeout": "must be 1-300, got 999"})
        assert err.invalid_items == {"timeout": "must be 1-300, got 999"}

    def test_message_includes_missing_keys(self):
        err = ConfigError(missing_keys=["embedding.url", "embedding.api_key"])
        msg = str(err)
        assert "embedding.url" in msg
        assert "embedding.api_key" in msg

    def test_message_includes_invalid_items(self):
        err = ConfigError(
            invalid_items={"timeout": "must be 1-300, got -5"}
        )
        msg = str(err)
        assert "timeout" in msg
        assert "must be 1-300, got -5" in msg

    def test_message_includes_both_missing_and_invalid(self):
        err = ConfigError(
            missing_keys=["reranker.url"],
            invalid_items={"embedding.timeout": "out of range"},
        )
        msg = str(err)
        assert "reranker.url" in msg
        assert "embedding.timeout" in msg
        assert "out of range" in msg

    def test_can_be_caught_as_rag_error(self):
        with pytest.raises(RAGError):
            raise ConfigError(missing_keys=["key"])


class TestEmbeddingError:
    """Tests for EmbeddingError exception."""

    def test_inherits_from_rag_error(self):
        assert issubclass(EmbeddingError, RAGError)

    def test_message_and_retries_stored(self):
        err = EmbeddingError("connection timeout", retries_attempted=3)
        assert str(err) == "connection timeout"
        assert err.retries_attempted == 3

    def test_zero_retries(self):
        err = EmbeddingError("immediate failure", retries_attempted=0)
        assert err.retries_attempted == 0

    def test_can_be_caught_as_rag_error(self):
        with pytest.raises(RAGError):
            raise EmbeddingError("fail", retries_attempted=2)


class TestEvalDatasetError:
    """Tests for EvalDatasetError exception."""

    def test_inherits_from_rag_error(self):
        assert issubclass(EvalDatasetError, RAGError)

    def test_message_and_record_index_stored(self):
        err = EvalDatasetError("missing query field", record_index=5)
        assert str(err) == "missing query field"
        assert err.record_index == 5

    def test_record_index_default_none(self):
        err = EvalDatasetError("dataset is empty")
        assert err.record_index is None

    def test_can_be_caught_as_rag_error(self):
        with pytest.raises(RAGError):
            raise EvalDatasetError("bad format", record_index=0)
