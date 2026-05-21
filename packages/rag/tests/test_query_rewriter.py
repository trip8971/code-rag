"""Unit tests for QueryRewriter."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag.config import ServiceConfig
from rag.query_rewriter import QueryRewriter


@pytest.fixture
def valid_config():
    """A valid ServiceConfig for query rewriter."""
    return ServiceConfig(
        url="https://api.openai.com/v1/chat/completions",
        api_key="sk-test-key-123",
        timeout=10,
        max_retries=3,
    )


@pytest.fixture
def rewriter(valid_config):
    """A QueryRewriter with valid config."""
    return QueryRewriter(valid_config)


class TestIsConfigValid:
    """Tests for _is_config_valid method."""

    def test_valid_config(self, rewriter):
        assert rewriter._is_config_valid() is True

    def test_empty_url(self):
        config = ServiceConfig(url="", api_key="sk-key", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is False

    def test_whitespace_url(self):
        config = ServiceConfig(url="   ", api_key="sk-key", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is False

    def test_invalid_url_no_http(self):
        config = ServiceConfig(url="ftp://example.com", api_key="sk-key", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is False

    def test_http_url_valid(self):
        config = ServiceConfig(url="http://localhost:8080/v1", api_key="sk-key", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is True

    def test_https_url_valid(self):
        config = ServiceConfig(url="https://api.example.com/v1", api_key="sk-key", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is True

    def test_empty_api_key(self):
        config = ServiceConfig(url="https://api.example.com", api_key="", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is False

    def test_whitespace_api_key(self):
        config = ServiceConfig(url="https://api.example.com", api_key="   ", timeout=10)
        rewriter = QueryRewriter(config)
        assert rewriter._is_config_valid() is False


class TestTruncateResult:
    """Tests for _truncate_result method."""

    def test_short_text_unchanged(self, rewriter):
        text = "short query"
        assert rewriter._truncate_result(text) == text

    def test_exactly_500_chars_unchanged(self, rewriter):
        text = "a" * 500
        assert rewriter._truncate_result(text) == text

    def test_over_500_chars_truncated(self, rewriter):
        text = "a" * 600
        result = rewriter._truncate_result(text)
        assert len(result) == 500
        assert result == "a" * 500

    def test_custom_max_length(self, rewriter):
        text = "a" * 100
        result = rewriter._truncate_result(text, max_length=50)
        assert len(result) == 50

    def test_empty_string(self, rewriter):
        assert rewriter._truncate_result("") == ""


class TestRewrite:
    """Tests for rewrite method."""

    @pytest.mark.asyncio
    async def test_invalid_config_returns_original(self):
        """When config is invalid, rewrite returns original query."""
        config = ServiceConfig(url="", api_key="", timeout=10)
        rewriter = QueryRewriter(config)
        result = await rewriter.rewrite("test query")
        assert result == "test query"

    @pytest.mark.asyncio
    async def test_successful_rewrite(self, valid_config):
        """When API returns valid response, rewrite returns rewritten query."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "rewritten test query"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "rewritten test query"

    @pytest.mark.asyncio
    async def test_truncates_long_response(self, valid_config):
        """When API returns long response, it gets truncated to 500 chars."""
        rewriter = QueryRewriter(valid_config)
        long_response = "x" * 600

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": long_response}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_timeout_returns_original(self, valid_config):
        """When request times out, returns original query."""
        rewriter = QueryRewriter(valid_config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "test query"

    @pytest.mark.asyncio
    async def test_http_error_returns_original(self, valid_config):
        """When API returns HTTP error, returns original query."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "test query"

    @pytest.mark.asyncio
    async def test_invalid_json_response_returns_original(self, valid_config):
        """When API returns invalid JSON structure, returns original query."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"invalid": "response"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "test query"

    @pytest.mark.asyncio
    async def test_connection_error_returns_original(self, valid_config):
        """When connection fails, returns original query."""
        rewriter = QueryRewriter(valid_config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "test query"

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self, valid_config):
        """Response content is stripped of leading/trailing whitespace."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  rewritten query  "}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await rewriter.rewrite("test query")

        assert result == "rewritten query"

    @pytest.mark.asyncio
    async def test_sends_correct_headers(self, valid_config):
        """Verify correct headers are sent to the API."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "rewritten"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await rewriter.rewrite("test query")

        # Verify the post call was made with correct arguments
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://api.openai.com/v1/chat/completions"
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_sends_correct_body(self, valid_config):
        """Verify correct request body is sent to the API."""
        rewriter = QueryRewriter(valid_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "rewritten"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await rewriter.rewrite("how to use torch.nn.Linear")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["json"]
        assert body["model"] == "gpt-4"
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "how to use torch.nn.Linear"
