"""Unit tests for EmbeddingEngine."""

import httpx
import pytest

from rag.config import ServiceConfig
from rag.embedding import EmbeddingEngine
from rag.exceptions import EmbeddingError


@pytest.fixture
def embedding_config():
    """Create a ServiceConfig for testing."""
    return ServiceConfig(
        url="https://api.example.com/v1/embeddings",
        api_key="test-api-key",
        timeout=10,
        max_retries=3,
    )


@pytest.fixture
def engine(embedding_config):
    """Create an EmbeddingEngine instance for testing."""
    return EmbeddingEngine(config=embedding_config, embedding_dim=3)


def make_success_response(embeddings: list[list[float]]) -> httpx.Response:
    """Helper to create a mock successful API response."""
    data = [{"embedding": emb} for emb in embeddings]
    return httpx.Response(
        status_code=200,
        json={"data": data},
        request=httpx.Request("POST", "https://api.example.com/v1/embeddings"),
    )


def make_error_response(status_code: int = 500) -> httpx.Response:
    """Helper to create a mock error API response."""
    return httpx.Response(
        status_code=status_code,
        json={"error": "Internal Server Error"},
        request=httpx.Request("POST", "https://api.example.com/v1/embeddings"),
    )


@pytest.mark.asyncio
async def test_embed_texts_success(engine, monkeypatch):
    """Test successful batch text embedding."""
    expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def mock_post(self, url, **kwargs):
        return make_success_response(expected_vectors)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await engine.embed_texts(["hello", "world"])
    assert result == expected_vectors


@pytest.mark.asyncio
async def test_embed_query_success(engine, monkeypatch):
    """Test successful single query embedding."""
    expected_vector = [0.1, 0.2, 0.3]

    async def mock_post(self, url, **kwargs):
        return make_success_response([expected_vector])

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await engine.embed_query("test query")
    assert result == expected_vector


@pytest.mark.asyncio
async def test_embed_texts_dimension_mismatch(engine, monkeypatch):
    """Test that dimension mismatch raises EmbeddingError."""
    wrong_dim_vectors = [[0.1, 0.2]]  # dim=2, expected dim=3

    async def mock_post(self, url, **kwargs):
        return make_success_response(wrong_dim_vectors)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(EmbeddingError) as exc_info:
        await engine.embed_texts(["hello"])
    assert "Dimension mismatch" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_dimension(engine):
    """Test _validate_dimension method."""
    assert engine._validate_dimension([0.1, 0.2, 0.3]) is True
    assert engine._validate_dimension([0.1, 0.2]) is False
    assert engine._validate_dimension([0.1, 0.2, 0.3, 0.4]) is False
    assert engine._validate_dimension([]) is False


@pytest.mark.asyncio
async def test_retry_on_server_error(engine, monkeypatch):
    """Test exponential backoff retry on server errors."""
    call_count = 0
    expected_vector = [[0.1, 0.2, 0.3]]

    async def mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("POST", url),
                response=httpx.Response(500),
            )
        return make_success_response(expected_vector)

    # Patch asyncio.sleep to avoid actual delays in tests
    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr("rag.embedding.asyncio.sleep", mock_sleep)

    result = await engine.embed_texts(["hello"])
    assert result == expected_vector
    assert call_count == 3
    # Verify exponential backoff: 1*2^0=1, 1*2^1=2
    assert sleep_calls == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retry_exhausted_raises_embedding_error(engine, monkeypatch):
    """Test that EmbeddingError is raised after all retries are exhausted."""

    async def mock_post(self, url, **kwargs):
        raise httpx.RequestError("Connection failed", request=httpx.Request("POST", url))

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr("rag.embedding.asyncio.sleep", mock_sleep)

    with pytest.raises(EmbeddingError) as exc_info:
        await engine.embed_texts(["hello"])

    assert exc_info.value.retries_attempted == 3
    assert "failed after 3 retries" in str(exc_info.value)
    # Verify backoff intervals: 1*2^0=1, 1*2^1=2, 1*2^2=4
    assert sleep_calls == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_retry_with_zero_max_retries(monkeypatch):
    """Test behavior with max_retries=0 (no retries)."""
    config = ServiceConfig(
        url="https://api.example.com/v1/embeddings",
        api_key="test-key",
        timeout=10,
        max_retries=0,
    )
    engine = EmbeddingEngine(config=config, embedding_dim=3)

    async def mock_post(self, url, **kwargs):
        raise httpx.RequestError("Connection failed", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(EmbeddingError) as exc_info:
        await engine.embed_texts(["hello"])

    assert exc_info.value.retries_attempted == 0


@pytest.mark.asyncio
async def test_api_call_sends_correct_headers_and_body(engine, monkeypatch):
    """Test that the API call sends correct headers and request body."""
    captured_kwargs = {}

    async def mock_post(self, url, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["url"] = url
        return make_success_response([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await engine.embed_texts(["test text"])

    assert captured_kwargs["url"] == "https://api.example.com/v1/embeddings"
    assert captured_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
    assert captured_kwargs["headers"]["Content-Type"] == "application/json"
    assert captured_kwargs["json"]["input"] == ["test text"]
    assert captured_kwargs["json"]["model"] == "text-embedding-ada-002"


@pytest.mark.asyncio
async def test_http_status_error_triggers_retry(engine, monkeypatch):
    """Test that HTTP 4xx/5xx status errors trigger retry."""
    call_count = 0

    async def mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        # Return a response that will trigger raise_for_status
        response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", url),
        )
        response.raise_for_status()

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr("rag.embedding.asyncio.sleep", mock_sleep)

    with pytest.raises(EmbeddingError) as exc_info:
        await engine.embed_texts(["hello"])

    # 1 initial + 3 retries = 4 total calls
    assert call_count == 4
    assert exc_info.value.retries_attempted == 3
