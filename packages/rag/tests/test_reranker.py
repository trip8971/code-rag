"""Unit tests for Reranker."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag.chunker import Chunk, ChunkType
from rag.config import ServiceConfig
from rag.reranker import Reranker


@pytest.fixture
def valid_config():
    """A valid ServiceConfig for reranker."""
    return ServiceConfig(
        url="https://api.cohere.ai/v1/rerank",
        api_key="test-api-key-123",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def reranker(valid_config):
    """A Reranker with valid config and default weights."""
    return Reranker(valid_config)


def make_chunk(chunk_id: str, content: str = "test content") -> Chunk:
    """Helper to create a Chunk with given id."""
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        source_file="test.md",
        heading_level=1,
        chunk_type=ChunkType.TEXT,
        start_line=1,
        heading_text="# Test",
    )


class TestMergeAndDeduplicate:
    """Tests for _merge_and_deduplicate method."""

    def test_empty_inputs(self, reranker):
        result = reranker._merge_and_deduplicate([], [])
        assert result == []

    def test_only_dense_results(self, reranker):
        c1 = make_chunk("a")
        c2 = make_chunk("b")
        dense = [(c1, 0.9), (c2, 0.8)]
        result = reranker._merge_and_deduplicate(dense, [])
        assert result == [c1, c2]

    def test_only_bm25_results(self, reranker):
        c1 = make_chunk("a")
        c2 = make_chunk("b")
        bm25 = [(c1, 0.5), (c2, 0.4)]
        result = reranker._merge_and_deduplicate([], bm25)
        assert result == [c1, c2]

    def test_no_overlap(self, reranker):
        c1 = make_chunk("a")
        c2 = make_chunk("b")
        c3 = make_chunk("c")
        dense = [(c1, 0.9)]
        bm25 = [(c2, 0.5), (c3, 0.4)]
        result = reranker._merge_and_deduplicate(dense, bm25)
        assert result == [c1, c2, c3]

    def test_with_overlap_preserves_first_occurrence(self, reranker):
        """Duplicate chunk_id in both lists: keep the one from dense (first)."""
        c1_dense = make_chunk("a", content="dense version")
        c2 = make_chunk("b")
        c1_bm25 = make_chunk("a", content="bm25 version")
        c3 = make_chunk("c")

        dense = [(c1_dense, 0.9), (c2, 0.8)]
        bm25 = [(c1_bm25, 0.5), (c3, 0.4)]

        result = reranker._merge_and_deduplicate(dense, bm25)
        assert len(result) == 3
        assert result[0].chunk_id == "a"
        assert result[0].content == "dense version"  # First occurrence preserved
        assert result[1].chunk_id == "b"
        assert result[2].chunk_id == "c"

    def test_all_duplicates(self, reranker):
        """All chunks appear in both lists."""
        c1 = make_chunk("a")
        c2 = make_chunk("b")
        dense = [(c1, 0.9), (c2, 0.8)]
        bm25 = [(c1, 0.5), (c2, 0.4)]
        result = reranker._merge_and_deduplicate(dense, bm25)
        assert len(result) == 2
        assert result[0].chunk_id == "a"
        assert result[1].chunk_id == "b"

    def test_order_dense_first_then_bm25(self, reranker):
        """Dense results come first in merged order."""
        c1 = make_chunk("d1")
        c2 = make_chunk("d2")
        c3 = make_chunk("b1")
        c4 = make_chunk("b2")
        dense = [(c1, 0.9), (c2, 0.8)]
        bm25 = [(c3, 0.5), (c4, 0.4)]
        result = reranker._merge_and_deduplicate(dense, bm25)
        assert [c.chunk_id for c in result] == ["d1", "d2", "b1", "b2"]


class TestFallbackScore:
    """Tests for _fallback_score method."""

    def test_empty_inputs(self, reranker):
        result = reranker._fallback_score([], [])
        assert result == []

    def test_only_dense_results(self, reranker):
        c1 = make_chunk("a")
        dense = [(c1, 0.8)]
        result = reranker._fallback_score(dense, [])
        assert len(result) == 1
        assert result[0][0].chunk_id == "a"
        # score = 0.7 * 0.8 + 0.3 * 0 = 0.56
        assert abs(result[0][1] - 0.56) < 1e-9

    def test_only_bm25_results(self, reranker):
        c1 = make_chunk("a")
        bm25 = [(c1, 0.6)]
        result = reranker._fallback_score([], bm25)
        assert len(result) == 1
        assert result[0][0].chunk_id == "a"
        # score = 0.7 * 0 + 0.3 * 0.6 = 0.18
        assert abs(result[0][1] - 0.18) < 1e-9

    def test_chunk_in_both_lists(self, reranker):
        c1 = make_chunk("a")
        dense = [(c1, 0.8)]
        bm25 = [(c1, 0.6)]
        result = reranker._fallback_score(dense, bm25)
        assert len(result) == 1
        # score = 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert abs(result[0][1] - 0.74) < 1e-9

    def test_sorted_descending(self, reranker):
        c1 = make_chunk("a")
        c2 = make_chunk("b")
        c3 = make_chunk("c")
        # c1: dense=0.9, bm25=0.1 -> 0.7*0.9 + 0.3*0.1 = 0.63 + 0.03 = 0.66
        # c2: dense=0.5, bm25=0.9 -> 0.7*0.5 + 0.3*0.9 = 0.35 + 0.27 = 0.62
        # c3: dense=0.0, bm25=0.8 -> 0.7*0.0 + 0.3*0.8 = 0.0 + 0.24 = 0.24
        dense = [(c1, 0.9), (c2, 0.5)]
        bm25 = [(c1, 0.1), (c2, 0.9), (c3, 0.8)]
        result = reranker._fallback_score(dense, bm25)
        assert len(result) == 3
        assert result[0][0].chunk_id == "a"
        assert result[1][0].chunk_id == "b"
        assert result[2][0].chunk_id == "c"

    def test_custom_weights(self, valid_config):
        reranker = Reranker(valid_config, dense_weight=0.5, bm25_weight=0.5)
        c1 = make_chunk("a")
        dense = [(c1, 0.8)]
        bm25 = [(c1, 0.4)]
        result = reranker._fallback_score(dense, bm25)
        # score = 0.5 * 0.8 + 0.5 * 0.4 = 0.4 + 0.2 = 0.6
        assert abs(result[0][1] - 0.6) < 1e-9


class TestRerank:
    """Tests for rerank method."""

    @pytest.mark.asyncio
    async def test_empty_inputs(self, reranker):
        result = await reranker.rerank("query", [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_api_call(self, valid_config):
        reranker = Reranker(valid_config)
        c1 = make_chunk("a", content="content a")
        c2 = make_chunk("b", content="content b")
        c3 = make_chunk("c", content="content c")

        dense = [(c1, 0.9), (c2, 0.8)]
        bm25 = [(c3, 0.5)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.90},
                {"index": 1, "relevance_score": 0.70},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await reranker.rerank("test query", dense, bm25, top_n=2)

        assert len(result) == 2
        assert result[0][0].chunk_id == "c"  # highest score
        assert result[0][1] == 0.95
        assert result[1][0].chunk_id == "a"
        assert result[1][1] == 0.90

    @pytest.mark.asyncio
    async def test_api_failure_falls_back(self, valid_config):
        """When API fails, falls back to weighted fusion."""
        reranker = Reranker(valid_config)
        c1 = make_chunk("a")
        c2 = make_chunk("b")

        dense = [(c1, 0.9)]
        bm25 = [(c2, 0.8)]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await reranker.rerank("test query", dense, bm25, top_n=5)

        # Fallback: c1 score = 0.7*0.9 + 0.3*0 = 0.63, c2 score = 0.7*0 + 0.3*0.8 = 0.24
        assert len(result) == 2
        assert result[0][0].chunk_id == "a"
        assert abs(result[0][1] - 0.63) < 1e-9
        assert result[1][0].chunk_id == "b"
        assert abs(result[1][1] - 0.24) < 1e-9

    @pytest.mark.asyncio
    async def test_api_timeout_falls_back(self, valid_config):
        """When API times out, falls back to weighted fusion."""
        reranker = Reranker(valid_config)
        c1 = make_chunk("a")
        dense = [(c1, 0.5)]
        bm25 = [(c1, 0.5)]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await reranker.rerank("query", dense, bm25, top_n=5)

        # Fallback: score = 0.7*0.5 + 0.3*0.5 = 0.35 + 0.15 = 0.5
        assert len(result) == 1
        assert abs(result[0][1] - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_top_n_limits_results(self, valid_config):
        """top_n limits the number of returned results."""
        reranker = Reranker(valid_config)
        chunks = [make_chunk(f"chunk_{i}") for i in range(10)]
        dense = [(c, 0.9 - i * 0.05) for i, c in enumerate(chunks[:5])]
        bm25 = [(c, 0.8 - i * 0.05) for i, c in enumerate(chunks[5:])]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await reranker.rerank("query", dense, bm25, top_n=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_fewer_results_than_top_n(self, valid_config):
        """When fewer results available than top_n, return all."""
        reranker = Reranker(valid_config)
        c1 = make_chunk("a")
        dense = [(c1, 0.9)]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await reranker.rerank("query", dense, [], top_n=10)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_sends_correct_request(self, valid_config):
        """Verify correct request format is sent to the API."""
        reranker = Reranker(valid_config)
        c1 = make_chunk("a", content="hello world")
        c2 = make_chunk("b", content="foo bar")

        dense = [(c1, 0.9)]
        bm25 = [(c2, 0.5)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await reranker.rerank("my query", dense, bm25, top_n=5)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://api.cohere.ai/v1/rerank"
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-api-key-123"
        assert headers["Content-Type"] == "application/json"
        body = call_kwargs[1]["json"]
        assert body["query"] == "my query"
        assert body["documents"] == ["hello world", "foo bar"]
        assert body["top_n"] == 5

    @pytest.mark.asyncio
    async def test_http_status_error_falls_back(self, valid_config):
        """When API returns HTTP error status, falls back."""
        reranker = Reranker(valid_config)
        c1 = make_chunk("a")
        dense = [(c1, 0.8)]

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
            result = await reranker.rerank("query", dense, [], top_n=5)

        assert len(result) == 1
        # Fallback: 0.7 * 0.8 = 0.56
        assert abs(result[0][1] - 0.56) < 1e-9
