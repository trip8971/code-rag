"""Unit tests for DenseSearch."""

import pytest

from rag.chunker import Chunk, ChunkType
from rag.database import LocalDatabase
from rag.dense_search import DenseSearch
from rag.embedding import EmbeddingEngine
from rag.exceptions import EmbeddingError


@pytest.fixture
def mock_embedding_engine(mocker):
    """Create a mock EmbeddingEngine."""
    engine = mocker.AsyncMock(spec=EmbeddingEngine)
    return engine


@pytest.fixture
def mock_database(mocker):
    """Create a mock LocalDatabase."""
    db = mocker.Mock(spec=LocalDatabase)
    return db


@pytest.fixture
def dense_search(mock_embedding_engine, mock_database):
    """Create a DenseSearch instance with mocked dependencies."""
    return DenseSearch(
        embedding_engine=mock_embedding_engine,
        database=mock_database,
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            content="How to use torch.nn.Linear",
            source_file="docs/api.md",
            heading_level=2,
            chunk_type=ChunkType.TEXT,
            start_line=10,
            heading_text="## Linear Layer",
        ),
        Chunk(
            chunk_id="chunk_2",
            content="torch.optim.Adam optimizer",
            source_file="docs/optim.md",
            heading_level=2,
            chunk_type=ChunkType.TEXT,
            start_line=20,
            heading_text="## Adam Optimizer",
        ),
    ]


@pytest.mark.asyncio
async def test_search_returns_results(dense_search, mock_embedding_engine, mock_database, sample_chunks):
    """Test that search returns results from database sorted by similarity."""
    query_vector = [0.1, 0.2, 0.3]
    mock_embedding_engine.embed_query.return_value = query_vector

    expected_results = [
        (sample_chunks[0], 0.95),
        (sample_chunks[1], 0.80),
    ]
    mock_database.search_by_cosine.return_value = expected_results

    results = await dense_search.search("how to use linear layer", top_k=5)

    mock_embedding_engine.embed_query.assert_called_once_with("how to use linear layer")
    mock_database.search_by_cosine.assert_called_once_with(query_vector, 5)
    assert results == expected_results


@pytest.mark.asyncio
async def test_search_uses_default_top_k(dense_search, mock_embedding_engine, mock_database):
    """Test that search uses default top_k=20 when not specified."""
    query_vector = [0.5, 0.5, 0.5]
    mock_embedding_engine.embed_query.return_value = query_vector
    mock_database.search_by_cosine.return_value = []

    await dense_search.search("test query")

    mock_database.search_by_cosine.assert_called_once_with(query_vector, 20)


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_no_results(dense_search, mock_embedding_engine, mock_database):
    """Test that search returns empty list when database has no matching chunks."""
    mock_embedding_engine.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_database.search_by_cosine.return_value = []

    results = await dense_search.search("nonexistent topic")

    assert results == []


@pytest.mark.asyncio
async def test_search_propagates_embedding_error(dense_search, mock_embedding_engine, mock_database):
    """Test that EmbeddingError propagates when embedding service is unavailable."""
    mock_embedding_engine.embed_query.side_effect = EmbeddingError(
        "Embedding API request failed after 3 retries: connection timeout",
        retries_attempted=3,
    )

    with pytest.raises(EmbeddingError) as exc_info:
        await dense_search.search("test query")

    assert exc_info.value.retries_attempted == 3
    assert "connection timeout" in str(exc_info.value)
    # Database should not be called when embedding fails
    mock_database.search_by_cosine.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_fewer_results_when_database_has_less(
    dense_search, mock_embedding_engine, mock_database, sample_chunks
):
    """Test that search returns all available chunks when fewer than top_k exist."""
    mock_embedding_engine.embed_query.return_value = [0.1, 0.2, 0.3]
    # Database only has 2 chunks but we request top_k=100
    mock_database.search_by_cosine.return_value = [
        (sample_chunks[0], 0.9),
        (sample_chunks[1], 0.7),
    ]

    results = await dense_search.search("test query", top_k=100)

    assert len(results) == 2
    mock_database.search_by_cosine.assert_called_once_with([0.1, 0.2, 0.3], 100)
