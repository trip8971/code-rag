"""Unit tests for LocalDatabase storage layer (ChromaDB backend)."""

import tempfile
import shutil

import pytest

from rag.chunker import Chunk, ChunkType, generate_chunk_id
from rag.database import LocalDatabase


@pytest.fixture
def db(tmp_path):
    """Create a temporary ChromaDB database for testing."""
    db_path = str(tmp_path / "test_chroma")
    database = LocalDatabase(db_path=db_path, embedding_dim=4)
    database.initialize()
    yield database
    database.close()


def make_chunk(content: str, source_file: str = "test.md", heading_level: int = 1,
               chunk_type: ChunkType = ChunkType.TEXT, start_line: int = 1,
               heading_text: str = "# Test") -> Chunk:
    """Helper to create a Chunk for testing."""
    return Chunk(
        chunk_id=generate_chunk_id(content, source_file, start_line),
        content=content,
        source_file=source_file,
        heading_level=heading_level,
        chunk_type=chunk_type,
        start_line=start_line,
        heading_text=heading_text,
    )


class TestInitialize:
    """Tests for database initialization."""

    def test_collection_exists(self, db: LocalDatabase):
        """Collection should exist after initialization."""
        assert db.collection is not None
        assert db.collection.count() == 0

    def test_idempotent(self, db: LocalDatabase):
        """Calling initialize multiple times should not raise."""
        db.initialize()
        db.initialize()


class TestStoreChunks:
    """Tests for store_chunks method."""

    def test_store_single_chunk(self, db: LocalDatabase):
        """Should store a single chunk and its embedding."""
        chunk = make_chunk("Hello world")
        embedding = [1.0, 0.0, 0.0, 0.0]

        db.store_chunks([chunk], [embedding])

        result = db.get_chunk_by_id(chunk.chunk_id)
        assert result is not None
        assert result.content == "Hello world"
        assert result.chunk_type == ChunkType.TEXT

    def test_store_multiple_chunks(self, db: LocalDatabase):
        """Should store multiple chunks at once."""
        chunks = [
            make_chunk("First chunk", start_line=1),
            make_chunk("Second chunk", start_line=2),
            make_chunk("Third chunk", start_line=3),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

        db.store_chunks(chunks, embeddings)

        all_chunks = db.get_all_chunks()
        assert len(all_chunks) == 3

    def test_store_replaces_existing(self, db: LocalDatabase):
        """Should replace existing chunk with same ID (upsert)."""
        chunk = make_chunk("Original content")
        db.store_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

        # Store again with same chunk_id
        db.store_chunks([chunk], [[0.0, 1.0, 0.0, 0.0]])

        all_chunks = db.get_all_chunks()
        assert len(all_chunks) == 1

    def test_mismatched_lengths_raises(self, db: LocalDatabase):
        """Should raise ValueError when chunks and embeddings have different lengths."""
        chunk = make_chunk("Hello")
        with pytest.raises(ValueError, match="same length"):
            db.store_chunks([chunk], [[1.0, 0.0], [0.0, 1.0]])

    def test_store_code_chunk(self, db: LocalDatabase):
        """Should correctly store code-type chunks."""
        chunk = make_chunk("def foo(): pass", chunk_type=ChunkType.CODE)
        db.store_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

        result = db.get_chunk_by_id(chunk.chunk_id)
        assert result is not None
        assert result.chunk_type == ChunkType.CODE


class TestDeleteBySource:
    """Tests for delete_by_source method."""

    def test_delete_existing_source(self, db: LocalDatabase):
        """Should delete all chunks from a given source file."""
        chunks = [
            make_chunk("Chunk A", source_file="file1.md", start_line=1),
            make_chunk("Chunk B", source_file="file1.md", start_line=2),
            make_chunk("Chunk C", source_file="file2.md", start_line=1),
        ]
        embeddings = [[1.0, 0.0, 0.0, 0.0]] * 3
        db.store_chunks(chunks, embeddings)

        deleted = db.delete_by_source("file1.md")

        assert deleted == 2
        all_chunks = db.get_all_chunks()
        assert len(all_chunks) == 1
        assert all_chunks[0].source_file == "file2.md"

    def test_delete_nonexistent_source(self, db: LocalDatabase):
        """Should return 0 when source file doesn't exist."""
        deleted = db.delete_by_source("nonexistent.md")
        assert deleted == 0


class TestSearchByCosine:
    """Tests for search_by_cosine method."""

    def test_returns_sorted_by_similarity(self, db: LocalDatabase):
        """Results should be sorted by cosine similarity descending."""
        chunks = [
            make_chunk("Orthogonal", start_line=1),
            make_chunk("Similar", start_line=2),
            make_chunk("Identical direction", start_line=3),
        ]
        embeddings = [
            [0.0, 1.0, 0.0, 0.0],  # orthogonal to query
            [0.5, 0.5, 0.0, 0.0],  # partially similar
            [1.0, 0.0, 0.0, 0.0],  # identical direction
        ]
        db.store_chunks(chunks, embeddings)

        results = db.search_by_cosine([1.0, 0.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        # Scores should be descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # First result should be the identical direction
        assert results[0][0].content == "Identical direction"

    def test_respects_top_k(self, db: LocalDatabase):
        """Should return at most top_k results."""
        chunks = [make_chunk(f"Chunk {i}", start_line=i) for i in range(5)]
        embeddings = [[float(i + 1), 0.0, 0.0, 0.0] for i in range(5)]
        db.store_chunks(chunks, embeddings)

        results = db.search_by_cosine([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_empty_database(self, db: LocalDatabase):
        """Should return empty list when database is empty."""
        results = db.search_by_cosine([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_fewer_results_than_top_k(self, db: LocalDatabase):
        """Should return all available results when fewer than top_k."""
        chunks = [make_chunk("Only one")]
        embeddings = [[1.0, 0.0, 0.0, 0.0]]
        db.store_chunks(chunks, embeddings)

        results = db.search_by_cosine([1.0, 0.0, 0.0, 0.0], top_k=10)
        assert len(results) == 1


class TestGetAllChunks:
    """Tests for get_all_chunks method."""

    def test_empty_database(self, db: LocalDatabase):
        """Should return empty list for empty database."""
        assert db.get_all_chunks() == []

    def test_returns_all_stored_chunks(self, db: LocalDatabase):
        """Should return all stored chunks."""
        chunks = [make_chunk(f"Chunk {i}", start_line=i) for i in range(3)]
        embeddings = [[1.0, 0.0, 0.0, 0.0]] * 3
        db.store_chunks(chunks, embeddings)

        result = db.get_all_chunks()
        assert len(result) == 3


class TestGetChunkById:
    """Tests for get_chunk_by_id method."""

    def test_existing_chunk(self, db: LocalDatabase):
        """Should return the chunk with matching ID."""
        chunk = make_chunk("Find me")
        db.store_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

        result = db.get_chunk_by_id(chunk.chunk_id)
        assert result is not None
        assert result.content == "Find me"
        assert result.chunk_id == chunk.chunk_id

    def test_nonexistent_chunk(self, db: LocalDatabase):
        """Should return None for non-existent ID."""
        result = db.get_chunk_by_id("nonexistent_id")
        assert result is None

    def test_preserves_all_fields(self, db: LocalDatabase):
        """Should preserve all chunk fields through storage and retrieval."""
        chunk = Chunk(
            chunk_id="test_id_123",
            content="Test content here",
            source_file="path/to/file.md",
            heading_level=3,
            chunk_type=ChunkType.CODE,
            start_line=42,
            heading_text="### API Reference",
        )
        db.store_chunks([chunk], [[1.0, 2.0, 3.0, 4.0]])

        result = db.get_chunk_by_id("test_id_123")
        assert result is not None
        assert result.chunk_id == "test_id_123"
        assert result.content == "Test content here"
        assert result.source_file == "path/to/file.md"
        assert result.heading_level == 3
        assert result.chunk_type == ChunkType.CODE
        assert result.start_line == 42
        assert result.heading_text == "### API Reference"
