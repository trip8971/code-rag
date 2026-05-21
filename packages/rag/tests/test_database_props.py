"""Property-based tests for LocalDatabase storage layer (ChromaDB backend)."""

import os
import shutil
import tempfile

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rag.chunker import Chunk, ChunkType, generate_chunk_id
from rag.database import LocalDatabase


EMBEDDING_DIM = 8


# --- Strategies ---


def source_file_strategy():
    """Generate valid source file path strings."""
    return st.from_regex(r"[a-z][a-z0-9_/]{0,30}\.(md|py|txt)", fullmatch=True)


def content_strategy():
    """Generate non-empty content strings."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=200,
    )


def chunk_type_strategy():
    """Generate a random ChunkType."""
    return st.sampled_from([ChunkType.TEXT, ChunkType.CODE])


def vector_strategy(dim: int = EMBEDDING_DIM):
    """Generate a random non-zero vector of given dimension."""
    return st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ).filter(lambda v: any(x != 0.0 for x in v))


# --- Property 5: 数据库存储往返一致性 ---


@st.composite
def chunk_with_vector_strategy(draw):
    """Generate a random Chunk with a corresponding non-zero vector."""
    content = draw(content_strategy())
    source_file = draw(source_file_strategy())
    heading_level = draw(st.integers(min_value=0, max_value=6))
    chunk_type = draw(chunk_type_strategy())
    start_line = draw(st.integers(min_value=1, max_value=10000))
    heading_text = draw(st.text(min_size=0, max_size=100))
    vector = draw(vector_strategy())

    chunk_id = generate_chunk_id(content, source_file, start_line)
    chunk = Chunk(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        heading_level=heading_level,
        chunk_type=chunk_type,
        start_line=start_line,
        heading_text=heading_text,
    )
    return chunk, vector


class TestProperty5DatabaseRoundTripConsistency:
    """Property 5: 数据库存储往返一致性

    For any valid Chunk, storing it in the database and then retrieving it
    by ID should yield all fields identical to the original.

    **Validates: Requirements 2.2**
    """

    @settings(max_examples=50, deadline=None)
    @given(data=chunk_with_vector_strategy())
    def test_chunk_roundtrip_all_fields_match(self, data):
        """After storing a Chunk, retrieving it by ID returns all fields unchanged."""
        chunk, vector = data

        db_dir = tempfile.mkdtemp(prefix="rag_prop5_")
        try:
            db = LocalDatabase(db_path=db_dir, embedding_dim=EMBEDDING_DIM)
            db.initialize()

            db.store_chunks([chunk], [vector])

            retrieved = db.get_chunk_by_id(chunk.chunk_id)

            assert retrieved is not None
            assert retrieved.chunk_id == chunk.chunk_id
            assert retrieved.content == chunk.content
            assert retrieved.source_file == chunk.source_file
            assert retrieved.heading_level == chunk.heading_level
            assert retrieved.chunk_type == chunk.chunk_type
            assert retrieved.start_line == chunk.start_line
            assert retrieved.heading_text == chunk.heading_text
        finally:
            db.close()
            shutil.rmtree(db_dir, ignore_errors=True)


# --- Property 6: 向量搜索结果按余弦相似度排序 ---


class TestProperty6VectorSearchSortedByCosine:
    """Property 6: 向量搜索结果按余弦相似度排序

    **Validates: Requirements 2.3, 4.1, 4.2, 4.4**
    """

    @settings(max_examples=30, deadline=None)
    @given(
        num_chunks=st.integers(min_value=2, max_value=10),
        top_k=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    def test_results_sorted_by_cosine_similarity_descending(
        self, num_chunks, top_k, data
    ):
        """Results from search_by_cosine are sorted descending and count <= min(top_k, total)."""
        # Generate non-zero vectors
        vectors = []
        for _ in range(num_chunks):
            vec = data.draw(vector_strategy())
            vectors.append(vec)

        query_vec = data.draw(vector_strategy())

        # Create chunks
        chunks = []
        for i in range(num_chunks):
            content = f"chunk_content_{i}_{data.draw(st.text(min_size=1, max_size=10))}"
            chunk = Chunk(
                chunk_id=generate_chunk_id(content, "test.md", i),
                content=content,
                source_file="test.md",
                heading_level=1,
                chunk_type=ChunkType.TEXT,
                start_line=i + 1,
                heading_text="# Test",
            )
            chunks.append(chunk)

        db_dir = tempfile.mkdtemp(prefix="rag_prop6_")
        try:
            db = LocalDatabase(db_path=db_dir, embedding_dim=EMBEDDING_DIM)
            db.initialize()
            db.store_chunks(chunks, vectors)

            results = db.search_by_cosine(query_vec, top_k)

            # Results sorted descending
            if len(results) > 1:
                scores = [s for _, s in results]
                for i in range(len(scores) - 1):
                    assert scores[i] >= scores[i + 1] - 1e-9

            # Count constraint
            assert len(results) <= min(top_k, num_chunks)
        finally:
            db.close()
            shutil.rmtree(db_dir, ignore_errors=True)


# --- Property 7: 重新索引幂等性 ---


class TestProperty7ReindexIdempotency:
    """Property 7: 重新索引幂等性

    **Validates: Requirements 2.4**
    """

    @settings(max_examples=30, deadline=None)
    @given(
        source_file=source_file_strategy(),
        data=st.data(),
    )
    def test_reindex_only_keeps_second_set(self, source_file, data):
        """After re-indexing, only the second set of chunks exists."""
        # Generate two different content sets
        first_contents = data.draw(
            st.lists(content_strategy(), min_size=1, max_size=3, unique=True)
        )
        second_contents = data.draw(
            st.lists(content_strategy(), min_size=1, max_size=3, unique=True)
        )
        assume(set(first_contents) != set(second_contents))

        db_dir = tempfile.mkdtemp(prefix="rag_prop7_")
        try:
            db = LocalDatabase(db_path=db_dir, embedding_dim=EMBEDDING_DIM)
            db.initialize()

            # First indexing
            first_chunks = []
            first_vectors = []
            for i, content in enumerate(first_contents):
                chunk = Chunk(
                    chunk_id=generate_chunk_id(content, source_file, i + 1),
                    content=content,
                    source_file=source_file,
                    heading_level=1,
                    chunk_type=ChunkType.TEXT,
                    start_line=i + 1,
                    heading_text="# First",
                )
                first_chunks.append(chunk)
                rng = np.random.default_rng(hash(content) % (2**32))
                first_vectors.append(rng.random(EMBEDDING_DIM).tolist())

            db.store_chunks(first_chunks, first_vectors)

            # Re-index: delete + store new
            db.delete_by_source(source_file)

            second_chunks = []
            second_vectors = []
            for i, content in enumerate(second_contents):
                chunk = Chunk(
                    chunk_id=generate_chunk_id(content, source_file, i + 100),
                    content=content,
                    source_file=source_file,
                    heading_level=2,
                    chunk_type=ChunkType.TEXT,
                    start_line=i + 100,
                    heading_text="# Second",
                )
                second_chunks.append(chunk)
                rng = np.random.default_rng(hash(content) % (2**32))
                second_vectors.append(rng.random(EMBEDDING_DIM).tolist())

            db.store_chunks(second_chunks, second_vectors)

            # Verify only second set exists
            all_chunks = db.get_all_chunks()
            second_ids = {c.chunk_id for c in second_chunks}
            stored_ids = {c.chunk_id for c in all_chunks}

            assert stored_ids == second_ids
            assert len(all_chunks) == len(second_chunks)
        finally:
            db.close()
            shutil.rmtree(db_dir, ignore_errors=True)
