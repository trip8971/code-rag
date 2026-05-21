"""Property-based tests for Reranker module."""

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rag.chunker import Chunk, ChunkType, generate_chunk_id
from rag.config import ServiceConfig
from rag.reranker import Reranker


# --- Strategies ---


def chunk_strategy(id_suffix: st.SearchStrategy[str] | None = None):
    """Generate a random Chunk with a unique id suffix."""
    if id_suffix is None:
        id_suffix = st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=5,
            max_size=15,
        )
    return id_suffix.map(
        lambda suffix: Chunk(
            chunk_id=generate_chunk_id(suffix, "test.md"),
            content=f"content_{suffix}",
            source_file="test.md",
            heading_level=1,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="# Test",
        )
    )


@st.composite
def overlapping_results_strategy(draw):
    """Generate two result lists (dense and bm25) with some overlapping chunk_ids.

    Returns (dense_results, bm25_results, shared_ids) where:
    - dense_results contains chunks from dense_only_ids + shared_ids
    - bm25_results contains chunks from bm25_only_ids + shared_ids
    - shared_ids are chunk_ids that appear in both lists (with different content/metadata)
    """
    # Generate a pool of unique chunk_ids
    all_ids = draw(
        st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                min_size=5,
                max_size=15,
            ),
            min_size=3,
            max_size=20,
            unique=True,
        )
    )

    n = len(all_ids)
    # At least 1 shared ID to test deduplication behavior
    num_shared = draw(st.integers(min_value=1, max_value=max(1, n // 2)))
    remaining = n - num_shared
    num_dense_only = draw(st.integers(min_value=0, max_value=remaining))
    num_bm25_only = remaining - num_dense_only

    shared_ids = all_ids[:num_shared]
    dense_only_ids = all_ids[num_shared : num_shared + num_dense_only]
    bm25_only_ids = all_ids[num_shared + num_dense_only :]

    # Build dense_results: dense_only + shared (with dense-specific metadata)
    dense_results: list[tuple[Chunk, float]] = []
    for suffix in dense_only_ids + shared_ids:
        chunk = Chunk(
            chunk_id=generate_chunk_id(suffix, "dense.md"),
            content=f"dense_content_{suffix}",
            source_file="dense.md",
            heading_level=1,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="# Dense",
        )
        score = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
        dense_results.append((chunk, score))

    # Build bm25_results: bm25_only + shared (with bm25-specific metadata)
    bm25_results: list[tuple[Chunk, float]] = []
    for suffix in bm25_only_ids + shared_ids:
        chunk = Chunk(
            chunk_id=generate_chunk_id(suffix, "bm25.md"),
            content=f"bm25_content_{suffix}",
            source_file="bm25.md",
            heading_level=2,
            chunk_type=ChunkType.TEXT,
            start_line=10,
            heading_text="# BM25",
        )
        score = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
        bm25_results.append((chunk, score))

    return (dense_results, bm25_results, shared_ids, dense_only_ids, bm25_only_ids)


# --- Property 11: Reranker 合并去重保留首次出现 ---


@settings(max_examples=100)
@given(data=overlapping_results_strategy())
def test_property_11_no_duplicate_chunk_ids(data):
    """Property 11: Merged results should contain no duplicate chunk_ids.

    **Validates: Requirements 6.1**
    """
    dense_results, bm25_results, shared_ids, dense_only_ids, bm25_only_ids = data

    config = ServiceConfig(url="http://fake.example.com/rerank", api_key="fake-key")
    reranker = Reranker(config=config, dense_weight=0.7, bm25_weight=0.3)

    merged = reranker._merge_and_deduplicate(dense_results, bm25_results)

    # (a) No duplicate chunk_ids
    merged_ids = [chunk.chunk_id for chunk in merged]
    assert len(merged_ids) == len(set(merged_ids)), (
        f"Duplicate chunk_ids found in merged results"
    )


@settings(max_examples=100)
@given(data=overlapping_results_strategy())
def test_property_11_preserves_first_occurrence_from_dense(data):
    """Property 11: For shared chunk_ids, the entry from dense_results (first traversed) is preserved.

    **Validates: Requirements 6.1**
    """
    dense_results, bm25_results, shared_ids, dense_only_ids, bm25_only_ids = data

    config = ServiceConfig(url="http://fake.example.com/rerank", api_key="fake-key")
    reranker = Reranker(config=config, dense_weight=0.7, bm25_weight=0.3)

    merged = reranker._merge_and_deduplicate(dense_results, bm25_results)

    # Build lookup from merged results
    merged_map = {chunk.chunk_id: chunk for chunk in merged}

    # Build lookup from dense_results (first occurrence per chunk_id)
    dense_map: dict[str, Chunk] = {}
    for chunk, _ in dense_results:
        if chunk.chunk_id not in dense_map:
            dense_map[chunk.chunk_id] = chunk

    # For shared chunk_ids, the dense version should be preserved
    for suffix in shared_ids:
        dense_chunk_id = generate_chunk_id(suffix, "dense.md")
        bm25_chunk_id = generate_chunk_id(suffix, "bm25.md")

        # The dense chunk_id should be in merged (since dense is traversed first)
        if dense_chunk_id in dense_map:
            assert dense_chunk_id in merged_map, (
                f"Dense chunk_id {dense_chunk_id} for shared suffix '{suffix}' "
                f"missing from merged results"
            )
            # Verify it's the dense version (content starts with "dense_content_")
            assert merged_map[dense_chunk_id].content == dense_map[dense_chunk_id].content
            assert merged_map[dense_chunk_id].source_file == "dense.md"


@settings(max_examples=100)
@given(data=overlapping_results_strategy())
def test_property_11_total_count_equals_unique_ids(data):
    """Property 11: Total count of merged results equals the number of unique chunk_ids.

    **Validates: Requirements 6.1**
    """
    dense_results, bm25_results, shared_ids, dense_only_ids, bm25_only_ids = data

    config = ServiceConfig(url="http://fake.example.com/rerank", api_key="fake-key")
    reranker = Reranker(config=config, dense_weight=0.7, bm25_weight=0.3)

    merged = reranker._merge_and_deduplicate(dense_results, bm25_results)

    # (c) Total count equals number of unique chunk_ids across both lists
    all_ids = set()
    for chunk, _ in dense_results:
        all_ids.add(chunk.chunk_id)
    for chunk, _ in bm25_results:
        all_ids.add(chunk.chunk_id)

    assert len(merged) == len(all_ids), (
        f"Expected {len(all_ids)} unique chunks but got {len(merged)}"
    )


def scored_results_strategy(min_size: int = 1, max_size: int = 10):
    """Generate a list of (Chunk, score) tuples with unique chunk_ids and scores in [0, 1]."""
    return st.lists(
        st.tuples(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                min_size=5,
                max_size=15,
            ),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=min_size,
        max_size=max_size,
        unique_by=lambda x: x[0],
    ).map(
        lambda items: [
            (
                Chunk(
                    chunk_id=generate_chunk_id(suffix, "test.md"),
                    content=f"content_{suffix}",
                    source_file="test.md",
                    heading_level=1,
                    chunk_type=ChunkType.TEXT,
                    start_line=1,
                    heading_text="# Test",
                ),
                score,
            )
            for suffix, score in items
        ]
    )


def weight_pair_strategy():
    """Generate (dense_weight, bm25_weight) positive floats that sum to 1.0."""
    return st.floats(
        min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
    ).map(lambda w: (w, 1.0 - w))


# --- Property 12: Reranker 回退加权融合正确性 ---


@settings(max_examples=100)
@given(
    weights=weight_pair_strategy(),
    dense_data=st.lists(
        st.tuples(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                min_size=5,
                max_size=15,
            ),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
        unique_by=lambda x: x[0],
    ),
    bm25_data=st.lists(
        st.tuples(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                min_size=5,
                max_size=15,
            ),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
        unique_by=lambda x: x[0],
    ),
)
def test_property_12_fallback_weighted_fusion_correctness(
    weights, dense_data, bm25_data
):
    """Property 12: Reranker 回退加权融合正确性

    For any set of dense and bm25 results with scores, the fallback scoring should produce:
    score = dense_weight × dense_score + bm25_weight × bm25_score
    and results should be sorted by score in descending order.

    **Validates: Requirements 6.5**
    """
    dense_weight, bm25_weight = weights

    # Build Chunk objects from generated data
    dense_results: list[tuple[Chunk, float]] = [
        (
            Chunk(
                chunk_id=generate_chunk_id(suffix, "test.md"),
                content=f"content_{suffix}",
                source_file="test.md",
                heading_level=1,
                chunk_type=ChunkType.TEXT,
                start_line=1,
                heading_text="# Test",
            ),
            score,
        )
        for suffix, score in dense_data
    ]

    bm25_results: list[tuple[Chunk, float]] = [
        (
            Chunk(
                chunk_id=generate_chunk_id(suffix, "test.md"),
                content=f"content_{suffix}",
                source_file="test.md",
                heading_level=1,
                chunk_type=ChunkType.TEXT,
                start_line=1,
                heading_text="# Test",
            ),
            score,
        )
        for suffix, score in bm25_data
    ]

    # Create Reranker with the generated weights
    config = ServiceConfig(url="http://fake.example.com/rerank", api_key="fake-key")
    reranker = Reranker(config=config, dense_weight=dense_weight, bm25_weight=bm25_weight)

    # Call _fallback_score
    results = reranker._fallback_score(dense_results, bm25_results)

    # Build expected score maps
    dense_score_map: dict[str, float] = {}
    for chunk, score in dense_results:
        if chunk.chunk_id not in dense_score_map:
            dense_score_map[chunk.chunk_id] = score

    bm25_score_map: dict[str, float] = {}
    for chunk, score in bm25_results:
        if chunk.chunk_id not in bm25_score_map:
            bm25_score_map[chunk.chunk_id] = score

    all_chunk_ids = set(dense_score_map.keys()) | set(bm25_score_map.keys())

    # Property 1: Each result's score equals dense_weight * dense_score + bm25_weight * bm25_score
    for chunk, fused_score in results:
        d_score = dense_score_map.get(chunk.chunk_id, 0.0)
        b_score = bm25_score_map.get(chunk.chunk_id, 0.0)
        expected_score = dense_weight * d_score + bm25_weight * b_score
        assert math.isclose(fused_score, expected_score, rel_tol=1e-9, abs_tol=1e-9), (
            f"Chunk {chunk.chunk_id}: expected score {expected_score}, got {fused_score}"
        )

    # Property 2: Results are sorted by score in descending order
    scores = [score for _, score in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Results not sorted descending at index {i}: {scores[i]} < {scores[i + 1]}"
        )

    # Property 3: All unique chunk_ids from both lists appear in the result
    result_chunk_ids = {chunk.chunk_id for chunk, _ in results}
    assert result_chunk_ids == all_chunk_ids, (
        f"Missing chunk_ids: {all_chunk_ids - result_chunk_ids}, "
        f"Extra chunk_ids: {result_chunk_ids - all_chunk_ids}"
    )
