"""Property-based tests for Evaluator metrics.

Property 13: 评估指标数学正确性
For any retrieved list and relevant set, verifies that Precision@K, Recall@K,
and MRR@K are computed correctly according to their mathematical definitions.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from rag.evaluator import Evaluator


# --- Strategies ---

# Generate chunk IDs as short strings
chunk_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=10,
)

# Generate a list of retrieved chunk_ids (1-20 items)
retrieved_strategy = st.lists(chunk_id_strategy, min_size=1, max_size=20)

# Generate a list of relevant chunk_ids (1-10 items)
relevant_strategy = st.lists(chunk_id_strategy, min_size=1, max_size=10)

# Generate K value (1-20)
k_strategy = st.integers(min_value=1, max_value=20)


@settings(max_examples=100)
@given(
    retrieved=retrieved_strategy,
    relevant=relevant_strategy,
    k=k_strategy,
)
def test_property_13_precision_at_k_mathematical_correctness(
    retrieved: list[str], relevant: list[str], k: int
):
    """Property 13: Precision@K = |set(retrieved[:K]) ∩ set(relevant)| / K

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
    """
    evaluator = Evaluator(retriever=lambda q: [], k=10)

    # Compute expected value independently
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    expected_precision = len(retrieved_at_k & relevant_set) / k

    # Compute actual value from Evaluator
    actual_precision = evaluator._compute_precision_at_k(retrieved, relevant, k)

    assert abs(actual_precision - expected_precision) < 1e-9, (
        f"Precision@{k} mismatch: expected {expected_precision}, got {actual_precision}. "
        f"retrieved[:k]={retrieved[:k]}, relevant={relevant}"
    )


@settings(max_examples=100)
@given(
    retrieved=retrieved_strategy,
    relevant=relevant_strategy,
    k=k_strategy,
)
def test_property_13_recall_at_k_mathematical_correctness(
    retrieved: list[str], relevant: list[str], k: int
):
    """Property 13: Recall@K = |set(retrieved[:K]) ∩ set(relevant)| / |relevant| (0 if relevant is empty)

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
    """
    evaluator = Evaluator(retriever=lambda q: [], k=10)

    # Compute expected value independently
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        expected_recall = 0.0
    else:
        expected_recall = len(retrieved_at_k & relevant_set) / len(relevant_set)

    # Compute actual value from Evaluator
    actual_recall = evaluator._compute_recall_at_k(retrieved, relevant, k)

    assert abs(actual_recall - expected_recall) < 1e-9, (
        f"Recall@{k} mismatch: expected {expected_recall}, got {actual_recall}. "
        f"retrieved[:k]={retrieved[:k]}, relevant={relevant}"
    )


@settings(max_examples=100)
@given(
    retrieved=retrieved_strategy,
    relevant=relevant_strategy,
    k=k_strategy,
)
def test_property_13_mrr_at_k_mathematical_correctness(
    retrieved: list[str], relevant: list[str], k: int
):
    """Property 13: MRR@K = 1/rank of first relevant in retrieved[:K] (1-indexed), or 0 if none found

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
    """
    evaluator = Evaluator(retriever=lambda q: [], k=10)

    # Compute expected value independently
    relevant_set = set(relevant)
    expected_mrr = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant_set:
            expected_mrr = 1.0 / (i + 1)
            break

    # Compute actual value from Evaluator
    actual_mrr = evaluator._compute_mrr_at_k(retrieved, relevant, k)

    assert abs(actual_mrr - expected_mrr) < 1e-9, (
        f"MRR@{k} mismatch: expected {expected_mrr}, got {actual_mrr}. "
        f"retrieved[:k]={retrieved[:k]}, relevant={relevant}"
    )


@settings(max_examples=100)
@given(
    retrieved=retrieved_strategy,
    relevant=relevant_strategy,
    k=k_strategy,
)
def test_property_13_all_metrics_combined(
    retrieved: list[str], relevant: list[str], k: int
):
    """Property 13: All three metrics computed together maintain mathematical correctness.

    Verifies Precision@K, Recall@K, and MRR@K simultaneously for the same inputs.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**
    """
    evaluator = Evaluator(retriever=lambda q: [], k=10)

    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_at_k & relevant_set)

    # Precision@K
    expected_precision = hits / k
    actual_precision = evaluator._compute_precision_at_k(retrieved, relevant, k)
    assert abs(actual_precision - expected_precision) < 1e-9

    # Recall@K
    if len(relevant_set) == 0:
        expected_recall = 0.0
    else:
        expected_recall = hits / len(relevant_set)
    actual_recall = evaluator._compute_recall_at_k(retrieved, relevant, k)
    assert abs(actual_recall - expected_recall) < 1e-9

    # MRR@K
    expected_mrr = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant_set:
            expected_mrr = 1.0 / (i + 1)
            break
    actual_mrr = evaluator._compute_mrr_at_k(retrieved, relevant, k)
    assert abs(actual_mrr - expected_mrr) < 1e-9

    # Additional invariants:
    # Precision is always in [0, 1]
    assert 0.0 <= actual_precision <= 1.0
    # Recall is always in [0, 1]
    assert 0.0 <= actual_recall <= 1.0
    # MRR is always in [0, 1]
    assert 0.0 <= actual_mrr <= 1.0
