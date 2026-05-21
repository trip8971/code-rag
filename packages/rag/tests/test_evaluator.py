"""Unit tests for the Evaluator module."""

import pytest

from rag.evaluator import EvalRecord, EvalReport, Evaluator
from rag.exceptions import EvalDatasetError


class TestEvalRecord:
    """Tests for EvalRecord dataclass."""

    def test_create_eval_record(self):
        record = EvalRecord(query="test query", relevant_chunk_ids=["id1", "id2"])
        assert record.query == "test query"
        assert record.relevant_chunk_ids == ["id1", "id2"]


class TestEvalReport:
    """Tests for EvalReport dataclass."""

    def test_create_eval_report(self):
        report = EvalReport(
            precision_at_k=0.8,
            recall_at_k=0.6,
            mrr_at_k=0.5,
            per_query_details=[],
        )
        assert report.precision_at_k == 0.8
        assert report.recall_at_k == 0.6
        assert report.mrr_at_k == 0.5
        assert report.previous_report is None

    def test_comparison_without_previous_report(self):
        report = EvalReport(
            precision_at_k=0.8,
            recall_at_k=0.6,
            mrr_at_k=0.5,
            per_query_details=[],
        )
        assert report.comparison is None

    def test_comparison_with_previous_report(self):
        previous = EvalReport(
            precision_at_k=0.7,
            recall_at_k=0.5,
            mrr_at_k=0.4,
            per_query_details=[],
        )
        current = EvalReport(
            precision_at_k=0.8,
            recall_at_k=0.6,
            mrr_at_k=0.5,
            per_query_details=[],
            previous_report=previous,
        )
        comparison = current.comparison
        assert comparison is not None
        assert abs(comparison["precision_delta"] - 0.1) < 1e-9
        assert abs(comparison["recall_delta"] - 0.1) < 1e-9
        assert abs(comparison["mrr_delta"] - 0.1) < 1e-9


class TestEvaluatorValidateDataset:
    """Tests for Evaluator.validate_dataset()."""

    def test_empty_dataset_raises_error(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        with pytest.raises(EvalDatasetError) as exc_info:
            evaluator.validate_dataset([])
        assert exc_info.value.record_index is None
        assert "空" in str(exc_info.value)

    def test_empty_query_raises_error(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        dataset = [EvalRecord(query="", relevant_chunk_ids=["id1"])]
        with pytest.raises(EvalDatasetError) as exc_info:
            evaluator.validate_dataset(dataset)
        assert exc_info.value.record_index == 0

    def test_whitespace_only_query_raises_error(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        dataset = [EvalRecord(query="   ", relevant_chunk_ids=["id1"])]
        with pytest.raises(EvalDatasetError) as exc_info:
            evaluator.validate_dataset(dataset)
        assert exc_info.value.record_index == 0

    def test_empty_relevant_chunk_ids_raises_error(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        dataset = [EvalRecord(query="valid query", relevant_chunk_ids=[])]
        with pytest.raises(EvalDatasetError) as exc_info:
            evaluator.validate_dataset(dataset)
        assert exc_info.value.record_index == 0

    def test_valid_dataset_returns_empty_errors(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        dataset = [EvalRecord(query="valid query", relevant_chunk_ids=["id1"])]
        errors = evaluator.validate_dataset(dataset)
        assert errors == []

    def test_second_record_invalid(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        dataset = [
            EvalRecord(query="valid query", relevant_chunk_ids=["id1"]),
            EvalRecord(query="", relevant_chunk_ids=["id2"]),
        ]
        with pytest.raises(EvalDatasetError) as exc_info:
            evaluator.validate_dataset(dataset)
        assert exc_info.value.record_index == 1


class TestEvaluatorMetrics:
    """Tests for individual metric computation methods."""

    def setup_method(self):
        self.evaluator = Evaluator(retriever=lambda q: [], k=10)

    def test_precision_at_k_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        # Precision@3 = 3/3 = 1.0
        assert self.evaluator._compute_precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_at_k_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        # Precision@3 = 0/3 = 0.0
        assert self.evaluator._compute_precision_at_k(retrieved, relevant, 3) == 0.0

    def test_precision_at_k_partial(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = ["a", "b", "c"]
        # Precision@4 = |{a, x, b, y} ∩ {a, b, c}| / 4 = 2/4 = 0.5
        assert self.evaluator._compute_precision_at_k(retrieved, relevant, 4) == 0.5

    def test_precision_at_k_with_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c"]
        # Precision@5 = |{a, b} ∩ {a, b, c}| / 5 = 2/5 = 0.4
        assert self.evaluator._compute_precision_at_k(retrieved, relevant, 5) == 0.4

    def test_recall_at_k_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        # Recall@3 = 3/3 = 1.0
        assert self.evaluator._compute_recall_at_k(retrieved, relevant, 3) == 1.0

    def test_recall_at_k_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        # Recall@3 = 0/3 = 0.0
        assert self.evaluator._compute_recall_at_k(retrieved, relevant, 3) == 0.0

    def test_recall_at_k_partial(self):
        retrieved = ["a", "x", "b"]
        relevant = ["a", "b", "c", "d"]
        # Recall@3 = |{a, x, b} ∩ {a, b, c, d}| / 4 = 2/4 = 0.5
        assert self.evaluator._compute_recall_at_k(retrieved, relevant, 3) == 0.5

    def test_recall_at_k_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = []
        # Empty relevant → 0.0
        assert self.evaluator._compute_recall_at_k(retrieved, relevant, 3) == 0.0

    def test_mrr_at_k_first_is_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        # First relevant at position 1 → 1/1 = 1.0
        assert self.evaluator._compute_mrr_at_k(retrieved, relevant, 3) == 1.0

    def test_mrr_at_k_second_is_relevant(self):
        retrieved = ["x", "a", "b"]
        relevant = ["a"]
        # First relevant at position 2 → 1/2 = 0.5
        assert self.evaluator._compute_mrr_at_k(retrieved, relevant, 3) == 0.5

    def test_mrr_at_k_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        # No relevant in top-3 → 0.0
        assert self.evaluator._compute_mrr_at_k(retrieved, relevant, 3) == 0.0

    def test_mrr_at_k_relevant_beyond_k(self):
        retrieved = ["x", "y", "z", "a"]
        relevant = ["a"]
        # "a" is at position 4, but k=3 → 0.0
        assert self.evaluator._compute_mrr_at_k(retrieved, relevant, 3) == 0.0


class TestEvaluatorEvaluate:
    """Tests for Evaluator.evaluate() method."""

    def test_evaluate_perfect_retrieval(self):
        def retriever(query: str) -> list[str]:
            return ["id1", "id2", "id3"]

        evaluator = Evaluator(retriever=retriever, k=3)
        dataset = [
            EvalRecord(query="query1", relevant_chunk_ids=["id1", "id2", "id3"]),
        ]
        report = evaluator.evaluate(dataset)
        assert report.precision_at_k == 1.0
        assert report.recall_at_k == 1.0
        assert report.mrr_at_k == 1.0
        assert len(report.per_query_details) == 1
        assert report.per_query_details[0]["hits"] == 3
        assert report.per_query_details[0]["misses"] == []

    def test_evaluate_no_hits(self):
        def retriever(query: str) -> list[str]:
            return ["x", "y", "z"]

        evaluator = Evaluator(retriever=retriever, k=3)
        dataset = [
            EvalRecord(query="query1", relevant_chunk_ids=["id1", "id2"]),
        ]
        report = evaluator.evaluate(dataset)
        assert report.precision_at_k == 0.0
        assert report.recall_at_k == 0.0
        assert report.mrr_at_k == 0.0
        assert report.per_query_details[0]["hits"] == 0
        assert set(report.per_query_details[0]["misses"]) == {"id1", "id2"}

    def test_evaluate_multiple_queries_averaged(self):
        call_count = [0]

        def retriever(query: str) -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                return ["id1", "id2"]  # Perfect for first query
            else:
                return ["x", "y"]  # No hits for second query

        evaluator = Evaluator(retriever=retriever, k=2)
        dataset = [
            EvalRecord(query="query1", relevant_chunk_ids=["id1", "id2"]),
            EvalRecord(query="query2", relevant_chunk_ids=["id3", "id4"]),
        ]
        report = evaluator.evaluate(dataset)
        # First query: precision=1.0, recall=1.0, mrr=1.0
        # Second query: precision=0.0, recall=0.0, mrr=0.0
        # Average: 0.5, 0.5, 0.5
        assert report.precision_at_k == 0.5
        assert report.recall_at_k == 0.5
        assert report.mrr_at_k == 0.5

    def test_evaluate_with_previous_report(self):
        def retriever(query: str) -> list[str]:
            return ["id1", "id2"]

        evaluator = Evaluator(retriever=retriever, k=2)
        dataset = [
            EvalRecord(query="query1", relevant_chunk_ids=["id1", "id2"]),
        ]

        previous = EvalReport(
            precision_at_k=0.5,
            recall_at_k=0.5,
            mrr_at_k=0.5,
            per_query_details=[],
        )

        report = evaluator.evaluate(dataset, previous_report=previous)
        assert report.previous_report is previous
        comparison = report.comparison
        assert comparison is not None
        assert abs(comparison["precision_delta"] - 0.5) < 1e-9
        assert abs(comparison["recall_delta"] - 0.5) < 1e-9
        assert abs(comparison["mrr_delta"] - 0.5) < 1e-9

    def test_evaluate_invalid_dataset_raises(self):
        evaluator = Evaluator(retriever=lambda q: [], k=10)
        with pytest.raises(EvalDatasetError):
            evaluator.evaluate([])

    def test_evaluate_per_query_details_structure(self):
        def retriever(query: str) -> list[str]:
            return ["id1", "id3", "id5"]

        evaluator = Evaluator(retriever=retriever, k=3)
        dataset = [
            EvalRecord(query="test query", relevant_chunk_ids=["id1", "id2", "id3"]),
        ]
        report = evaluator.evaluate(dataset)
        detail = report.per_query_details[0]
        assert detail["query"] == "test query"
        assert detail["hits"] == 2  # id1 and id3
        assert detail["misses"] == ["id2"]  # id2 not in retrieved
        assert detail["retrieved_ids"] == ["id1", "id3", "id5"]
