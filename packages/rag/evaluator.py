"""Retrieval quality evaluation module.

Provides the Evaluator class for measuring retrieval quality using
Precision@K, Recall@K, and MRR@K metrics against annotated datasets.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .exceptions import EvalDatasetError


@dataclass
class EvalRecord:
    """评估数据集中的单条记录。

    Attributes:
        query: 查询文本
        relevant_chunk_ids: 标注的相关 Chunk ID 列表
    """

    query: str
    relevant_chunk_ids: list[str]


@dataclass
class EvalReport:
    """评估报告。

    Attributes:
        precision_at_k: Precision@K 指标均值
        recall_at_k: Recall@K 指标均值
        mrr_at_k: MRR@K 指标均值
        per_query_details: 每个查询的详细匹配信息
        previous_report: 上一次评估报告（用于对比）
    """

    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    per_query_details: list[dict]
    previous_report: Optional["EvalReport"] = None

    @property
    def comparison(self) -> Optional[dict[str, float]]:
        """计算与上一次评估报告的数值变化对比。

        Returns:
            包含 precision_delta、recall_delta、mrr_delta 的字典，
            如果没有上一次报告则返回 None。
        """
        if self.previous_report is None:
            return None
        return {
            "precision_delta": self.precision_at_k - self.previous_report.precision_at_k,
            "recall_delta": self.recall_at_k - self.previous_report.recall_at_k,
            "mrr_delta": self.mrr_at_k - self.previous_report.mrr_at_k,
        }


class Evaluator:
    """检索质量评估器。

    使用标注数据集评估检索系统的 Precision@K、Recall@K 和 MRR@K 指标。

    Args:
        retriever: 检索函数，接受查询字符串，返回 chunk_id 字符串列表
        k: 评估时考虑的最大排名深度，默认为 10
    """

    def __init__(self, retriever: Callable[[str], list[str]], k: int = 10):
        self._retriever = retriever
        self._k = k

    def evaluate(
        self, dataset: list[EvalRecord], previous_report: Optional[EvalReport] = None
    ) -> EvalReport:
        """执行评估，返回评估报告。

        对数据集中每条记录执行检索并计算指标，最终返回所有查询的平均指标。

        Args:
            dataset: 评估数据集，包含查询和标注的相关 Chunk ID
            previous_report: 上一次评估报告，用于对比

        Returns:
            EvalReport 评估报告

        Raises:
            EvalDatasetError: 数据集格式无效时抛出
        """
        self.validate_dataset(dataset)

        precisions: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        per_query_details: list[dict] = []

        for record in dataset:
            # 调用检索器获取结果
            retrieved_ids = self._retriever(record.query)

            # 计算指标
            precision = self._compute_precision_at_k(
                retrieved_ids, record.relevant_chunk_ids, self._k
            )
            recall = self._compute_recall_at_k(
                retrieved_ids, record.relevant_chunk_ids, self._k
            )
            mrr = self._compute_mrr_at_k(
                retrieved_ids, record.relevant_chunk_ids, self._k
            )

            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)

            # 计算每个查询的详细信息
            retrieved_top_k = retrieved_ids[: self._k]
            relevant_set = set(record.relevant_chunk_ids)
            hits = len(set(retrieved_top_k) & relevant_set)
            misses = [
                cid for cid in record.relevant_chunk_ids if cid not in set(retrieved_top_k)
            ]

            per_query_details.append(
                {
                    "query": record.query,
                    "hits": hits,
                    "misses": misses,
                    "retrieved_ids": retrieved_top_k,
                }
            )

        # 计算平均指标
        n = len(dataset)
        avg_precision = sum(precisions) / n
        avg_recall = sum(recalls) / n
        avg_mrr = sum(mrrs) / n

        return EvalReport(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            mrr_at_k=avg_mrr,
            per_query_details=per_query_details,
            previous_report=previous_report,
        )

    def _compute_precision_at_k(
        self, retrieved: list[str], relevant: list[str], k: int
    ) -> float:
        """计算 Precision@K。

        Precision@K = |retrieved[:K] ∩ relevant| / K

        Args:
            retrieved: 检索结果 ID 列表（按排名顺序）
            relevant: 标注的相关 ID 列表
            k: 截断深度

        Returns:
            Precision@K 值，范围 [0, 1]
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(retrieved_at_k & relevant_set)
        return hits / k

    def _compute_recall_at_k(
        self, retrieved: list[str], relevant: list[str], k: int
    ) -> float:
        """计算 Recall@K。

        Recall@K = |retrieved[:K] ∩ relevant| / |relevant|
        如果 relevant 为空，返回 0。

        Args:
            retrieved: 检索结果 ID 列表（按排名顺序）
            relevant: 标注的相关 ID 列表
            k: 截断深度

        Returns:
            Recall@K 值，范围 [0, 1]
        """
        if not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(retrieved_at_k & relevant_set)
        return hits / len(relevant_set)

    def _compute_mrr_at_k(
        self, retrieved: list[str], relevant: list[str], k: int
    ) -> float:
        """计算 MRR@K (Mean Reciprocal Rank)。

        MRR@K = 1/rank，其中 rank 为 retrieved[:K] 中第一个 relevant 元素的位置（1-indexed）。
        如果 retrieved[:K] 中没有 relevant 元素，返回 0。

        Args:
            retrieved: 检索结果 ID 列表（按排名顺序）
            relevant: 标注的相关 ID 列表
            k: 截断深度

        Returns:
            Reciprocal Rank 值，范围 [0, 1]
        """
        relevant_set = set(relevant)
        for i, chunk_id in enumerate(retrieved[:k]):
            if chunk_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def validate_dataset(self, dataset: list[EvalRecord]) -> list[str]:
        """验证数据集格式。

        检查数据集非空，且每条记录包含非空的 query 和非空的 relevant_chunk_ids 列表。
        无效时抛出 EvalDatasetError。

        Args:
            dataset: 待验证的评估数据集

        Returns:
            错误信息列表（空列表表示验证通过）

        Raises:
            EvalDatasetError: 数据集为空或记录格式无效时抛出
        """
        errors: list[str] = []

        if not dataset:
            raise EvalDatasetError(
                message="评估数据集为空",
                record_index=None,
            )

        for i, record in enumerate(dataset):
            if not record.query or not record.query.strip():
                raise EvalDatasetError(
                    message=f"记录 {i} 的查询文本为空",
                    record_index=i,
                )
            if not record.relevant_chunk_ids:
                raise EvalDatasetError(
                    message=f"记录 {i} 的相关 Chunk ID 列表为空",
                    record_index=i,
                )

        return errors
