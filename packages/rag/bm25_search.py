"""BM25 sparse keyword search with Chinese + code identifier tokenization."""

import logging
import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .database import LocalDatabase

logger = logging.getLogger(__name__)

# 预编译正则：检测是否包含中文字符
_HAS_CHINESE = re.compile(r"[\u4e00-\u9fff]")

# 预编译正则：camelCase 拆分
_CAMEL_SPLIT = re.compile(r"([A-Z])")

# BM25 索引持久化文件名
_BM25_INDEX_FILE = "bm25_index.pkl"


class BM25Search:
    """基于 BM25 算法的稀疏关键词检索。

    使用 jieba 分词处理中文文本，同时支持驼峰命名和下划线命名的
    代码标识符拆分。索引支持 pickle 持久化，避免每次启动重建。
    """

    def __init__(self, database: LocalDatabase):
        """初始化 BM25 检索器。

        Args:
            database: 本地数据库实例，用于加载 Chunk 数据
        """
        self.database = database
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

        # 持久化路径与 ChromaDB 放在同一目录
        self._persist_path = Path(database.db_path) / _BM25_INDEX_FILE

    def build_index(self) -> None:
        """从数据库加载所有 Chunk 构建 BM25 索引并持久化。"""
        self._chunks = self.database.get_all_chunks()

        if not self._chunks:
            self._index = None
            return

        tokenized_corpus = [self._tokenize(chunk.embedding_text) for chunk in self._chunks]
        self._index = BM25Okapi(tokenized_corpus)

        # 持久化到磁盘
        self._save()

    def load_index(self) -> bool:
        """从磁盘加载持久化的 BM25 索引。

        Returns:
            加载成功返回 True，文件不存在或加载失败返回 False
        """
        if not self._persist_path.exists():
            return False

        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._index = data["index"]
            self._chunks = data["chunks"]
            logger.info(
                "Loaded BM25 index from disk: %d chunks", len(self._chunks)
            )
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 index from disk: %s", str(e))
            return False

    def _save(self) -> None:
        """将 BM25 索引持久化到磁盘。"""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "wb") as f:
                pickle.dump(
                    {"index": self._index, "chunks": self._chunks},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(
                "Saved BM25 index to disk: %d chunks", len(self._chunks)
            )
        except Exception as e:
            logger.warning("Failed to save BM25 index: %s", str(e))

    def ensure_index(self) -> None:
        """确保索引可用：先尝试从磁盘加载，失败则从数据库重建。"""
        if self._index is not None:
            return

        if not self.load_index():
            self.build_index()

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """执行 BM25 检索，返回 (Chunk, score) 列表。

        仅返回评分 > 0 的结果，按评分降序排列。

        Args:
            query: 查询文本
            top_k: 返回的最大结果数量，范围 1-100，默认 10

        Returns:
            按 BM25 评分降序排列的 (Chunk, score) 列表
        """
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        scored_results: list[tuple[Chunk, float]] = []
        for chunk, score in zip(self._chunks, scores):
            if score > 0:
                scored_results.append((chunk, float(score)))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """混合分词：中文用 jieba，英文/代码标识符用 camelCase + snake_case 拆分。

        处理流程：
        1. 按空白字符拆分为 token
        2. 对每个 token：
           - 如果包含中文 → 用 jieba 精确模式分词
           - 否则 → 按下划线拆分（snake_case），再按大写字母拆分（camelCase）
        3. 所有结果转为小写，过滤空串和单字符标点

        Args:
            text: 输入文本

        Returns:
            分词后的小写 token 列表

        Examples:
            >>> bm25._tokenize("getUserName")
            ['get', 'user', 'name']
            >>> bm25._tokenize("get_user_name")
            ['get', 'user', 'name']
            >>> bm25._tokenize("如何使用线性层")
            ['如何', '使用', '线性', '层']
            >>> bm25._tokenize("torch.nn.Linear 线性层")
            ['torch', 'nn', 'linear', '线性', '层']
        """
        tokens: list[str] = []

        # Split by whitespace first
        words = text.split()

        for word in words:
            if _HAS_CHINESE.search(word):
                # 包含中文：用 jieba 分词
                seg_list = jieba.cut(word, cut_all=False)
                for seg in seg_list:
                    seg = seg.strip()
                    if seg and len(seg) > 0:
                        tokens.append(seg.lower())
            else:
                # 纯英文/代码：按 snake_case 和 camelCase 拆分
                parts = word.split("_")
                for part in parts:
                    if not part:
                        continue
                    # 按 . 拆分（如 torch.nn.Linear）
                    dot_parts = part.split(".")
                    for dp in dot_parts:
                        if not dp:
                            continue
                        # camelCase 拆分
                        camel_split = _CAMEL_SPLIT.sub(r" \1", dp).split()
                        for token in camel_split:
                            stripped = token.strip()
                            if stripped:
                                tokens.append(stripped.lower())

        return tokens

    def update_index(self, chunks: list[Chunk]) -> None:
        """更新 BM25 索引以反映数据库当前状态。"""
        self.build_index()

    def rebuild_index(self) -> None:
        """从数据库完全重建 BM25 索引。"""
        self.build_index()
