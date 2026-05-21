"""Local vector database using ChromaDB for chunk and vector storage."""

import logging
from typing import Optional

import chromadb

from .chunker import Chunk, ChunkType

logger = logging.getLogger(__name__)


class LocalDatabase:
    """基于 ChromaDB 的本地向量数据库。

    使用 ChromaDB 持久化存储文档片段及其向量表示，
    支持高效的余弦相似度搜索。
    """

    def __init__(self, db_path: str, embedding_dim: int):
        """初始化数据库连接。

        Args:
            db_path: ChromaDB 持久化目录路径
            embedding_dim: 向量维度
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端，懒初始化。"""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """获取 ChromaDB collection，懒初始化。"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="rag_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def initialize(self) -> None:
        """初始化数据库。

        确保 ChromaDB collection 已创建。
        """
        # Trigger lazy initialization
        _ = self.collection

    def store_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """存储文档片段及其向量。

        Args:
            chunks: 文档片段列表
            embeddings: 对应的向量列表，每个向量为 float 列表

        Raises:
            ValueError: 当 chunks 和 embeddings 长度不一致时
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length, "
                f"got {len(chunks)} and {len(embeddings)}"
            )

        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "heading_level": chunk.heading_level,
                "chunk_type": chunk.chunk_type.value,
                "start_line": chunk.start_line,
                "heading_text": chunk.heading_text,
                "context": chunk.context,
            }
            for chunk in chunks
        ]

        # ChromaDB upsert handles both insert and update
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_source(self, source_file: str) -> int:
        """删除指定源文件的所有 Chunk，返回删除数量。

        Args:
            source_file: 源文件路径

        Returns:
            删除的 Chunk 数量
        """
        # Query for all chunks with this source_file
        results = self.collection.get(
            where={"source_file": source_file},
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        return count

    def search_by_cosine(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """基于余弦相似度搜索，返回 (Chunk, score) 列表，按相似度降序。

        Args:
            query_vector: 查询向量
            top_k: 返回的最大结果数量

        Returns:
            按相似度降序排列的 (Chunk, score) 列表
        """
        # Check if collection is empty
        if self.collection.count() == 0:
            return []

        # Clamp top_k to collection size
        actual_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output: list[tuple[Chunk, float]] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            # ChromaDB cosine distance = 1 - cosine_similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            chunk = Chunk(
                chunk_id=chunk_id,
                content=document,
                source_file=metadata["source_file"],
                heading_level=metadata["heading_level"],
                chunk_type=ChunkType(metadata["chunk_type"]),
                start_line=metadata["start_line"],
                heading_text=metadata["heading_text"],
                context=metadata.get("context", ""),
            )
            output.append((chunk, similarity))

        return output

    def get_all_chunks(self) -> list[Chunk]:
        """获取所有 Chunk。

        Returns:
            所有存储的 Chunk 列表
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        chunks: list[Chunk] = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            chunk = Chunk(
                chunk_id=chunk_id,
                content=document,
                source_file=metadata["source_file"],
                heading_level=metadata["heading_level"],
                chunk_type=ChunkType(metadata["chunk_type"]),
                start_line=metadata["start_line"],
                heading_text=metadata["heading_text"],
                context=metadata.get("context", ""),
            )
            chunks.append(chunk)

        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """根据 ID 获取单个 Chunk。

        Args:
            chunk_id: Chunk 的唯一标识符

        Returns:
            对应的 Chunk，如果不存在则返回 None
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
        except Exception:
            return None

        if not results["ids"]:
            return None

        metadata = results["metadatas"][0]
        document = results["documents"][0]

        return Chunk(
            chunk_id=chunk_id,
            content=document,
            source_file=metadata["source_file"],
            heading_level=metadata["heading_level"],
            chunk_type=ChunkType(metadata["chunk_type"]),
            start_line=metadata["start_line"],
            heading_text=metadata["heading_text"],
            context=metadata.get("context", ""),
        )

    def close(self) -> None:
        """关闭数据库连接。"""
        self._collection = None
        self._client = None
