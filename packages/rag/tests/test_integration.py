"""End-to-end integration tests for the RAG system.

Tests the complete indexing and retrieval pipelines with mocked external APIs.
Validates: Requirements 1.1, 2.1, 3.1, 4.1, 5.1, 6.1
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest

from rag.chunker import Chunk, ChunkType
from rag.config import RAGConfig, ServiceConfig
from rag.rag_system import RAGSystem


# --- Fixtures ---


@pytest.fixture
def embedding_dim():
    """Embedding dimension used across tests."""
    return 64


@pytest.fixture
def rag_config(tmp_path, embedding_dim):
    """Create a RAGConfig with fake URLs/keys for testing."""
    return RAGConfig(
        embedding=ServiceConfig(
            url="http://fake-embedding.local/v1/embeddings",
            api_key="fake-embedding-key",
            timeout=10,
            max_retries=1,
        ),
        reranker=ServiceConfig(
            url="http://fake-reranker.local/v1/rerank",
            api_key="fake-reranker-key",
            timeout=10,
            max_retries=1,
        ),
        query_rewriter=ServiceConfig(
            url="http://fake-rewriter.local/v1/chat/completions",
            api_key="fake-rewriter-key",
            timeout=10,
            max_retries=1,
        ),
        max_chunk_size=1500,
        chunk_overlap=200,
        dense_top_k=20,
        bm25_top_k=10,
        rerank_top_n=5,
        dense_weight=0.7,
        bm25_weight=0.3,
        db_path=str(tmp_path / "test_rag.db"),
        chroma_persist_dir=str(tmp_path / "chroma_db"),
        embedding_dim=embedding_dim,
    )


def make_fake_embedding_response(num_texts: int, dim: int) -> dict:
    """Create a fake embedding API response with random vectors."""
    data = []
    for i in range(num_texts):
        # Generate a deterministic but non-zero vector
        rng = np.random.default_rng(seed=i)
        vector = rng.standard_normal(dim).tolist()
        data.append({"embedding": vector, "index": i})
    return {"data": data, "model": "text-embedding-ada-002", "usage": {"total_tokens": 10}}


def make_fake_rewriter_response(rewritten_query: str) -> dict:
    """Create a fake query rewriter (chat completions) response."""
    return {
        "choices": [
            {
                "message": {"content": rewritten_query},
                "index": 0,
                "finish_reason": "stop",
            }
        ]
    }


def make_fake_reranker_response(num_docs: int, top_n: int) -> dict:
    """Create a fake reranker API response."""
    results = []
    for i in range(min(num_docs, top_n)):
        results.append({"index": i, "relevance_score": 1.0 - i * 0.1})
    return {"results": results}


@pytest.fixture
def mock_embedding_api(embedding_dim):
    """Mock httpx.AsyncClient.post to simulate embedding API responses."""

    async def mock_post(self, url, **kwargs):
        """Route mock responses based on URL."""
        if "embedding" in url:
            # Determine number of texts from the request body
            body = kwargs.get("json", {})
            texts = body.get("input", [])
            num_texts = len(texts) if isinstance(texts, list) else 1
            response_data = make_fake_embedding_response(num_texts, embedding_dim)
            response = httpx.Response(
                status_code=200,
                json=response_data,
                request=httpx.Request("POST", url),
            )
            return response
        elif "rerank" in url:
            body = kwargs.get("json", {})
            documents = body.get("documents", [])
            top_n = body.get("top_n", 5)
            response_data = make_fake_reranker_response(len(documents), top_n)
            response = httpx.Response(
                status_code=200,
                json=response_data,
                request=httpx.Request("POST", url),
            )
            return response
        elif "rewriter" in url or "chat" in url:
            response_data = make_fake_rewriter_response("improved search query")
            response = httpx.Response(
                status_code=200,
                json=response_data,
                request=httpx.Request("POST", url),
            )
            return response
        else:
            # Unknown URL - return 404
            return httpx.Response(
                status_code=404,
                request=httpx.Request("POST", url),
            )

    return mock_post


# --- Test: Full Indexing Flow ---


@pytest.mark.asyncio
async def test_full_indexing_flow(rag_config, mock_embedding_api, tmp_path):
    """Test complete indexing pipeline: document → chunking → vectorization (mock) → storage.

    Validates Requirements 1.1, 2.1
    """
    # Create a test Markdown document
    md_content = """# Introduction

This is a test document for the RAG system integration test.
It contains multiple sections with different content.

## API Reference

The `getUserName` function returns the current user's display name.
It accepts a user ID parameter and queries the database.

### Parameters

- `userId` (string): The unique identifier of the user
- `format` (string): Optional output format

## Code Example

```python
def get_user_name(user_id: str) -> str:
    user = db.query(user_id)
    return user.display_name
```

## Conclusion

This concludes the API documentation for the user module.
"""
    md_file = tmp_path / "test_doc.md"
    md_file.write_text(md_content, encoding="utf-8")

    with patch("httpx.AsyncClient.post", new=mock_embedding_api):
        system = RAGSystem(rag_config)
        chunk_count = await system.index_document(str(md_file))

    # Verify chunks were created
    assert chunk_count > 0

    # Verify chunks are stored in the database
    all_chunks = system.database.get_all_chunks()
    assert len(all_chunks) == chunk_count

    # Verify chunk metadata
    for chunk in all_chunks:
        assert chunk.source_file == str(md_file)
        assert chunk.heading_level >= 0
        assert chunk.heading_level <= 6
        assert chunk.chunk_type in (ChunkType.TEXT, ChunkType.CODE)
        assert chunk.start_line >= 1
        assert chunk.content.strip() != ""

    # Verify at least one code chunk exists (contains code blocks)
    code_chunks = [c for c in all_chunks if c.chunk_type == ChunkType.CODE]
    assert len(code_chunks) >= 1
    # Code chunk should contain the python code
    assert any("def get_user_name" in c.content for c in code_chunks)

    # Verify all content is preserved (text content exists in some chunk)
    all_content = "\n".join(c.content for c in all_chunks)
    assert "getUserName" in all_content
    assert "API Reference" in all_content


# --- Test: Full Retrieval Flow ---


@pytest.mark.asyncio
async def test_full_retrieval_flow(rag_config, mock_embedding_api, tmp_path):
    """Test complete retrieval pipeline: query → rewrite (mock) → retrieve → rerank (mock) → results.

    Validates Requirements 3.1, 4.1, 5.1, 6.1
    """
    # First, index a document
    md_content = """# User Authentication

The authentication module handles user login and session management.

## Login Function

The `authenticateUser` function validates credentials against the database.
It supports both password and token-based authentication.

```python
async def authenticate_user(username: str, password: str) -> bool:
    hashed = hash_password(password)
    user = await db.find_user(username)
    return user.password_hash == hashed
```

## Session Management

Sessions are stored in Redis with a configurable TTL.
The default session timeout is 30 minutes.
"""
    md_file = tmp_path / "auth_doc.md"
    md_file.write_text(md_content, encoding="utf-8")

    with patch("httpx.AsyncClient.post", new=mock_embedding_api):
        system = RAGSystem(rag_config)
        await system.index_document(str(md_file))

        # Now perform a retrieval query
        results = await system.retrieve("how to authenticate a user")

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) > 0

    # Each result should be a (Chunk, score) tuple
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)
        assert chunk.content.strip() != ""
        assert chunk.source_file == str(md_file)

    # Results should not exceed rerank_top_n
    assert len(results) <= rag_config.rerank_top_n


# --- Test: Re-indexing ---


@pytest.mark.asyncio
async def test_reindexing_replaces_old_chunks(rag_config, mock_embedding_api, tmp_path):
    """Test that re-indexing a document replaces old chunks with new ones.

    Validates Requirements 2.4
    """
    md_file = tmp_path / "evolving_doc.md"

    # First version of the document
    original_content = """# Original Document

This is the original content that will be replaced.

## Section A

Original section A content with specific keywords: alpha beta gamma.
"""
    md_file.write_text(original_content, encoding="utf-8")

    with patch("httpx.AsyncClient.post", new=mock_embedding_api):
        system = RAGSystem(rag_config)
        first_count = await system.index_document(str(md_file))

    assert first_count > 0
    first_chunks = system.database.get_all_chunks()
    first_chunk_ids = {c.chunk_id for c in first_chunks}

    # Second version of the document (different content)
    updated_content = """# Updated Document

This is completely new content after the update.

## Section B

Updated section B with different keywords: delta epsilon zeta.

## Section C

An entirely new section that didn't exist before.
"""
    md_file.write_text(updated_content, encoding="utf-8")

    with patch("httpx.AsyncClient.post", new=mock_embedding_api):
        second_count = await system.index_document(str(md_file))

    assert second_count > 0

    # Verify only new chunks exist
    current_chunks = system.database.get_all_chunks()
    current_chunk_ids = {c.chunk_id for c in current_chunks}

    # No old chunk IDs should remain
    assert first_chunk_ids.isdisjoint(current_chunk_ids), (
        "Old chunks should be completely replaced after re-indexing"
    )

    # All current chunks should be from the updated document
    for chunk in current_chunks:
        assert chunk.source_file == str(md_file)

    # The number of chunks should match the second indexing
    assert len(current_chunks) == second_count


# --- Test: Directory Indexing ---


@pytest.mark.asyncio
async def test_directory_indexing(rag_config, mock_embedding_api, tmp_path):
    """Test indexing all .md files in a directory.

    Validates Requirements 1.1, 2.1
    """
    # Create a temp directory with multiple .md files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    file1_content = """# Module A

Module A provides utility functions for string processing.

## Functions

- `trim_whitespace`: Removes leading and trailing whitespace
- `capitalize_words`: Capitalizes the first letter of each word
"""

    file2_content = """# Module B

Module B handles file I/O operations.

## Reading Files

```python
def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()
```
"""

    file3_content = """# Module C

Module C implements the caching layer.

## Cache Configuration

The cache uses LRU eviction with a configurable max size.
Default max size is 1000 entries.
"""

    (docs_dir / "module_a.md").write_text(file1_content, encoding="utf-8")
    (docs_dir / "module_b.md").write_text(file2_content, encoding="utf-8")
    (docs_dir / "module_c.md").write_text(file3_content, encoding="utf-8")

    # Also create a non-.md file that should be ignored
    (docs_dir / "readme.txt").write_text("This should be ignored", encoding="utf-8")

    with patch("httpx.AsyncClient.post", new=mock_embedding_api):
        system = RAGSystem(rag_config)
        results = await system.index_directory(str(docs_dir))

    # Verify all 3 .md files were indexed
    assert len(results) == 3

    # Verify each file has chunks
    for file_path, chunk_count in results.items():
        assert file_path.endswith(".md")
        assert chunk_count > 0

    # Verify all chunks are in the database
    all_chunks = system.database.get_all_chunks()
    total_chunks = sum(results.values())
    assert len(all_chunks) == total_chunks

    # Verify chunks come from all three source files
    source_files = {c.source_file for c in all_chunks}
    assert len(source_files) == 3

    # Verify the .txt file was not indexed
    for chunk in all_chunks:
        assert not chunk.source_file.endswith(".txt")
