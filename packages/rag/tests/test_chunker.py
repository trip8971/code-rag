"""Unit tests for Chunk data model and ChunkType enum."""

import hashlib

from rag.chunker import Chunk, ChunkType, generate_chunk_id


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_text_value(self):
        assert ChunkType.TEXT.value == "text"

    def test_code_value(self):
        assert ChunkType.CODE.value == "code"

    def test_enum_members(self):
        assert set(ChunkType) == {ChunkType.TEXT, ChunkType.CODE}


class TestGenerateChunkId:
    """Tests for generate_chunk_id helper function."""

    def test_generates_sha256_hash(self):
        content = "Hello, world!"
        source_file = "docs/readme.md"
        expected = hashlib.sha256(f"{content}{source_file}0".encode("utf-8")).hexdigest()
        assert generate_chunk_id(content, source_file) == expected

    def test_different_content_produces_different_id(self):
        source_file = "docs/readme.md"
        id1 = generate_chunk_id("content A", source_file)
        id2 = generate_chunk_id("content B", source_file)
        assert id1 != id2

    def test_different_source_file_produces_different_id(self):
        content = "same content"
        id1 = generate_chunk_id(content, "file_a.md")
        id2 = generate_chunk_id(content, "file_b.md")
        assert id1 != id2

    def test_same_inputs_produce_same_id(self):
        content = "deterministic"
        source_file = "test.md"
        id1 = generate_chunk_id(content, source_file)
        id2 = generate_chunk_id(content, source_file)
        assert id1 == id2

    def test_returns_64_char_hex_string(self):
        chunk_id = generate_chunk_id("text", "file.md")
        assert len(chunk_id) == 64
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_empty_content(self):
        chunk_id = generate_chunk_id("", "file.md")
        expected = hashlib.sha256("file.md0".encode("utf-8")).hexdigest()
        assert chunk_id == expected

    def test_unicode_content(self):
        content = "你好世界"
        source_file = "文档/说明.md"
        expected = hashlib.sha256(f"{content}{source_file}0".encode("utf-8")).hexdigest()
        assert generate_chunk_id(content, source_file) == expected


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_text_chunk(self):
        chunk = Chunk(
            chunk_id="abc123",
            content="Some text content",
            source_file="docs/guide.md",
            heading_level=2,
            chunk_type=ChunkType.TEXT,
            start_line=10,
            heading_text="## Getting Started",
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.content == "Some text content"
        assert chunk.source_file == "docs/guide.md"
        assert chunk.heading_level == 2
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.start_line == 10
        assert chunk.heading_text == "## Getting Started"

    def test_create_code_chunk(self):
        chunk = Chunk(
            chunk_id="def456",
            content="def hello():\n    print('hi')",
            source_file="docs/api.md",
            heading_level=3,
            chunk_type=ChunkType.CODE,
            start_line=25,
            heading_text="### Examples",
        )
        assert chunk.chunk_type == ChunkType.CODE
        assert chunk.heading_level == 3

    def test_heading_level_zero_for_no_heading(self):
        chunk = Chunk(
            chunk_id="ghi789",
            content="Content without heading",
            source_file="notes.md",
            heading_level=0,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="",
        )
        assert chunk.heading_level == 0
        assert chunk.heading_text == ""

    def test_chunk_with_generated_id(self):
        content = "Test content"
        source_file = "test.md"
        chunk_id = generate_chunk_id(content, source_file)
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            heading_level=1,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="# Title",
        )
        expected_id = hashlib.sha256(f"{content}{source_file}0".encode("utf-8")).hexdigest()
        assert chunk.chunk_id == expected_id
