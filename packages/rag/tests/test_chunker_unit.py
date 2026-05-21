"""Unit tests for Chunker class implementation."""

from rag.chunker import Chunk, ChunkType, Chunker


class TestChunkerInit:
    """Tests for Chunker initialization."""

    def test_default_params(self):
        chunker = Chunker()
        assert chunker.max_chunk_size == 1500
        assert chunker.overlap == 200

    def test_custom_params(self):
        chunker = Chunker(max_chunk_size=500, overlap=50)
        assert chunker.max_chunk_size == 500
        assert chunker.overlap == 50


class TestChunkDocumentEmpty:
    """Tests for empty document handling."""

    def test_empty_string(self):
        chunker = Chunker()
        assert chunker.chunk_document("", "test.md") == []

    def test_whitespace_only(self):
        chunker = Chunker()
        assert chunker.chunk_document("   \n\n  ", "test.md") == []

    def test_none_like_empty(self):
        chunker = Chunker()
        assert chunker.chunk_document("", "test.md") == []


class TestChunkDocumentHeadings:
    """Tests for heading-based splitting."""

    def test_single_heading_with_content(self):
        content = "# Title\n\nSome content here."
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        assert len(chunks) >= 1
        assert chunks[0].heading_level == 1
        assert chunks[0].heading_text == "# Title"
        assert chunks[0].chunk_type == ChunkType.TEXT

    def test_multiple_headings(self):
        content = "# H1\n\nContent 1\n\n## H2\n\nContent 2\n\n### H3\n\nContent 3"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        # Short sections may be merged; verify all content is preserved
        assert len(chunks) >= 1
        all_content = "\n".join(c.content for c in chunks)
        assert "Content 1" in all_content
        assert "Content 2" in all_content
        assert "Content 3" in all_content

    def test_content_before_heading(self):
        content = "Some intro text.\n\n# Title\n\nContent under title."
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        # First chunk should have heading_level 0
        assert chunks[0].heading_level == 0
        assert chunks[0].heading_text == ""

    def test_heading_levels_1_to_6(self):
        content = "# H1\ntext\n## H2\ntext\n### H3\ntext\n#### H4\ntext\n##### H5\ntext\n###### H6\ntext"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        # Short sections get merged; verify all heading text is preserved in output
        all_content = "\n".join(c.content for c in chunks)
        for level in range(1, 7):
            assert f"H{level}" in all_content


class TestChunkDocumentCodeBlocks:
    """Tests for code block handling."""

    def test_code_block_kept_intact(self):
        content = "# Title\n\n```python\ndef hello():\n    print('world')\n```\n"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 1
        assert "def hello():" in code_chunks[0].content
        assert "print('world')" in code_chunks[0].content

    def test_code_block_not_split_even_if_long(self):
        long_code = "x = 1\n" * 500  # Very long code block
        content = f"# Title\n\n```python\n{long_code}```\n"
        chunker = Chunker(max_chunk_size=100)
        chunks = chunker.chunk_document(content, "test.md")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 1
        assert len(code_chunks[0].content) > 100  # Not split

    def test_code_block_with_language_tag(self):
        content = "# Title\n\n```javascript\nconsole.log('hi');\n```\n"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 1
        assert "console.log" in code_chunks[0].content

    def test_multiple_code_blocks(self):
        content = "# Title\n\nText between.\n\n```python\ncode1\n```\n\nMore text.\n\n```js\ncode2\n```\n"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        # Short code blocks are merged with surrounding text into CODE-typed chunks
        assert len(code_chunks) >= 1
        # All code block content must be preserved
        all_content = "".join(c.content for c in chunks)
        assert "code1" in all_content
        assert "code2" in all_content


class TestSplitLongText:
    """Tests for _split_long_text method."""

    def test_short_text_not_split(self):
        chunker = Chunker(max_chunk_size=100)
        result = chunker._split_long_text("Short text.", 1)
        assert result == ["Short text."]

    def test_long_text_split_at_sentence_boundary(self):
        chunker = Chunker(max_chunk_size=50)
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = chunker._split_long_text(text, 1)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 50

    def test_split_at_chinese_sentence_endings(self):
        chunker = Chunker(max_chunk_size=20)
        text = "这是第一句话内容。这是第二句话内容。这是第三句话内容。"
        result = chunker._split_long_text(text, 1)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 20

    def test_split_at_newline(self):
        chunker = Chunker(max_chunk_size=30)
        text = "Line one content\nLine two content\nLine three content"
        result = chunker._split_long_text(text, 1)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 30

    def test_no_sentence_boundary_force_split(self):
        chunker = Chunker(max_chunk_size=10)
        text = "abcdefghijklmnopqrstuvwxyz"  # No sentence boundaries
        result = chunker._split_long_text(text, 1)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 10


class TestApplyOverlap:
    """Tests for _apply_overlap method."""

    def test_single_chunk_no_overlap(self):
        chunker = Chunker(overlap=50)
        result = chunker._apply_overlap(["only one chunk"])
        assert result == ["only one chunk"]

    def test_two_chunks_overlap_applied(self):
        chunker = Chunker(overlap=5)
        chunks = ["Hello World", "Next chunk"]
        result = chunker._apply_overlap(chunks)
        assert result[0] == "Hello World"
        # Second chunk should start with last 5 chars of first chunk
        assert result[1] == "WorldNext chunk"

    def test_overlap_larger_than_chunk(self):
        chunker = Chunker(overlap=100)
        chunks = ["Short", "Next"]
        result = chunker._apply_overlap(chunks)
        assert result[0] == "Short"
        # When overlap > chunk length, use entire previous chunk
        assert result[1] == "ShortNext"

    def test_zero_overlap(self):
        chunker = Chunker(overlap=0)
        chunks = ["First", "Second", "Third"]
        result = chunker._apply_overlap(chunks)
        assert result == ["First", "Second", "Third"]

    def test_multiple_chunks_overlap(self):
        chunker = Chunker(overlap=3)
        chunks = ["ABCDE", "FGHIJ", "KLMNO"]
        result = chunker._apply_overlap(chunks)
        assert result[0] == "ABCDE"
        assert result[1] == "CDEFGHIJ"
        assert result[2] == "HIJKLMNO"


class TestChunkDocumentMetadata:
    """Tests for chunk metadata correctness."""

    def test_source_file_preserved(self):
        content = "# Title\n\nContent here."
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "docs/guide.md")
        for chunk in chunks:
            assert chunk.source_file == "docs/guide.md"

    def test_chunk_id_generated(self):
        content = "# Title\n\nContent here."
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        for chunk in chunks:
            assert len(chunk.chunk_id) == 64
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_start_line_positive(self):
        content = "# Title\n\nContent here.\n\n## Section\n\nMore content."
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        for chunk in chunks:
            assert chunk.start_line >= 1

    def test_chunk_type_correct(self):
        content = "# Title\n\nText content.\n\n```python\ncode\n```\n"
        chunker = Chunker()
        chunks = chunker.chunk_document(content, "test.md")
        # Short code blocks are merged with text, resulting in CODE-typed chunks
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) >= 1
        # Code content must be preserved
        all_content = "".join(c.content for c in chunks)
        assert "code" in all_content


class TestSplitByHeadings:
    """Tests for _split_by_headings method."""

    def test_no_headings(self):
        chunker = Chunker()
        sections = chunker._split_by_headings("Just plain text.\nMore text.")
        assert len(sections) == 1
        assert sections[0]["heading_level"] == 0
        assert sections[0]["start_line"] == 1

    def test_single_heading(self):
        chunker = Chunker()
        sections = chunker._split_by_headings("# Title\n\nContent.")
        assert len(sections) == 1
        assert sections[0]["heading_level"] == 1
        assert sections[0]["heading_text"] == "# Title"

    def test_multiple_headings_same_level(self):
        chunker = Chunker()
        content = "## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        sections = chunker._split_by_headings(content)
        assert len(sections) == 2
        assert sections[0]["heading_text"] == "## Section A"
        assert sections[1]["heading_text"] == "## Section B"

    def test_nested_headings(self):
        chunker = Chunker()
        content = "# Main\n\nIntro.\n\n## Sub\n\nDetails."
        sections = chunker._split_by_headings(content)
        assert len(sections) == 2
        assert sections[0]["heading_level"] == 1
        assert sections[1]["heading_level"] == 2
