"""Property-based tests for Chunker module."""

import re
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.chunker import ChunkType, Chunker


# --- Strategies ---


def language_tag_strategy():
    """Generate random programming language tags for code blocks."""
    return st.sampled_from([
        "python", "javascript", "typescript", "java", "c", "cpp", "go",
        "rust", "ruby", "bash", "sql", "html", "css", "json", "yaml",
        "markdown", "shell", "kotlin", "swift", "scala", "",
    ])


def code_content_strategy():
    """Generate random code content (printable ASCII, no triple backticks)."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z", "S"),
            blacklist_characters="`",
        ),
        min_size=1,
        max_size=500,
    ).map(lambda s: s.replace("```", "---"))


def long_code_content_strategy(min_size: int = 2000):
    """Generate long code content that exceeds typical max_chunk_size."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z", "S"),
            blacklist_characters="`",
        ),
        min_size=min_size,
        max_size=5000,
    ).map(lambda s: s.replace("```", "---"))


@st.composite
def markdown_with_code_blocks_strategy(draw):
    """Generate a Markdown document containing one or more code blocks.

    The document has optional text before/after/between code blocks,
    and code blocks with random language tags and content.
    """
    num_code_blocks = draw(st.integers(min_value=1, max_value=4))

    parts = []

    # Optional heading at the start
    if draw(st.booleans()):
        level = draw(st.integers(min_value=1, max_value=6))
        heading = "#" * level + " " + draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=30,
        ))
        parts.append(heading + "\n\n")

    for i in range(num_code_blocks):
        # Optional text before code block
        if draw(st.booleans()):
            text = draw(st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                    blacklist_characters="`",
                ),
                min_size=1,
                max_size=100,
            )).replace("```", "---")
            parts.append(text + "\n\n")

        # Code block
        lang = draw(language_tag_strategy())
        # Decide whether to generate a long code block (exceeding max_chunk_size)
        if draw(st.booleans()):
            code = draw(long_code_content_strategy(min_size=2000))
        else:
            code = draw(code_content_strategy())

        # Ensure code content has at least one newline for realistic code
        if "\n" not in code:
            code = code + "\n"

        code_block = f"```{lang}\n{code}\n```"
        parts.append(code_block + "\n\n")

    # Optional trailing text
    if draw(st.booleans()):
        trailing = draw(st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_characters="`",
            ),
            min_size=1,
            max_size=100,
        )).replace("```", "---")
        parts.append(trailing + "\n")

    document = "".join(parts)
    return document


# --- Property 2: 代码块完整性不变量 ---


class TestProperty2CodeBlockIntegrity:
    """Property 2: 代码块完整性不变量

    For any Markdown 文档中的代码块，无论其长度是否超过 max_chunk_size，
    该代码块应作为一个完整的 Chunk 出现在结果中，不被拆分。

    **Validates: Requirements 1.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(document=markdown_with_code_blocks_strategy())
    def test_code_blocks_appear_as_complete_chunks(self, document):
        """Each code block from the original document is preserved intact
        within some CODE-typed chunk. Short code blocks may be merged with
        surrounding text, but their content is never split or lost.

        **Validates: Requirements 1.3**
        """
        # Use a small max_chunk_size to stress-test that code blocks are never split
        chunker = Chunker(max_chunk_size=200, overlap=50)
        chunks = chunker.chunk_document(document, "test_doc.md")

        # Extract all code blocks from the original document using the same
        # regex pattern the Chunker uses
        code_block_pattern = re.compile(
            r"^(```[^\n]*\n.*?^```)", re.MULTILINE | re.DOTALL
        )
        original_code_blocks = code_block_pattern.findall(document)

        # Get all CODE type chunks from the result
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]

        # Verification 1: There is at least one CODE chunk if there are code blocks
        if original_code_blocks:
            assert len(code_chunks) >= 1, (
                f"Expected at least 1 code chunk for {len(original_code_blocks)} "
                f"code blocks, got 0."
            )

        # Verification 2: Each code block from the original document appears
        # intact within some CODE-typed chunk (either standalone or merged).
        all_code_content = "\n".join(c.content for c in code_chunks)
        for code_block in original_code_blocks:
            assert code_block in all_code_content, (
                f"Code block content not found in any CODE chunk.\n"
                f"Code block (first 100 chars): {code_block[:100]!r}"
            )

        # Verification 3: All CODE chunks have the correct chunk_type
        for chunk in code_chunks:
            assert chunk.chunk_type == ChunkType.CODE, (
                f"Code chunk should have chunk_type CODE, got {chunk.chunk_type}"
            )

        # Verification 5: Even long code blocks (> max_chunk_size) are single chunks
        for code_block in original_code_blocks:
            if len(code_block) > chunker.max_chunk_size:
                matching = [c for c in code_chunks if c.content == code_block]
                assert len(matching) >= 1, (
                    f"Long code block (len={len(code_block)}) should still be "
                    f"a single chunk even though it exceeds max_chunk_size "
                    f"({chunker.max_chunk_size})"
                )
                for m in matching:
                    assert len(m.content) == len(code_block), (
                        f"Long code block content was truncated: "
                        f"expected {len(code_block)} chars, "
                        f"got {len(m.content)} chars"
                    )
