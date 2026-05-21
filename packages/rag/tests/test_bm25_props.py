"""Property-based tests for BM25Search module."""

from unittest.mock import MagicMock

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rag.bm25_search import BM25Search
from rag.chunker import Chunk, ChunkType, generate_chunk_id


# --- Strategies ---


def ascii_word_strategy():
    """Generate lowercase ASCII alphabetic words (3-8 chars) for code identifiers."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=3,
        max_size=8,
    )


def camel_case_strategy():
    """Generate camelCase/PascalCase identifiers from 2-5 capitalized words.

    Examples: "GetUserName", "ParseJsonData", "MyClass"
    """
    return st.lists(ascii_word_strategy(), min_size=2, max_size=5).map(
        lambda words: "".join(w.capitalize() for w in words)
    )


def snake_case_strategy():
    """Generate snake_case identifiers from 2-5 lowercase words joined by underscores.

    Examples: "get_user_name", "parse_json_data"
    """
    return st.lists(ascii_word_strategy(), min_size=2, max_size=5).map(
        lambda words: "_".join(words)
    )

# Word pool for generating realistic document content and queries
WORD_POOL = [
    "python", "function", "class", "module", "import", "return", "variable",
    "data", "list", "dict", "string", "integer", "float", "boolean", "array",
    "object", "method", "attribute", "parameter", "argument", "value", "key",
    "index", "loop", "condition", "exception", "error", "file", "path", "name",
    "type", "code", "test", "debug", "build", "run", "install", "package",
    "library", "framework", "server", "client", "request", "response", "api",
    "database", "query", "table", "record", "field", "schema", "model",
    "config", "setting", "option", "flag", "enable", "disable", "create",
    "update", "delete", "read", "write", "open", "close", "start", "stop",
    "async", "await", "thread", "process", "memory", "cache", "buffer",
    "network", "socket", "port", "host", "url", "http", "json", "xml",
    "parse", "format", "encode", "decode", "compress", "extract", "search",
    "sort", "filter", "map", "reduce", "transform", "convert", "validate",
]


@st.composite
def document_content_strategy(draw):
    """Generate a document content string (10-200 chars) using real words."""
    num_words = draw(st.integers(min_value=3, max_value=30))
    words = draw(
        st.lists(
            st.sampled_from(WORD_POOL),
            min_size=num_words,
            max_size=num_words,
        )
    )
    content = " ".join(words)
    # Ensure content is within 10-200 chars
    if len(content) < 10:
        content = content + " " + " ".join(["code"] * 3)
    if len(content) > 200:
        content = content[:200]
    return content


@st.composite
def query_strategy(draw):
    """Generate a query string (1-50 chars) using real words."""
    num_words = draw(st.integers(min_value=1, max_value=6))
    words = draw(
        st.lists(
            st.sampled_from(WORD_POOL),
            min_size=num_words,
            max_size=num_words,
        )
    )
    query = " ".join(words)
    if len(query) > 50:
        query = query[:50]
    return query


@st.composite
def document_collection_strategy(draw):
    """Generate a collection of 2-20 document content strings."""
    num_docs = draw(st.integers(min_value=2, max_value=20))
    docs = draw(
        st.lists(
            document_content_strategy(),
            min_size=num_docs,
            max_size=num_docs,
        )
    )
    return docs


# --- Property 8: BM25 分词器正确拆分代码标识符 ---


class TestProperty8BM25TokenizerSplitsCodeIdentifiers:
    """Property 8: BM25 分词器正确拆分代码标识符

    For any camelCase or snake_case identifier, _tokenize() should correctly
    split it into its component words (all lowercased).

    **Validates: Requirements 5.3**
    """

    @settings(max_examples=100)
    @given(words=st.lists(ascii_word_strategy(), min_size=2, max_size=5))
    def test_camel_case_identifiers_split_correctly(self, words: list[str]):
        """CamelCase identifiers are split into their component words.

        **Validates: Requirements 5.3**
        """
        mock_db = MagicMock()
        mock_db.get_all_chunks.return_value = []
        bm25 = BM25Search(database=mock_db)

        # Build a PascalCase identifier: e.g., ["get", "user", "name"] -> "GetUserName"
        identifier = "".join(w.capitalize() for w in words)

        tokens = bm25._tokenize(identifier)

        # All component words should appear in the tokenized output (lowercased)
        for word in words:
            assert word.lower() in tokens, (
                f"Expected '{word.lower()}' in tokens {tokens} "
                f"for identifier '{identifier}'"
            )

        # No empty tokens in the result
        assert all(t != "" for t in tokens), (
            f"Found empty token in {tokens} for identifier '{identifier}'"
        )

    @settings(max_examples=100)
    @given(words=st.lists(ascii_word_strategy(), min_size=2, max_size=5))
    def test_snake_case_identifiers_split_correctly(self, words: list[str]):
        """Snake_case identifiers are split into their component words.

        **Validates: Requirements 5.3**
        """
        mock_db = MagicMock()
        mock_db.get_all_chunks.return_value = []
        bm25 = BM25Search(database=mock_db)

        # Build a snake_case identifier: e.g., ["get", "user", "name"] -> "get_user_name"
        identifier = "_".join(words)

        tokens = bm25._tokenize(identifier)

        # All component words should appear in the tokenized output (lowercased)
        for word in words:
            assert word.lower() in tokens, (
                f"Expected '{word.lower()}' in tokens {tokens} "
                f"for identifier '{identifier}'"
            )

        # No empty tokens in the result
        assert all(t != "" for t in tokens), (
            f"Found empty token in {tokens} for identifier '{identifier}'"
        )

    @settings(max_examples=100)
    @given(words=st.lists(ascii_word_strategy(), min_size=2, max_size=5))
    def test_no_empty_tokens_in_output(self, words: list[str]):
        """Tokenizer never produces empty tokens for valid identifiers.

        **Validates: Requirements 5.3**
        """
        mock_db = MagicMock()
        mock_db.get_all_chunks.return_value = []
        bm25 = BM25Search(database=mock_db)

        # Test both camelCase and snake_case
        camel_id = "".join(w.capitalize() for w in words)
        snake_id = "_".join(words)

        camel_tokens = bm25._tokenize(camel_id)
        snake_tokens = bm25._tokenize(snake_id)

        assert all(len(t) > 0 for t in camel_tokens), (
            f"Empty token found in camelCase result: {camel_tokens}"
        )
        assert all(len(t) > 0 for t in snake_tokens), (
            f"Empty token found in snake_case result: {snake_tokens}"
        )


# --- Property 9: BM25 仅返回非零评分结果且按分数排序 ---


class TestProperty9BM25NonZeroScoresAndSorted:
    """Property 9: BM25 仅返回非零评分结果且按分数排序

    For any 查询和文档集合，BM25_Search 返回的结果应满足：
    (a) 所有结果的 BM25 评分 > 0，
    (b) 结果按评分降序排列，
    (c) 结果数量不超过 min(K, 评分非零的 Chunk 数量)。

    **Validates: Requirements 5.1, 5.2, 5.4, 5.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        documents=document_collection_strategy(),
        query=query_strategy(),
        top_k=st.integers(min_value=1, max_value=50),
    )
    def test_bm25_returns_only_nonzero_scores_sorted_descending(
        self, documents, query, top_k
    ):
        """BM25 search results should have all scores > 0, be sorted in
        descending order by score, and the count should not exceed
        min(top_k, number of documents with non-zero BM25 score).

        **Validates: Requirements 5.1, 5.2, 5.4, 5.5**
        """
        # Create chunks from the generated documents
        chunks = []
        for i, content in enumerate(documents):
            chunk_id = generate_chunk_id(content, f"docs/doc_{i}.md")
            chunk = Chunk(
                chunk_id=chunk_id,
                content=content,
                source_file=f"docs/doc_{i}.md",
                heading_level=1,
                chunk_type=ChunkType.TEXT,
                start_line=1,
                heading_text=f"Document {i}",
            )
            chunks.append(chunk)

        # Mock the database's get_all_chunks() to return our generated chunks
        mock_database = MagicMock()
        mock_database.get_all_chunks.return_value = chunks

        # Create BM25Search and build index
        bm25_search = BM25Search(database=mock_database)
        bm25_search.build_index()

        # Execute search
        results = bm25_search.search(query=query, top_k=top_k)

        # --- Verification 1: All returned scores are > 0 ---
        for chunk, score in results:
            assert score > 0, (
                f"BM25 score should be > 0, but got {score} "
                f"for chunk '{chunk.content[:50]}...'"
            )

        # --- Verification 2: Results are sorted by score in descending order ---
        if len(results) > 1:
            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Results not sorted in descending order: "
                    f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
                )

        # --- Verification 3: Number of results <= min(top_k, non-zero count) ---
        # Compute the total number of documents with non-zero BM25 score
        # by running a search with a very large top_k to get all non-zero results
        all_results = bm25_search.search(query=query, top_k=len(chunks))
        non_zero_count = len(all_results)

        expected_max = min(top_k, non_zero_count)
        assert len(results) <= expected_max, (
            f"Too many results: got {len(results)}, "
            f"expected at most min({top_k}, {non_zero_count}) = {expected_max}"
        )


# --- Strategies for Property 10 ---


def unique_keyword_strategy():
    """Generate a unique keyword that is unlikely to appear in random text.

    Uses a prefix to ensure uniqueness and avoid collisions with other content.
    """
    return st.from_regex(r"xkw[a-z]{4,8}", fullmatch=True)


# --- Property 10: BM25 索引与文档同步 ---


class TestProperty10BM25IndexDocumentSync:
    """Property 10: BM25 索引与文档同步

    For any 文档更新操作（新增或修改），更新后的 BM25 搜索应能检索到新内容，
    且不再检索到已被替换的旧内容。

    **Validates: Requirements 5.6**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        old_keyword=unique_keyword_strategy(),
        new_keyword=unique_keyword_strategy(),
        data=st.data(),
    )
    def test_index_sync_after_update(self, old_keyword, new_keyword, data):
        """After updating the index, searching for the new keyword returns results
        and searching for the old keyword returns empty results.

        **Validates: Requirements 5.6**
        """
        # Ensure old and new keywords are different
        assume(old_keyword != new_keyword)

        # Generate old content containing the old keyword
        old_prefix = data.draw(
            st.from_regex(r"[a-z]{3,8}( [a-z]{3,8}){1,3}", fullmatch=True),
            label="old_prefix",
        )
        old_suffix = data.draw(
            st.from_regex(r"[a-z]{3,8}( [a-z]{3,8}){1,3}", fullmatch=True),
            label="old_suffix",
        )
        old_content = f"{old_prefix} {old_keyword} {old_suffix}"

        # Generate new content containing the new keyword
        new_prefix = data.draw(
            st.from_regex(r"[a-z]{3,8}( [a-z]{3,8}){1,3}", fullmatch=True),
            label="new_prefix",
        )
        new_suffix = data.draw(
            st.from_regex(r"[a-z]{3,8}( [a-z]{3,8}){1,3}", fullmatch=True),
            label="new_suffix",
        )
        new_content = f"{new_prefix} {new_keyword} {new_suffix}"

        # Ensure old keyword doesn't appear in new content and vice versa
        assume(old_keyword not in new_content)
        assume(new_keyword not in old_content)

        source_file = "docs/test_sync.md"

        # Create old chunk with the unique keyword
        old_chunk = Chunk(
            chunk_id=generate_chunk_id(old_content, source_file),
            content=old_content,
            source_file=source_file,
            heading_level=1,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="Old Section",
        )

        # Create new chunk with the new unique keyword
        new_chunk = Chunk(
            chunk_id=generate_chunk_id(new_content, source_file),
            content=new_content,
            source_file=source_file,
            heading_level=1,
            chunk_type=ChunkType.TEXT,
            start_line=1,
            heading_text="New Section",
        )

        # BM25 IDF requires multiple documents to produce positive scores for
        # a unique term. Add background documents that don't contain either keyword.
        background_chunks = [
            Chunk(
                chunk_id=generate_chunk_id(f"background doc {i} with various words", "docs/bg.md"),
                content=f"background doc {i} with various words about programming and code",
                source_file="docs/bg.md",
                heading_level=1,
                chunk_type=ChunkType.TEXT,
                start_line=i * 10,
                heading_text=f"Background {i}",
            )
            for i in range(3)
        ]

        # Create a mock database
        mock_db = MagicMock()

        # Phase 1: Build index with old chunk + background docs
        mock_db.get_all_chunks.return_value = [old_chunk] + background_chunks
        bm25 = BM25Search(database=mock_db)
        bm25.build_index()

        # Verify old keyword is searchable
        old_results = bm25.search(old_keyword, top_k=10)
        assert len(old_results) > 0, (
            f"Expected to find results for old keyword '{old_keyword}' "
            f"in content '{old_content}', but got empty results"
        )

        # Phase 2: Update index with new chunk replacing old chunk
        # (simulating document update - database now returns new chunks)
        mock_db.get_all_chunks.return_value = [new_chunk] + background_chunks
        bm25.update_index([new_chunk])

        # Verify new keyword is searchable after update
        new_results = bm25.search(new_keyword, top_k=10)
        assert len(new_results) > 0, (
            f"Expected to find results for new keyword '{new_keyword}' "
            f"in content '{new_content}' after update, but got empty results"
        )

        # Verify old keyword is no longer searchable after update
        old_results_after_update = bm25.search(old_keyword, top_k=10)
        assert len(old_results_after_update) == 0, (
            f"Expected no results for old keyword '{old_keyword}' after update, "
            f"but got {len(old_results_after_update)} results"
        )
