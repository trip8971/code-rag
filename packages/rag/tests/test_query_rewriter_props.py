"""Property-based tests for QueryRewriter."""

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.config import ServiceConfig
from rag.query_rewriter import QueryRewriter


def _make_rewriter() -> QueryRewriter:
    """Create a QueryRewriter instance for testing _truncate_result."""
    config = ServiceConfig(
        url="https://api.openai.com/v1/chat/completions",
        api_key="sk-test-key",
        timeout=10,
        max_retries=3,
    )
    return QueryRewriter(config)


class TestProperty17QueryRewriteOutputLengthConstraint:
    """Property 17: 查询改写输出长度约束

    For any rewritten query text, _truncate_result() ensures the output
    never exceeds 500 characters (or the specified max_length).

    **Validates: Requirements 3.1**
    """

    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=2000))
    def test_output_never_exceeds_default_max_length(self, text: str):
        """Output of _truncate_result() never exceeds 500 characters.

        **Validates: Requirements 3.1**
        """
        rewriter = _make_rewriter()
        result = rewriter._truncate_result(text)
        assert len(result) <= 500, (
            f"Expected output length <= 500, got {len(result)} "
            f"for input of length {len(text)}"
        )

    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=500))
    def test_short_input_unchanged(self, text: str):
        """If input <= 500 chars, output equals input unchanged.

        **Validates: Requirements 3.1**
        """
        rewriter = _make_rewriter()
        result = rewriter._truncate_result(text)
        assert result == text, (
            f"Expected input of length {len(text)} to be returned unchanged, "
            f"but got different result"
        )

    @settings(max_examples=100)
    @given(text=st.text(min_size=501, max_size=2000))
    def test_long_input_truncated_to_prefix(self, text: str):
        """If input > 500 chars, output is exactly 500 chars and is a prefix of input.

        **Validates: Requirements 3.1**
        """
        rewriter = _make_rewriter()
        result = rewriter._truncate_result(text)
        assert len(result) == 500, (
            f"Expected output length == 500 for input of length {len(text)}, "
            f"got {len(result)}"
        )
        assert text.startswith(result), (
            "Expected output to be a prefix of the input"
        )

    @settings(max_examples=100)
    @given(
        text=st.text(min_size=0, max_size=2000),
        max_length=st.integers(min_value=1, max_value=1000),
    )
    def test_output_respects_custom_max_length(self, text: str, max_length: int):
        """Output of _truncate_result() never exceeds the specified max_length.

        **Validates: Requirements 3.1**
        """
        rewriter = _make_rewriter()
        result = rewriter._truncate_result(text, max_length=max_length)
        assert len(result) <= max_length, (
            f"Expected output length <= {max_length}, got {len(result)} "
            f"for input of length {len(text)}"
        )
