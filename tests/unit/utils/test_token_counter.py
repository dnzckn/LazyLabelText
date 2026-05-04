"""Tests for token counter."""

from lazylabeltext.utils.token_counter import count_tokens, estimate_tokens


def test_estimate_tokens():
    assert estimate_tokens("") == 1  # min 1
    assert estimate_tokens("hello world test") > 0


def test_count_tokens():
    # Should work regardless of tiktoken availability
    result = count_tokens("The quick brown fox jumps over the lazy dog.")
    assert result > 0
    assert isinstance(result, int)


def test_count_tokens_consistency():
    text = "This is a test sentence."
    a = count_tokens(text)
    b = count_tokens(text)
    assert a == b
