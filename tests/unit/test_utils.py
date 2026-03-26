"""Tests for ottawa_assistant.utils — shared utilities."""

import pytest

from ottawa_assistant.utils import dedupe_preserve_order, sanitize_user_input


# ---------------------------------------------------------------------------
# dedupe_preserve_order
# ---------------------------------------------------------------------------

class TestDedupePreserveOrder:

    def test_removes_duplicates(self):
        assert dedupe_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_preserves_insertion_order(self):
        assert dedupe_preserve_order(["c", "b", "a"]) == ["c", "b", "a"]

    def test_strips_whitespace(self):
        assert dedupe_preserve_order(["  hello ", "hello"]) == ["hello"]

    def test_skips_empty_strings(self):
        assert dedupe_preserve_order(["", " ", "a", ""]) == ["a"]

    def test_empty_input(self):
        assert dedupe_preserve_order([]) == []

    def test_accepts_generator(self):
        gen = (x for x in ["a", "b", "a"])
        assert dedupe_preserve_order(gen) == ["a", "b"]

    def test_all_duplicates(self):
        assert dedupe_preserve_order(["x", "x", "x"]) == ["x"]

    def test_single_item(self):
        assert dedupe_preserve_order(["only"]) == ["only"]


# ---------------------------------------------------------------------------
# sanitize_user_input
# ---------------------------------------------------------------------------

class TestSanitizeUserInput:

    def test_normal_input_unchanged(self):
        assert sanitize_user_input("How do I get OHIP?") == "How do I get OHIP?"

    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_user_input("  hello  ") == "hello"

    def test_removes_control_characters(self):
        assert sanitize_user_input("hello\x00world") == "helloworld"
        assert sanitize_user_input("test\x07\x08input") == "testinput"

    def test_preserves_newlines_and_tabs(self):
        # \n and \t are NOT stripped (they are valid user input characters)
        result = sanitize_user_input("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_raises_on_too_long_input(self):
        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_user_input("x" * 2001)

    def test_custom_max_length(self):
        assert sanitize_user_input("short", max_length=100) == "short"
        with pytest.raises(ValueError):
            sanitize_user_input("too long", max_length=3)

    def test_empty_string(self):
        assert sanitize_user_input("") == ""

    def test_exactly_at_max_length(self):
        text = "a" * 2000
        assert sanitize_user_input(text) == text

    def test_control_chars_then_length_check(self):
        # After removing control chars, should still check length
        text = "\x00" * 100 + "a" * 2000
        assert sanitize_user_input(text) == "a" * 2000
