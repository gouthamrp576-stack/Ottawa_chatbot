"""Tests for ottawa_assistant.web_fallback — tokenization, ranking, context building."""

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from ottawa_assistant.web_fallback import (
    _tokenize,
    _rank_documents,
    _build_context_block,
    _extract_answer_text,
    _format_sources,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:

    def test_returns_set_of_strings(self):
        result = _tokenize("SIN card immigration")
        assert isinstance(result, set)
        assert all(isinstance(t, str) for t in result)

    def test_lowercases_tokens(self):
        result = _tokenize("Immigration Service")
        assert all(t == t.lower() for t in result)

    def test_removes_stopwords(self):
        result = _tokenize("the immigration to and from Ottawa")
        assert "the" not in result
        assert "and" not in result
        assert "ottawa" not in result  # ottawa is in stopwords
        assert "immigration" in result

    def test_removes_single_char_tokens(self):
        result = _tokenize("I a the x y")
        # Single character tokens should be removed
        assert "x" not in result
        assert "y" not in result

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_extracts_alphanumeric(self):
        result = _tokenize("job-search 2024 housing!")
        assert "job" in result
        assert "search" in result
        assert "2024" in result
        assert "housing" in result


# ---------------------------------------------------------------------------
# _rank_documents
# ---------------------------------------------------------------------------

class TestRankDocuments:

    def _make_doc(self, content: str, source: str = "https://test.ca") -> Document:
        return Document(page_content=content, metadata={"source": source})

    def test_returns_list(self):
        docs = [self._make_doc("immigration services in Ottawa")]
        result = _rank_documents("immigration", docs)
        assert isinstance(result, list)

    def test_empty_documents(self):
        assert _rank_documents("query", []) == []

    def test_relevant_doc_ranked_higher(self):
        relevant = self._make_doc("SIN card application immigration newcomer")
        irrelevant = self._make_doc("weather forecast for Monday sunny")
        result = _rank_documents("SIN card immigration", [irrelevant, relevant])
        assert result[0].page_content == relevant.page_content

    def test_max_docs_limit(self):
        docs = [self._make_doc(f"doc {i}") for i in range(20)]
        result = _rank_documents("test", docs, max_docs=5)
        assert len(result) <= 5

    def test_empty_content_docs_handled(self):
        docs = [
            self._make_doc(""),
            self._make_doc("real content about housing"),
        ]
        result = _rank_documents("housing", docs)
        assert len(result) >= 1

    def test_no_matching_terms_still_returns_docs(self):
        docs = [self._make_doc("completely unrelated content about xyz")]
        result = _rank_documents("immigration housing jobs", docs)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:

    def test_returns_string(self):
        docs = [Document(page_content="test content", metadata={"source": "url", "title": "T"})]
        result = _build_context_block(docs)
        assert isinstance(result, str)

    def test_includes_source_and_snippet(self):
        docs = [Document(
            page_content="Ottawa immigration guide for newcomers.",
            metadata={"source": "https://ottawa.ca", "title": "Immigration"},
        )]
        result = _build_context_block(docs)
        assert "https://ottawa.ca" in result
        assert "immigration" in result.lower()

    def test_numbering(self):
        docs = [
            Document(page_content="first", metadata={"source": "s1", "title": ""}),
            Document(page_content="second", metadata={"source": "s2", "title": ""}),
        ]
        result = _build_context_block(docs)
        assert "[1]" in result
        assert "[2]" in result

    def test_truncates_long_content(self):
        long_content = "x" * 5000
        docs = [Document(page_content=long_content, metadata={"source": "s", "title": ""})]
        result = _build_context_block(docs, max_chars_per_doc=100)
        # The snippet should be much shorter than 5000 chars
        assert len(result) < 1000

    def test_empty_docs(self):
        result = _build_context_block([])
        assert isinstance(result, str)
        assert result == ""


# ---------------------------------------------------------------------------
# _extract_answer_text
# ---------------------------------------------------------------------------

class TestExtractAnswerText:

    def test_string_input(self):
        assert _extract_answer_text("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        assert _extract_answer_text("  answer  ") == "answer"

    def test_list_input(self):
        result = _extract_answer_text(["part1", "part2"])
        assert "part1" in result
        assert "part2" in result

    def test_non_string_input(self):
        result = _extract_answer_text(42)
        assert result == "42"

    def test_empty_string(self):
        assert _extract_answer_text("") == ""

    def test_none_input(self):
        result = _extract_answer_text(None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _format_sources
# ---------------------------------------------------------------------------

class TestFormatSources:

    def test_deduplicates(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://ottawa.ca"}),
            Document(page_content="b", metadata={"source": "https://ottawa.ca"}),
        ]
        result = _format_sources(docs)
        assert result.count("ottawa.ca") == 1

    def test_empty_docs_returns_placeholder(self):
        result = _format_sources([])
        assert "no trusted source" in result.lower()

    def test_multiple_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://ottawa.ca"}),
            Document(page_content="b", metadata={"source": "https://ontario.ca"}),
        ]
        result = _format_sources(docs)
        assert "ottawa.ca" in result
        assert "ontario.ca" in result
