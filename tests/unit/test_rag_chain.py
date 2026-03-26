"""Tests for ottawa_assistant.rag_chain — format_sources utility."""

import pytest
from langchain_core.documents import Document

from ottawa_assistant.rag_chain import format_sources


class TestFormatSources:

    def test_returns_string(self):
        docs = [Document(page_content="x", metadata={"source": "https://ottawa.ca"})]
        result = format_sources(docs)
        assert isinstance(result, str)

    def test_deduplicates_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://ottawa.ca"}),
            Document(page_content="b", metadata={"source": "https://ottawa.ca"}),
        ]
        result = format_sources(docs)
        assert result.count("ottawa.ca") == 1

    def test_preserves_order(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://first.ca"}),
            Document(page_content="b", metadata={"source": "https://second.ca"}),
        ]
        result = format_sources(docs)
        assert result.index("first.ca") < result.index("second.ca")

    def test_multiple_unique_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://ottawa.ca"}),
            Document(page_content="b", metadata={"source": "https://ontario.ca"}),
            Document(page_content="c", metadata={"source": "https://canada.ca"}),
        ]
        result = format_sources(docs)
        assert "ottawa.ca" in result
        assert "ontario.ca" in result
        assert "canada.ca" in result

    def test_empty_documents_returns_placeholder(self):
        result = format_sources([])
        assert "no official source" in result.lower() or result.startswith("-")

    def test_missing_source_metadata_skipped(self):
        docs = [
            Document(page_content="a", metadata={"source": "https://ottawa.ca"}),
            Document(page_content="b", metadata={}),
        ]
        result = format_sources(docs)
        assert "ottawa.ca" in result
        # Empty source should be skipped
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 1

    def test_accepts_iterator(self):
        docs = [Document(page_content="a", metadata={"source": "https://ottawa.ca"})]
        result = format_sources(iter(docs))
        assert "ottawa.ca" in result

    def test_whitespace_only_source_skipped(self):
        docs = [Document(page_content="a", metadata={"source": "   "})]
        result = format_sources(docs)
        assert "no official source" in result.lower() or result.startswith("-")
