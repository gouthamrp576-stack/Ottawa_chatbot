"""Tests for ottawa_assistant.retriever.ingest — document loading and splitting."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from ottawa_assistant.retriever.ingest import (
    load_web_documents,
    load_pdf_documents,
    split_documents,
)


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------

class TestSplitDocuments:

    def test_returns_list_of_documents(self):
        docs = [Document(page_content="A" * 2000, metadata={"source": "test"})]
        chunks = split_documents(docs)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_long_doc_is_split_into_chunks(self):
        # 2200 chars with chunk_size=1100 should produce at least 2 chunks
        docs = [Document(page_content="word " * 440, metadata={"source": "test"})]
        chunks = split_documents(docs)
        assert len(chunks) >= 2

    def test_short_doc_stays_single_chunk(self):
        docs = [Document(page_content="Short text.", metadata={"source": "test"})]
        chunks = split_documents(docs)
        assert len(chunks) == 1

    def test_preserves_metadata(self):
        docs = [Document(
            page_content="Some content here.",
            metadata={"source": "https://ottawa.ca", "custom_key": "value"},
        )]
        chunks = split_documents(docs)
        assert chunks[0].metadata["source"] == "https://ottawa.ca"
        assert chunks[0].metadata["custom_key"] == "value"

    def test_empty_list(self):
        assert split_documents([]) == []


# ---------------------------------------------------------------------------
# load_web_documents — requires mocking WebBaseLoader
# ---------------------------------------------------------------------------

class TestLoadWebDocuments:

    @patch("ottawa_assistant.retriever.ingest.WebBaseLoader")
    @patch("ottawa_assistant.retriever.ingest.is_trusted_url", return_value=True)
    def test_loads_trusted_url(self, mock_trusted, mock_loader_class):
        mock_doc = Document(page_content="content", metadata={})
        mock_loader_class.return_value.load.return_value = [mock_doc]
        docs = load_web_documents(["https://ottawa.ca/page"])
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://ottawa.ca/page"
        assert docs[0].metadata["source_type"] == "web"

    @patch("ottawa_assistant.retriever.ingest.is_trusted_url", return_value=False)
    def test_skips_untrusted_url(self, mock_trusted):
        docs = load_web_documents(["https://evil.com/page"])
        assert docs == []

    @patch("ottawa_assistant.retriever.ingest.WebBaseLoader")
    @patch("ottawa_assistant.retriever.ingest.is_trusted_url", return_value=True)
    def test_handles_loader_failure_gracefully(self, mock_trusted, mock_loader_class):
        mock_loader_class.return_value.load.side_effect = Exception("network error")
        docs = load_web_documents(["https://ottawa.ca/page"])
        assert docs == []

    def test_empty_url_list(self):
        docs = load_web_documents([])
        assert docs == []

    @patch("ottawa_assistant.retriever.ingest.WebBaseLoader")
    @patch("ottawa_assistant.retriever.ingest.is_trusted_url", return_value=True)
    def test_deduplicates_urls(self, mock_trusted, mock_loader_class):
        mock_doc = Document(page_content="content", metadata={})
        mock_loader_class.return_value.load.return_value = [mock_doc]
        docs = load_web_documents([
            "https://ottawa.ca/page",
            "https://ottawa.ca/page",  # duplicate
        ])
        # WebBaseLoader should only be called once
        assert mock_loader_class.call_count == 1


# ---------------------------------------------------------------------------
# load_pdf_documents
# ---------------------------------------------------------------------------

class TestLoadPdfDocuments:

    def test_skips_nonexistent_file(self):
        docs = load_pdf_documents(["/nonexistent/file.pdf"])
        assert docs == []

    def test_skips_non_pdf_file(self, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("not a pdf")
        docs = load_pdf_documents([str(txt_file)])
        assert docs == []

    def test_empty_list(self):
        docs = load_pdf_documents([])
        assert docs == []
