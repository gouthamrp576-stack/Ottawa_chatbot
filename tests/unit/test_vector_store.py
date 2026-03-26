"""Tests for ottawa_assistant.retriever.vector_store — index metadata and compatibility."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ottawa_assistant.retriever.vector_store import (
    _index_exists,
    _index_metadata_path,
    _save_index_metadata,
    _read_index_embedding_signature,
    _validate_embedding_compatibility,
    INDEX_METADATA_FILE,
)


# ---------------------------------------------------------------------------
# _index_exists
# ---------------------------------------------------------------------------

class TestIndexExists:

    def test_returns_true_when_both_files_present(self, tmp_path):
        (tmp_path / "index.faiss").write_text("fake")
        (tmp_path / "index.pkl").write_text("fake")
        assert _index_exists(tmp_path) is True

    def test_returns_false_when_faiss_missing(self, tmp_path):
        (tmp_path / "index.pkl").write_text("fake")
        assert _index_exists(tmp_path) is False

    def test_returns_false_when_pkl_missing(self, tmp_path):
        (tmp_path / "index.faiss").write_text("fake")
        assert _index_exists(tmp_path) is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        assert _index_exists(tmp_path) is False


# ---------------------------------------------------------------------------
# _index_metadata_path
# ---------------------------------------------------------------------------

class TestIndexMetadataPath:

    def test_returns_correct_path(self, tmp_path):
        result = _index_metadata_path(tmp_path)
        assert result == tmp_path / INDEX_METADATA_FILE


# ---------------------------------------------------------------------------
# _save_index_metadata and _read_index_embedding_signature (round-trip)
# ---------------------------------------------------------------------------

class TestIndexMetadataRoundTrip:

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    def test_save_and_read_signature(self, mock_sig, tmp_path):
        _save_index_metadata(tmp_path)
        sig = _read_index_embedding_signature(tmp_path)
        assert sig == "openai:text-embedding-3-large"

    def test_read_returns_none_when_file_missing(self, tmp_path):
        assert _read_index_embedding_signature(tmp_path) is None

    def test_read_returns_none_on_corrupt_json(self, tmp_path):
        (tmp_path / INDEX_METADATA_FILE).write_text("not json!", encoding="utf-8")
        assert _read_index_embedding_signature(tmp_path) is None

    def test_read_returns_none_when_key_missing(self, tmp_path):
        (tmp_path / INDEX_METADATA_FILE).write_text(
            json.dumps({"other_key": "value"}), encoding="utf-8"
        )
        assert _read_index_embedding_signature(tmp_path) is None


# ---------------------------------------------------------------------------
# _validate_embedding_compatibility
# ---------------------------------------------------------------------------

class TestValidateEmbeddingCompatibility:

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    def test_passes_when_signatures_match(self, mock_sig, tmp_path):
        # Write matching metadata
        payload = {"embedding_signature": "openai:text-embedding-3-large"}
        (tmp_path / INDEX_METADATA_FILE).write_text(json.dumps(payload), encoding="utf-8")
        # Should not raise
        _validate_embedding_compatibility(tmp_path)

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    @patch("ottawa_assistant.retriever.vector_store.settings")
    def test_raises_when_signatures_differ(self, mock_settings, mock_sig, tmp_path):
        mock_settings.allow_embedding_mismatch = False
        payload = {"embedding_signature": "ollama:nomic-embed-text@http://localhost:11434"}
        (tmp_path / INDEX_METADATA_FILE).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="different embedding"):
            _validate_embedding_compatibility(tmp_path)

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    @patch("ottawa_assistant.retriever.vector_store.settings")
    def test_passes_when_mismatch_allowed(self, mock_settings, mock_sig, tmp_path):
        mock_settings.allow_embedding_mismatch = True
        payload = {"embedding_signature": "different:model"}
        (tmp_path / INDEX_METADATA_FILE).write_text(json.dumps(payload), encoding="utf-8")
        _validate_embedding_compatibility(tmp_path)  # should not raise

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    @patch("ottawa_assistant.retriever.vector_store.settings")
    def test_missing_metadata_raises_by_default(self, mock_settings, mock_sig, tmp_path):
        mock_settings.allow_embedding_mismatch = False
        with pytest.raises(RuntimeError, match="metadata is missing"):
            _validate_embedding_compatibility(tmp_path)

    @patch(
        "ottawa_assistant.retriever.vector_store.embedding_signature",
        return_value="openai:text-embedding-3-large",
    )
    @patch("ottawa_assistant.retriever.vector_store.settings")
    def test_missing_metadata_passes_when_mismatch_allowed(self, mock_settings, mock_sig, tmp_path):
        mock_settings.allow_embedding_mismatch = True
        _validate_embedding_compatibility(tmp_path)  # should not raise
