"""Tests for ottawa_assistant.config — settings, validation, and URL trust."""

import pytest

from ottawa_assistant.config import (
    Settings,
    _env_bool,
    _env_int,
    is_trusted_url,
    validate_settings,
    SUPPORTED_MODEL_PROVIDERS,
)


# ---------------------------------------------------------------------------
# _env_bool
# ---------------------------------------------------------------------------

class TestEnvBool:

    @pytest.mark.parametrize("raw,expected", [
        ("1", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("y", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("random", False),
    ])
    def test_truthy_and_falsy_values(self, raw, expected, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", raw)
        import os
        assert _env_bool("TEST_BOOL", False) == expected

    def test_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL_MISSING", raising=False)
        assert _env_bool("TEST_BOOL_MISSING", True) is True
        assert _env_bool("TEST_BOOL_MISSING", False) is False


# ---------------------------------------------------------------------------
# _env_int
# ---------------------------------------------------------------------------

class TestEnvInt:

    def test_parses_valid_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert _env_int("TEST_INT", 0) == 42

    def test_returns_default_on_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_MISSING", raising=False)
        assert _env_int("TEST_INT_MISSING", 99) == 99

    def test_returns_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_BAD", "not_a_number")
        assert _env_int("TEST_INT_BAD", 7) == 7

    def test_negative_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_NEG", "-5")
        assert _env_int("TEST_INT_NEG", 0) == -5


# ---------------------------------------------------------------------------
# is_trusted_url
# ---------------------------------------------------------------------------

class TestIsTrustedUrl:

    TRUSTED = ("ottawa.ca", "ontario.ca", "canada.ca")

    @pytest.mark.parametrize("url,expected", [
        ("https://ottawa.ca/en/services", True),
        ("https://www.ottawa.ca/en/services", True),
        ("https://sub.ottawa.ca/page", True),
        ("https://ontario.ca/page", True),
        ("https://canada.ca/en/immigration", True),
        ("https://CANADA.CA/path", True),
        ("https://evil.com", False),
        ("https://notottawa.ca/page", False),
        ("https://ottawa.ca.evil.com/phish", False),
        ("", False),
        ("not-a-url", False),
        ("ftp://ottawa.ca/file", True),
    ])
    def test_trusted_url_variants(self, url, expected):
        result = is_trusted_url(url, trusted_domains=self.TRUSTED)
        assert result is expected

    def test_uses_default_domains_from_settings(self):
        # Should use settings.trusted_domains when no override provided
        assert is_trusted_url("https://ottawa.ca/test") is True
        assert is_trusted_url("https://randomsite.xyz/page") is False

    def test_empty_domain_list(self):
        assert is_trusted_url("https://ottawa.ca/page", trusted_domains=()) is False


# ---------------------------------------------------------------------------
# validate_settings — uses mock Settings to avoid needing real API keys
# ---------------------------------------------------------------------------

class TestValidateSettings:

    def test_valid_openai_config(self, monkeypatch):
        """Valid OpenAI config should not raise."""
        # Patch the settings module-level singleton
        monkeypatch.setattr(
            "ottawa_assistant.config.settings",
            Settings(
                model_provider="openai",
                embedding_provider="openai",
                openai_api_key="sk-test",
            ),
        )
        validate_settings(require_embeddings=True)  # should not raise

    def test_missing_openai_key_raises(self, monkeypatch):
        monkeypatch.setattr(
            "ottawa_assistant.config.settings",
            Settings(
                model_provider="openai",
                embedding_provider="openai",
                openai_api_key="",
            ),
        )
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            validate_settings(require_embeddings=False)

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setattr(
            "ottawa_assistant.config.settings",
            Settings(model_provider="fakeai"),
        )
        with pytest.raises(RuntimeError, match="MODEL_PROVIDER"):
            validate_settings()

    def test_ollama_does_not_require_api_key(self, monkeypatch):
        monkeypatch.setattr(
            "ottawa_assistant.config.settings",
            Settings(
                model_provider="ollama",
                embedding_provider="ollama",
                openai_api_key="",
            ),
        )
        validate_settings(require_embeddings=True)  # should not raise

    def test_zero_retrieval_k_raises(self, monkeypatch):
        monkeypatch.setattr(
            "ottawa_assistant.config.settings",
            Settings(
                model_provider="openai",
                embedding_provider="openai",
                openai_api_key="sk-test",
                retrieval_k=0,
            ),
        )
        with pytest.raises(RuntimeError, match="RETRIEVAL_K"):
            validate_settings()
