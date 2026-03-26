"""Central settings for the Ottawa newcomer assistant project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(raw_path: str, default: str) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(raw_path or default).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    model_provider: str = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    embedding_provider: str = os.getenv(
        "EMBEDDING_PROVIDER",
        os.getenv("MODEL_PROVIDER", "openai"),
    ).strip().lower()

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-large",
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_num_ctx: int = _env_int("OLLAMA_NUM_CTX", 8192)

    vector_store_dir: Path = _resolve_path(
        os.getenv("VECTOR_STORE_DIR", ""),
        "src/ottawa_assistant/retriever/index",
    )
    retrieval_k: int = _env_int("RETRIEVAL_K", 6)
    allow_embedding_mismatch: bool = _env_bool("ALLOW_EMBEDDING_MISMATCH", False)
    enable_web_fallback: bool = _env_bool("ENABLE_WEB_FALLBACK", True)
    google_fallback_results: int = _env_int("GOOGLE_FALLBACK_RESULTS", 10)
    google_fallback_pages: int = _env_int("GOOGLE_FALLBACK_PAGES", 5)
    trusted_domains: tuple[str, ...] = (
        "ottawa.ca",
        "ottawapublichealth.ca",
        "ontario.ca",
        "canada.ca",
        "octranspo.com",
        "uottawa.ca",
        "algonquincollege.com",
        "carleton.ca",
        "ociso.org",
        "ymcaywca.ca",
    )
    seed_web_sources: tuple[str, ...] = (
        "https://ottawa.ca/en/family-and-social-services/immigration-and-settlement",
        "https://ottawa.ca/en/parking-roads-and-travel/public-transit",
        "https://www.ontario.ca/page/apply-ohip-and-get-health-card",
        "https://www.ontario.ca/page/renting-ontario-your-rights",
        "https://www.ontario.ca/page/settle-ontario",
        "https://www.canada.ca/en/immigration-refugees-citizenship/services/new-immigrants.html",
        "https://www.octranspo.com/en/fares/payment/where-how-to-pay/",
        "https://ociso.org/",
        "https://www.ymcaywca.ca/newcomer-services/",
        "https://www.uottawa.ca/about-us/campus-life",
    )


SUPPORTED_MODEL_PROVIDERS = ("openai", "ollama")


def is_trusted_url(url: str, trusted_domains: tuple[str, ...] | None = None) -> bool:
    """Return True if the URL belongs to a trusted official/reliable domain."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    domains = trusted_domains or settings.trusted_domains
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def validate_settings(require_embeddings: bool = False) -> None:
    """Validate provider setup before running retrieval or generation."""
    if settings.log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise RuntimeError(
            "LOG_LEVEL must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG."
        )
    if settings.model_provider not in SUPPORTED_MODEL_PROVIDERS:
        raise RuntimeError(
            "MODEL_PROVIDER must be one of: "
            f"{', '.join(SUPPORTED_MODEL_PROVIDERS)}"
        )
    if settings.embedding_provider not in SUPPORTED_MODEL_PROVIDERS:
        raise RuntimeError(
            "EMBEDDING_PROVIDER must be one of: "
            f"{', '.join(SUPPORTED_MODEL_PROVIDERS)}"
        )

    if settings.model_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when MODEL_PROVIDER=openai."
        )
    if require_embeddings and settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai."
        )

    if settings.model_provider == "ollama" and not settings.ollama_chat_model:
        raise RuntimeError("OLLAMA_CHAT_MODEL is required when MODEL_PROVIDER=ollama.")
    if require_embeddings and settings.embedding_provider == "ollama" and not settings.ollama_embedding_model:
        raise RuntimeError(
            "OLLAMA_EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=ollama."
        )

    if settings.retrieval_k <= 0:
        raise RuntimeError("RETRIEVAL_K must be greater than 0.")
    if settings.google_fallback_results <= 0:
        raise RuntimeError("GOOGLE_FALLBACK_RESULTS must be greater than 0.")
    if settings.google_fallback_pages <= 0:
        raise RuntimeError("GOOGLE_FALLBACK_PAGES must be greater than 0.")


settings = Settings()
