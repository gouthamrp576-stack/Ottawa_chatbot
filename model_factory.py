"""Factories for chat models and embeddings across providers."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings, validate_settings


def create_chat_model(temperature: float = 0.15) -> BaseChatModel:
    """Build chat model from MODEL_PROVIDER."""
    validate_settings(require_embeddings=False)

    if settings.model_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )

    if settings.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
        )

    raise RuntimeError(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")


def create_embeddings() -> Embeddings:
    """Build embeddings model from EMBEDDING_PROVIDER."""
    validate_settings(require_embeddings=True)

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def embedding_signature() -> str:
    """Return compact embedding setup signature for index compatibility checks."""
    if settings.embedding_provider == "openai":
        return f"openai:{settings.openai_embedding_model}"
    if settings.embedding_provider == "ollama":
        return f"ollama:{settings.ollama_embedding_model}@{settings.ollama_base_url}"
    return f"unknown:{settings.embedding_provider}"


def runtime_summary() -> str:
    """Human-readable runtime config summary for UI and logs."""
    return (
        f"LLM={settings.model_provider} | "
        f"Embeddings={settings.embedding_provider} | "
        f"TopK={settings.retrieval_k} | "
        f"WebFallback={'on' if settings.enable_web_fallback else 'off'}"
    )
