"""Google-search fallback when local vector index is unavailable."""

from __future__ import annotations

import re

from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .config import is_trusted_url, settings
from .logging_utils import get_logger
from .model_factory import create_chat_model
from .prompts import build_qa_prompt
from .utils import dedupe_preserve_order

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "a",
    "an",
    "is",
    "are",
    "what",
    "how",
    "where",
    "when",
    "with",
    "can",
    "i",
    "my",
    "about",
    "ottawa",
    "ontario",
    "canada",
}


def _tokenize(text: str) -> set[str]:
    tokens = {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1}
    return {token for token in tokens if token not in _STOPWORDS}


def _search_trusted_urls(question: str) -> list[str]:
    if not settings.enable_web_fallback:
        return []

    query = (
        f"{question} Ottawa newcomer services official site "
        "City of Ottawa Ontario Government"
    )
    candidates: list[str] = []

    try:
        for url in search(query, num_results=settings.google_fallback_results):
            if is_trusted_url(url):
                candidates.append(url)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Google search failed: {exc}") from exc

    trusted_urls = dedupe_preserve_order(candidates)
    if len(trusted_urls) < settings.google_fallback_pages:
        for seed_url in settings.seed_web_sources:
            if is_trusted_url(seed_url):
                trusted_urls.append(seed_url)
            trusted_urls = dedupe_preserve_order(trusted_urls)
            if len(trusted_urls) >= settings.google_fallback_pages:
                break

    return trusted_urls[: settings.google_fallback_pages]


def _load_web_documents(urls: list[str]) -> list[Document]:
    documents: list[Document] = []

    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=(url,))
            loaded_docs = loader.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("google_fallback_loader_failed url=%s error=%s", url, exc)
            continue

        for doc in loaded_docs:
            doc.metadata["source"] = url
            doc.metadata["source_type"] = "google_fallback"
        documents.extend(loaded_docs)

    return documents


def _rank_documents(question: str, documents: list[Document], max_docs: int = 8) -> list[Document]:
    if not documents:
        return []

    query_terms = _tokenize(question)
    scored: list[tuple[int, int, Document]] = []

    for doc in documents:
        text = " ".join(doc.page_content.split())
        if not text:
            continue
        text_sample = text[:5000]
        score = len(query_terms & _tokenize(text_sample))
        scored.append((score, len(text_sample), doc))

    if not scored:
        return documents[:max_docs]

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    relevant = [doc for score, _, doc in scored if score > 0]
    if relevant:
        return relevant[:max_docs]
    return [doc for _, _, doc in scored[:max_docs]]


def _build_context_block(documents: list[Document], max_chars_per_doc: int = 1200) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = str(doc.metadata.get("source", "Unknown source"))
        title = str(doc.metadata.get("title", ""))
        snippet = " ".join(doc.page_content.split())[:max_chars_per_doc]
        blocks.append(
            f"[{idx}] Source: {source}\n"
            f"Title: {title}\n"
            f"Snippet: {snippet}"
        )
    return "\n\n".join(blocks)


def _extract_answer_text(response_content: object) -> str:
    if isinstance(response_content, str):
        return response_content.strip()
    if isinstance(response_content, list):
        return " ".join(str(item) for item in response_content).strip()
    return str(response_content).strip()


def _format_sources(documents: list[Document]) -> str:
    sources = dedupe_preserve_order(
        str(doc.metadata.get("source", "")).strip()
        for doc in documents
    )
    if not sources:
        return "- No trusted source found."
    return "\n".join(f"- {source}" for source in sources)


def answer_with_google_fallback(
    question: str,
    chat_history: list[BaseMessage],
) -> tuple[str, str]:
    """Answer a user question from trusted Google-discovered web sources."""
    logger.info("google_fallback_started question_length=%s", len(question))
    urls = _search_trusted_urls(question)
    if not urls:
        raise RuntimeError("No trusted URLs were found by Google fallback search.")

    documents = _load_web_documents(urls)
    if not documents:
        raise RuntimeError("Could not load trusted pages from Google fallback search.")

    selected_docs = _rank_documents(question, documents, max_docs=8)
    context_block = _build_context_block(selected_docs)

    llm = create_chat_model(temperature=0.15)
    qa_prompt = build_qa_prompt()
    prompt_messages = qa_prompt.format_messages(
        chat_history=chat_history,
        input=question,
        context=context_block,
    )
    response = llm.invoke(prompt_messages)
    answer_text = _extract_answer_text(getattr(response, "content", response))
    if not answer_text:
        answer_text = "I could not find enough information from trusted web sources."

    logger.info("google_fallback_completed sources=%s", len(selected_docs))
    return answer_text, _format_sources(selected_docs)
