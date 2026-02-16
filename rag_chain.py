"""Build the LangChain RAG pipeline used by the Streamlit app."""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document

from config import settings, validate_settings
from model_factory import create_chat_model
from prompts import build_contextualize_prompt, build_qa_prompt
from retriever.vector_store import create_retriever

try:
    # LangChain 0.3.x style imports.
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    try:
        # LangChain 1.x moved legacy chains into langchain_classic.
        from langchain_classic.chains import (  # type: ignore
            create_history_aware_retriever,
            create_retrieval_chain,
        )
        from langchain_classic.chains.combine_documents import (  # type: ignore
            create_stuff_documents_chain,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import retrieval chain helpers. "
            "Install compatible packages with:\n"
            "pip install \"langchain<1.0\" \"langchain-openai<1.0\" "
            "\"langchain-community<1.0\" \"langchain-text-splitters<1.0\""
        ) from exc


def build_rag_chain():
    """Create a conversational RAG chain with history-aware retrieval."""
    validate_settings(require_embeddings=True)

    # Main chat model used for both question rewriting and final answering.
    llm = create_chat_model(temperature=0.15)

    # 1) Retrieve documents.
    retriever = create_retriever(k=settings.retrieval_k)
    history_prompt = build_contextualize_prompt()
    qa_prompt = build_qa_prompt()

    # 2) Rewrite follow-up questions into standalone retrieval queries.
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=history_prompt,
    )
    # 3) Generate grounded answer from retrieved context.
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    # 4) Combine retrieval + QA into one runnable chain.
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def format_sources(documents: Iterable[Document]) -> str:
    """Render unique source links/paths from retrieved documents."""
    seen: set[str] = set()
    ordered_sources: list[str] = []

    for doc in documents:
        source = str(doc.metadata.get("source", "")).strip()
        if source and source not in seen:
            seen.add(source)
            ordered_sources.append(source)

    if not ordered_sources:
        return "- No official source available for this answer."
    return "\n".join(f"- {src}" for src in ordered_sources)
