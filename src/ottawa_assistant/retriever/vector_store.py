"""Vector store helpers using FAISS with configurable embeddings."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import settings
from ..model_factory import create_embeddings, embedding_signature

INDEX_METADATA_FILE = "index_metadata.json"

def _index_metadata_path(index_dir: Path) -> Path:
    return index_dir / INDEX_METADATA_FILE


def _index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def _save_index_metadata(index_dir: Path) -> None:
    payload = {"embedding_signature": embedding_signature()}
    _index_metadata_path(index_dir).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _read_index_embedding_signature(index_dir: Path) -> str | None:
    path = _index_metadata_path(index_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    signature = payload.get("embedding_signature")
    return str(signature) if isinstance(signature, str) else None


def _validate_embedding_compatibility(index_dir: Path) -> None:
    saved_signature = _read_index_embedding_signature(index_dir)
    current_signature = embedding_signature()
    if saved_signature is None:
        if settings.allow_embedding_mismatch:
            return
        raise RuntimeError(
            "Vector index metadata is missing, so embedding compatibility cannot be verified.\n"
            "Rebuild index with `python -m ottawa_assistant.retriever.ingest --use-seed` "
            "or set ALLOW_EMBEDDING_MISMATCH=true."
        )
    if saved_signature == current_signature:
        return
    if settings.allow_embedding_mismatch:
        return
    raise RuntimeError(
        "Vector index was built with a different embedding setup.\n"
        f"Current: {current_signature}\n"
        f"Index:   {saved_signature}\n"
        "Rebuild index with `python -m ottawa_assistant.retriever.ingest --use-seed` "
        "or set ALLOW_EMBEDDING_MISMATCH=true."
    )


def save_vector_store(documents: list[Document], index_dir: Path | None = None) -> None:
    """Build and persist the FAISS index from split documents."""
    if not documents:
        raise ValueError("No documents provided to build vector store.")

    target_dir = index_dir or settings.vector_store_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(target_dir))
    _save_index_metadata(target_dir)


def load_vector_store(index_dir: Path | None = None) -> FAISS:
    """Load persisted FAISS index from disk."""
    target_dir = index_dir or settings.vector_store_dir
    if not _index_exists(target_dir):
        raise FileNotFoundError(
            f"Vector index not found at {target_dir}. "
            "Run `python -m ottawa_assistant.retriever.ingest --use-seed` first."
        )

    _validate_embedding_compatibility(target_dir)
    embeddings = create_embeddings()
    return FAISS.load_local(
        str(target_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def create_retriever(k: int | None = None) -> BaseRetriever:
    """Create similarity retriever from the stored FAISS index."""
    store = load_vector_store()
    top_k = k if k is not None else settings.retrieval_k
    return store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
