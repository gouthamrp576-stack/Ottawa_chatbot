"""Ingest trusted web pages and PDFs into the FAISS vector index.

Examples:
1) Use trusted seed websites:
   python -m ottawa_assistant.retriever.ingest --use-seed

2) Ingest specific web pages:
   python -m ottawa_assistant.retriever.ingest --urls https://ottawa.ca/... https://www.ontario.ca/...

3) Ingest local PDFs:
   python -m ottawa_assistant.retriever.ingest --pdfs ./docs/guide1.pdf ./docs/guide2.pdf

4) Mix web and PDF sources:
   python -m ottawa_assistant.retriever.ingest --use-seed --pdfs ./docs/newcomer_guide.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import is_trusted_url, settings, validate_settings
from .vector_store import save_vector_store


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = raw_item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def load_web_documents(urls: list[str]) -> list[Document]:
    """Load content from trusted web pages only."""
    docs: list[Document] = []

    for url in _dedupe_preserve_order(urls):
        if not is_trusted_url(url):
            print(f"[skip] Untrusted domain: {url}")
            continue

        try:
            loader = WebBaseLoader(web_paths=(url,))
            loaded = loader.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Could not load URL {url}: {exc}")
            continue

        for item in loaded:
            item.metadata["source"] = url
            item.metadata["source_type"] = "web"
        docs.extend(loaded)
        print(f"[ok] Loaded web source: {url}")

    return docs


def load_pdf_documents(pdf_paths: list[str]) -> list[Document]:
    """Load content from local PDF files."""
    docs: list[Document] = []

    for raw_path in pdf_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(f"[skip] PDF not found: {path}")
            continue
        if path.suffix.lower() != ".pdf":
            print(f"[skip] Not a PDF file: {path}")
            continue

        try:
            loader = PyPDFLoader(str(path))
            loaded = loader.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Could not load PDF {path}: {exc}")
            continue

        for item in loaded:
            page = item.metadata.get("page")
            page_label = f" (page {page + 1})" if isinstance(page, int) else ""
            item.metadata["source"] = f"{path.resolve()}{page_label}"
            item.metadata["source_type"] = "pdf"
        docs.extend(loaded)
        print(f"[ok] Loaded PDF: {path}")

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks optimized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(urls: list[str], pdfs: list[str]) -> None:
    """Load, split, and store documents in FAISS."""
    validate_settings(require_embeddings=True)
    all_docs = load_web_documents(urls) + load_pdf_documents(_dedupe_preserve_order(pdfs))
    if not all_docs:
        raise RuntimeError("No documents loaded. Provide valid URLs/PDF paths.")

    chunks = split_documents(all_docs)
    save_vector_store(chunks)
    print(
        f"[done] Indexed {len(all_docs)} documents as {len(chunks)} chunks in "
        f"{settings.vector_store_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents for Ottawa newcomer assistant")
    parser.add_argument(
        "--use-seed",
        action="store_true",
        help="Ingest default trusted web sources from config.",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="Trusted web URLs to ingest.",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=[],
        help="Local PDF file paths to ingest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = list(args.urls)
    if args.use_seed:
        urls.extend(settings.seed_web_sources)
    urls = _dedupe_preserve_order(urls)

    try:
        ingest(urls=urls, pdfs=list(args.pdfs))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
