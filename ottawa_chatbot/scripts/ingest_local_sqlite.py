import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

from backend.vector_store_sqlite import reset_db, add_chunks

load_dotenv()

DATA_DIR = Path("data/raw_sources")

# Faster chunking
CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP = 200

EMBED_MODEL = "text-embedding-3-small"


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def chunk_text(text: str):
    chunks = []
    i = 0
    step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP
    if step <= 0:
        step = CHUNK_SIZE_CHARS

    while i < len(text):
        chunk = text[i : i + CHUNK_SIZE_CHARS]
        if chunk.strip():
            chunks.append(chunk)
        i += step

    return chunks


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in your .env file.")

    client = OpenAI()

    if not DATA_DIR.exists():
        raise RuntimeError(f"Missing folder: {DATA_DIR.resolve()}")

    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found under {DATA_DIR.resolve()}")

    print(f"Found {len(pdfs)} PDFs")
    print("Resetting SQLite vector DB...")
    reset_db()

    batch = []
    total_chunks = 0

    for pdf_path in pdfs:
        print(f"\nLoading: {pdf_path}")
        raw = extract_text_from_pdf(pdf_path)
        text = clean_text(raw)
        chunks = chunk_text(text)
        print(f"Chunks: {len(chunks)}")

        category = pdf_path.parent.name.lower().strip()

        for ch in chunks:
            emb = client.embeddings.create(model=EMBED_MODEL, input=ch).data[0].embedding

            batch.append(
                {
                    "text": ch,
                    "embedding": emb,
                    "source_file": pdf_path.name,
                    "url": "",
                    "category": category,
                }
            )
            total_chunks += 1

            if len(batch) >= 50:
                add_chunks(batch)
                batch.clear()
                print(f"Stored {total_chunks} chunks so far...")

    if batch:
        add_chunks(batch)

    print(f"\nDONE ✅ Stored {total_chunks} chunks into vector_db.sqlite")


if __name__ == "__main__":
    main()