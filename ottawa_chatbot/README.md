# Ottawa Newcomer Support Chatbot (Local RAG with Chroma)

This project runs **fully locally** (except OpenAI API calls for embeddings + generation). It uses:
- **FastAPI** backend (RAG retrieval + response)
- Simple HTML/JS frontend
- **ChromaDB** as a local vector database (persisted on disk)
- **pypdf** for PDF text extraction
- **OpenAI embeddings** (`text-embedding-3-small`) + an LLM (default: `gpt-4.1-mini`)

## 1) Setup
1. Copy `.env.example` to `.env` and set:
   - `OPENAI_API_KEY=...`

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 2) Ingest PDFs into local Chroma
Put PDFs under `data/raw_sources/**.pdf` (sample PDFs are already included).

Run:
```bash
python scripts/ingest_local_chroma.py
```

This creates a local persisted DB in `./chroma_db`.

## 3) Run backend
```bash
python backend/run_server.py
```

Backend will be available at `http://localhost:8000`.

## 4) Run frontend
Open:
- `frontend/index.html`

Then ask a question.

## Notes
- If answers say ingestion is missing, run step (2) again.
- To add official URLs/titles, fill `data/metadata/trusted_sources.csv` with:
  `category,title,url,notes` + ensure filename matches in ingestion (optional).
