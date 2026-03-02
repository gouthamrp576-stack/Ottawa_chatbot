# Ottawa Newcomer Assistant (Streamlit + LangChain + RAG, OpenAI or Ollama)

This project is a ready-to-run AI assistant for newcomers in Ottawa, Canada.

It helps with:
- Housing (rentals, tenant rights, official resources)
- Healthcare (OHIP and local health services)
- Community integration (events, language programs, newcomer support)
- Administrative tasks (documents, transport, essential setup)

The app uses RAG with trusted sources and shows source links in every factual response.
It supports both cloud (`OpenAI`) and local (`Ollama`) models.
If the local FAISS index is unavailable, it can automatically fall back to trusted Google search.

## Project Structure

```text
.
├── src/
│   └── ottawa_assistant/
│       ├── __init__.py
│       ├── main.py              # Streamlit app implementation
│       ├── prompts.py           # Custom newcomer prompt templates
│       ├── rag_chain.py         # LangChain conversational retrieval chain
│       ├── config.py            # Environment settings + trusted domain list
│       ├── model_factory.py     # OpenAI/Ollama model + embedding factories
│       ├── web_fallback.py      # Trusted Google fallback answer flow
│       └── retriever/
│           ├── __init__.py
│           ├── ingest.py        # Ingest web pages and PDFs
│           ├── vector_store.py  # FAISS load/save helpers
│           └── index/           # Persisted vector index + metadata
├── requirements.txt
├── .env.example
└── public/
    ├── maple-leaf.svg
    └── style.css
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you already installed packages before, force-refresh compatible versions:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

Google fallback dependency can also be installed directly:

```bash
pip install googlesearch-python
```

3. Create `.env`:

```bash
cp .env.example .env
```

4. Configure model provider in `.env` (OpenAI or Ollama).
5. Expose the `src` package path:

```bash
export PYTHONPATH="$PWD/src"
```

## Model Provider Setup

### Option A: OpenAI (default)

Use:

```env
MODEL_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=...
```

### Option B: Local Ollama

1. Install and start Ollama:
   - [https://ollama.com/download](https://ollama.com/download)
2. Pull models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

3. Set `.env`:

```env
MODEL_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

You can also run hybrid mode (for example `MODEL_PROVIDER=ollama` with `EMBEDDING_PROVIDER=openai`).

## Ingest Data (RAG Index)

Important:
- Rebuild the index after changing embedding provider/model.
- The app checks embedding/index compatibility automatically.
- If `EMBEDDING_PROVIDER=ollama`, ensure `OLLAMA_EMBEDDING_MODEL` is pulled before ingest.
- If your index was created with an older version of this project, rebuild once to add index metadata.

Google fallback behavior:
- When index is missing or incompatible, the app automatically searches trusted domains via Google.
- You can control this in `.env`:
  - `ENABLE_WEB_FALLBACK=true`
  - `GOOGLE_FALLBACK_RESULTS=10`
  - `GOOGLE_FALLBACK_PAGES=5`

### Option A: Ingest default trusted Ottawa/Ontario/Canada sources

```bash
python -m ottawa_assistant.retriever.ingest --use-seed
```

### Option B: Ingest specific trusted web pages

```bash
python -m ottawa_assistant.retriever.ingest --urls \
  https://ottawa.ca/en/family-and-social-services/immigration-and-settlement \
  https://www.ontario.ca/page/apply-ohip-and-get-health-card
```

### Option C: Ingest local PDFs

```bash
python -m ottawa_assistant.retriever.ingest --pdfs ./docs/newcomer-guide.pdf ./docs/tenant-rights.pdf
```

### Option D: Combine web + PDFs

```bash
python -m ottawa_assistant.retriever.ingest --use-seed --pdfs ./docs/newcomer-guide.pdf
```

## Run the Streamlit App

```bash
streamlit run src/ottawa_assistant/main.py
```

Open the URL shown in terminal (usually [http://localhost:8501](http://localhost:8501)).

## Notes

- The assistant only ingests trusted domains from `src/ottawa_assistant/config.py`.
- If no vector index exists, the app tells you to run ingestion first.
- The interface is Canada-themed (`public/style.css`, `public/maple-leaf.svg`), with maple-leaf visual branding.
- Runtime details (LLM, embeddings, retrieval top-k) are shown in the UI sidebar.
