# Developer Guide — Ottawa Newcomer Assistant

Welcome to the Ottawa Newcomer Assistant project! This guide will help you get your local environment set up, explain how the system works, and show you how to add new features.

---

## 1. Quick Start (Running Locally)

### Prerequisites
- Python 3.10+
- Access to OpenAI API or a local Ollama instance.

### Installation

1. **Clone and setup a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the project in editable mode** with development dependencies:
   ```bash
   # This makes `src/ottawa_assistant` importable anywhere in the project
   pip install -e ".[dev]"
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   *Note: If using OpenAI, you must set `OPENAI_API_KEY`.*

### Running the Application

To start the Streamlit frontend:
```bash
streamlit run src/ottawa_assistant/main.py
```
This will open the chatbot interface in your browser (usually at `http://localhost:8501`).

---

## 2. System Architecture

The project is structured under `src/ottawa_assistant/`:

- **`main.py`**: The Streamlit UI entrypoint. Handles state, chat rendering, and invokes the RAG pipeline.
- **`chat_service.py`**: Central application service for validating input, running RAG, handling fallback, and updating chat history.
- **`config.py`**: Centralized configuration management (reads from `.env`). Validates provider setups (OpenAI vs Ollama).
- **`logging_utils.py`**: Shared logging setup so the UI, ingestion flow, and fallback logic produce consistent logs.
- **`model_factory.py`**: Instantiates LLMs and Embedding models based on config.
- **`prompts.py`**: Contains the system prompts and question-rewriting logic.
- **`rag_chain.py`**: Combines the retriever and LLM into a LangChain pipeline.
- **`web_fallback.py`**: A fallback mechanism that searches Google for trusted domains (like `ottawa.ca`) if the local FAISS index is unavailable or fails. Uses lightweight lexical ranking to find the best snippets.
- **`utils.py`**: Shared utility functions (e.g., input sanitization, list deduplication).

### `retriever/` (Data Ingestion)
- **`vector_store.py`**: Manages saving/loading the FAISS index to disk (`src/ottawa_assistant/retriever/index/`). Asserts embedding compatibility before loading.
- **`ingest.py`**: Script to scrape websites or parse PDFs and build the FAISS index.

---

## 3. Data Ingestion (Building the Index)

For the RAG pipeline to work without relying entirely on the web fallback, you need to build a local FAISS index.

1. **Ingest Default Seed URLs** (defined in `config.py`):
   ```bash
   python -m ottawa_assistant.retriever.ingest --use-seed
   ```

2. **Ingest Specific URLs**:
   ```bash
   python -m ottawa_assistant.retriever.ingest --urls https://ottawa.ca/en/services https://www.ontario.ca/page
   ```

3. **Ingest Local PDFs**:
   ```bash
   python -m ottawa_assistant.retriever.ingest --pdfs ./docs/guide1.pdf ./docs/guide2.pdf
   ```

*Tip: If you change your `EMBEDDING_PROVIDER` or model in `.env`, you **must** rebuild the index. The app will refuse to load an index built with different embeddings.*

---

## 4. Running Tests

We use `pytest` for unit and integration testing. Real logic (like URL validation, input sanitization, prompt construction, and ranking algorithms) is tested thoroughly.

```bash
# Quick syntax smoke test
python -m compileall src tests

# Run all tests
pytest tests/

# Run unit tests with verbose output
pytest tests/unit/ -v

# Run tests and stop on the first failure
pytest -x
```

---

## 5. How to Implement New Features

Here is the general workflow for adding new functionality:

### Example: Adding a New Quick Prompt Button

1. **Update the UI (`main.py`)**:
   Locate the `QUICK_PROMPTS` list at the top of the file and add your new tuple `("Button Label", "Prompt text")`.
   Or, manually add a button inside `_render_left_rail()`:
   ```python
   if st.button("New Topic", use_container_width=True):
       _process_input("Tell me about [New Topic] in Ottawa.")
       st.rerun()
   ```

### Example: Enhancing the RAG Pipeline

If you want the bot to reason differently or use a different chain structure (e.g., adding an intermediate verification step):

1. **Update `prompts.py`** to adjust how the LLM receives context.
2. **Modify `rag_chain.py`** (`build_rag_chain()`). You can swap `create_stuff_documents_chain` for something more complex, or wrap the final `create_retrieval_chain` output.
3. Write/update a test in `tests/unit/test_rag_chain.py`.

### Example: Adding a New Supported LLM Provider (e.g., Anthropic)

1. **Update Config**: Add to `SUPPORTED_MODEL_PROVIDERS` in `config.py`. Add dataclass fields for API keys (e.g., `anthropic_api_key`). Update `validate_settings()`.
2. **Update Factory**: In `model_factory.py`, update `create_chat_model()` and `create_embeddings()` to return the appropriate LangChain objects when `settings.model_provider == "anthropic"`.
3. **Update Tests**: Add unit tests in `test_config.py` and `test_model_factory.py` (if applicable) for the new provider.

### Best Practices

- **Use the shared logger**: Prefer `logging_utils.configure_logging()` and module loggers instead of bare `print()`.
- **Fail Fast**: Use `validate_settings()` to catch configuration errors at startup, rather than deep inside an API call.
- **Test Real Logic**: When adding a new utility or ranking function, write a test in `tests/unit/` that calls the function with real inputs (not mocks) to verify behavior.
- **Keep UI Thin**: Put validation, orchestration, and fallback logic in `chat_service.py` so `main.py` stays focused on rendering.
