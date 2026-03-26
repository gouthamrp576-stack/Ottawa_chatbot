# For Beginner

This file explains the project in very simple language.
If you are new to Python, AI apps, or this codebase, start here.

## 1. What this project is

This project is a chatbot for newcomers in Ottawa, Canada.

It can help answer questions about:
- housing
- healthcare
- jobs
- study resources
- local services

The app has two main jobs:
- show a chat interface in the browser
- look up trusted information and ask an AI model to answer clearly

## 2. Big picture: how the app works

Think of the app like this:

1. A user types a question in the chat page.
2. The app cleans and checks that question.
3. The app looks for useful documents in its local knowledge index.
4. The app sends the question and the found documents to an AI model.
5. The app shows the answer and the sources.
6. If the local index is missing, the app can try a trusted web fallback.

Short version:

- `main.py` = the screen the user sees
- `chat_service.py` = the brain that decides how to answer
- `rag_chain.py` = the retrieval + AI answering pipeline
- `retriever/` = code that builds and loads the local search index

## 3. What you need before running it

You need:
- Python 3.10 or newer
- a terminal
- this project folder on your computer
- either:
  - an OpenAI API key
  - or Ollama installed locally

If you are a complete beginner, OpenAI is usually the easiest first option.

## 4. First-time setup

Open a terminal in this project folder and run these commands one by one.

### Step 1: Create a virtual environment

```bash
python3 -m venv .venv
```

What this does:
- creates a private Python environment for this project
- keeps this project's packages separate from other projects

### Step 2: Turn it on

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

You know it worked when you see something like `(.venv)` in the terminal.

### Step 3: Install the project

```bash
pip install -e ".[dev]"
```

What this does:
- installs the app
- installs test tools
- makes the `src/ottawa_assistant` package importable

### Step 4: Create your environment file

```bash
cp .env.example .env
```

What this does:
- makes your own local settings file
- this file is where API keys and app settings go

### Step 5: Pick a model provider

Open `.env` and choose one of the options below.

#### Option A: OpenAI

Put something like this in `.env`:

```env
LOG_LEVEL=INFO
MODEL_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

#### Option B: Ollama

First install Ollama and pull models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Then put this in `.env`:

```env
LOG_LEVEL=INFO
MODEL_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## 5. Build the local knowledge index

Before the chatbot can answer from trusted local sources, it needs an index.

Run:

```bash
python -m ottawa_assistant.retriever.ingest --use-seed
```

What this does:
- downloads content from trusted websites
- splits the content into smaller pieces
- stores those pieces in a FAISS index for fast search

This index is saved in:

- `src/ottawa_assistant/retriever/index/`

You usually rebuild the index when:
- you change the embedding model
- you add new documents or sources

## 6. Run the app

Start the app with:

```bash
streamlit run src/ottawa_assistant/main.py
```

If `streamlit` is not found, use:

```bash
python -m streamlit run src/ottawa_assistant/main.py
```

Then open the address shown in the terminal.
It is usually:

- `http://localhost:8501`

## 7. Important files and what they do

### Main app files

- `src/ottawa_assistant/main.py`
  - starts the Streamlit app
  - builds the page layout
  - shows the chat messages and buttons

- `src/ottawa_assistant/chat_service.py`
  - handles one chat message from start to finish
  - cleans input
  - runs the retrieval pipeline
  - uses web fallback if needed
  - builds the final assistant message

- `src/ottawa_assistant/config.py`
  - reads values from `.env`
  - stores settings like provider, model names, and index path
  - checks that the settings are valid

- `src/ottawa_assistant/model_factory.py`
  - creates the chat model
  - creates the embedding model
  - supports OpenAI and Ollama

- `src/ottawa_assistant/prompts.py`
  - contains the instructions sent to the AI model
  - controls tone, scope, and grounding behavior

- `src/ottawa_assistant/rag_chain.py`
  - creates the retrieval chain
  - connects the retriever to the model
  - formats source links

- `src/ottawa_assistant/web_fallback.py`
  - used when the local index is missing or unusable
  - searches trusted sites and builds an answer from them

- `src/ottawa_assistant/utils.py`
  - contains small helper functions
  - for example: deduplicating items and sanitizing user input

- `src/ottawa_assistant/logging_utils.py`
  - sets up logging
  - helps the app print useful logs in a consistent format

### Retriever files

- `src/ottawa_assistant/retriever/ingest.py`
  - builds the FAISS index from web pages or PDFs

- `src/ottawa_assistant/retriever/vector_store.py`
  - saves and loads the FAISS index
  - checks that the embeddings still match the stored index

### Frontend assets

- `public/style.css`
  - controls the look of the UI

- `public/maple-leaf.svg`
  - image asset used by the UI

### Project files

- `.env.example`
  - example settings file

- `pyproject.toml`
  - project metadata
  - dependencies
  - test configuration

- `README.md`
  - short project overview

- `docs/developer_guide.md`
  - more detailed guide for developers

### Tests

- `tests/unit/`
  - small focused tests for individual functions

- `tests/integration/`
  - broader tests for how multiple parts work together

## 8. A simple mental model of the code flow

When a user types a question:

1. `main.py` collects the text from the chat box.
2. `chat_service.py` cleans the text and checks it.
3. `chat_service.py` calls the RAG pipeline from `rag_chain.py`.
4. `rag_chain.py` uses the retriever from `vector_store.py`.
5. The retriever searches the FAISS index built by `ingest.py`.
6. The model is created by `model_factory.py`.
7. The model uses prompts from `prompts.py`.
8. The answer comes back to `main.py`.
9. The answer and sources are shown in the UI.

If the local index fails:

1. `chat_service.py` switches to `web_fallback.py`
2. `web_fallback.py` looks for trusted web pages
3. the app still tries to return an answer with sources

## 9. Common beginner tasks

### I want to change the text the bot sees at startup

Look in:

- `src/ottawa_assistant/main.py`

Search for:

- `DEFAULT_ASSISTANT_MESSAGE`

### I want to change the AI instructions

Look in:

- `src/ottawa_assistant/prompts.py`

This is where the main system prompt lives.

### I want to change the page design

Look in:

- `public/style.css`

### I want to add more trusted websites

Look in:

- `src/ottawa_assistant/config.py`

Search for:

- `trusted_domains`
- `seed_web_sources`

Then rebuild the index:

```bash
python -m ottawa_assistant.retriever.ingest --use-seed
```

### I want to add a quick prompt button

Look in:

- `src/ottawa_assistant/main.py`

Search for:

- `QUICK_PROMPTS`

## 10. How to run tests

Run:

```bash
pytest tests/
```

If you only want a quick syntax check:

```bash
python -m compileall src tests
```

## 11. Common errors and how to fix them

### Error: `OPENAI_API_KEY is required`

Cause:
- you selected OpenAI but did not add the API key in `.env`

Fix:
- open `.env`
- add your real `OPENAI_API_KEY`

### Error: `Vector index not found`

Cause:
- the local knowledge index has not been built yet

Fix:

```bash
python -m ottawa_assistant.retriever.ingest --use-seed
```

### Error: `pytest: command not found`

Cause:
- test tools are not installed in the active environment

Fix:

```bash
pip install -e ".[dev]"
```

### The app runs but answers are weak

Possible causes:
- no local index was built
- the index was built with a different embedding model
- the fallback path is being used too often

Try:
- rebuilding the index
- checking `.env`
- reading the logs in the terminal

## 12. Words you may hear in this project

- `LLM`
  - large language model
  - the AI that writes the answer

- `Embeddings`
  - numerical representations of text
  - used to search for similar documents

- `RAG`
  - Retrieval-Augmented Generation
  - the app first finds documents, then asks the AI to answer using them

- `FAISS`
  - a library used for fast vector search
  - this is how the app searches the local index

- `Streamlit`
  - the Python library used to build the web UI

- `Fallback`
  - backup behavior when the main path fails

## 13. If you are new, what should you read first?

Read in this order:

1. this file
2. `README.md`
3. `src/ottawa_assistant/main.py`
4. `src/ottawa_assistant/chat_service.py`
5. `src/ottawa_assistant/config.py`
6. `src/ottawa_assistant/prompts.py`

That order gives you:
- what the app is
- how to run it
- what the screen does
- where the chat logic lives
- where settings come from
- what instructions the model follows

## 14. Final advice for beginners

Do not try to understand everything at once.

A good beginner workflow is:

1. run the app
2. ask one question
3. change one small thing
4. run it again
5. read only the file related to that small change

That is the fastest way to learn this codebase without getting overwhelmed.
