from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
import os

from backend.app.config import settings
from backend.app.rag.safety import needs_emergency_redirect, emergency_message
from backend.app.rag.vector_store import query

load_dotenv()

GENERATION_MODEL = os.getenv("OPENAI_MODEL", settings.openai_model)


def _format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    seen = set()

    for r in results:
        key = (r.get("source_file", ""), r.get("url", ""))
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            {
                "file_name": r.get("source_file", ""),
                "title": "",
                "url": r.get("url", ""),
                "category": r.get("category", ""),
            }
        )

    return sources[:8]


def _auto_citations(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    seen = set()

    for r in results:
        file_name = (r.get("source_file") or "").strip()
        url = (r.get("url") or "").strip()

        if not file_name:
            continue
        if (file_name, url) in seen:
            continue
        seen.add((file_name, url))

        if url:
            lines.append(f"- [{file_name}] ({url}) — retrieved from official document chunk")
        else:
            lines.append(f"- [{file_name}] — retrieved from official document chunk")

        if len(lines) >= 6:
            break

    if not lines:
        return "- [unknown] — retrieved context"

    return "\n".join(lines)


def _build_prompt(user_message: str, results: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for r in results:
        file_name = r.get("source_file", "unknown")
        url = r.get("url", "")
        category = r.get("category", "")
        content = r.get("text", "")

        cite = f"[{file_name}]"
        if url:
            cite = f"[{file_name} | {url}]"

        context_blocks.append(
            f"""SOURCE {cite}
CATEGORY: {category}
CONTENT:
{content}
"""
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are the Ottawa Newcomer Support Chatbot.

STRICT RULES:
- Use ONLY the provided SOURCES to answer.
- If the SOURCES do not contain the answer, say: "I don't have enough information in the provided official documents to answer that."
- Do NOT guess.
- Do NOT add outside knowledge.
- Write a clear, helpful answer for newcomers in Ottawa.
- Keep the answer direct and practical.
- Do NOT invent citations or filenames.
- Do NOT list citations yourself. The system will add citations automatically.

USER QUESTION:
{user_message}

SOURCES:
{context}
"""


async def generate_answer(user_message: str, category: Optional[str] = None) -> Dict[str, Any]:
    if needs_emergency_redirect(user_message):
        return emergency_message()

    if not settings.openai_api_key:
        return {
            "answer": "OPENAI_API_KEY is not set. Add it to your .env file and restart the backend.",
            "sources": [],
        }

    # Faster + category-aware retrieval
    results = query(user_message, n_results=3, category=category)

    if not results:
        return {
            "answer": "I couldn't find relevant information in the local knowledge base yet. Please try another question or add more official documents.",
            "sources": [],
        }

    prompt = _build_prompt(user_message, results)

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        resp = client.responses.create(
            model=GENERATION_MODEL,
            input=prompt,
            temperature=0.1,
        )
        answer_text = resp.output_text
    except Exception:
        chat = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": "You follow the user's instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        answer_text = chat.choices[0].message.content or ""

    final_answer = (answer_text or "").strip()
    final_answer = final_answer + "\n\nCitations:\n" + _auto_citations(results)

    return {
        "answer": final_answer,
        "sources": _format_sources(results),
    }