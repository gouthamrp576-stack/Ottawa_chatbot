from __future__ import annotations

from typing import Dict, Any, List, Optional
import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.vector_store_sqlite import query_similar

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _normalize_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return None

    cat = category.strip().lower()

    mapping = {
        "all": None,
        "all topics": None,
        "healthcare": "healthcare",
        "housing": "housing",
        "jobs": "jobs",
        "transportation": "transportation",
        "community": "community_events",
        "community events": "community_events",
        "study": "study",
    }

    return mapping.get(cat, cat)


def _guess_category_from_question(question: str) -> Optional[str]:
    q = question.lower()

    if any(w in q for w in ["ohip", "doctor", "clinic", "health", "hospital", "mental health", "988", "911"]):
        return "healthcare"

    if any(w in q for w in ["rent", "lease", "tenant", "housing", "landlord", "shelter", "residence"]):
        return "housing"

    if any(w in q for w in ["job", "resume", "interview", "employment", "work permit", "job bank"]):
        return "jobs"

    if any(w in q for w in ["bus", "oc transpo", "presto", "lrt", "fare", "route", "train", "transit"]):
        return "transportation"

    if any(w in q for w in ["community", "festival", "volunteer", "library", "ociso", "ymca", "linc"]):
        return "community_events"

    if any(w in q for w in ["study", "student", "university", "tuition", "scholarship", "permit", "academic"]):
        return "study"

    return None


def query(user_message: str, n_results: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
    client = OpenAI()

    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=user_message
    ).data[0].embedding

    normalized_category = _normalize_category(category)

    if not normalized_category:
        normalized_category = _guess_category_from_question(user_message)

    return query_similar(
        query_embedding=q_emb,
        top_k=n_results,
        category=normalized_category,
    )