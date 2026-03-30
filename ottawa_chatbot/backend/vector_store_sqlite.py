import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db.sqlite")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            source_file TEXT,
            url TEXT,
            category TEXT
        )
        """
    )
    conn.commit()
    return conn


def reset_db():
    conn = _connect()
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.commit()
    conn.close()
    _connect().close()


def add_chunks(items: List[Dict[str, Any]]):
    conn = _connect()
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO chunks (text, embedding, source_file, url, category) VALUES (?, ?, ?, ?, ?)",
        [
            (
                it["text"],
                json.dumps(it["embedding"]),
                it.get("source_file", ""),
                it.get("url", ""),
                it.get("category", ""),
            )
            for it in items
        ],
    )
    conn.commit()
    conn.close()


def query_similar(
    query_embedding: List[float],
    top_k: int = 3,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _connect()

    if category and category != "all":
        rows = conn.execute(
            "SELECT text, embedding, source_file, url, category FROM chunks WHERE LOWER(category) = ?",
            (category.lower(),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT text, embedding, source_file, url, category FROM chunks"
        ).fetchall()

    conn.close()

    if not rows:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q) + 1e-10

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for text, emb_json, source_file, url, cat in rows:
        e = np.array(json.loads(emb_json), dtype=np.float32)
        score = float(np.dot(q, e) / (q_norm * (np.linalg.norm(e) + 1e-10)))
        scored.append(
            (
                score,
                {
                    "text": text,
                    "source_file": source_file,
                    "url": url,
                    "category": cat,
                    "score": score,
                },
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]