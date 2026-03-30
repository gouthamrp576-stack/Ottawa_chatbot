from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import tiktoken

@dataclass
class Chunk:
    text: str
    start_token: int
    end_token: int

def chunk_text(text: str, *, chunk_tokens: int = 1000, overlap_tokens: int = 150, encoding_name: str = "cl100k_base") -> List[Chunk]:
    """Token-aware chunking using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text or "")
    if not tokens:
        return []

    chunks: List[Chunk] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + chunk_tokens)
        chunk_tokens_slice = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens_slice).strip()
        if chunk_text_str:
            chunks.append(Chunk(text=chunk_text_str, start_token=start, end_token=end))
        if end >= len(tokens):
            break
    return chunks
