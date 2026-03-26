"""Shared utility functions for the Ottawa Newcomer Assistant."""

from __future__ import annotations

import re
from typing import Iterable


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique non-empty strings while preserving insertion order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = raw_item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_INPUT_LENGTH = 2000


def sanitize_user_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Strip control characters and enforce a maximum length.

    Returns the sanitised string.  Raises ``ValueError`` if the result
    exceeds *max_length* after stripping.
    """
    cleaned = _CONTROL_CHAR_RE.sub("", text).strip()
    if len(cleaned) > max_length:
        raise ValueError(
            f"Input exceeds maximum length of {max_length} characters "
            f"(got {len(cleaned)})."
        )
    return cleaned
