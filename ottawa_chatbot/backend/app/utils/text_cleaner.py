from __future__ import annotations

import re

def clean_text(text: str) -> str:
    """Light cleanup for PDF-extracted text.

    - Collapses repeated whitespace
    - Removes common header/footer artifacts (heuristic)
    """
    if not text:
        return ""
    # Remove non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Heuristic: remove page numbers like "Page 1 of 10"
    text = re.sub(r"\bPage\s+\d+\s+(of|/)\s*\d+\b", " ", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
