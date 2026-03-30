import re

EMERGENCY_PATTERNS = [
    r"suicide",
    r"self-harm",
    r"kill myself",
    r"overdose",
    r"chest pain",
    r"can't breathe",
    r"stroke",
    r"emergency",
]

def needs_emergency_redirect(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in EMERGENCY_PATTERNS)

def emergency_message():
    return {
        "answer": (
            "If this is an emergency or someone is in immediate danger, call 911 right now.\n"
            "If you are in Canada and need urgent mental health support, you can call or text 988.\n"
            "I can share general information and official resources, but I can’t provide emergency or medical advice."
        ),
        "sources": [],
    }
