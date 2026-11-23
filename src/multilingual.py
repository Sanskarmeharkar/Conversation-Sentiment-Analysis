"""
Multilingual Text Handling and Language Detection Module
Identifies conversation language and supports preprocessing for multilingual conversational setups.
"""

from typing import Dict, Any, List
import re

# Common ISO-639-1 language names
LANG_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese"
}


def detect_conversation_language(texts: List[str]) -> Dict[str, Any]:
    """Detect dominant language of the conversation turns with lightweight heuristics."""
    combined = " ".join(texts)
    
    # Check if langdetect is installed
    try:
        from langdetect import detect_langs
        predictions = detect_langs(combined)
        primary = predictions[0]
        lang_code = primary.lang
        confidence = float(primary.prob)
        return {
            "language_code": lang_code,
            "language_name": LANG_MAP.get(lang_code, lang_code.upper()),
            "confidence": round(confidence * 100, 1),
            "is_english": lang_code == "en"
        }
    except Exception:
        # Simple ASCII / Latin heuristic fallback
        ascii_chars = sum(1 for c in combined if ord(c) < 128)
        ratio = ascii_chars / max(1, len(combined))
        return {
            "language_code": "en" if ratio > 0.85 else "other",
            "language_name": "English" if ratio > 0.85 else "Non-English / Multilingual",
            "confidence": round(ratio * 100, 1),
            "is_english": ratio > 0.85
        }


def normalize_multilingual_text(text: str) -> str:
    """Normalize unicode representations, quotes, and punctuation marks across languages."""
    if not text:
        return ""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", " - ")
    return re.sub(r"\s+", " ", text).strip()
