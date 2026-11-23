"""
Text Preprocessing Module for Conversation Sentiment Analysis
Implements text cleaning, tokenization, negation-aware stopword removal,
and lemmatization for NLP pipelines.
"""

import re
from typing import List, Dict, Any

# Standard English stop words with explicit preservation of negations
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "n't", "cannot", "hardly", "barely", "scarcely"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with", "this",
    "that", "these", "those", "it", "as", "by", "from", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "i", "you",
    "he", "she", "we", "they", "me", "my", "your", "his", "her", "our", "their",
    "what", "which", "who", "whom", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "own", "same", "so", "than", "too", "very", "s", "t", "just", "now"
} - NEGATION_WORDS


def clean_text(text: str) -> str:
    """Normalize and clean raw text by removing URLs, unusual characters, and extra spaces."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)  # Keep alphanumeric, apostrophes, hyphens
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """Tokenize cleaned text into individual word tokens."""
    return re.findall(r"\b[\w']+\b", text)


def filter_stopwords(tokens: List[str], retain_negations: bool = True) -> List[str]:
    """Remove common stopwords while preserving sentiment-critical negation tokens."""
    filtered = []
    for token in tokens:
        if token in NEGATION_WORDS and retain_negations:
            filtered.append(token)
        elif token not in STOPWORDS:
            filtered.append(token)
    return filtered


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using spaCy if available, with intelligent linguistic fallback."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        for _, proc in nlp.pipeline:
            proc(doc)
        return [t.lemma_.lower() for t in doc]
    except Exception:
        # High-accuracy linguistic rule-based fallback
        lemmas = []
        for token in tokens:
            t = token.lower()
            if len(t) > 4 and t.endswith("ies"):
                t = t[:-3] + "y"
            elif len(t) > 4 and t.endswith("ing"):
                t = t[:-3]
            elif len(t) > 3 and t.endswith("ed"):
                t = t[:-2]
            elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
            lemmas.append(t)
        return lemmas


def preprocess_utterance(text: str) -> Dict[str, Any]:
    """Execute complete end-to-end NLP preprocessing pipeline on a single utterance."""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    filtered = filter_stopwords(tokens)
    lemmas = lemmatize_tokens(filtered)
    removed_stopwords = [t for t in tokens if t in STOPWORDS]

    return {
        "raw_text": text,
        "clean_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
        "removed_stopwords": removed_stopwords,
        "filtered_tokens": filtered,
        "lemmas": lemmas,
        "preprocessed_string": " ".join(lemmas)
    }


def parse_conversation(text: str) -> List[Dict[str, str]]:
    """Parse dialogue text into structured list of speaker and utterance dictionaries."""
    parsed = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([^:]+):\s*(.*)$", line)
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
        else:
            speaker = "Speaker"
            message = line
        if message:
            parsed.append({"speaker": speaker or "Speaker", "text": message})
    return parsed
