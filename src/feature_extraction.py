"""
Feature Extraction Module for Conversation NLP
Implements TF-IDF keyword analysis, n-gram extraction, and Named Entity Recognition (NER).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(texts: List[str], max_features: int = 15, ngram_range: Tuple[int, int] = (1, 2)) -> pd.DataFrame:
    """Extract salient keywords and phrases using TF-IDF weighting."""
    valid_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not valid_texts or len(valid_texts) < 1:
        return pd.DataFrame(columns=["Keyword/Phrase", "Score"])

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
        )
        matrix = vectorizer.fit_transform(valid_texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        feature_names = vectorizer.get_feature_names_out()

        df = pd.DataFrame({
            "Keyword/Phrase": feature_names,
            "Score": [round(float(s), 4) for s in scores]
        }).sort_values("Score", ascending=False)

        return df
    except ValueError:
        return pd.DataFrame(columns=["Keyword/Phrase", "Score"])


def extract_named_entities(messages: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Named Entities (Persons, Organizations, Locations, Dates) using spaCy."""
    entities = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")

        for i, item in enumerate(messages):
            speaker = item.get("speaker", "Unknown")
            text = str(item.get("text", ""))
            doc = nlp(text)

            for ent in doc.ents:
                entities.append({
                    "Turn": i + 1,
                    "Message": i + 1,
                    "Speaker": speaker,
                    "Entity": ent.text,
                    "Type": ent.label_,
                    "Explanation": spacy.explain(ent.label_) or ent.label_
                })

        if entities:
            return pd.DataFrame(entities)
        return pd.DataFrame(columns=["Turn", "Message", "Speaker", "Entity", "Type", "Explanation"])
    except Exception:
        return pd.DataFrame(columns=["Turn", "Message", "Speaker", "Entity", "Type", "Explanation"])
