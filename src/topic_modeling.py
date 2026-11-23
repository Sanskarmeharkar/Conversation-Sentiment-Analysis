"""
Topic Modeling Module for Conversational Utterances
Implements Non-negative Matrix Factorization (NMF) and Latent Dirichlet Allocation (LDA).
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def extract_conversation_topics(
    texts: List[str],
    n_topics: int = 3,
    top_words_count: int = 6,
    method: str = "nmf"
) -> Dict[str, Any]:
    """
    Decompose conversational turns into semantic topics and assign dominant topic to each utterance.
    """
    valid_texts = [str(t).strip() for t in texts if str(t).strip()]
    if len(valid_texts) < 2:
        return {"topics": [], "assigned": [], "topic_distribution": {}}

    num_topics = max(1, min(n_topics, len(valid_texts)))

    try:
        if method.lower() == "lda":
            vec = CountVectorizer(stop_words="english", max_features=500, min_df=1)
            matrix = vec.fit_transform(valid_texts)
            model = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
        else:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000, min_df=1)
            matrix = vec.fit_transform(valid_texts)
            model = NMF(n_components=num_topics, random_state=42, init="nndsvda", max_iter=300)

        topic_weights = model.fit_transform(matrix)
        feature_names = vec.get_feature_names_out()

        topics = []
        for topic_idx, component in enumerate(model.components_):
            top_indices = component.argsort()[-top_words_count:][::-1]
            words = [feature_names[i] for i in top_indices]
            topics.append({
                "topic_id": topic_idx + 1,
                "label": f"Topic {topic_idx + 1}: " + ", ".join(words[:3]),
                "keywords": words
            })

        assigned = (topic_weights.argmax(axis=1) + 1).tolist()
        topic_counts = pd.Series(assigned).value_counts().to_dict()

        return {
            "topics": topics,
            "assigned": assigned,
            "topic_distribution": {f"Topic {k}": v for k, v in topic_counts.items()}
        }
    except Exception as exc:
        return {"topics": [], "assigned": [], "topic_distribution": {}, "error": str(exc)}
