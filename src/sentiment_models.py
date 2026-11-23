"""
Sentiment and Emotion Models Module
Encapsulates inference logic for classical ML baselines (Naive Bayes, Logistic Regression, Linear SVM)
and fine-tuned transformer architectures (DistilBERT, Context-DistilBERT, RoBERTa).
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def load_classical_bundle(model_path: Path) -> Dict[str, Any]:
    """Load pre-trained classical models and TF-IDF vectorizer bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Classical model artifact missing at {model_path}")
    return joblib.load(model_path)


def load_transformer_checkpoint(checkpoint_dir: Path, use_gpu: bool = False) -> Tuple[Any, Any, torch.device]:
    """Load HuggingFace transformer tokenizer and sequence classifier onto target device."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Transformer checkpoint directory missing at {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_classical_sentiment(
    messages: List[Dict[str, str]],
    model_key: str,
    classical_bundle: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], float]:
    """Run inference using classical TF-IDF model and record latency."""
    start_time = time.perf_counter()
    clf = classical_bundle[model_key]
    vectorizer = classical_bundle["tfidf_vectorizer"]
    class_order = list(getattr(clf, "classes_", [0, 1, 2]))
    label_map = {str(label).strip().lower(): i for i, label in enumerate(class_order)}

    predictions = []
    for i, msg in enumerate(messages):
        text = msg["text"]
        feats = vectorizer.transform([text])
        raw_pred = clf.predict(feats)[0]

        if isinstance(raw_pred, str):
            pred_id = label_map.get(raw_pred.strip().lower(), 1)
        else:
            pred_id = int(raw_pred)

        # Calculate class probabilities
        probs = None
        if hasattr(clf, "predict_proba"):
            probs = np.asarray(clf.predict_proba(feats)[0], dtype=float)
        elif hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(feats)[0], dtype=float)
            if scores.ndim == 0:
                scores = np.array([-scores, 0.0, scores])
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum()

        if probs is not None:
            prob_dict = {
                label_map.get(str(l).strip().lower(), idx): float(p)
                for idx, (l, p) in enumerate(zip(class_order, probs))
            }
            neg = prob_dict.get(0, 0.0)
            neu = prob_dict.get(1, 0.0)
            pos = prob_dict.get(2, 0.0)
            conf = float(np.max(probs))
        else:
            neg, neu, pos, conf = 0.0, 0.0, 0.0, 0.0

        predictions.append({
            "message_id": i + 1,
            "speaker": msg["speaker"],
            "text": text,
            "sentiment": SENTIMENT_LABELS.get(pred_id, str(pred_id)),
            "confidence": conf,
            "negative_probability": neg,
            "neutral_probability": neu,
            "positive_probability": pos,
            "context_text": text
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return predictions, elapsed_ms


def predict_transformer_sentiment(
    messages: List[Dict[str, str]],
    context_strings: List[str],
    tokenizer: Any,
    model: Any,
    device: torch.device
) -> Tuple[List[Dict[str, Any]], float]:
    """Run batch or sequential transformer inference with device acceleration and latency tracking."""
    start_time = time.perf_counter()
    predictions = []

    for i, (msg, ctx_text) in enumerate(zip(messages, context_strings)):
        encoded = tokenizer(ctx_text, return_tensors="pt", truncation=True, padding=True, max_length=192)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            encoded.pop("token_type_ids", None)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        predictions.append({
            "message_id": i + 1,
            "speaker": msg["speaker"],
            "text": msg["text"],
            "sentiment": SENTIMENT_LABELS.get(pred_id, str(pred_id)),
            "confidence": float(probs[pred_id]),
            "negative_probability": float(probs[0]),
            "neutral_probability": float(probs[1]),
            "positive_probability": float(probs[2]),
            "context_text": ctx_text
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return predictions, elapsed_ms


def predict_emotions(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    model: Any,
    device: torch.device
) -> Tuple[List[Dict[str, Any]], float]:
    """Predict utterance-level 7-class emotion distribution using the fine-tuned emotion checkpoint."""
    start_time = time.perf_counter()
    results = []

    for msg in messages:
        encoded = tokenizer(msg["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            encoded.pop("token_type_ids", None)
            probs = torch.softmax(model(**encoded).logits, dim=1)[0].cpu().numpy()

        labels = model.config.id2label
        pred_id = int(np.argmax(probs))
        emotion = str(labels.get(pred_id, EMOTION_LABELS[pred_id])).lower()

        results.append({
            "emotion": emotion,
            "confidence": float(probs[pred_id]),
            "emotion_probabilities": {
                str(labels.get(j, EMOTION_LABELS[j])).lower(): float(p) for j, p in enumerate(probs)
            }
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return results, elapsed_ms
