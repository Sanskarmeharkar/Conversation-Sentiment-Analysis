"""
Context Analysis Module for Multi-Turn Conversation Sentiment
Handles conversational windowing, speaker-history formatting, and emotional trajectory tracking.
"""

from typing import List, Dict, Any
import pandas as pd


def build_context_text(messages: List[Dict[str, str]], index: int, previous_turns: int = 2, is_context_model: bool = True) -> str:
    """
    Construct multi-turn conversational context string.
    
    If is_context_model is True, uses the exact template used during fine-tuning of Context-DistilBERT:
    'Previous Speaker: {spk} | Previous Utterance: {utt} [SEP] Current Speaker: {spk} | Current Utterance: {utt}'
    """
    if not messages or index < 0 or index >= len(messages):
        return ""

    if previous_turns <= 0 and not is_context_model:
        return messages[index]["text"]

    start = max(0, index - int(previous_turns))

    if is_context_model:
        context_parts = []
        # Preceding turns
        for j in range(start, index):
            spk = messages[j].get("speaker", "Unknown")
            utt = messages[j].get("text", "")
            context_parts.append(f"Previous Speaker: {spk} | Previous Utterance: {utt}")

        # Current target utterance
        cur_spk = messages[index].get("speaker", "Unknown")
        cur_utt = messages[index].get("text", "")
        context_parts.append(f"Current Speaker: {cur_spk} | Current Utterance: {cur_utt}")

        return " [SEP] ".join(context_parts)
    else:
        # Standard speaker-utterance prefix
        selected = messages[start:index + 1]
        return " [SEP] ".join(f"{item.get('speaker', 'Speaker')}: {item.get('text', '')}" for item in selected)


def analyze_sentiment_transitions(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze shift dynamics from one utterance to the next (e.g., negative to positive escalation or de-escalation)."""
    if not predictions or len(predictions) < 2:
        return pd.DataFrame(columns=["From Sentiment", "To Sentiment", "Count"])

    transitions = []
    for i in range(1, len(predictions)):
        from_sent = predictions[i - 1]["sentiment"]
        to_sent = predictions[i]["sentiment"]
        transitions.append({"From Sentiment": from_sent, "To Sentiment": to_sent})

    df = pd.DataFrame(transitions)
    summary = df.groupby(["From Sentiment", "To Sentiment"]).size().reset_index(name="Count")
    return summary.sort_values("Count", ascending=False)


def compute_speaker_profiles(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute aggregate emotional profile, dominant tone, and average confidence per participant."""
    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame(predictions)
    speakers = df["speaker"].unique()
    profiles = []

    for spk in speakers:
        spk_df = df[df["speaker"] == spk]
        total = len(spk_df)
        counts = spk_df["sentiment"].value_counts().to_dict()

        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral", 0)

        dominant = spk_df["sentiment"].mode().iloc[0] if not spk_df.empty else "Neutral"
        avg_conf = spk_df["confidence"].mean() if "confidence" in spk_df else 0.0

        profiles.append({
            "Speaker": spk,
            "Total Messages": total,
            "Dominant Sentiment": dominant,
            "Positive %": round((pos / total) * 100, 1),
            "Neutral %": round((neu / total) * 100, 1),
            "Negative %": round((neg / total) * 100, 1),
            "Avg Confidence (%)": round(avg_conf * 100, 1)
        })

    return pd.DataFrame(profiles)


def calculate_emotional_trajectory(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute sentiment trajectory score (+1 = highly positive, -1 = highly negative)."""
    if not predictions:
        return pd.DataFrame()

    records = []
    for p in predictions:
        pos_prob = p.get("positive_probability", 0.0)
        neg_prob = p.get("negative_probability", 0.0)
        records.append({
            "Message ID": p.get("message_id", 0),
            "Speaker": p.get("speaker", "Speaker"),
            "Utterance": p.get("text", "")[:40] + ("..." if len(p.get("text", "")) > 40 else ""),
            "Sentiment Score": round(pos_prob - neg_prob, 3),
            "Sentiment": p.get("sentiment", "Neutral"),
            "Confidence": round(p.get("confidence", 0.0) * 100, 1)
        })

    return pd.DataFrame(records)
