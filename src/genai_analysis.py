"""
Generative AI Context Preparation and Insight Orchestration
Formats rich conversational payloads and prompts for LLM synthesis.
"""

from typing import List, Dict, Any


def compile_analysis_payload(
    messages: List[Dict[str, str]],
    sentiment_predictions: List[Dict[str, Any]],
    emotion_predictions: List[Dict[str, Any]] = None
) -> str:
    """Format full conversational transcripts and model predictions into an LLM analysis context."""
    lines = ["CONVERSATION TRANSCRIPT:"]
    for m in messages:
        lines.append(f"{m.get('speaker', 'Speaker')}: {m.get('text', '')}")

    lines.append("\nPREDICTED SENTIMENT BY UTTERANCE:")
    for p in sentiment_predictions:
        lines.append(
            f"Message {p['message_id']} | Speaker: {p['speaker']} | "
            f"Sentiment: {p['sentiment']} | Confidence: {p['confidence'] * 100:.1f}% | "
            f"Text: \"{p['text']}\""
        )

    if emotion_predictions:
        lines.append("\nPREDICTED EMOTIONS:")
        for i, e in enumerate(emotion_predictions):
            spk = messages[i].get("speaker", "Speaker") if i < len(messages) else "Speaker"
            lines.append(
                f"Message {i + 1} | Speaker: {spk} | "
                f"Emotion: {e['emotion'].title()} | Confidence: {e['confidence'] * 100:.1f}%"
            )

    return "\n".join(lines)


def build_insight_prompt(payload: str, prompt_type: str = "summary") -> str:
    """Generate structured analytical prompt for conversational intelligence."""
    templates = {
        "summary": (
            "Analyze and summarize the following multi-turn dialogue with emotional trajectory:\n\n"
            f"{payload}\n\n"
            "Provide:\n"
            "1. Concise executive summary of the dialogue\n"
            "2. Key friction or collaboration points between speakers\n"
            "3. Overall conversational sentiment trajectory\n"
            "4. Actionable takeaway for conversational outcome"
        ),
        "explanation": (
            "Explain the sentiment and emotion predictions from our NLP transformer models:\n\n"
            f"{payload}\n\n"
            "Provide:\n"
            "1. Interpretation of the detected emotional shifts\n"
            "2. Potential linguistic triggers (words, sarcasm, context) causing sentiment changes\n"
            "3. Evaluation of model confidence and potential ambiguities"
        ),
        "recommendations": (
            "Based on the conversational dynamics and emotional friction detected:\n\n"
            f"{payload}\n\n"
            "Provide 3 actionable communication recommendations to de-escalate tension or improve rapport."
        )
    }

    return templates.get(prompt_type, templates["summary"])
