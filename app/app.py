# ============================================================
# CONVERSATION SENTIMENT + EMOTION ANALYSIS DASHBOARD
# Fully Modular Architecture & Live Demonstration System
# ============================================================

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
# ---------------- PATH CONFIGURATION ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------- MODULAR SRC IMPORTS ----------------
from src.preprocessing import (
    clean_text,
    tokenize_text,
    filter_stopwords,
    lemmatize_tokens,
    preprocess_utterance,
    parse_conversation,
    STOPWORDS,
    NEGATION_WORDS,
)
from src.context_analysis import (
    build_context_text,
    analyze_sentiment_transitions,
    compute_speaker_profiles,
    calculate_emotional_trajectory,
)
from src.feature_extraction import extract_keywords_tfidf, extract_named_entities
from src.topic_modeling import extract_conversation_topics
from src.sentiment_models import (
    SENTIMENT_LABELS,
    EMOTION_LABELS,
    predict_classical_sentiment,
    predict_transformer_sentiment,
    predict_emotions as run_emotion_inference,
)
from src.genai_analysis import compile_analysis_payload

try:
    from src.llm_service import (
        generate_llm_response,
        generate_conversation_summary,
        generate_sentiment_explanation,
        generate_recommendations,
    )
    LLM_AVAILABLE = True
    LLM_IMPORT_ERROR = None
except Exception as llm_import_exc:
    LLM_AVAILABLE = False
    LLM_IMPORT_ERROR = str(llm_import_exc)

# ---------------- PAGE CONFIG & MODERN CSS ----------------
st.set_page_config(
    page_title="Conversational Sentiment & Emotion Analysis",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""<style>
/* Modern typography and base dark background */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}
.stApp {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1450px;
}

/* Hero Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    color: #ffffff;
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-top: 6px;
    margin-bottom: 12px;
}
.hero-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #f1f5f9;
}

/* Metric Cards with Accent Top Borders */
div[data-testid="stMetric"] {
    background: #161f30 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    border-top: 4px solid #3b82f6 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35) !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Sidebar Dark Theme & Crisp High Contrast */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1f2937 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stCaption p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
h2 {
    font-size: 1.35rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
}
.stCaption p {
    color: #94a3b8 !important;
}

/* Chat Turn Cards */
.chat-turn {
    background: #161f30 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color 0.15s ease;
}
.chat-turn:hover {
    border-color: #3b82f6 !important;
}
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 8px;
}
.chat-speaker {
    font-weight: 700;
    color: #60a5fa !important;
    font-size: 0.95rem;
}
.chat-badges {
    display: flex;
    gap: 6px;
    align-items: center;
}
.chat-text {
    font-size: 0.95rem;
    color: #e2e8f0 !important;
    line-height: 1.5;
}

/* Badges for Sentiment & Emotion (Vibrant on Dark) */
.badge-pos {
    background: rgba(34, 197, 94, 0.18) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(74, 222, 128, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-neg {
    background: rgba(239, 68, 68, 0.18) !important;
    color: #f87171 !important;
    border: 1px solid rgba(248, 113, 113, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-neu {
    background: rgba(148, 163, 184, 0.18) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(203, 213, 225, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Emotion Badges */
.badge-joy { background: rgba(59, 130, 246, 0.18) !important; color: #60a5fa !important; border: 1px solid rgba(96, 165, 250, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-anger { background: rgba(239, 68, 68, 0.18) !important; color: #f87171 !important; border: 1px solid rgba(248, 113, 113, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-surprise { background: rgba(245, 158, 11, 0.18) !important; color: #fbbf24 !important; border: 1px solid rgba(251, 191, 36, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-sadness { background: rgba(99, 102, 241, 0.18) !important; color: #818cf8 !important; border: 1px solid rgba(129, 140, 248, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-fear { background: rgba(168, 85, 247, 0.18) !important; color: #c084fc !important; border: 1px solid rgba(192, 132, 252, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-disgust { background: rgba(20, 184, 166, 0.18) !important; color: #2dd4bf !important; border: 1px solid rgba(45, 212, 191, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }
.badge-neutral { background: rgba(148, 163, 184, 0.18) !important; color: #cbd5e1 !important; border: 1px solid rgba(203, 213, 225, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; }

/* Buttons & Inputs */
.stButton button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease;
}
.stButton button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
}
</style>""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🎭 Conversational Sentiment & Emotion Intelligence</span>
    </div>
    <div class="hero-subtitle">
        Deep multi-turn conversation dynamics, 7-class emotion recognition, and contextual transformer inference benchmarked on MELD.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">⚡ Context-Aware DistilBERT</span>
        <span class="hero-pill">🎭 7-Class Emotion Model</span>
        <span class="hero-pill">📊 MELD Benchmark</span>
        <span class="hero-pill">🔍 TF-IDF & spaCy NER</span>
        <span class="hero-pill">🤖 Groq Llama-3.3 GenAI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL PATHS & REGISTRY ----------------

HF_MODEL_REPO = "Sanskarmeharkar/conversation-sentiment-models"

CLASSICAL_MODEL_FILE = "classical/classical_models.joblib"

MODEL_OPTIONS = {
    "Context-Aware DistilBERT (Recommended)": "context_distilbert",
    "DistilBERT (Utterance Only)": "distilbert",
    "RoBERTa (Utterance Only)": "roberta",
    "Logistic Regression (TF-IDF Baseline)": "logistic_regression",
    "Linear SVM (TF-IDF Baseline)": "linear_svm",
    "Naive Bayes (TF-IDF Baseline)": "naive_bayes",
}

TRANSFORMER_PATHS = {
    "distilbert": "distilbert_meld_sentiment",
    "context_distilbert": "distilbert_meld_context",
    "roberta": "roberta_meld_sentiment",
}

EMOTION_MODEL_PATH = "distilbert_meld_emotion"

EMOTION_EMOJIS = {
    "anger": "😡 Anger",
    "disgust": "🤢 Disgust",
    "fear": "😨 Fear",
    "joy": "😊 Joy",
    "neutral": "😐 Neutral",
    "sadness": "😢 Sadness",
    "surprise": "😲 Surprise"
}

## ---------------- CACHED MODEL LOADERS ----------------

# ---------------- CACHED MODEL LOADERS ----------------

@st.cache_resource
def load_classical_models():
    model_file = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=CLASSICAL_MODEL_FILE
    )
    return joblib.load(model_file)


@st.cache_resource
def load_transformer_model(model_key, use_gpu=False):
    model_folder = TRANSFORMER_PATHS.get(model_key)

    if model_folder is None:
        raise ValueError(f"Unknown transformer model: {model_key}")

    from huggingface_hub import snapshot_download

    downloaded_repo = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=f"{model_folder}/*"
    )

    local_model_path = Path(downloaded_repo) / model_folder

    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_path)
    )

    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


@st.cache_resource
def load_emotion_model():
    from huggingface_hub import snapshot_download

    downloaded_repo = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=f"{EMOTION_MODEL_PATH}/*"
    )

    local_model_path = Path(downloaded_repo) / EMOTION_MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_path)
    )

    # Emotion model runs on CPU
    device = torch.device("cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device

# ---------------- PRESET CONVERSATIONS FOR DEMO ----------------
DEMO_PRESETS = {
    "Preset 1: Exam Anxiety & Relief (Short Demo)": (
        "Person A: Hey, how was your exam today?\n"
        "Person B: It was really difficult. I think I did badly.\n"
        "Person A: Don't worry, you prepared well and tried your best.\n"
        "Person B: Thanks, that makes me feel a little better."
    ),
    "Preset 2: Project Deadline Escalation (Conflict Resolution)": (
        "Alex: Did you see the revised deadline for the NLP project? It was moved up to tomorrow morning!\n"
        "Jordan: What? Are you serious? There's no way we can finish the evaluation benchmarks by then!\n"
        "Alex: I know, I panicked at first too. But look, most of our fine-tuning is already complete.\n"
        "Jordan: True, but we still haven't generated the confusion matrix plots or tested the context windowing.\n"
        "Alex: Don't stress. If you run the validation script, I can take care of the dashboard integration and reports.\n"
        "Jordan: Okay, that actually sounds manageable. Thanks for stepping up, Alex.\n"
        "Alex: Anytime! Let's get this done and make the demo look great.\n"
        "Jordan: Agreed! I'm feeling much more confident now. Let's do this!"
    ),
    "Preset 3: Customer Support Billing Complaint (Customer Resolution)": (
        "Customer: Hello, I have been trying to access my subscription for two days and it keeps saying unauthorized!\n"
        "Support Agent: I am so sorry to hear that you are experiencing this frustration. Let me look up your account right now.\n"
        "Customer: Thank you, I really need this resolved because our team has an urgent demonstration scheduled.\n"
        "Support Agent: I completely understand the urgency. I found the issue—a billing token had expired. I have refreshed your active license.\n"
        "Customer: Wait, let me try logging in again... Yes! It's working now. That was fast!\n"
        "Support Agent: Wonderful! I am very glad we could get that sorted out for you. Is there anything else I can assist with?\n"
        "Customer: No, that's everything! You have been extremely helpful. Have a fantastic day!"
    ),
    "Preset 4: Friends Dialogue (Chandler & Phoebe Sarcasm)": (
        "Chandler: Hey Pheebs, how's the guitar playing going?\n"
        "Phoebe: Oh it's terrible! The chords are hurting my fingers and the cat ran away when I started singing!\n"
        "Chandler: Well to be fair, that song was about a smelly cat.\n"
        "Phoebe: Hey! That song is an emotional masterpiece!\n"
        "Chandler: I know, I was just teasing. You're actually really creative.\n"
        "Phoebe: Aww thanks Chandler, that is so sweet of you to say!\n"
        "Chandler: Don't tell Joey I was being nice, I have a sarcastic reputation to protect."
    ),
    "Custom Text Input": "",
    "Upload Conversation File (.txt, .json, .csv)": ""
}

# ---------------- SIDEBAR CONFIGURATION ----------------
st.sidebar.title("🛠️ Analysis Configuration")
selected_model_name = st.sidebar.selectbox("Select Model Architecture", list(MODEL_OPTIONS.keys()), key="selected_model_name")
model_key = MODEL_OPTIONS[selected_model_name]

use_gpu = st.sidebar.checkbox(
    "Enable CUDA Acceleration",
    value=False,
    help="Utilizes GPU for transformer inference if CUDA is available. Classical models and emotion model run on CPU."
)

if model_key == "context_distilbert":
    context_window = 2
    st.sidebar.slider("Context Window (Previous Turns)", min_value=0, max_value=4, value=2, disabled=True)
    st.sidebar.caption("🔒 Fixed at 2 turns: Context-DistilBERT was fine-tuned on 2 preceding turns via `[SEP]` alignment.")
else:
    context_window = st.sidebar.slider("Context Window (Previous Turns)", min_value=0, max_value=4, value=0, step=1)

st.sidebar.divider()
st.sidebar.markdown("### 📚 Architecture Stack")
st.sidebar.caption(
    "• **Pretrained Transformers:** DistilBERT & RoBERTa\n"
    "• **Context Windowing:** Sliding turns with speaker history\n"
    "• **Emotion Recognition:** 7-Class MELD model\n"
    "• **NLP Inspection:** Tokenization, stopword filter, lemmatization\n"
    "• **Entity & Topic Modeling:** spaCy NER & NMF Topic extraction\n"
    "• **Generative AI:** Groq Llama-3.3 conversation intelligence"
)

# ---------------- INPUT SECTION ----------------
st.header("1. Input Conversation")
preset_choice = st.selectbox("Choose a Demonstration Dialogue or Custom Input", list(DEMO_PRESETS.keys()), index=1)

conversation_text = ""
if preset_choice in [
    "Preset 1: Exam Anxiety & Relief (Short Demo)",
    "Preset 2: Project Deadline Escalation (Conflict Resolution)",
    "Preset 3: Customer Support Billing Complaint (Customer Resolution)",
    "Preset 4: Friends Dialogue (Chandler & Phoebe Sarcasm)"
]:
    conversation_text = st.text_area(
        "Dialogue Script (Speaker: Message)",
        value=DEMO_PRESETS[preset_choice],
        height=180
    )
elif preset_choice == "Custom Text Input":
    conversation_text = st.text_area(
        "Enter custom conversation turns",
        height=180,
        placeholder="Speaker A: Hello there!\nSpeaker B: Hi, how are you feeling today?"
    )
else:
    uploaded_file = st.file_uploader("Upload conversation file", type=["txt", "json", "csv"])
    if uploaded_file is not None:
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".txt"):
                conversation_text = uploaded_file.getvalue().decode("utf-8-sig")
            elif name.endswith(".json"):
                data = json.loads(uploaded_file.getvalue().decode("utf-8-sig"))
                if isinstance(data, list):
                    conversation_text = "\n".join(f"{x.get('speaker', 'Speaker')}: {x.get('text', '')}" for x in data)
                else:
                    st.error("JSON must contain a list of objects with 'speaker' and 'text' keys.")
            elif name.endswith(".csv"):
                upload_df = pd.read_csv(uploaded_file)
                st.caption("CSV Preview")
                st.dataframe(upload_df.head(), use_container_width=True)
                if {"speaker", "text"}.issubset(upload_df.columns):
                    conversation_text = "\n".join(upload_df["speaker"].astype(str) + ": " + upload_df["text"].astype(str))
                else:
                    st.error("CSV must contain columns named 'speaker' and 'text'.")
        except Exception as exc:
            st.error(f"Error reading uploaded file: {exc}")

run_analysis = st.button("🚀 Analyze Conversation & Predict Emotions", type="primary")

if run_analysis:
    if not conversation_text.strip():
        st.warning("Please provide or select a conversation dialogue before running analysis.")
    else:
        parsed = parse_conversation(conversation_text)
        if not parsed:
            st.warning("No valid conversation turns found. Please use the format 'Speaker: Message'.")
        else:
            st.session_state["messages"] = parsed
            st.session_state["analyzed"] = True
            st.session_state.pop("sentiment_cache_key", None)
            st.session_state.pop("emotion_cache_key", None)
            for key in ("llm_summary", "llm_explanation", "llm_recommendations", "llm_chat_answer"):
                st.session_state.pop(key, None)

# ---------------- EXECUTE ANALYSIS ----------------
if st.session_state.get("analyzed", False):
    messages = st.session_state.get("messages", [])
    cache_key = (tuple((m["speaker"], m["text"]) for m in messages), model_key, context_window, use_gpu)
    emotion_key = tuple((m["speaker"], m["text"]) for m in messages)

    try:
        if st.session_state.get("sentiment_cache_key") != cache_key:
            with st.spinner(f"Running sentiment inference with {selected_model_name}..."):
                if model_key in {"naive_bayes", "logistic_regression", "linear_svm"}:
                    bundle = load_classical_models()
                    preds, lat_ms = predict_classical_sentiment(messages, model_key, bundle)
                else:
                    tok, mod, dev = load_transformer_model(model_key, use_gpu)
                    is_ctx = (model_key == "context_distilbert")
                    ctx_strings = [
                        build_context_text(messages, idx, previous_turns=context_window, is_context_model=is_ctx)
                        for idx in range(len(messages))
                    ]
                    preds, lat_ms = predict_transformer_sentiment(messages, ctx_strings, tok, mod, dev)
                st.session_state["sentiment_results"] = preds
                st.session_state["sentiment_latency"] = lat_ms
                st.session_state["sentiment_cache_key"] = cache_key

        predictions = st.session_state.get("sentiment_results", [])
        sent_latency = st.session_state.get("sentiment_latency", 0.0)

        if st.session_state.get("emotion_cache_key") != emotion_key:
            with st.spinner("Classifying 7-Class Emotions (MELD Emotion Model)..."):
                e_tok, e_mod, e_dev = load_emotion_model()
                emo_results, emo_lat_ms = run_emotion_inference(messages, e_tok, e_mod, e_dev)
                st.session_state["emotion_predictions"] = emo_results
                st.session_state["emotion_latency"] = emo_lat_ms
                st.session_state["emotion_cache_key"] = emotion_key

        emotion_predictions = st.session_state.get("emotion_predictions", [])
        emo_latency = st.session_state.get("emotion_latency", 0.0)

    except Exception as exc:
        st.error(f"Inference pipeline error: {exc}")
        st.exception(exc)
        predictions = st.session_state.get("sentiment_results", [])
        emotion_predictions = st.session_state.get("emotion_predictions", [])

    # ---------------- OVERVIEW METRICS ----------------
    st.divider()
    st.header("2. Real-Time Conversation Performance")
    
    # Calculate dominant emotion
    all_emotions = [e["emotion"].lower() for e in emotion_predictions] if emotion_predictions else []
    dom_emotion = Counter(all_emotions).most_common(1)[0][0].title() if all_emotions else "N/A"
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Turns", len(messages))
    m2.metric("Participants", len({m["speaker"] for m in messages}))
    m3.metric("Dominant Emotion", f"{EMOTION_EMOJIS.get(dom_emotion.lower(), dom_emotion)}")
    m4.metric("Sentiment Latency", f"{sent_latency:.1f} ms", f"{sent_latency/max(1, len(messages)):.1f} ms/turn")
    m5.metric("Emotion Latency", f"{emo_latency:.1f} ms", f"{emo_latency/max(1, len(messages)):.1f} ms/turn")

    # ---------------- CONVERSATION TRANSCRIPT & DETECTED EMOTIONS ----------------
    st.subheader("Turn-by-Turn Sentiment & Emotion Recognition")
    
    # Custom Styled Chat Bubble Layout
    for i, p in enumerate(predictions):
        e_info = emotion_predictions[i] if i < len(emotion_predictions) else {"emotion": "neutral", "confidence": 0.0}
        emo_name = e_info["emotion"].lower()
        emo_label = EMOTION_EMOJIS.get(emo_name, emo_name.title())
        
        sent = p["sentiment"].lower()
        if "pos" in sent:
            sent_badge_class = "badge-pos"
            sent_text = "🟢 Positive"
        elif "neg" in sent:
            sent_badge_class = "badge-neg"
            sent_text = "🔴 Negative"
        else:
            sent_badge_class = "badge-neu"
            sent_text = "⚪ Neutral"
            
        emo_badge_class = f"badge-{emo_name}" if emo_name in EMOTION_EMOJIS else "badge-neutral"

        st.markdown(f"""
        <div class="chat-turn">
            <div class="chat-header">
                <span class="chat-speaker">Turn {i+1} • {p['speaker']}</span>
                <div class="chat-badges">
                    <span class="{sent_badge_class}">{sent_text} ({p['confidence']*100:.1f}%)</span>
                    <span class="{emo_badge_class}">{emo_label} ({e_info['confidence']*100:.1f}%)</span>
                </div>
            </div>
            <div class="chat-text">"{p['text']}"</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 View Detailed Data Table"):
        table_rows = []
        for i, p in enumerate(predictions):
            e_info = emotion_predictions[i] if i < len(emotion_predictions) else {"emotion": "N/A", "confidence": 0.0}
            table_rows.append({
                "Turn": i + 1,
                "Speaker": p["speaker"],
                "Utterance": p["text"],
                "Predicted Sentiment": p["sentiment"],
                "Sentiment Conf": f"{p['confidence'] * 100:.1f}%",
                "Predicted Emotion": e_info["emotion"].title(),
                "Emotion Conf": f"{e_info['confidence'] * 100:.1f}%"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ---------------- SENTIMENT & EMOTION VISUAL ANALYTICS ----------------
    st.divider()
    st.header("3. Sentiment & Emotion Visual Analytics")

    tab_sent, tab_emo = st.tabs(["📈 Sentiment Dynamics & Trajectory", "🎭 7-Class Emotion Analytics"])

    with tab_sent:
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            st.subheader("Speaker Profiles")
            profiles_df = compute_speaker_profiles(predictions)
            st.dataframe(profiles_df, use_container_width=True, hide_index=True)

            pred_df = pd.DataFrame(predictions)
            counts = pred_df.groupby("speaker")["sentiment"].value_counts().unstack(fill_value=0)
            for lbl in ["Negative", "Neutral", "Positive"]:
                if lbl not in counts:
                    counts[lbl] = 0
            st.bar_chart(counts[["Negative", "Neutral", "Positive"]])

        with col_s2:
            st.subheader("Sentiment Trajectory")
            trajectory_df = calculate_emotional_trajectory(predictions)
            if not trajectory_df.empty:
                trajectory_df["Turn_Label"] = "Turn " + trajectory_df["Message ID"].astype(str) + " (" + trajectory_df["Speaker"] + ")"
                st.line_chart(trajectory_df.set_index("Turn_Label")["Sentiment Score"], height=260)
                st.caption("📈 Score: +1.0 = Highly Positive | 0.0 = Balanced/Neutral | -1.0 = Highly Negative")

            transitions_df = analyze_sentiment_transitions(predictions)
            if not transitions_df.empty:
                st.subheader("Sentiment Transitions")
                st.dataframe(transitions_df, use_container_width=True, hide_index=True)

    with tab_emo:
        col_e1, col_e2 = st.columns([1, 1])

        with col_e1:
            st.subheader("Overall Emotion Distribution (7 Classes)")
            if emotion_predictions:
                emo_series = pd.Series([e["emotion"].title() for e in emotion_predictions]).value_counts()
                st.bar_chart(emo_series)
                st.caption("Frequency of detected emotions across all conversation turns.")

        with col_e2:
            st.subheader("Speaker-wise Emotion Breakdown")
            if emotion_predictions:
                emo_speaker_df = pd.DataFrame([
                    {"Speaker": m["speaker"], "Emotion": e["emotion"].title()}
                    for m, e in zip(messages, emotion_predictions)
                ])
                speaker_emo_counts = emo_speaker_df.groupby("Speaker")["Emotion"].value_counts().unstack(fill_value=0)
                st.bar_chart(speaker_emo_counts)
                
                # Dominant Emotion per Speaker
                st.subheader("Speaker Dominant Emotion")
                speaker_dom = emo_speaker_df.groupby("Speaker")["Emotion"].agg(lambda x: x.mode().iloc[0]).reset_index()
                speaker_dom.columns = ["Speaker", "Dominant Emotion"]
                st.dataframe(speaker_dom, use_container_width=True, hide_index=True)

    # ---------------- INFORMATION EXTRACTION & TOPIC MODELING ----------------
    st.divider()
    st.header("4. Information Extraction & Topic Modeling")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.subheader("Salient Keywords (TF-IDF N-Grams)")
        utterance_texts = [p["text"] for p in predictions]
        kw_df = extract_keywords_tfidf(utterance_texts, max_features=10)
        if not kw_df.empty:
            st.dataframe(kw_df, use_container_width=True, hide_index=True)
            st.bar_chart(kw_df.set_index("Keyword/Phrase")["Score"])
        else:
            st.info("Insufficient vocabulary diversity for TF-IDF keyword extraction.")

    with info_col2:
        st.subheader("Named Entities (spaCy NER)")
        ner_df = extract_named_entities(messages)
        if not ner_df.empty:
            disp_cols = [c for c in ["Turn", "Speaker", "Entity", "Type", "Explanation"] if c in ner_df.columns]
            st.dataframe(ner_df[disp_cols], use_container_width=True, hide_index=True)
            st.caption(f"Found {len(ner_df)} named entities across conversation.")
        else:
            st.info("No named entities detected in this conversation dialogue.")

    # Topic Modeling
    st.subheader("Semantic Topic Clusters (NMF)")
    topic_results = extract_conversation_topics(utterance_texts, n_topics=3)
    if topic_results.get("topics"):
        for t in topic_results["topics"]:
            with st.expander(f"🔹 {t['label']}"):
                st.write("**Salient Terms:**", ", ".join(t["keywords"]))
                assigned_turns = [
                    f"**{predictions[idx]['speaker']}**: {predictions[idx]['text']}"
                    for idx, topic_id in enumerate(topic_results["assigned"])
                    if topic_id == t["topic_id"]
                ]
                st.write("**Assigned Turns:**")
                for at in assigned_turns[:4]:
                    st.markdown(f"- {at}")
    else:
        st.info("Dialogue turns are too concise for multi-topic decomposition.")

    # ---------------- GENERATIVE AI CONVERSATION INTELLIGENCE ----------------
    st.divider()
    st.header("5. Generative AI Conversation Intelligence")
    analysis_payload = compile_analysis_payload(messages, predictions, emotion_predictions)

    if not LLM_AVAILABLE:
        st.warning(f"LLM Service offline: {LLM_IMPORT_ERROR}")
    else:
        st.caption("Generates executive summaries, emotional root-cause explanations, and communication recommendations via Groq Llama-3.3.")
        g1, g2, g3 = st.columns(3)

        if g1.button("📄 Generate Summary", key="btn_summary"):
            with st.spinner("Synthesizing conversation summary..."):
                st.session_state["llm_summary"] = generate_conversation_summary(analysis_payload)
        if g2.button("🔍 Explain Dynamics", key="btn_explain"):
            with st.spinner("Analyzing emotional inflection points..."):
                st.session_state["llm_explanation"] = generate_sentiment_explanation(analysis_payload)
        if g3.button("💡 Recommendations", key="btn_recs"):
            with st.spinner("Formulating actionable guidance..."):
                st.session_state["llm_recommendations"] = generate_recommendations(analysis_payload)

        if st.session_state.get("llm_summary"):
            st.subheader("AI Conversation Summary")
            st.markdown(st.session_state["llm_summary"])
        if st.session_state.get("llm_explanation"):
            st.subheader("AI Sentiment & Emotion Interpretation")
            st.markdown(st.session_state["llm_explanation"])
        if st.session_state.get("llm_recommendations"):
            st.subheader("Actionable Recommendations")
            st.markdown(st.session_state["llm_recommendations"])

        # Interactive Q&A
        st.subheader("Ask AI about this Conversation")
        user_query = st.text_input("Enter a question about the conversation", placeholder="Why did the speaker's tone improve near the end?")
        if st.button("Submit Question", key="btn_qa") and user_query.strip():
            with st.spinner("Generating data-grounded response..."):
                prompt = (
                    f"Use this conversation and model predictions to answer:\n\n{analysis_payload}\n\n"
                    f"Question: {user_query.strip()}"
                )
                st.session_state["llm_chat_answer"] = generate_llm_response(prompt)
        if st.session_state.get("llm_chat_answer"):
            st.markdown("**AI Response:**")
            st.markdown(st.session_state["llm_chat_answer"])

    # ---------------- BENCHMARK & MODEL PERFORMANCE ----------------
    st.divider()
    st.header("6. Model Benchmark Comparison (MELD Dataset)")

    benchmark_csv = PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv"
    if benchmark_csv.exists():
        model_results = pd.read_csv(benchmark_csv)
    else:
        model_results = pd.DataFrame([
            {"Model": "Naive Bayes (TF-IDF)", "Type": "Classical", "Accuracy (%)": 51.13, "Macro F1 (%)": 41.00, "Weighted F1 (%)": 45.00},
            {"Model": "Logistic Regression (TF-IDF)", "Type": "Classical", "Accuracy (%)": 51.22, "Macro F1 (%)": 49.24, "Weighted F1 (%)": 51.00},
            {"Model": "Linear SVM (TF-IDF)", "Type": "Classical", "Accuracy (%)": 48.69, "Macro F1 (%)": 46.00, "Weighted F1 (%)": 48.00},
            {"Model": "DistilBERT", "Type": "Transformer", "Accuracy (%)": 67.99, "Macro F1 (%)": 65.45, "Weighted F1 (%)": 67.55},
            {"Model": "Context DistilBERT", "Type": "Context-Aware Transformer", "Accuracy (%)": 68.71, "Macro F1 (%)": 66.94, "Weighted F1 (%)": 68.32},
            {"Model": "RoBERTa", "Type": "Transformer", "Accuracy (%)": 68.89, "Macro F1 (%)": 66.91, "Weighted F1 (%)": 68.70},
        ])

    b1, b2 = st.columns(2)
    best_acc = model_results.loc[model_results["Accuracy (%)"].idxmax()]
    best_f1 = model_results.loc[model_results["Macro F1 (%)"].idxmax()]
    b1.metric("Highest Test Accuracy", f"{best_acc['Accuracy (%)']:.2f}%", best_acc["Model"])
    b2.metric("Highest Macro F1", f"{best_f1['Macro F1 (%)']:.2f}%", best_f1["Model"])

    st.dataframe(model_results, use_container_width=True, hide_index=True)

    c_plot1, c_plot2 = st.columns(2)
    with c_plot1:
        st.subheader("Accuracy Comparison")
        st.bar_chart(model_results.set_index("Model")["Accuracy (%)"])
    with c_plot2:
        st.subheader("Macro F1 vs Weighted F1")
        st.bar_chart(model_results.set_index("Model")[["Macro F1 (%)", "Weighted F1 (%)"]])

    st.caption("Evaluated on 2,610 MELD test utterances. Context DistilBERT integrates sliding dialogue history via [SEP] tokenization.")

# ---------------- NLP PROCESSING PIPELINE SIDEBAR INSPECTOR ----------------
if "messages" in locals() and messages:
    with st.sidebar:
        st.divider()
        st.subheader("🔬 NLP Pipeline Inspector")
        inspect_idx = st.selectbox(
            "Select Turn to Inspect",
            range(len(messages)),
            format_func=lambda i: f"Turn {i+1} | {messages[i]['speaker']}: {messages[i]['text'][:40]}..."
        )

        selected_text = messages[inspect_idx]["text"]
        prep_info = preprocess_utterance(selected_text)

        with st.expander("1. Raw Utterance"):
            st.code(selected_text)

        with st.expander("2. Cleaned Text"):
            st.code(prep_info["clean_text"])

        with st.expander("3. Tokenization"):
            st.write(prep_info["tokens"])
            st.caption(f"Total tokens: {prep_info['token_count']}")

        with st.expander("4. Stopword Filter (Negation-Preserving)"):
            c_rem, c_ret = st.columns(2)
            c_rem.write("**Removed:**")
            c_rem.write(prep_info["removed_stopwords"])
            c_ret.write("**Retained:**")
            c_ret.write(prep_info["filtered_tokens"])

        with st.expander("5. Lemmatization"):
            st.dataframe(
                pd.DataFrame({"Token": prep_info["filtered_tokens"], "Lemma": prep_info["lemmas"]}),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("6. Transformer Context String"):
            if predictions:
                st.code(predictions[inspect_idx]["context_text"])
            else:
                st.info("No context available.")

        with st.expander("7. Sentiment Probability"):
            if predictions:
                res = predictions[inspect_idx]
                prob_data = pd.DataFrame({
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Probability": [res["negative_probability"], res["neutral_probability"], res["positive_probability"]]
                })
                st.bar_chart(prob_data.set_index("Sentiment"))

        with st.expander("8. 7-Class Emotion Probabilities"):
            if emotion_predictions and inspect_idx < len(emotion_predictions):
                e_res = emotion_predictions[inspect_idx]
                emo_probs = e_res["emotion_probabilities"]
                st.write(f"**Predicted:** {e_res['emotion'].title()} ({e_res['confidence']*100:.1f}%)")
                emo_prob_df = pd.DataFrame(
                    list(emo_probs.items()),
                    columns=["Emotion", "Probability"]
                )
                emo_prob_df["Emotion"] = emo_prob_df["Emotion"].str.title()
                st.bar_chart(emo_prob_df.set_index("Emotion"))
