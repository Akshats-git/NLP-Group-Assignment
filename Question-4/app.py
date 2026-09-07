"""
app.py - Streamlit Live NLP Editor (Question 4)

Integrates:
1. Joint Segmentation + POS tagging (Question 1)
2. Spelling Correction via SymDel (Question 3)
3. Language Model Grammar Checking + PCFG Constituency Parser (Question 4)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Path resolution
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
Q1_DIR = REPO_ROOT / "Question-1"
Q3_DIR = REPO_ROOT / "Question-3" / "q3_spelling_corrector"

for p in (str(BASE_DIR), str(Q1_DIR), str(Q3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(
    page_title="Q4 - Live Text Editor with NLP Pipeline",
    page_icon="📝",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .editor-container {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 16px;
        color: #cdd6f4;
        min-height: 140px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .alert-card {
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 14px;
        font-family: monospace;
    }
    .seg-alert {
        background-color: #312e81;
        border-left: 4px solid #6366f1;
        color: #e0e7ff;
    }
    .spell-alert {
        background-color: #451a03;
        border-left: 4px solid #f97316;
        color: #ffedd5;
    }
    .grammar-alert {
        background-color: #3f0713;
        border-left: 4px solid #f43f5e;
        color: #ffe4e6;
    }
    .tree-container {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 13px;
        color: #34d399;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session():
    defaults = {
        "models": None,
        "pcfg": None,
        "editor_text": "",
        "alerts": [],
        "word_buffer": [],
        "word_count": 0,
        "sim_running": False,
        "parse_history": [],
        "latency_seg": [],
        "latency_grammar": [],
        "passage_tokens": [],
        "passage_idx": 0,
        "live_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


@st.cache_resource(show_spinner="Loading Q1 and Q3 language models...")
def get_models():
    from q4.model_loader import load_all_models
    return load_all_models()


@st.cache_resource(show_spinner="Building PCFG grammar from Penn Treebank...")
def get_pcfg():
    from q4.pcfg import train_pcfg
    return train_pcfg(cache=True, min_count=2)


from q4.grammar import TRIGGER_N, check_grammar_window
from q4.passage import MERGE_PROB, passage_token_stream, sample_passage
from q4.pcfg import parse_sentence
from q4.segmentation import check_token_segmentation
from q4.spelling import check_token_spelling


def clean_word(token: str) -> str:
    return "".join(c for c in token.lower() if c.isalpha())


def process_token(token: str, models, pcfg) -> list[dict]:
    alerts = []
    word = clean_word(token)
    if not word:
        return alerts

    t0 = time.perf_counter()

    # 1. Segmentation Check
    seg_res = check_token_segmentation(
        word,
        lm=models["q1_lm"],
        config=models["q1_config"],
        tagger=models["q1_tagger"],
        vocab=models["q1_vocab"],
    )
    if seg_res.fired:
        alerts.append({
            "type": "SEGMENT",
            "token": token,
            "detail": f"Split '{token}' -> {' + '.join(seg_res.words)} [{', '.join(seg_res.pos_tags)}]",
            "latency": seg_res.latency_ms,
        })
        word = seg_res.words[0] if seg_res.words else word

    # 2. Spelling Check
    spell_res = check_token_spelling(word, models["corrector"], models["q3_vocab"])
    if spell_res.fired:
        alerts.append({
            "type": "SPELL",
            "token": token,
            "detail": f"Spelling: '{token}' -> suggest '{spell_res.suggestion}'",
            "latency": spell_res.latency_ms,
        })

    st.session_state.latency_seg.append((time.perf_counter() - t0) * 1000)
    return alerts


def process_grammar(models) -> list[dict]:
    alerts = []
    window = list(st.session_state.word_buffer)[-TRIGGER_N:]
    if len(window) < 2:
        return alerts

    res = check_grammar_window(
        window,
        lm=models["q1_lm"],
        corrector=models["corrector"],
        q3_bigram=models["q3_bigram"],
        q3_unigram=models["q3_unigram"],
        q3_vocab_size=models["q3_vocab_size"],
        q3_k=models["q3_k"],
        q3_vocab=models["q3_vocab"],
    )
    st.session_state.latency_grammar.append(res.latency_ms)

    if res.fired:
        msg = f"Grammar check (PPL: {res.ppl:.1f})"
        if res.real_word_fixes:
            fixes = ", ".join(f"{f.original}->{f.suggestion}" for f in res.real_word_fixes)
            msg += f" | Real-word: {fixes}"
        alerts.append({
            "type": "GRAMMAR",
            "detail": msg,
            "latency": res.latency_ms,
        })
    return alerts


def process_parse(sentence_tokens: list[str], models, pcfg):
    words = [clean_word(t) for t in sentence_tokens if clean_word(t)]
    if not words or not pcfg:
        return
    tags = models["q1_tagger"].tag(tuple(words))
    res = parse_sentence(tuple(words), tags, pcfg)
    st.session_state.parse_history.append(res)


# Header
st.title("NLP Group Assignment - Question 4 Editor")
st.subheader("Joint Segmentation, Spelling Correction, Grammar Check & PCFG Parser")

# Sidebar
with st.sidebar:
    st.header("Settings & Simulation")
    mode = st.radio("Input Mode", ["Simulated Live Typing", "Manual Typing"])

    words_per_sec = st.slider("Typing Speed (words/sec)", 0.5, 5.0, 2.0, 0.5)
    merge_p = st.slider("Merge Error Probability (p)", 0.0, 0.25, MERGE_PROB, 0.01)

    st.markdown("---")
    st.header("Latency Metrics")

    l_seg = st.session_state.latency_seg
    l_gram = st.session_state.latency_grammar
    avg_seg = (sum(l_seg) / len(l_seg)) if l_seg else 0.0
    avg_gram = (sum(l_gram) / len(l_gram)) if l_gram else 0.0

    st.metric("Seg + Spell Check (avg)", f"{avg_seg:.2f} ms")
    st.metric("Grammar Check (avg)", f"{avg_gram:.2f} ms")

    if st.button("Reset Session"):
        for k in ["editor_text", "alerts", "word_buffer", "word_count", "parse_history", "latency_seg", "latency_grammar", "sim_running", "passage_tokens", "passage_idx", "live_text"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else (False if isinstance(st.session_state[k], bool) else (0 if isinstance(st.session_state[k], int) else ""))
        st.rerun()

models = get_models()
pcfg = get_pcfg()

col1, col2 = st.columns([3, 2])

with col1:
    if mode == "Simulated Live Typing":
        c1, c2 = st.columns(2)
        if c1.button("Start Passage Stream"):
            passage, cname = sample_passage()
            st.session_state.passage_tokens = list(passage_token_stream(passage, p=merge_p))
            st.session_state.passage_idx = 0
            st.session_state.sim_running = True
            st.session_state.editor_text = ""
            st.session_state.alerts = []
            st.session_state.word_buffer = []
            st.session_state.parse_history = []
            st.session_state.word_count = 0

        if c2.button("Stop Stream"):
            st.session_state.sim_running = False

        box = st.empty()
        status = st.empty()

        if st.session_state.sim_running:
            tokens = st.session_state.passage_tokens
            idx = st.session_state.passage_idx
            delay = 1.0 / words_per_sec

            sentence_accum = []
            while idx < len(tokens) and st.session_state.sim_running:
                token, is_sent_end = tokens[idx]
                idx += 1
                st.session_state.passage_idx = idx

                st.session_state.editor_text += (" " if st.session_state.editor_text else "") + token
                box.markdown(f'<div class="editor-container">{st.session_state.editor_text}▌</div>', unsafe_allow_html=True)

                new_alerts = process_token(token, models, pcfg)
                st.session_state.alerts.extend(new_alerts)

                w = clean_word(token)
                if w:
                    st.session_state.word_buffer.append(w)
                    sentence_accum.append(token)
                    st.session_state.word_count += 1

                if st.session_state.word_count > 0 and st.session_state.word_count % TRIGGER_N == 0:
                    st.session_state.alerts.extend(process_grammar(models))

                if is_sent_end and sentence_accum:
                    process_parse(sentence_accum, models, pcfg)
                    sentence_accum = []

                status.text(f"Streaming token {idx}/{len(tokens)}...")
                time.sleep(delay)

            box.markdown(f'<div class="editor-container">{st.session_state.editor_text}</div>', unsafe_allow_html=True)
            if idx >= len(tokens):
                st.session_state.sim_running = False
                status.text("Stream finished.")
        else:
            text = st.session_state.editor_text or "Click 'Start Passage Stream' to simulate typing."
            box.markdown(f'<div class="editor-container">{text}</div>', unsafe_allow_html=True)

    else:
        st.subheader("Manual Text Input")
        user_input = st.text_area("Type text here", value=st.session_state.live_text, height=140)

        if user_input != st.session_state.live_text:
            old_w = st.session_state.live_text.split()
            new_w = user_input.split()
            added = new_w[len(old_w):]
            for t in added:
                st.session_state.alerts.extend(process_token(t, models, pcfg))
                w = clean_word(t)
                if w:
                    st.session_state.word_buffer.append(w)
                    st.session_state.word_count += 1
                if st.session_state.word_count % TRIGGER_N == 0 and st.session_state.word_count > 0:
                    st.session_state.alerts.extend(process_grammar(models))
            st.session_state.live_text = user_input

        if st.button("Parse Current Text"):
            if user_input:
                process_parse(user_input.split(), models, pcfg)

with col2:
    st.subheader("Live Pipeline Alerts")
    if not st.session_state.alerts:
        st.info("No pipeline alerts triggered yet.")
    else:
        for alert in reversed(st.session_state.alerts[-15:]):
            atype = alert["type"]
            css_cls = "seg-alert" if atype == "SEGMENT" else ("spell-alert" if atype == "SPELL" else "grammar-alert")
            st.markdown(
                f'<div class="alert-card {css_cls}"><b>[{atype}-ALERT]</b> {alert["detail"]} <br><small>Latency: {alert["latency"]:.2f} ms</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("PCFG Constituency Parse Trees")
    if not st.session_state.parse_history:
        st.caption("Parse trees for completed sentences will appear here.")
    else:
        for i, res in enumerate(reversed(st.session_state.parse_history[-5:])):
            sent = " ".join(res["words"])
            with st.expander(f"Sentence: {sent[:40]}..."):
                st.write("**Penn Treebank POS tags:**", " ".join(res["ptb_tags"]))
                if res["parseable"]:
                    st.markdown(f'<div class="tree-container">{res["parse"]}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Unparseable under PCFG grammar.")
