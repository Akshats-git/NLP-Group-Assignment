"""
app.py — Q4 Streamlit Live NLP Editor

A Streamlit application that streams a randomly sampled English passage
word-by-word, running three real-time NLP checks as each token arrives:

    [SEGMENT-ALERT]  Q1 beam-search segmentation (merged-token detection)
    [SPELL-ALERT]    Q3 SymDel spelling correction (Method B)
    [GRAMMAR-ALERT]  Trigram perplexity + real-word error check (every N=10 words)

Part 2 adds a PCFG constituency parser (Penn Treebank) that parses each
completed sentence and displays the bracket-notation parse tree.

Design decisions are justified inline and documented in the individual
q4/ modules.
"""
from __future__ import annotations

import sys
import time
import random
from collections import deque
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — must happen before importing q4 or Q1/Q3 modules
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent           # Question-4/
_REPO = _HERE.parent                              # NLP-Group-Assignment/
_Q1   = _REPO / "Question-1"
_Q3   = _REPO / "Question-3" / "q3_spelling_corrector"
for _p in (str(_HERE), str(_Q1), str(_Q3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Q4 — Live NLP Editor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Editor text area */
.editor-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    line-height: 1.9;
    min-height: 160px;
    color: #e2e8f0;
    word-wrap: break-word;
    white-space: pre-wrap;
    backdrop-filter: blur(10px);
}

/* Alert badges */
.alert-segment {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #1a1a2e;
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0 3px;
    font-family: 'JetBrains Mono', monospace;
}
.alert-spell {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0 3px;
    font-family: 'JetBrains Mono', monospace;
}
.alert-grammar {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    color: white;
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* Alert log entries */
.log-entry {
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.84rem;
    border-left: 3px solid;
    font-family: 'JetBrains Mono', monospace;
}
.log-segment {
    background: rgba(245,158,11,0.1);
    border-color: #f59e0b;
    color: #fde68a;
}
.log-spell {
    background: rgba(239,68,68,0.1);
    border-color: #ef4444;
    color: #fca5a5;
}
.log-grammar {
    background: rgba(139,92,246,0.1);
    border-color: #8b5cf6;
    color: #c4b5fd;
}

/* Parse tree */
.parse-tree {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #6ee7b7;
    white-space: pre-wrap;
    word-break: break-all;
}
.parse-unparseable {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #fca5a5;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Section headers */
h1, h2, h3 { color: #e2e8f0 !important; }

/* Status dot */
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.3; }
    100% { opacity: 1; }
}
.dot-running { background: #22c55e; }
.dot-idle    { background: #64748b; }

div[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "models_loaded": False,
        "pcfg_loaded": False,
        "models": None,
        "pcfg_grammar": None,
        "editor_text": "",
        "alerts": [],           # list of alert dicts
        "word_buffer": [],      # clean words accumulated for grammar window
        "word_count": 0,
        "sim_running": False,
        "parse_results": [],    # list of parse result dicts (per sentence)
        "latency_seg": [],      # per-token seg+spell latency (ms)
        "latency_grammar": [],  # per-trigger grammar latency (ms)
        "passage_tokens": [],   # flat list of (token, is_sent_end) from passage
        "passage_idx": 0,
        "passages_used": [],
        "live_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Model loading (cached with st.cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_models():
    from q4.model_loader import load_all_models
    return load_all_models()


@st.cache_resource(show_spinner=False)
def _load_pcfg():
    from q4.pcfg import train_pcfg
    return train_pcfg(cache=True)


# ---------------------------------------------------------------------------
# Alert processing helpers
# ---------------------------------------------------------------------------
from q4.segmentation import check_token_segmentation
from q4.spelling import check_token_spelling
from q4.grammar import check_grammar_window, TRIGGER_N
from q4.pcfg import parse_sentence
from q4.passage import sample_passage, passage_token_stream, inject_merges, MERGE_PROB


def _clean_token(tok: str) -> str:
    """Lowercase, strip punctuation tails for LM/vocab lookup."""
    return "".join(ch for ch in tok.lower() if ch.isalpha())


def process_token(token: str, models: dict, pcfg_grammar) -> list[dict]:
    """Run per-token checks (segmentation + spelling) and return alert dicts."""
    alerts = []
    cleaned = _clean_token(token)
    if not cleaned:
        return alerts

    t0 = time.perf_counter()

    # ------ SEGMENT-ALERT ------
    seg = check_token_segmentation(
        cleaned,
        lm=models["q1_lm"],
        config=models["q1_config"],
        tagger=models["q1_tagger"],
        vocab=models["q1_vocab"],
    )
    if seg.fired:
        alerts.append({
            "type": "SEGMENT",
            "token": token,
            "split": list(seg.words),
            "pos": list(seg.pos_tags),
            "latency_ms": seg.latency_ms,
            "detail": f"'{token}' → {' + '.join(seg.words)} [{', '.join(seg.pos_tags)}] "
                      f"(split score {seg.split_score:.2f} > single {seg.single_score:.2f})",
        })
        # After splitting, the first sub-word becomes the token for spell check
        cleaned = seg.words[0] if seg.words else cleaned

    # ------ SPELL-ALERT ------
    spell = check_token_spelling(cleaned, models["corrector"], models["q3_vocab"])
    if spell.fired:
        alerts.append({
            "type": "SPELL",
            "token": token,
            "suggestion": spell.suggestion,
            "candidates": spell.candidates[:5],
            "latency_ms": spell.latency_ms,
            "detail": f"'{token}' → suggest '{spell.suggestion}' "
                      f"(top candidates: {spell.candidates[:3]})",
        })

    total_latency = (time.perf_counter() - t0) * 1000
    st.session_state.latency_seg.append(total_latency)
    return alerts


def process_grammar_window(models: dict) -> list[dict]:
    """Run the grammar check on the current word buffer."""
    alerts = []
    window = list(st.session_state.word_buffer)[-TRIGGER_N:]
    if len(window) < 2:
        return alerts

    result = check_grammar_window(
        window,
        lm=models["q1_lm"],
        corrector=models["corrector"],
        q3_bigram=models["q3_bigram"],
        q3_unigram=models["q3_unigram"],
        q3_vocab_size=models["q3_vocab_size"],
        q3_k=models["q3_k"],
        q3_vocab=models["q3_vocab"],
    )
    st.session_state.latency_grammar.append(result.latency_ms)

    if result.fired:
        detail_parts = []
        if result.ppl > result.threshold:
            detail_parts.append(f"PPL={result.ppl:.1f} > threshold({result.threshold:.0f})")
        for fix in result.real_word_fixes:
            detail_parts.append(
                f"real-word: '{fix.original}' → '{fix.suggestion}' (+{fix.score_gain:.2f} nats)"
            )
        alerts.append({
            "type": "GRAMMAR",
            "window": window,
            "ppl": result.ppl,
            "threshold": result.threshold,
            "real_word_fixes": [
                {"original": f.original, "suggestion": f.suggestion, "gain": f.score_gain}
                for f in result.real_word_fixes
            ],
            "latency_ms": result.latency_ms,
            "detail": " | ".join(detail_parts) if detail_parts else "Grammar anomaly detected",
        })
    return alerts


def process_sentence_parse(words: list[str], models: dict, pcfg_grammar):
    """Parse a completed sentence and store result."""
    if not words or pcfg_grammar is None:
        return
    clean_words = [_clean_token(w) for w in words if _clean_token(w)]
    if not clean_words:
        return
    tags = models["q1_tagger"].tag(tuple(clean_words))
    result = parse_sentence(tuple(clean_words), tags, pcfg_grammar)
    st.session_state.parse_results.append(result)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center; font-size:2rem; font-weight:700;
           background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           margin-bottom: 0;'>
  📝 Live NLP Editor — Q4
</h1>
<p style='text-align:center; color:#64748b; margin-top:6px; font-size:0.9rem;'>
  Segmentation • Spelling Correction • Grammar Checking • PCFG Constituency Parsing
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    mode = st.radio(
        "Mode",
        ["Simulated Typing", "Live Typing"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ Parameters")

    typing_speed = st.slider("Words/second (sim)", 0.5, 5.0, 2.0, 0.5)
    ppl_threshold = st.slider("Grammar PPL threshold", 100, 1000, 300, 50)
    merge_prob = st.slider("Merge probability (p)", 0.0, 0.3, MERGE_PROB, 0.01,
                           help="Prob. of dropping space between words to test segmenter")
    show_parse = st.checkbox("Show PCFG parse tree", value=True)

    st.markdown("---")
    st.markdown("### 📊 Latency (ms)")

    seg_lats = st.session_state.latency_seg
    gram_lats = st.session_state.latency_grammar
    avg_seg  = (sum(seg_lats) / len(seg_lats)) if seg_lats else 0.0
    avg_gram = (sum(gram_lats) / len(gram_lats)) if gram_lats else 0.0

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg_seg:.1f}</div>
            <div class='metric-label'>Seg+Spell (avg)</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg_gram:.1f}</div>
            <div class='metric-label'>Grammar (avg)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔢 Session Stats")
    st.markdown(f"**Words processed:** {st.session_state.word_count}")
    st.markdown(f"**Alerts fired:** {len(st.session_state.alerts)}")
    st.markdown(f"**Sentences parsed:** {len(st.session_state.parse_results)}")

    if st.button("🗑️ Clear session"):
        for k in ["editor_text", "alerts", "word_buffer", "word_count",
                  "parse_results", "latency_seg", "latency_grammar",
                  "passage_tokens", "passage_idx", "sim_running"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else \
                                   (False if isinstance(st.session_state[k], bool) else
                                    (0 if isinstance(st.session_state[k], int) else ""))
        st.rerun()

# ---- Main content ----
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # ---- Model loading ----
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading Q1 + Q3 models (may train Q1 on first run ~30s) …"):
            try:
                st.session_state.models = _load_models()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.stop()

    if not st.session_state.pcfg_loaded and show_parse:
        with st.spinner("🌲 Training PCFG from Penn Treebank (cached after first run) …"):
            try:
                st.session_state.pcfg_grammar = _load_pcfg()
                st.session_state.pcfg_loaded = True
            except Exception as e:
                st.warning(f"⚠️ PCFG unavailable: {e}")
                st.session_state.pcfg_grammar = None
                st.session_state.pcfg_loaded = True

    models = st.session_state.models
    pcfg_grammar = st.session_state.pcfg_grammar

    # ------------------------------------------------------------------
    # MODE 1: Simulated Typing
    # ------------------------------------------------------------------
    if mode == "Simulated Typing":
        st.markdown("#### 🤖 Simulated Typing")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶️ Start new passage", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)

        if stop_btn:
            st.session_state.sim_running = False

        if start_btn:
            # Sample a fresh passage each run (no fixed seed)
            passage, corpus_name = sample_passage(seed=None)
            st.session_state.passages_used.append(corpus_name)
            tokens = list(passage_token_stream(passage, p=merge_prob))
            st.session_state.passage_tokens = tokens
            st.session_state.passage_idx = 0
            st.session_state.sim_running = True
            st.session_state.editor_text = ""
            st.session_state.alerts = []
            st.session_state.word_buffer = []
            st.session_state.parse_results = []
            st.session_state.word_count = 0

        # Display editor
        editor_placeholder = st.empty()
        status_placeholder = st.empty()

        # Typing loop
        if st.session_state.sim_running:
            tokens = st.session_state.passage_tokens
            idx = st.session_state.passage_idx

            if idx >= len(tokens):
                st.session_state.sim_running = False
                st.success("✅ Passage complete!")
            else:
                sentence_words: list[str] = []
                inter_word_delay = 1.0 / typing_speed

                while idx < len(tokens) and st.session_state.sim_running:
                    token, is_sent_end = tokens[idx]
                    idx += 1
                    st.session_state.passage_idx = idx

                    # Append to editor display
                    st.session_state.editor_text += (" " if st.session_state.editor_text else "") + token
                    editor_placeholder.markdown(
                        f"<div class='editor-box'>{st.session_state.editor_text}█</div>",
                        unsafe_allow_html=True,
                    )

                    # Per-token checks
                    new_alerts = process_token(token, models, pcfg_grammar)
                    st.session_state.alerts.extend(new_alerts)

                    cleaned = _clean_token(token)
                    if cleaned:
                        st.session_state.word_buffer.append(cleaned)
                        sentence_words.append(token)
                        st.session_state.word_count += 1

                    # Trigger-based grammar check every N words
                    if st.session_state.word_count > 0 and st.session_state.word_count % TRIGGER_N == 0:
                        gram_alerts = process_grammar_window(models)
                        st.session_state.alerts.extend(gram_alerts)

                    # Sentence-end: parse the sentence
                    if is_sent_end and show_parse and sentence_words:
                        process_sentence_parse(sentence_words, models, pcfg_grammar)
                        sentence_words = []

                    status_placeholder.markdown(
                        f"<span class='status-dot dot-running'></span>"
                        f"<span style='color:#94a3b8; font-size:0.8rem;'>Typing… word {st.session_state.word_count}</span>",
                        unsafe_allow_html=True,
                    )

                    time.sleep(inter_word_delay)

                # Final render when done
                editor_placeholder.markdown(
                    f"<div class='editor-box'>{st.session_state.editor_text}</div>",
                    unsafe_allow_html=True,
                )
                if idx >= len(tokens):
                    st.session_state.sim_running = False
                    status_placeholder.markdown(
                        "<span style='color:#22c55e; font-size:0.8rem;'>✅ Passage complete</span>",
                        unsafe_allow_html=True,
                    )
        else:
            # Static display when not running
            text = st.session_state.editor_text or "_Passage will appear here when you press Start…_"
            st.markdown(f"<div class='editor-box'>{text}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # MODE 2: Live Typing
    # ------------------------------------------------------------------
    else:
        st.markdown("#### ⌨️ Live Typing Mode")
        st.markdown("<span style='color:#94a3b8; font-size:0.85rem;'>Type text below — checks run after each word (press space/enter).</span>", unsafe_allow_html=True)

        user_input = st.text_area(
            "Your text",
            value=st.session_state.live_input,
            height=150,
            placeholder="Start typing here…",
            key="live_textarea",
            label_visibility="collapsed",
        )

        if user_input != st.session_state.live_input:
            # Find new words since last update
            old_words = st.session_state.live_input.split()
            new_words = user_input.split()
            added = new_words[len(old_words):]
            for tok in added:
                new_alerts = process_token(tok, models, pcfg_grammar)
                st.session_state.alerts.extend(new_alerts)
                cleaned = _clean_token(tok)
                if cleaned:
                    st.session_state.word_buffer.append(cleaned)
                    st.session_state.word_count += 1
                if st.session_state.word_count % TRIGGER_N == 0 and st.session_state.word_count > 0:
                    gram_alerts = process_grammar_window(models)
                    st.session_state.alerts.extend(gram_alerts)
            st.session_state.live_input = user_input

        if st.button("🔍 Parse current text as single sentence"):
            words = [_clean_token(w) for w in user_input.split() if _clean_token(w)]
            if words:
                process_sentence_parse(list(user_input.split()), models, pcfg_grammar)

# ---- Right column: Alerts + Parse ----
with col_right:
    st.markdown("#### 🚨 Alert Log")

    if not st.session_state.alerts:
        st.markdown("<span style='color:#475569; font-size:0.85rem;'>No alerts yet — start the simulation.</span>", unsafe_allow_html=True)
    else:
        # Show latest 20 alerts, newest first
        for alert in reversed(st.session_state.alerts[-20:]):
            atype = alert["type"]
            if atype == "SEGMENT":
                st.markdown(f"""
                <div class='log-entry log-segment'>
                    <span class='alert-segment'>SEGMENT-ALERT</span>
                    &nbsp;{alert['detail']}<br>
                    <span style='color:#6b7280; font-size:0.75rem;'>⏱ {alert['latency_ms']:.1f} ms</span>
                </div>""", unsafe_allow_html=True)
            elif atype == "SPELL":
                st.markdown(f"""
                <div class='log-entry log-spell'>
                    <span class='alert-spell'>SPELL-ALERT</span>
                    &nbsp;{alert['detail']}<br>
                    <span style='color:#6b7280; font-size:0.75rem;'>⏱ {alert['latency_ms']:.1f} ms</span>
                </div>""", unsafe_allow_html=True)
            elif atype == "GRAMMAR":
                fixes = alert.get("real_word_fixes", [])
                fix_str = " | ".join(f"{f['original']}→{f['suggestion']}" for f in fixes)
                st.markdown(f"""
                <div class='log-entry log-grammar'>
                    <span class='alert-grammar'>GRAMMAR-ALERT</span>
                    &nbsp;{alert['detail']}<br>
                    {"<span style='color:#a78bfa;'>Real-word: " + fix_str + "</span><br>" if fixes else ""}
                    <span style='color:#6b7280; font-size:0.75rem;'>⏱ {alert['latency_ms']:.1f} ms | PPL={alert['ppl']:.1f}</span>
                </div>""", unsafe_allow_html=True)

    # ---- PCFG Parse Trees ----
    if show_parse and st.session_state.parse_results:
        st.markdown("---")
        st.markdown("#### 🌲 PCFG Parse Trees")
        for i, res in enumerate(reversed(st.session_state.parse_results[-5:])):
            sent_str = " ".join(res["words"])
            ptb_str = " ".join(res["ptb_tags"])
            with st.expander(f"Sentence {len(st.session_state.parse_results) - i}: «{sent_str[:50]}{'…' if len(sent_str) > 50 else ''}»"):
                st.markdown(f"<span style='color:#94a3b8; font-size:0.78rem;'>PTB tags (reconciled): {ptb_str}</span>", unsafe_allow_html=True)
                if res["parseable"]:
                    # Pretty-print the bracket notation
                    parse_pretty = res["parse"] or ""
                    st.markdown(f"<div class='parse-tree'>{parse_pretty}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='parse-unparseable'>⚠️ UNPARSEABLE — No spanning parse found for this sentence.</div>", unsafe_allow_html=True)

    # ---- Corpus info ----
    if st.session_state.passages_used:
        st.markdown("---")
        st.markdown(f"<span style='color:#475569; font-size:0.75rem;'>📚 Corpora used: {', '.join(set(st.session_state.passages_used))}</span>", unsafe_allow_html=True)
