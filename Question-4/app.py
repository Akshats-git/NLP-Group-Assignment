"""
app.py - Streamlit Live NLP Editor (Question 4)

Integrates:
1. Joint Segmentation + POS tagging (Question 1)
2. Spelling Correction via SymDel (Question 3)
3. Language Model Grammar Checking + PCFG Constituency Parser (Question 4)

Text arrives one token at a time, either streamed from a sampled passage or
typed by hand, and the same checks run on it either way. Once the passage is
finished the editor scores every sentence with the parser and both n-gram
models and shows the Part 4 comparison table.
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

from q4.analysis import agreement_summary, analyse_document, load_floors, summary_rows
from q4.grammar import TRIGGER_N
from q4.passage import MERGE_PROB, passage_token_stream, sample_passage
from q4.pipeline import LiveDocument, check_token, check_window


def new_session_state() -> dict:
    return {
        "editor_text": "",
        "alerts": [],
        "document": LiveDocument(),
        "live_flags": [],
        "word_count": 0,
        "next_trigger": TRIGGER_N,
        "sim_running": False,
        "latency_seg": [],
        "latency_grammar": [],
        "latency_analysis": 0.0,
        "passage_tokens": [],
        "passage_idx": 0,
        "live_text": "",
        "analysis": None,
        "agreement": None,
        "benchmark": None,
    }


def init_session() -> None:
    for key, value in new_session_state().items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


@st.cache_resource(show_spinner="Loading Q1, Q3 and Q4 language models...")
def get_models():
    from q4.model_loader import load_all_models
    return load_all_models()


@st.cache_resource(show_spinner="Building PCFG grammar from Penn Treebank...")
def get_pcfg():
    from q4.pcfg import train_pcfg
    return train_pcfg(cache=True, min_count=2)


@st.cache_resource(show_spinner="Loading the calibrated score floors...")
def get_floors(_models, _pcfg):
    return load_floors(_models, _pcfg, _models["q1_tagger"])


def flag_current_sentence() -> None:
    """Mark the sentence being typed as one the live layer complained about."""
    index = len(st.session_state.document.sentences)
    flags = st.session_state.live_flags
    while len(flags) <= index:
        flags.append(False)
    flags[index] = True


def process_token(token: str, models, sentence_end: bool = False) -> None:
    """Run the live checks on one token and fold it into the document."""
    outcome = check_token(token, models)
    st.session_state.latency_seg.append(outcome.latency_ms)

    for alert in outcome.alerts:
        flag_current_sentence()
        st.session_state.alerts.append(
            {"type": alert.kind, "detail": alert.detail, "latency": alert.latency_ms}
        )

    st.session_state.word_count += len(outcome.words)
    st.session_state.document.add(outcome, force_sentence_end=sentence_end)


def process_grammar(models, trigger_n: int, threshold: float) -> None:
    """Run the grammar and real-word check over the last trigger_n words."""
    window = st.session_state.document.words()[-trigger_n:]
    if len(window) < 2:
        return

    result = check_window(window, models, ppl_threshold=threshold)
    st.session_state.latency_grammar.append(result.latency_ms)
    if not result.fired:
        return

    flag_current_sentence()
    detail = f"Window perplexity {result.ppl:.0f} against threshold {threshold:.0f}"
    if result.real_word_fixes:
        fixes = ", ".join(
            f"{fix.original} -> {fix.suggestion}" for fix in result.real_word_fixes
        )
        detail += f" | Real-word: {fixes}"
    st.session_state.alerts.append(
        {"type": "GRAMMAR", "detail": detail, "latency": result.latency_ms}
    )


def finish_and_analyse(models, pcfg, floors) -> None:
    """Close the passage and run the end-of-passage analysis over it."""
    document = st.session_state.document
    document.close()

    started = time.perf_counter()
    analyses = analyse_document(document.sentences, pcfg, models, floors)
    st.session_state.latency_analysis = (time.perf_counter() - started) * 1000

    flags = st.session_state.live_flags
    while len(flags) < len(analyses):
        flags.append(False)

    st.session_state.analysis = analyses
    st.session_state.agreement = agreement_summary(analyses, flags)


# Header
st.title("NLP Group Assignment - Question 4 Editor")
st.subheader("Joint Segmentation, Spelling Correction, Grammar Check & PCFG Parser")

models = get_models()
pcfg = get_pcfg()
floors = get_floors(models, pcfg)

# Sidebar
with st.sidebar:
    st.header("Settings & Simulation")
    mode = st.radio("Input Mode", ["Simulated Live Typing", "Manual Typing"])

    words_per_sec = st.slider("Typing Speed (words/sec)", 0.5, 5.0, 2.0, 0.5)
    merge_p = st.slider("Merge Error Probability (p)", 0.0, 0.25, MERGE_PROB, 0.01)
    trigger_n = st.slider("Grammar Trigger Interval (N words)", 5, 30, TRIGGER_N, 1)
    threshold = st.slider(
        "Perplexity Threshold",
        1000.0,
        40000.0,
        float(floors["window_ppl_p95"]),
        500.0,
        help="Default is the level only 5 percent of clean Brown windows exceed.",
    )

    st.markdown("---")
    st.header("Latency Metrics")

    seg_latencies = st.session_state.latency_seg
    grammar_latencies = st.session_state.latency_grammar
    avg_seg = (sum(seg_latencies) / len(seg_latencies)) if seg_latencies else 0.0
    avg_grammar = (
        (sum(grammar_latencies) / len(grammar_latencies)) if grammar_latencies else 0.0
    )
    total_ms = sum(seg_latencies) + sum(grammar_latencies) + st.session_state.latency_analysis

    st.metric("Seg + Spell Check (avg)", f"{avg_seg:.2f} ms")
    st.metric("Grammar Check (avg)", f"{avg_grammar:.2f} ms")
    st.metric("Total Pipeline Time", f"{total_ms:.0f} ms")

    if st.button("Reset Session"):
        for key, value in new_session_state().items():
            st.session_state[key] = value
        st.rerun()

    st.markdown("---")
    with st.expander("Speed Demon benchmark"):
        st.caption("1,000 corrupted words through both layers, as in Part 5.")
        if st.button("Run benchmark"):
            from q4.speed_demon import format_report, run_speed_demon

            with st.spinner("Timing both layers..."):
                st.session_state.benchmark = format_report(
                    run_speed_demon(models, trigger_n=trigger_n)
                )
        if st.session_state.benchmark:
            st.code(st.session_state.benchmark)

col1, col2 = st.columns([3, 2])

with col1:
    if mode == "Simulated Live Typing":
        c1, c2, c3 = st.columns(3)
        if c1.button("Start Passage Stream"):
            passage, corpus_name = sample_passage()
            for key, value in new_session_state().items():
                st.session_state[key] = value
            st.session_state.passage_tokens = list(
                passage_token_stream(passage, p=merge_p)
            )
            st.session_state.sim_running = True

        if c2.button("Stop Stream"):
            st.session_state.sim_running = False

        if c3.button("Analyse Passage"):
            st.session_state.sim_running = False
            finish_and_analyse(models, pcfg, floors)

        box = st.empty()
        status = st.empty()

        if st.session_state.sim_running:
            tokens = st.session_state.passage_tokens
            idx = st.session_state.passage_idx
            delay = 1.0 / words_per_sec

            while idx < len(tokens) and st.session_state.sim_running:
                token, sentence_end = tokens[idx]
                idx += 1
                st.session_state.passage_idx = idx

                st.session_state.editor_text += (
                    " " if st.session_state.editor_text else ""
                ) + token
                box.markdown(
                    f'<div class="editor-container">{st.session_state.editor_text}▌</div>',
                    unsafe_allow_html=True,
                )

                process_token(token, models, sentence_end=sentence_end)

                if st.session_state.word_count >= st.session_state.next_trigger:
                    st.session_state.next_trigger += trigger_n
                    process_grammar(models, trigger_n, threshold)

                status.text(f"Streaming token {idx}/{len(tokens)}...")
                time.sleep(delay)

            box.markdown(
                f'<div class="editor-container">{st.session_state.editor_text}</div>',
                unsafe_allow_html=True,
            )
            if idx >= len(tokens):
                st.session_state.sim_running = False
                status.text("Stream finished, running the end-of-passage analysis...")
                finish_and_analyse(models, pcfg, floors)
                st.rerun()
        else:
            text = st.session_state.editor_text or "Click 'Start Passage Stream' to simulate typing."
            box.markdown(
                f'<div class="editor-container">{text}</div>', unsafe_allow_html=True
            )

    else:
        st.subheader("Manual Text Input")
        user_input = st.text_area(
            "Type text here", value=st.session_state.live_text, height=140
        )

        if user_input != st.session_state.live_text:
            already_seen = len(st.session_state.live_text.split())
            for token in user_input.split()[already_seen:]:
                process_token(token, models)
                if st.session_state.word_count >= st.session_state.next_trigger:
                    st.session_state.next_trigger += trigger_n
                    process_grammar(models, trigger_n, threshold)
            st.session_state.live_text = user_input

        if st.button("Analyse Text"):
            finish_and_analyse(models, pcfg, floors)

with col2:
    st.subheader("Live Pipeline Alerts")
    if not st.session_state.alerts:
        st.info("No pipeline alerts triggered yet.")
    else:
        for alert in reversed(st.session_state.alerts[-15:]):
            kind = alert["type"]
            css_class = (
                "seg-alert"
                if kind == "SEGMENT"
                else ("spell-alert" if kind == "SPELL" else "grammar-alert")
            )
            st.markdown(
                f'<div class="alert-card {css_class}"><b>[{kind}-ALERT]</b> {alert["detail"]}'
                f'<br><small>Latency: {alert["latency"]:.2f} ms</small></div>',
                unsafe_allow_html=True,
            )

st.markdown("---")
st.subheader("Final Passage Analysis")

if not st.session_state.analysis:
    st.caption(
        "Finish the passage to score every sentence with the PCFG parser and both n-gram models."
    )
else:
    analyses = st.session_state.analysis
    agreement = st.session_state.agreement

    st.dataframe(summary_rows(analyses), width="stretch", hide_index=True)
    st.caption(
        "PCFG and n-gram columns are per-word log probabilities. The chosen method is the"
        " parser when it finds a parse that is not a probability outlier, the trigram when"
        " it has seen at least 80 percent of the sentence, and the bigram otherwise."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Live alerts against the final verdict**")
        st.write(
            {
                "sentences": agreement["sentences"],
                "flagged by both": agreement["both_flagged"],
                "live alert only": agreement["live_only"],
                "final verdict only": agreement["final_only"],
                "clean for both": agreement["neither"],
                "agreement": f"{agreement['agreement']:.0%}",
            }
        )
    with right:
        st.markdown("**Timing**")
        st.write(
            {
                "tokens checked": len(st.session_state.latency_seg),
                "grammar triggers": len(st.session_state.latency_grammar),
                "end-of-passage analysis": f"{st.session_state.latency_analysis:.1f} ms",
                "total pipeline time": f"{total_ms:.0f} ms",
            }
        )

    st.markdown("**PCFG Constituency Parse Trees**")
    for item in analyses:
        with st.expander(f"{item.index}. {item.text[:60]}"):
            st.write("Penn Treebank POS tags:", " ".join(item.ptb_tags))
            st.write("Chosen method:", item.chosen_method, "because", item.chosen_reason)
            if item.pcfg_parseable:
                st.markdown(
                    f'<div class="tree-container">{item.pcfg_parse}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Unparseable under the pruned Penn Treebank grammar.")
