"""
model_loader.py — Q4 Model Loading

Loads Q1 (English LM + tagger + decoder config) and Q3 (vocab, unigram,
bigram, SymDel index, SpellingCorrector) models into a single shared dict.

Q1 models:  loaded from Question-1/models/q1_english.pkl
            If the file does not exist, training is done automatically
            (one-time ~30s cost) and the result is cached.
Q3 models:  loaded from Question-3/q3_spelling_corrector/models/q3_language_models.pkl
            Always present (pre-trained by Q3 author).

The shared trigram LM from Q1 is the single LM used for grammar perplexity
checks in Q4 — one model, three sub-systems, as the assignment requires.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# NLTK data — download anything missing silently on first use
# ---------------------------------------------------------------------------
import nltk

_NLTK_REQUIRED = [
    "brown",           # Q1 training corpus + Q3 corpus
    "universal_tagset",# needed by load_brown(tagset='universal')
    "treebank",        # Q4 PCFG training
    "gutenberg",       # passage sampling
    "reuters",         # passage sampling fallback
    "punkt",           # sentence tokeniser
    "punkt_tab",       # punkt (newer NLTK)
]

def _ensure_nltk_data() -> None:
    """Download any missing NLTK resources, silently."""
    for resource in _NLTK_REQUIRED:
        try:
            # Try to find it first (cheap)
            if resource in ("universal_tagset",):
                nltk.data.find(f"taggers/{resource}")
            elif resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Repository root and path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]        # NLP-Group-Assignment/
Q1_ROOT = REPO_ROOT / "Question-1"
Q3_SRC = REPO_ROOT / "Question-3" / "q3_spelling_corrector"

Q1_MODEL_PATH = Q1_ROOT / "models" / "q1_english.pkl"
Q3_MODEL_PATH = Q3_SRC / "models" / "q3_language_models.pkl"

# Add Q1 and Q3 source roots to sys.path so their modules are importable.
for _p in (str(Q1_ROOT), str(Q3_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Q1 model loading / training fallback
# ---------------------------------------------------------------------------

def _train_q1_english() -> dict[str, Any]:
    """Train Q1 English models from scratch and return them as a dict.

    Uses the exact same pipeline as Question-1/scripts/run_q1.py:
    - Brown corpus, universal tagset (12 classes)
    - Trigram NgramLM (Witten-Bell smoothing)
    - Trigram HMMTagger (deleted interpolation + suffix model)
    - DecoderConfig with tuned max_word_len=20, beam_width=8, unk_penalty=0.0

    This is the fallback when models/q1_english.pkl does not yet exist.
    It is safe to call multiple times (idempotent after the first save).
    """
    from q1.data import load_brown, vocabulary, max_word_length
    from q1.lm import NgramLM
    from q1.segment import DecoderConfig, decode_segmentation
    from q1.tagger import HMMTagger, TaggerConfig, tagged_pairs

    print("[Q4] Q1 model file not found — training English models from Brown corpus …")
    t0 = time.perf_counter()

    corpus = load_brown(tagset="universal")
    lm = NgramLM(order=3, smoothing="witten_bell").fit(
        s.words for s in corpus.train
    )
    train_vocab = vocabulary(corpus.train)
    max_len = min(20, max_word_length(corpus.train, 0.999) + 4)
    config = DecoderConfig(max_word_len=max_len, beam_width=8, unk_penalty=0.0)

    train_pairs = tagged_pairs(corpus.train)
    tagger = HMMTagger(TaggerConfig(order=3)).fit(train_pairs)

    elapsed = time.perf_counter() - t0
    print(f"[Q4] Q1 training complete in {elapsed:.1f}s — saving to {Q1_MODEL_PATH}")

    Q1_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": "English",
        "corpus": "Brown",
        "tagset": "universal",
        "lm": lm,
        "config": config,
        "tagger": tagger,
        "tagger_config": tagger.config,
        "train_vocab": train_vocab,
    }
    with Q1_MODEL_PATH.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_q1_models() -> dict[str, Any]:
    """Return the Q1 English model bundle, training if necessary."""
    if Q1_MODEL_PATH.exists():
        with Q1_MODEL_PATH.open("rb") as fh:
            return pickle.load(fh)
    return _train_q1_english()


# ---------------------------------------------------------------------------
# Q3 model loading
# ---------------------------------------------------------------------------

def load_q3_models() -> dict[str, Any]:
    """Load Q3 vocabulary, unigram, bigram, and symdel index."""
    if not Q3_MODEL_PATH.exists():
        # Q3 models don't exist yet — build them on the fly.
        from corpus_models import build_and_save_all, load_models
        print("[Q4] Q3 model file not found — building from Brown corpus …")
        build_and_save_all(model_dir=Q3_MODEL_PATH.parent)
        return load_models(model_dir=Q3_MODEL_PATH.parent)

    from corpus_models import load_models
    return load_models(model_dir=Q3_MODEL_PATH.parent)


# ---------------------------------------------------------------------------
# Combined loader — call this once at Streamlit startup
# ---------------------------------------------------------------------------

_CACHE: dict[str, Any] | None = None


def load_all_models(force_retrain: bool = False) -> dict[str, Any]:
    """Load Q1 + Q3 models and build the SpellingCorrector.

    Returns a flat dict with keys:
        q1_lm          : NgramLM  — shared trigram LM (Q1, Brown English)
        q1_config      : DecoderConfig — tuned beam-search settings
        q1_tagger      : HMMTagger
        q1_vocab       : set[str]  — Q1 training vocabulary
        q3_vocab       : set[str]  — Q3 vocabulary (Brown, alphabetic)
        q3_unigram     : Counter[str, int]
        q3_bigram      : Counter[(str,str), int]
        q3_vocab_size  : int
        q3_k           : float (add-k smoothing constant)
        q3_symdel      : dict[str, list[str]]
        corrector      : SpellingCorrector (Method B — SymDel, fastest)

    The shared LM (q1_lm) is used for:
        - segmentation scoring  (Q1 reuse)
        - grammar perplexity    (Q4 new use)
    """
    global _CACHE
    if _CACHE is not None and not force_retrain:
        return _CACHE

    from candidates import build_symdel_index
    from corrector import SpellingCorrector

    q1 = load_q1_models()
    q3 = load_q3_models()

    symdel = build_symdel_index(q3["vocab"])

    corrector = SpellingCorrector(
        vocab=q3["vocab"],
        unigram_counts=q3["unigram_counts"],
        bigram_counts=q3["bigram_counts"],
        vocab_size=q3["vocab_size"],
        symdel_index=symdel,
        method="B",        # SymDel — fastest (Q3 Phase 5 benchmark confirmed)
        real_word_threshold=2.0,
        k=q3["k"],
    )

    _CACHE = {
        # Q1
        "q1_lm":      q1["lm"],
        "q1_config":  q1["config"],
        "q1_tagger":  q1["tagger"],
        "q1_vocab":   q1["train_vocab"],
        # Q3
        "q3_vocab":       q3["vocab"],
        "q3_unigram":     q3["unigram_counts"],
        "q3_bigram":      q3["bigram_counts"],
        "q3_vocab_size":  q3["vocab_size"],
        "q3_k":           q3["k"],
        "q3_symdel":      symdel,
        "corrector":      corrector,
    }
    return _CACHE
