"""
model_loader.py - Model Loader for Q1 and Q3

Loads pre-trained Q1 (trigram LM, HMM tagger) and Q3 (vocab, SymDel index, SpellingCorrector)
models into memory for Q4, along with Q4's own add-k grammar models. Trains Q1 on
first launch if model binary is missing.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any

import nltk

_NLTK_DEPS = ["brown", "universal_tagset", "treebank", "gutenberg", "reuters", "punkt"]

for dep in _NLTK_DEPS:
    try:
        if dep == "universal_tagset":
            nltk.data.find(f"taggers/{dep}")
        elif dep == "punkt":
            nltk.data.find(f"tokenizers/{dep}")
        else:
            nltk.data.find(f"corpora/{dep}")
    except LookupError:
        nltk.download(dep, quiet=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
Q1_ROOT = REPO_ROOT / "Question-1"
Q3_SRC = REPO_ROOT / "Question-3" / "q3_spelling_corrector"

Q1_MODEL_PATH = Q1_ROOT / "models" / "q1_english.pkl"
Q3_MODEL_PATH = Q3_SRC / "models" / "q3_language_models.pkl"

# Q4-owned fallback cache: used ONLY when Q1's/Q3's own artifact is missing.
# Q4 must never write into Question-1/models/ or Question-3/.../models/ —
# those directories belong to Q1 and Q3 respectively, and writing a
# Q4-trained substitute there (even a well-intentioned "train it if
# missing" fallback) risks silently replacing the real, fuller-schema
# artifact those questions' own scripts produce with a slimmer one Q4
# happens to need, which then breaks Q1's/Q3's own code the next time it
# tries to load its own file. See REPORT_Q4.md for the incident this
# guarded against.
Q4_MODEL_CACHE_DIR = REPO_ROOT / "Question-4" / "models"
Q4_Q1_CACHE_PATH = Q4_MODEL_CACHE_DIR / "q1_english_cache.pkl"
Q4_Q3_CACHE_DIR = Q4_MODEL_CACHE_DIR / "q3_cache"

for path in (str(Q1_ROOT), str(Q3_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _train_q1_english() -> dict[str, Any]:
    from q1.data import load_brown, vocabulary, max_word_length
    from q1.lm import NgramLM
    from q1.segment import DecoderConfig
    from q1.tagger import HMMTagger, TaggerConfig, tagged_pairs

    print(
        "[Q4] Question-1/models/q1_english.pkl not found — training a "
        "Q4-local fallback copy instead (cached at "
        f"{Q4_Q1_CACHE_PATH}). For a real 'reuse Q1's trained model' run, "
        "run Question-1/scripts/run_q1.py first."
    )
    corpus = load_brown(tagset="universal")
    lm = NgramLM(order=3, smoothing="witten_bell").fit(s.words for s in corpus.train)
    train_vocab = vocabulary(corpus.train)
    max_len = min(20, max_word_length(corpus.train, 0.999) + 4)
    config = DecoderConfig(max_word_len=max_len, beam_width=8, unk_penalty=0.0)

    train_pairs = tagged_pairs(corpus.train)
    tagger = HMMTagger(TaggerConfig(order=3)).fit(train_pairs)

    payload = {
        "language": "English",
        "corpus": "Brown",
        "tagset": "universal",
        "lm": lm,
        "config": config,
        "tagger": tagger,
        "train_vocab": train_vocab,
    }
    # Cached under Question-4/models/, never under Question-1/models/ —
    # that path is Q1's own, and this payload does not match the schema
    # Q1's own scripts save there (no tagger_config/morph_tagger/
    # baseline_tagger), so writing it to Q1's path would corrupt Q1's
    # artifact for anyone who loads it expecting the full schema.
    Q4_Q1_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Q4_Q1_CACHE_PATH.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_q1_models() -> dict[str, Any]:
    if Q1_MODEL_PATH.exists():
        with Q1_MODEL_PATH.open("rb") as f:
            return pickle.load(f)
    if Q4_Q1_CACHE_PATH.exists():
        with Q4_Q1_CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return _train_q1_english()


def load_q3_models() -> dict[str, Any]:
    from corpus_models import build_and_save_all, load_models
    if Q3_MODEL_PATH.exists():
        return load_models(model_dir=Q3_MODEL_PATH.parent)
    # Same reasoning as load_q1_models(): never write Q3's real path.
    if not Q4_Q3_CACHE_DIR.exists() or not (Q4_Q3_CACHE_DIR / "q3_language_models.pkl").exists():
        print(
            "[Q4] Question-3/q3_spelling_corrector/models/q3_language_models.pkl "
            f"not found — building a Q4-local fallback copy instead (cached at "
            f"{Q4_Q3_CACHE_DIR}). For a real 'reuse Q3's trained model' run, "
            "run Question-3/q3_spelling_corrector/corpus_models.py first."
        )
        build_and_save_all(model_dir=Q4_Q3_CACHE_DIR)
    return load_models(model_dir=Q4_Q3_CACHE_DIR)


_CACHE: dict[str, Any] | None = None


def load_all_models() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    from candidates import build_symdel_index
    from corrector import SpellingCorrector

    from q4.ngram_lm import load_grammar_lms

    q1 = load_q1_models()
    q3 = load_q3_models()
    q4_lms = load_grammar_lms()
    symdel = build_symdel_index(q3["vocab"])

    corrector = SpellingCorrector(
        vocab=q3["vocab"],
        unigram_counts=q3["unigram_counts"],
        bigram_counts=q3["bigram_counts"],
        vocab_size=q3["vocab_size"],
        symdel_index=symdel,
        method="B",
        # Matches Q3's own chosen default (see Question-3/REPORT_Q3.md §3.1)
        # rather than a second, independently hardcoded value — this is
        # meant to be Q3's corrector reused as-is, not a re-tuned copy.
        real_word_threshold=1.1,
        k=q3["k"],
    )

    _CACHE = {
        "q1_lm": q1["lm"],
        "q1_config": q1["config"],
        "q1_tagger": q1["tagger"],
        "q1_vocab": q1["train_vocab"],
        "q3_vocab": q3["vocab"],
        "q3_unigram": q3["unigram_counts"],
        "q3_bigram": q3["bigram_counts"],
        "q3_vocab_size": q3["vocab_size"],
        "q3_k": q3["k"],
        "q3_symdel": symdel,
        "corrector": corrector,
        "q4_bigram": q4_lms["bigram"],
        "q4_trigram": q4_lms["trigram"],
        "q4_floors": q4_lms["floors"],
        "q4_bigram_k": q4_lms["bigram_k"],
        "q4_trigram_k": q4_lms["trigram_k"],
    }
    return _CACHE
