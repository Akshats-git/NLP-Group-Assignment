"""
model_loader.py - Model Loader for Q1 and Q3

Loads pre-trained Q1 (trigram LM, HMM tagger) and Q3 (vocab, SymDel index, SpellingCorrector)
models into memory for Q4. Trains Q1 on first launch if model binary is missing.
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

for path in (str(Q1_ROOT), str(Q3_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _train_q1_english() -> dict[str, Any]:
    from q1.data import load_brown, vocabulary, max_word_length
    from q1.lm import NgramLM
    from q1.segment import DecoderConfig
    from q1.tagger import HMMTagger, TaggerConfig, tagged_pairs

    print("[Q4] Training Q1 English model on Brown corpus...")
    corpus = load_brown(tagset="universal")
    lm = NgramLM(order=3, smoothing="witten_bell").fit(s.words for s in corpus.train)
    train_vocab = vocabulary(corpus.train)
    max_len = min(20, max_word_length(corpus.train, 0.999) + 4)
    config = DecoderConfig(max_word_len=max_len, beam_width=8, unk_penalty=0.0)

    train_pairs = tagged_pairs(corpus.train)
    tagger = HMMTagger(TaggerConfig(order=3)).fit(train_pairs)

    Q1_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": "English",
        "corpus": "Brown",
        "tagset": "universal",
        "lm": lm,
        "config": config,
        "tagger": tagger,
        "train_vocab": train_vocab,
    }
    with Q1_MODEL_PATH.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_q1_models() -> dict[str, Any]:
    if Q1_MODEL_PATH.exists():
        with Q1_MODEL_PATH.open("rb") as f:
            return pickle.load(f)
    return _train_q1_english()


def load_q3_models() -> dict[str, Any]:
    from corpus_models import build_and_save_all, load_models
    if not Q3_MODEL_PATH.exists():
        build_and_save_all(model_dir=Q3_MODEL_PATH.parent)
    return load_models(model_dir=Q3_MODEL_PATH.parent)


_CACHE: dict[str, Any] | None = None


def load_all_models() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
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
        method="B",
        real_word_threshold=2.0,
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
    }
    return _CACHE
