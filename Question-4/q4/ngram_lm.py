"""
ngram_lm.py - Shared bigram and trigram grammar models (Q4 Part 3)

Q4 needs its own smoothed n-gram models for the live perplexity alert and for
the end-of-passage sentence scoring. Both are trained here on the Brown train
split with add-k smoothing and cached on disk.

The Q1 and Q3 models are deliberately left alone. Q1's Witten-Bell trigram LM
keeps scoring segmentation splits and Q3's bigram counts keep scoring real-word
corrections, so nothing from those questions is retrained while Q4 runs.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

Q1_DIR = Path(__file__).resolve().parents[2] / "Question-1"
if str(Q1_DIR) not in sys.path:
    sys.path.insert(0, str(Q1_DIR))

from q1.data import load_brown
from q1.lm import NgramLM

GRAMMAR_LM_PATH = Path(__file__).parent.parent / "models" / "q4_grammar_lms.pkl"

# Picked with scripts/tune_lm_k.py, which sweeps k on the Brown dev split. The
# trigram wants the smaller constant because its contexts are far sparser, so
# the same k spreads much more mass over unseen continuations.
DEFAULT_BIGRAM_K = 0.01
DEFAULT_TRIGRAM_K = 0.001

# Live alerts score the last ten words, so the calibration windows are the same
# length. A threshold measured on longer windows would not transfer.
CALIBRATION_WINDOW = 10
CALIBRATION_SENTENCES = 2000


def _train_pair(
    sentences: Sequence[Sequence[str]], bigram_k: float, trigram_k: float
) -> tuple[NgramLM, NgramLM]:
    """Fit a bigram and a trigram model with add-k smoothing on the same text."""
    bigram = NgramLM(order=2, smoothing="addk", k=bigram_k).fit(sentences, char_lm=False)
    trigram = NgramLM(order=3, smoothing="addk", k=trigram_k).fit(sentences, char_lm=False)
    return bigram, trigram


def _percentile(values: list[float], fraction: float) -> float:
    """Value below which the given fraction of the sample falls."""
    if not values:
        return float("-inf")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _windows(sentences: Sequence[Sequence[str]], size: int) -> Iterator[list[str]]:
    """Every full window of `size` consecutive words inside a sentence."""
    for words in sentences:
        if len(words) < size:
            continue
        for start in range(len(words) - size + 1):
            yield list(words[start : start + size])


def per_word_logprob(lm: NgramLM, words: Sequence[str]) -> float:
    """Sentence log-probability divided by its token count, end marker included.

    Raw log-probability drops with length, so a length-normalised score is the
    only fair way to compare a four-word sentence against a thirty-word one.
    """
    words = list(words)
    if not words:
        return float("-inf")
    return lm.sentence_logprob(words) / (len(words) + 1)


def coverage(lm: NgramLM, words: Sequence[str]) -> float:
    """Share of the sentence the model has actually seen in training."""
    words = list(words)
    if not words:
        return 0.0
    known = sum(1 for w in words if w in lm.vocab)
    return known / len(words)


def calibrate(
    bigram: NgramLM,
    trigram: NgramLM,
    sentences: Sequence[Sequence[str]],
    window: int = CALIBRATION_WINDOW,
) -> dict[str, float]:
    """Measure what the two models consider normal on held-out Brown text.

    Everything downstream (the live perplexity threshold and the end-of-passage
    verdicts) is a cut-off on these distributions, so the numbers come from real
    held-out sentences rather than from a hand-picked constant.
    """
    bigram_scores = [per_word_logprob(bigram, s) for s in sentences if s]
    trigram_scores = [per_word_logprob(trigram, s) for s in sentences if s]

    window_ppl = [trigram.perplexity([w]) for w in _windows(sentences[:400], window)]

    return {
        "bigram_p02": _percentile(bigram_scores, 0.02),
        "bigram_p10": _percentile(bigram_scores, 0.10),
        "trigram_p02": _percentile(trigram_scores, 0.02),
        "trigram_p10": _percentile(trigram_scores, 0.10),
        "window_ppl_p90": _percentile(window_ppl, 0.90),
        "window_ppl_p95": _percentile(window_ppl, 0.95),
        "n_sentences": float(len(bigram_scores)),
        "n_windows": float(len(window_ppl)),
    }


def train_grammar_lms(
    bigram_k: float = DEFAULT_BIGRAM_K,
    trigram_k: float = DEFAULT_TRIGRAM_K,
    cache: bool = True,
) -> dict[str, Any]:
    """Train both grammar models on Brown and calibrate them on the dev split."""
    if cache and GRAMMAR_LM_PATH.exists():
        with GRAMMAR_LM_PATH.open("rb") as f:
            return pickle.load(f)

    corpus = load_brown(tagset="universal")
    train_sentences = [list(s.words) for s in corpus.train]
    dev_sentences = [list(s.words) for s in corpus.dev[:CALIBRATION_SENTENCES]]

    bigram, trigram = _train_pair(train_sentences, bigram_k, trigram_k)

    # Add-k only behaves as a probability model if the mass still sums to one,
    # and the segmentation and verdict comparisons both assume that it does.
    bigram.check_normalised(("the",))
    trigram.check_normalised(("of", "the"))

    payload = {
        "bigram_k": bigram_k,
        "trigram_k": trigram_k,
        "corpus": "Brown (train split, 80/10/10 by genre)",
        "bigram": bigram,
        "trigram": trigram,
        "floors": calibrate(bigram, trigram, dev_sentences),
        "n_train_sentences": len(train_sentences),
    }

    if cache:
        GRAMMAR_LM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GRAMMAR_LM_PATH.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


_CACHE: dict[str, Any] | None = None


def load_grammar_lms() -> dict[str, Any]:
    """Load the trained grammar models, training them once if needed."""
    global _CACHE
    if _CACHE is None:
        _CACHE = train_grammar_lms()
    return _CACHE


def tune_k(
    candidates: Sequence[float] = (0.001, 0.01, 0.1, 0.5, 1.0),
    dev_limit: int = CALIBRATION_SENTENCES,
) -> list[dict[str, float]]:
    """Fit both models at each k and report dev perplexity for the sweep."""
    corpus = load_brown(tagset="universal")
    train_sentences = [list(s.words) for s in corpus.train]
    dev_sentences = [list(s.words) for s in corpus.dev[:dev_limit]]

    results = []
    for k in candidates:
        bigram, trigram = _train_pair(train_sentences, k, k)
        results.append(
            {
                "k": k,
                "bigram_dev_ppl": bigram.perplexity(dev_sentences),
                "trigram_dev_ppl": trigram.perplexity(dev_sentences),
            }
        )
    return results
