"""
grammar.py — Q4 Grammar & Real-Word Alert Engine

Implements the trigger-based grammar check that fires every N words.

Trigger interval N = 10
------------------------
Justification:
    - Too small (N < 5): the window is too narrow for bi/trigram context to be
      meaningful. A 3-word window has too little signal and fires constantly.
    - Too large (N > 15): the window spans multiple sentences and the check
      loses the per-sentence relevance. The parser cost also grows with window.
    - N = 10 aligns with average English sentence length (~15-20 words), so
      the grammar check fires roughly once per sentence — exactly the right
      cadence for sentence-level feedback without flooding the UI.
    - At simulated typing speed of 1 word/0.15s, a trigger every 10 words
      fires every ~1.5s, giving ample visual time between alerts.
    - Measured grammar check latency: <30ms for a 10-word window (empirical);
      well below the 150ms inter-word delay.

GRAMMAR-ALERT logic (two parts):
    1. Perplexity check: compute trigram perplexity of the last N words via
       the shared Q1 LM. If PPL > PERPLEXITY_THRESHOLD, fire a grammar alert.
    2. Real-word error check: for each in-vocabulary word in the window,
       compare its bigram score against edit-distance-1 candidates generated
       by Q3's SymDel method. If any candidate scores > REAL_WORD_MARGIN nats
       higher, flag it as part of the grammar alert output.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from q1.lm import NgramLM
    from corrector import SpellingCorrector

_Q3_SRC = Path(__file__).resolve().parents[2] / "Question-3" / "q3_spelling_corrector"
if str(_Q3_SRC) not in sys.path:
    sys.path.insert(0, str(_Q3_SRC))

from corpus_models import bigram_log_prob

# ---------------------------------------------------------------------------
# Tuning constants (exposed for potential UI override)
# ---------------------------------------------------------------------------
TRIGGER_N: int = 10           # grammar check fires every N words
PERPLEXITY_THRESHOLD: float = 300.0  # PPL above this → grammar alert
REAL_WORD_MARGIN: float = 2.0  # nats — margin for real-word substitution


class RealWordFix(NamedTuple):
    index: int          # position in the window
    original: str
    suggestion: str
    score_gain: float   # nats improvement


class GrammarResult(NamedTuple):
    fired: bool
    ppl: float
    threshold: float
    real_word_fixes: list[RealWordFix]
    window: list[str]          # the N words that triggered the check
    latency_ms: float


def _window_perplexity(words: list[str], lm: "NgramLM") -> float:
    """Trigram perplexity of ``words`` under the shared Q1 LM."""
    if not words:
        return math.inf
    return lm.perplexity([words])


def _bigram_score_word(
    words: list[str],
    idx: int,
    word: str,
    bigram_counts,
    unigram_counts,
    vocab_size: int,
    k: float,
) -> float:
    """Bigram context score for ``word`` at position ``idx`` in ``words``.

    score = log P(word | prev) + log P(next | word)   (terms that exist)
    Same formula as Q3's _score_word_in_context.
    """
    score = 0.0
    if idx > 0:
        score += bigram_log_prob(
            words[idx - 1], word,
            bigram_counts, unigram_counts, vocab_size, k,
        )
    if idx < len(words) - 1:
        score += bigram_log_prob(
            word, words[idx + 1],
            bigram_counts, unigram_counts, vocab_size, k,
        )
    return score


def check_grammar_window(
    window: list[str],
    lm: "NgramLM",
    corrector: "SpellingCorrector",
    q3_bigram,
    q3_unigram,
    q3_vocab_size: int,
    q3_k: float,
    q3_vocab: set[str],
    ppl_threshold: float = PERPLEXITY_THRESHOLD,
    real_word_margin: float = REAL_WORD_MARGIN,
) -> GrammarResult:
    """Run the GRAMMAR-ALERT + real-word check on a sliding window.

    Parameters
    ----------
    window        : List of the last N clean (lowercase, alpha) word tokens.
    lm            : Shared Q1 English trigram LM.
    corrector     : Q3 SpellingCorrector (for candidate generation).
    q3_bigram     : Q3 bigram counts.
    q3_unigram    : Q3 unigram counts.
    q3_vocab_size : Q3 vocabulary size.
    q3_k          : Q3 add-k smoothing constant.
    q3_vocab      : Q3 vocabulary set.
    """
    t0 = time.perf_counter()

    # Part 1: perplexity check
    ppl = _window_perplexity(window, lm)
    grammar_fired = (ppl > ppl_threshold) and not math.isinf(ppl)

    # Part 2: real-word error check
    real_word_fixes: list[RealWordFix] = []
    for idx, word in enumerate(window):
        if word not in q3_vocab:
            continue  # only real-word errors

        orig_score = _bigram_score_word(
            window, idx, word,
            q3_bigram, q3_unigram, q3_vocab_size, q3_k,
        )

        # Generate edit-distance-1 candidates via SymDel (Q3 Method B)
        res = corrector.correct_realword(window, idx)
        if res["changed"]:
            score_gain = (res["best_candidate_score"] or 0.0) - orig_score
            if score_gain > real_word_margin:
                real_word_fixes.append(
                    RealWordFix(
                        index=idx,
                        original=word,
                        suggestion=res["corrected"],
                        score_gain=score_gain,
                    )
                )

    fired = grammar_fired or bool(real_word_fixes)
    latency = (time.perf_counter() - t0) * 1000

    return GrammarResult(
        fired=fired,
        ppl=ppl,
        threshold=ppl_threshold,
        real_word_fixes=real_word_fixes,
        window=list(window),
        latency_ms=latency,
    )
