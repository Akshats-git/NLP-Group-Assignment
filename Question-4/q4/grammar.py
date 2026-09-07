"""
grammar.py - Grammar and Real-Word Error Checker

Triggers every N words (default N=10) to compute perplexity over the window
and check for real-word substitution errors using bigram scoring.
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

Q3_DIR = Path(__file__).resolve().parents[2] / "Question-3" / "q3_spelling_corrector"
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from corpus_models import bigram_log_prob

TRIGGER_N = 10
PERPLEXITY_THRESHOLD = 300.0
REAL_WORD_MARGIN = 2.0


class RealWordFix(NamedTuple):
    index: int
    original: str
    suggestion: str
    score_gain: float


class GrammarResult(NamedTuple):
    fired: bool
    ppl: float
    threshold: float
    real_word_fixes: list[RealWordFix]
    window: list[str]
    latency_ms: float


def _bigram_score_word(words: list[str], idx: int, word: str, bigram_counts, unigram_counts, vocab_size: int, k: float) -> float:
    score = 0.0
    if idx > 0:
        score += bigram_log_prob(words[idx - 1], word, bigram_counts, unigram_counts, vocab_size, k)
    if idx < len(words) - 1:
        score += bigram_log_prob(word, words[idx + 1], bigram_counts, unigram_counts, vocab_size, k)
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
    """Run perplexity and real-word error check on sliding window."""
    t0 = time.perf_counter()

    ppl = lm.perplexity([window]) if window else math.inf
    grammar_fired = (ppl > ppl_threshold) and not math.isinf(ppl)

    real_word_fixes: list[RealWordFix] = []
    for idx, word in enumerate(window):
        if word not in q3_vocab:
            continue

        orig_score = _bigram_score_word(window, idx, word, q3_bigram, q3_unigram, q3_vocab_size, q3_k)
        res = corrector.correct_realword(window, idx)
        if res["changed"]:
            gain = (res["best_candidate_score"] or 0.0) - orig_score
            if gain > real_word_margin:
                real_word_fixes.append(RealWordFix(idx, word, res["corrected"], gain))

    fired = grammar_fired or bool(real_word_fixes)
    return GrammarResult(
        fired=fired,
        ppl=ppl,
        threshold=ppl_threshold,
        real_word_fixes=real_word_fixes,
        window=list(window),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
