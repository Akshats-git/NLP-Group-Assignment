"""
segmentation.py - Segmentation Alert Wrapper

Uses Q1's beam-search segmentation decoder to detect and split merged tokens.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

Q1_DIR = Path(__file__).resolve().parents[2] / "Question-1"
if str(Q1_DIR) not in sys.path:
    sys.path.insert(0, str(Q1_DIR))

from q1.lm import NgramLM, BOS, EOS
from q1.segment import DecoderConfig, decode_segmentation
from q1.tagger import HMMTagger


class SegmentResult(NamedTuple):
    fired: bool
    original_token: str
    words: tuple[str, ...]
    pos_tags: tuple[str, ...]
    single_score: float
    split_score: float
    latency_ms: float


def _single_token_score(token: str, lm: NgramLM) -> float:
    context = (BOS,) * (lm.order - 1)
    return lm.logprob(token, context) + lm.logprob(EOS, (token,))


def check_token_segmentation(
    token: str,
    lm: NgramLM,
    config: DecoderConfig,
    tagger: HMMTagger,
    vocab: set[str],
) -> SegmentResult:
    """Check if token should be split using Q1 beam-search decoder."""
    t0 = time.perf_counter()

    cleaned = token.lower().strip()
    alpha = "".join(c for c in cleaned if c.isalpha())
    if not alpha:
        return SegmentResult(False, token, (token,), ("X",), 0.0, 0.0, (time.perf_counter() - t0) * 1000)

    # Skip decoder if word is in vocabulary and reasonable length
    if alpha in vocab and len(alpha) <= max(8, config.max_word_len // 2):
        pos = tagger.tag((alpha,))
        return SegmentResult(False, token, (alpha,), pos, 0.0, 0.0, (time.perf_counter() - t0) * 1000)

    split_words = decode_segmentation(alpha, lm, config)

    if len(split_words) <= 1:
        pos = tagger.tag((alpha,))
        s = _single_token_score(alpha, lm)
        return SegmentResult(False, token, (alpha,), pos, s, s, (time.perf_counter() - t0) * 1000)

    s_single = _single_token_score(alpha, lm)
    s_split = lm.sentence_logprob(split_words)
    fired = s_split > s_single

    pos_tags = tagger.tag(split_words) if fired else tagger.tag((alpha,))
    words = split_words if fired else (alpha,)

    return SegmentResult(
        fired=fired,
        original_token=token,
        words=words,
        pos_tags=pos_tags,
        single_score=s_single,
        split_score=s_split,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
