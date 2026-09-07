"""
segmentation.py — Q4 Segmentation Alert Wrapper

Wraps Q1's beam-search decoder for per-token segmentation checks.

SEGMENT-ALERT logic:
    If a token is not found in Q1's vocabulary as a single intact word
    (or is unusually long — longer than max_word_len / 2), run Q1's
    trained English joint beam-search decoder restricted to that token.
    Compare the log-score of the split hypothesis against the log-score of
    treating the token as a single (possibly OOV) word.  If the split wins,
    fire a SEGMENT-ALERT and return the split words + their POS tags.

The decoder is called with the SAME settings that were tuned during Q1
(max_word_len, unk_penalty, beam_width) — loaded directly from the saved
DecoderConfig rather than being re-chosen here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

# Ensure Q1 is importable
_Q1_ROOT = Path(__file__).resolve().parents[2] / "Question-1"
if str(_Q1_ROOT) not in sys.path:
    sys.path.insert(0, str(_Q1_ROOT))

from q1.lm import NgramLM, BOS, EOS
from q1.segment import DecoderConfig, decode_segmentation
from q1.tagger import HMMTagger


class SegmentResult(NamedTuple):
    fired: bool                   # True → SEGMENT-ALERT
    original_token: str
    words: tuple[str, ...]        # split words (len==1 means no split)
    pos_tags: tuple[str, ...]     # POS tags for each word
    single_score: float           # log-score treating token as one word
    split_score: float            # log-score of split
    latency_ms: float


def _single_token_score(token: str, lm: NgramLM) -> float:
    """Log-probability of treating ``token`` as a single word."""
    # Pad with BOS context (empty history) + score EOS — matches what the
    # beam decoder uses internally for a one-word sequence.
    context = (BOS,) * (lm.order - 1)
    return lm.logprob(token, context) + lm.logprob(EOS, (token,))


def _split_score(words: tuple[str, ...], lm: NgramLM) -> float:
    """Full sentence log-probability for the split hypothesis."""
    return lm.sentence_logprob(words)


def check_token_segmentation(
    token: str,
    lm: NgramLM,
    config: DecoderConfig,
    tagger: HMMTagger,
    vocab: set[str],
) -> SegmentResult:
    """Run the segmentation check for a single token.

    The token is checked if:
        - it is not in the vocabulary as a single word, OR
        - it is longer than ``config.max_word_len / 2`` (suggests a merge)

    If neither condition holds, the call returns immediately with
    ``fired=False`` to keep per-token latency minimal for normal words.

    Parameters
    ----------
    token   : The raw surface token (already normalised to lowercase by caller).
    lm      : Q1 English NgramLM (shared).
    config  : Q1 DecoderConfig with tuned max_word_len / beam_width.
    tagger  : Q1 HMMTagger.
    vocab   : Q1 training vocabulary.
    """
    t0 = time.perf_counter()

    cleaned = token.lower().strip()
    # Words with non-alphabetic characters are not run through the segmenter.
    alpha = "".join(ch for ch in cleaned if ch.isalpha())
    if not alpha:
        return SegmentResult(
            fired=False,
            original_token=token,
            words=(token,),
            pos_tags=("X",),
            single_score=0.0,
            split_score=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # Fast path: word is in vocab and not unusually long → no alert.
    long_threshold = max(8, config.max_word_len // 2)
    if alpha in vocab and len(alpha) <= long_threshold:
        return SegmentResult(
            fired=False,
            original_token=token,
            words=(alpha,),
            pos_tags=(tagger.tag((alpha,))[0],),
            single_score=0.0,
            split_score=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # Run the beam-search decoder on ``alpha`` as a single "unsegmented" string.
    split_words = decode_segmentation(alpha, lm, config)

    if len(split_words) <= 1:
        # Decoder cannot split it further.
        pos = tagger.tag((alpha,))
        latency = (time.perf_counter() - t0) * 1000
        return SegmentResult(
            fired=False,
            original_token=token,
            words=(alpha,),
            pos_tags=pos,
            single_score=_single_token_score(alpha, lm),
            split_score=_single_token_score(alpha, lm),
            latency_ms=latency,
        )

    s_single = _single_token_score(alpha, lm)
    s_split = _split_score(split_words, lm)
    fired = s_split > s_single

    pos_tags = tagger.tag(split_words) if fired else (tagger.tag((alpha,))[0],)
    final_words = split_words if fired else (alpha,)

    return SegmentResult(
        fired=fired,
        original_token=token,
        words=final_words,
        pos_tags=pos_tags,
        single_score=s_single,
        split_score=s_split,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
