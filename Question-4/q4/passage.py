"""
passage.py — Q4 Random Passage Sampler + Merge-Token Generator

Randomly samples a paragraph (5–8 contiguous sentences) from an NLTK corpus
and optionally injects merged tokens (dropped spaces) to exercise Q1's
segmentation model.

Merge probability p = 0.08
---------------------------
Justification:  At a typical sentence length of ~20 words, each pair of
adjacent words is independently merged with probability p.  With 19 pairs per
sentence the expected count of merges is 19 × 0.08 ≈ 1.5 merges per sentence,
meaning about 78% of sentences contain at least one merge.  This is enough to
exercise the segmenter on almost every sentence while keeping the vast majority
of tokens (>90%) intact and the passage readable.  Values below 0.05 produce
too few merges to demonstrate the segmenter; values above 0.15 make the text
difficult to follow in the UI.
"""
from __future__ import annotations

import random
from typing import Iterator

import nltk

# Default corpora to sample from (tried in order until one loads)
_CORPUS_NAMES = ["gutenberg", "brown", "reuters"]

MERGE_PROB: float = 0.08  # p = 0.08, see module docstring
MIN_SENTENCES: int = 5
MAX_SENTENCES: int = 8


def _ensure_corpus(name: str) -> bool:
    """Return True if ``name`` is available (download if needed, silent)."""
    try:
        corpus = getattr(nltk.corpus, name)
        # Try to access — will raise LookupError if not downloaded.
        _ = corpus.fileids()
        return True
    except Exception:
        try:
            nltk.download(name, quiet=True)
            return True
        except Exception:
            return False


def _sentences_from_corpus(name: str, rng: random.Random) -> list[list[str]] | None:
    """Return a list of token lists from a random file in ``name`` corpus."""
    corpus = getattr(nltk.corpus, name)
    try:
        fileids = corpus.fileids()
    except Exception:
        return None
    fileid = rng.choice(fileids)
    try:
        # brown / gutenberg expose .sents(); reuters exposes .sents(fileid)
        sents = corpus.sents(fileid)
        return [list(s) for s in sents]
    except Exception:
        return None


def sample_passage(
    seed: int | None = None,
    n_sentences: int | None = None,
) -> tuple[list[list[str]], str]:
    """Sample a random paragraph of 5–8 sentences.

    Returns
    -------
    (sentences, corpus_name)
        ``sentences`` : each sentence is a list of space-delimited token strings
        (punctuation may be attached to words as in raw NLTK corpora — the
        caller needs to handle that).
        ``corpus_name`` : which NLTK corpus was used.
    """
    # Use a time-based seed in production (different every run), but allow an
    # explicit seed for deterministic debugging.
    rng = random.Random(seed)

    all_sents: list[list[str]] | None = None
    used_corpus = "unknown"

    for corpus_name in _CORPUS_NAMES:
        if not _ensure_corpus(corpus_name):
            continue
        all_sents = _sentences_from_corpus(corpus_name, rng)
        if all_sents and len(all_sents) >= MAX_SENTENCES:
            used_corpus = corpus_name
            break

    if not all_sents or len(all_sents) < MIN_SENTENCES:
        # Absolute fallback: a hard-coded short passage so the app never crashes.
        return _fallback_passage(), "fallback"

    n = n_sentences or rng.randint(MIN_SENTENCES, MAX_SENTENCES)
    n = min(n, len(all_sents))
    start = rng.randint(0, max(0, len(all_sents) - n))
    return all_sents[start : start + n], used_corpus


def _fallback_passage() -> list[list[str]]:
    return [
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
        ["She", "said", "that", "the", "book", "was", "on", "the", "table", "."],
        ["The", "government", "announced", "a", "new", "plan", "last", "week", "."],
        ["He", "walked", "slowly", "towards", "the", "old", "house", "."],
        ["The", "children", "played", "happily", "in", "the", "garden", "."],
    ]


def inject_merges(
    tokens: list[str],
    p: float = MERGE_PROB,
    rng: random.Random | None = None,
) -> list[str]:
    """Randomly merge adjacent tokens (drop the space between them).

    With probability ``p`` between each consecutive pair, the two words are
    concatenated into a single string.  This simulates a real typist who
    occasionally fails to hit the spacebar in time, and gives Q1's segmentation
    model a genuine job to do (without merges the passage is already correctly
    space-delimited and segmentation would be a no-op).

    Parameters
    ----------
    tokens:
        A flat list of whitespace-delimited word-tokens for a single sentence.
    p:
        Merge probability (default 0.08 — see module docstring for justification).
    rng:
        Optional ``random.Random`` instance.  A fresh instance is used if None.

    Returns
    -------
    list[str]
        A new list of tokens, some of which may be concatenated pairs.
        The total character count is conserved (no characters are dropped).

    Examples
    --------
    >>> inject_merges(["the", "cat", "sat"], p=1.0)
    ['thecatsat']
    >>> inject_merges(["the", "cat", "sat"], p=0.0)
    ['the', 'cat', 'sat']
    """
    if rng is None:
        rng = random.Random()
    if not tokens:
        return list(tokens)

    result: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if rng.random() < p:
            # Merge: append to the last token without a space.
            result[-1] = result[-1] + tok
        else:
            result.append(tok)
    return result


def passage_token_stream(
    passage: list[list[str]],
    p: float = MERGE_PROB,
    seed: int | None = None,
) -> Iterator[tuple[str, bool]]:
    """Yield ``(token, is_sentence_end)`` pairs with optional merge injection.

    ``is_sentence_end`` is True for the last token of each sentence, so the
    caller can trigger grammar checks at sentence boundaries if desired.

    Parameters
    ----------
    passage:
        List of sentences; each sentence is a list of string tokens.
    p:
        Merge probability passed to :func:`inject_merges`.
    seed:
        Optional RNG seed (for debug reproducibility).
    """
    rng = random.Random(seed)
    for sentence in passage:
        merged = inject_merges(sentence, p=p, rng=rng)
        for i, token in enumerate(merged):
            yield token, (i == len(merged) - 1)
