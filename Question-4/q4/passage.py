"""
passage.py - Passage Sampler and Token Merger

Samples sentences from NLTK corpora and simulates space-bar typing errors
by merging adjacent tokens with probability p.
"""
from __future__ import annotations

import random
from typing import Iterator

import nltk

CORPORA = ["gutenberg", "brown", "reuters"]
MERGE_PROB = 0.08


def _ensure_corpus(name: str) -> bool:
    try:
        corpus = getattr(nltk.corpus, name)
        _ = corpus.fileids()
        return True
    except Exception:
        try:
            nltk.download(name, quiet=True)
            return True
        except Exception:
            return False


def sample_passage(seed: int | None = None, n_sentences: int = 6) -> tuple[list[list[str]], str]:
    """Sample a short passage (5-8 sentences) from available NLTK corpora."""
    rng = random.Random(seed)

    # Shuffled rather than tried in order, otherwise the first corpus in the
    # list answers every call and the passage is never drawn from the other two.
    order = list(CORPORA)
    rng.shuffle(order)

    for corpus_name in order:
        if not _ensure_corpus(corpus_name):
            continue
        corpus = getattr(nltk.corpus, corpus_name)
        try:
            fileids = corpus.fileids()
            fileid = rng.choice(fileids)
            sents = [list(s) for s in corpus.sents(fileid)]
            if len(sents) >= n_sentences:
                start = rng.randint(0, max(0, len(sents) - n_sentences))
                return sents[start : start + n_sentences], corpus_name
        except Exception:
            continue

    # Fallback sentences if NLTK corpus read fails
    fallback = [
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
        ["She", "said", "that", "the", "book", "was", "on", "the", "table", "."],
        ["The", "government", "announced", "a", "new", "plan", "last", "week", "."],
        ["He", "walked", "slowly", "towards", "the", "old", "house", "."],
        ["The", "children", "played", "happily", "in", "the", "garden", "."],
    ]
    return fallback, "fallback"


def inject_merges(tokens: list[str], p: float = MERGE_PROB, rng: random.Random | None = None) -> list[str]:
    """Simulate missing space errors by concatenating tokens with probability p."""
    if rng is None:
        rng = random.Random()
    if not tokens:
        return []

    res = [tokens[0]]
    for tok in tokens[1:]:
        if rng.random() < p:
            res[-1] += tok
        else:
            res.append(tok)
    return res


def passage_token_stream(passage: list[list[str]], p: float = MERGE_PROB, seed: int | None = None) -> Iterator[tuple[str, bool]]:
    """Yields (token, is_sentence_end) tuples for stream simulation."""
    rng = random.Random(seed)
    for sent in passage:
        merged = inject_merges(sent, p=p, rng=rng)
        for i, token in enumerate(merged):
            yield token, (i == len(merged) - 1)
