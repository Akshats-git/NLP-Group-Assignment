from __future__ import annotations

from dataclasses import dataclass, field
from typing import Container, Sequence

from q1.lm import BOS, EOS, NEG_INF, NgramLM


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Every hyper-parameter of the Q1 decoders, in one loadable object.

    Question 4 is required to run Q1's decoder "using the same maximum word
    length, alpha, beta, and beam-width settings selected during Q1", so the
    tuned values are persisted as this object alongside the models rather than
    being re-chosen by hand.
    """

    max_word_len: int = 20
    beam_width: int | None = 12
    unk_penalty: float = 0.0
    alpha: float = 1.0          # weight on the segmentation (word LM) score
    beta: float = 1.0           # weight on the tagging (HMM) score
    restrict_to_vocab: bool = False

    def replace(self, **changes) -> "DecoderConfig":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


@dataclass
class _Hypothesis:
    score: float
    words: tuple[str, ...]
    history: tuple[str, ...]           # last (order-1) words, the DP state


def spans_from_words(words: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Character spans of a word sequence, matching ``Sentence.gold_spans``."""
    spans = []
    cursor = 0
    for word in words:
        spans.append((cursor, cursor + len(word)))
        cursor += len(word)
    return tuple(spans)


def decode_segmentation(
    chars: str,
    lm: NgramLM,
    config: DecoderConfig = DecoderConfig(),
) -> tuple[str, ...]:
    """Most probable segmentation of ``chars`` under ``lm``.

    Returns the word sequence; use ``spans_from_words`` for character offsets.
    """
    n = len(chars)
    if n == 0:
        return ()

    order = lm.order
    max_len = min(config.max_word_len, n)
    # columns[i] maps a DP state (the last order-1 words) to the best
    # hypothesis reaching character position i in that state.
    columns: list[dict[tuple[str, ...], _Hypothesis]] = [{} for _ in range(n + 1)]
    start_history = (BOS,) * (order - 1)
    columns[0][start_history] = _Hypothesis(0.0, (), start_history)

    # Memoise LM lookups: the same (word, history) pair recurs constantly
    # across columns, and the recursive interpolation is the inner-loop cost.
    cache: dict[tuple[str, tuple[str, ...]], float] = {}

    def score_of(word: str, history: tuple[str, ...]) -> float:
        key = (word, history)
        value = cache.get(key)
        if value is None:
            value = lm.logprob(word, history)
            if word not in lm.vocab:
                value += config.unk_penalty
            cache[key] = value
        return value

    for i in range(n):
        column = columns[i]
        if not column:
            continue
        # Prune on arrival: every hypothesis that can reach position i has
        # already been added, because all earlier positions are processed.
        hypotheses = list(column.values())
        if config.beam_width is not None and len(hypotheses) > config.beam_width:
            hypotheses.sort(key=lambda h: h.score, reverse=True)
            hypotheses = hypotheses[: config.beam_width]

        for length in range(1, min(max_len, n - i) + 1):
            word = chars[i : i + length]
            if config.restrict_to_vocab and word not in lm.vocab:
                continue
            target = columns[i + length]
            for hypothesis in hypotheses:
                score = hypothesis.score + score_of(word, hypothesis.history)
                if score == NEG_INF:
                    continue
                history = (hypothesis.history + (word,))[-(order - 1) :] if order > 1 else ()
                incumbent = target.get(history)
                if incumbent is None or score > incumbent.score:
                    target[history] = _Hypothesis(
                        score, hypothesis.words + (word,), history
                    )

    final = columns[n]
    if not final:
        # Only reachable if max_word_len < the length of an unsplittable tail.
        return (chars,)
    best = max(
        final.values(), key=lambda h: h.score + score_of(EOS, h.history)
    )
    return best.words


def segment_corpus(
    sentences: Sequence,
    lm: NgramLM,
    config: DecoderConfig = DecoderConfig(),
) -> list[tuple[str, ...]]:
    """Segment every sentence's ``chars``; convenience wrapper for evaluation."""
    return [decode_segmentation(s.chars, lm, config) for s in sentences]


# --------------------------------------------------------------------------
# Part 4 -- baseline segmenter
# --------------------------------------------------------------------------
def greedy_longest_match(
    chars: str,
    vocabulary: Container[str],
    max_word_len: int = DecoderConfig().max_word_len,
) -> tuple[str, ...]:
    """Maximum matching: at each position take the longest word in ``vocabulary``.

    The baseline the DP decoder has to beat.  Its defining weakness is that the
    choice is *local and final* -- there is no way to reconsider a bite once
    taken, so one wrong early match corrupts every boundary after it.  The DP
    decoder differs precisely in scoring whole segmentations instead.

    A position where no vocabulary word matches emits a single character, the
    standard fallback: it keeps the output a partition of the input (so it can
    be scored by exactly the same span metric) and it cannot silently drop
    characters.

    ``vocabulary`` should be the *full* training vocabulary, hapax words
    included, even though the language model itself discards those -- the
    comparison is more honest for handing the baseline the larger lexicon
    rather than the model's own.
    """
    words: list[str] = []
    position, n = 0, len(chars)
    while position < n:
        longest = min(max_word_len, n - position)
        length = 1
        for length in range(longest, 0, -1):
            if length == 1 or chars[position : position + length] in vocabulary:
                break
        words.append(chars[position : position + length])
        position += length
    return tuple(words)


def greedy_corpus(
    sentences: Sequence,
    vocabulary: Container[str],
    max_word_len: int = DecoderConfig().max_word_len,
) -> list[tuple[str, ...]]:
    return [greedy_longest_match(s.chars, vocabulary, max_word_len) for s in sentences]
