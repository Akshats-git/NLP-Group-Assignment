"""
pipeline.py - Per-token live checks and the running document (Q4 Parts 1, 4, 5)

A single place where an incoming token is put through the segmentation check and
then the spelling check. The Streamlit app, the speed benchmark and the
end-of-passage analysis all call this, so the text that gets analysed at the end
is exactly the text the live alerts produced, and the benchmark times the same
work the editor does.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from q4.grammar import GrammarResult, check_grammar_window
from q4.segmentation import check_token_segmentation
from q4.spelling import check_token_spelling

SENTENCE_ENDINGS = ".!?"
_CLOSING_MARKS = "\"')]}"


def clean_word(token: str) -> str:
    """Lowercased alphabetic core of a token, which is all the models accept."""
    return "".join(c for c in token.lower() if c.isalpha())


def ends_sentence(token: str) -> bool:
    """True when a raw token carries sentence-final punctuation.

    A single letter before the full stop is an initial rather than the end of a
    sentence, which keeps names like "robert e. lee" in one piece.
    """
    if not token.rstrip(_CLOSING_MARKS).endswith(tuple(SENTENCE_ENDINGS)):
        return False
    return len(clean_word(token)) > 1


@dataclass
class Alert:
    kind: str
    token: str
    detail: str
    latency_ms: float


@dataclass
class TokenOutcome:
    """What the live checks made of one typed token."""

    raw_token: str
    words: list[str]
    tags: list[str]
    alerts: list[Alert] = field(default_factory=list)
    merges_resolved: int = 0
    spelling_fixes: int = 0
    latency_ms: float = 0.0
    ends_sentence: bool = False


def check_token(token: str, models: dict[str, Any]) -> TokenOutcome:
    """Run the segmentation check and then the spelling check on one token."""
    start = time.perf_counter()
    boundary = ends_sentence(token)
    word = clean_word(token)

    if not word:
        return TokenOutcome(
            raw_token=token,
            words=[],
            tags=[],
            latency_ms=(time.perf_counter() - start) * 1000,
            ends_sentence=boundary,
        )

    alerts: list[Alert] = []

    segment = check_token_segmentation(
        word,
        lm=models["q1_lm"],
        config=models["q1_config"],
        tagger=models["q1_tagger"],
        vocab=models["q1_vocab"],
    )
    words = list(segment.words) or [word]
    tags = list(segment.pos_tags) or ["X"] * len(words)
    merges = 0

    if segment.fired:
        merges = len(words) - 1
        alerts.append(
            Alert(
                kind="SEGMENT",
                token=token,
                detail=f"Split '{token}' -> {' + '.join(words)} [{', '.join(tags)}]",
                latency_ms=segment.latency_ms,
            )
        )

    fixes = 0
    for i, candidate in enumerate(words):
        spelling = check_token_spelling(candidate, models["corrector"], models["q3_vocab"])
        if spelling.fired:
            alerts.append(
                Alert(
                    kind="SPELL",
                    token=candidate,
                    detail=f"Spelling: '{candidate}' -> suggest '{spelling.suggestion}'",
                    latency_ms=spelling.latency_ms,
                )
            )
            words[i] = spelling.suggestion
            fixes += 1

    if fixes:
        # The tags were predicted from the misspelled forms, so they are worth
        # nothing once the words have changed underneath them.
        tags = list(models["q1_tagger"].tag(tuple(words)))

    return TokenOutcome(
        raw_token=token,
        words=words,
        tags=tags,
        alerts=alerts,
        merges_resolved=merges,
        spelling_fixes=fixes,
        latency_ms=(time.perf_counter() - start) * 1000,
        ends_sentence=boundary,
    )


def check_window(
    window: list[str], models: dict[str, Any], ppl_threshold: float | None = None
) -> GrammarResult:
    """Run the grammar and real-word check that fires every N words.

    The perplexity comes from Q4's own add-k trigram, and the threshold defaults
    to the level that only 5 percent of clean Brown windows exceed, which is
    where the calibration in q4/ngram_lm.py put it.
    """
    if ppl_threshold is None:
        ppl_threshold = models["q4_floors"]["window_ppl_p95"]

    return check_grammar_window(
        window,
        lm=models["q4_trigram"],
        corrector=models["corrector"],
        q3_bigram=models["q3_bigram"],
        q3_unigram=models["q3_unigram"],
        q3_vocab_size=models["q3_vocab_size"],
        q3_k=models["q3_k"],
        q3_vocab=models["q3_vocab"],
        ppl_threshold=ppl_threshold,
    )


@dataclass
class DocumentSentence:
    """One finished sentence of the corrected stream, with its repair counts."""

    words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    merges_resolved: int = 0
    spelling_fixes: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def __len__(self) -> int:
        return len(self.words)


class LiveDocument:
    """The corrected token stream as it builds up, cut into sentences.

    Part 4 has to score the passage after segmentation and spelling have already
    been applied, and it has to report how many merges and corrections landed in
    each sentence, so those counts are carried here as the tokens arrive rather
    than reconstructed afterwards.
    """

    def __init__(self) -> None:
        self.sentences: list[DocumentSentence] = []
        self._current = DocumentSentence()

    def add(self, outcome: TokenOutcome, force_sentence_end: bool = False) -> None:
        self._current.words.extend(outcome.words)
        self._current.tags.extend(outcome.tags)
        self._current.merges_resolved += outcome.merges_resolved
        self._current.spelling_fixes += outcome.spelling_fixes

        if (outcome.ends_sentence or force_sentence_end) and self._current.words:
            self.sentences.append(self._current)
            self._current = DocumentSentence()

    def close(self) -> None:
        """Finish the sentence still being typed, if there is one."""
        if self._current.words:
            self.sentences.append(self._current)
            self._current = DocumentSentence()

    def finished_sentences(self) -> list[DocumentSentence]:
        """Every sentence including the one in progress, without closing it."""
        if self._current.words:
            return self.sentences + [self._current]
        return list(self.sentences)

    def words(self) -> list[str]:
        """Flat corrected word stream, which is what the grammar window reads."""
        out: list[str] = []
        for sentence in self.finished_sentences():
            out.extend(sentence.words)
        return out

    @property
    def word_count(self) -> int:
        return len(self.words())
