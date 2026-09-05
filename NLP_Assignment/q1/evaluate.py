from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from q1.data import Sentence
from q1.segment import spans_from_words

# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
@dataclass
class SegmentationScores:
    n_sentences: int = 0
    n_exact: int = 0                 # sentences segmented perfectly
    gold_tokens: int = 0
    pred_tokens: int = 0
    matched_tokens: int = 0          # predicted spans that are also gold spans
    gold_boundaries: int = 0
    pred_boundaries: int = 0
    matched_boundaries: int = 0

    @property
    def token_precision(self) -> float:
        return self.matched_tokens / self.pred_tokens if self.pred_tokens else 0.0

    @property
    def token_recall(self) -> float:
        return self.matched_tokens / self.gold_tokens if self.gold_tokens else 0.0

    @property
    def token_f1(self) -> float:
        p, r = self.token_precision, self.token_recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def boundary_f1(self) -> float:
        p = self.matched_boundaries / self.pred_boundaries if self.pred_boundaries else 0.0
        r = self.matched_boundaries / self.gold_boundaries if self.gold_boundaries else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def sentence_accuracy(self) -> float:
        return self.n_exact / self.n_sentences if self.n_sentences else 0.0


def score_segmentation(
    sentences: Sequence[Sentence], predictions: Sequence[Sequence[str]]
) -> SegmentationScores:
    scores = SegmentationScores()
    for sentence, words in zip(sentences, predictions):
        gold_spans = set(sentence.gold_spans)
        pred_spans = spans_from_words(words)
        scores.n_sentences += 1
        scores.n_exact += int(tuple(words) == sentence.words)
        scores.gold_tokens += len(sentence.gold_spans)
        scores.pred_tokens += len(pred_spans)
        scores.matched_tokens += sum(1 for s in pred_spans if s in gold_spans)

        gold_cuts = {end for _, end in sentence.gold_spans[:-1]}
        pred_cuts = {end for _, end in pred_spans[:-1]}
        scores.gold_boundaries += len(gold_cuts)
        scores.pred_boundaries += len(pred_cuts)
        scores.matched_boundaries += len(gold_cuts & pred_cuts)
    return scores
