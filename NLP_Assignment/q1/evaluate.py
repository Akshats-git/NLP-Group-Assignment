from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Container, Sequence

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


# --------------------------------------------------------------------------
# Tagging (Part 2), on gold segmentation
# --------------------------------------------------------------------------
@dataclass
class TaggingScores:
    """Accuracy of a tagger given correctly segmented words.

    The known/unknown split is the informative one: overall accuracy is
    dominated by frequent unambiguous words, so two taggers can differ by a
    fraction of a point overall while differing enormously on the OOV tail --
    which is where a morphologically rich language spends most of its mass.
    """

    n_tokens: int = 0
    n_correct: int = 0
    n_known: int = 0
    n_known_correct: int = 0
    n_unknown: int = 0
    n_unknown_correct: int = 0
    n_sentences: int = 0
    n_exact: int = 0
    confusions: Counter = field(default_factory=Counter)   # (gold, predicted)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_tokens if self.n_tokens else 0.0

    @property
    def known_accuracy(self) -> float:
        return self.n_known_correct / self.n_known if self.n_known else 0.0

    @property
    def unknown_accuracy(self) -> float:
        return self.n_unknown_correct / self.n_unknown if self.n_unknown else 0.0

    @property
    def oov_rate(self) -> float:
        return self.n_unknown / self.n_tokens if self.n_tokens else 0.0

    @property
    def sentence_accuracy(self) -> float:
        return self.n_exact / self.n_sentences if self.n_sentences else 0.0

    def top_confusions(self, n: int = 10) -> list[tuple[tuple[str, str], int]]:
        return self.confusions.most_common(n)


def score_tagging(
    sentences: Sequence[Sentence],
    predictions: Sequence[Sequence[str]],
    known_vocab: Container[str] | None = None,
) -> TaggingScores:
    """Tag accuracy over gold-segmented words.

    ``known_vocab`` is the training vocabulary; supply it to get the
    known/unknown breakdown.
    """
    scores = TaggingScores()
    for sentence, tags in zip(sentences, predictions):
        gold = sentence.tags
        if len(tags) != len(gold):
            raise ValueError(
                f"{sentence.sent_id}: {len(tags)} tags for {len(gold)} gold tokens; "
                "use score_pipeline for predictions over predicted segmentations"
            )
        scores.n_sentences += 1
        scores.n_exact += int(tuple(tags) == gold)
        for word, gold_tag, predicted in zip(sentence.words, gold, tags):
            scores.n_tokens += 1
            correct = predicted == gold_tag
            scores.n_correct += int(correct)
            if not correct:
                scores.confusions[(gold_tag, predicted)] += 1
            if known_vocab is not None:
                if word in known_vocab:
                    scores.n_known += 1
                    scores.n_known_correct += int(correct)
                else:
                    scores.n_unknown += 1
                    scores.n_unknown_correct += int(correct)
    return scores


# --------------------------------------------------------------------------
# End-to-end: segment, then tag
# --------------------------------------------------------------------------
@dataclass
class PipelineScores:
    """Accuracy over gold tokens when segmentation is predicted, not given.

    A predicted (span, tag) counts only if the span is exactly a gold span and
    the tag matches, which splits every error into one of two kinds:

    * **segmentation-induced** -- the gold token's character span was never
      recovered, so no tag decision was even made for it;
    * **genuine** -- the span was recovered and the tag is wrong.

    Keeping them apart is what tells you which half of the pipeline to fix.
    """

    n_gold_tokens: int = 0
    n_correct: int = 0
    n_span_correct: int = 0          # gold span recovered, tag ignored
    n_seg_induced_errors: int = 0
    n_genuine_errors: int = 0
    n_sentences: int = 0
    n_exact: int = 0                 # whole sentence: every span and tag right

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_gold_tokens if self.n_gold_tokens else 0.0

    @property
    def tag_accuracy_given_span(self) -> float:
        """Tag accuracy restricted to the tokens the segmenter got right."""
        return self.n_correct / self.n_span_correct if self.n_span_correct else 0.0

    @property
    def n_errors(self) -> int:
        return self.n_seg_induced_errors + self.n_genuine_errors

    @property
    def share_segmentation_induced(self) -> float:
        return self.n_seg_induced_errors / self.n_errors if self.n_errors else 0.0

    @property
    def sentence_accuracy(self) -> float:
        return self.n_exact / self.n_sentences if self.n_sentences else 0.0


def score_pipeline(
    sentences: Sequence[Sentence],
    predicted_words: Sequence[Sequence[str]],
    predicted_tags: Sequence[Sequence[str]],
) -> PipelineScores:
    """Score (segmentation, tagging) jointly against the gold tokens.

    Predictions need not have the same length as the gold: they are aligned by
    character span, which is well defined because both sides partition the same
    character string.
    """
    scores = PipelineScores()
    for sentence, words, tags in zip(sentences, predicted_words, predicted_tags):
        predicted = dict(zip(spans_from_words(words), tags))
        scores.n_sentences += 1
        sentence_exact = tuple(words) == sentence.words
        for span, gold_tag in zip(sentence.gold_spans, sentence.tags):
            scores.n_gold_tokens += 1
            predicted_tag = predicted.get(span)
            if predicted_tag is None:
                scores.n_seg_induced_errors += 1
                sentence_exact = False
                continue
            scores.n_span_correct += 1
            if predicted_tag == gold_tag:
                scores.n_correct += 1
            else:
                scores.n_genuine_errors += 1
                sentence_exact = False
        scores.n_exact += int(sentence_exact)
    return scores
