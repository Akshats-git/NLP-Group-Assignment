from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Container, Sequence

from q1.data import (
    MORPH_ABBREVIATIONS,
    MORPH_FEATURES,
    Sentence,
    coarse_tag,
    tag_features,
)
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
    morph: bool = False,
) -> TaggingScores:
    """Tag accuracy over gold-segmented words.

    ``known_vocab`` is the training vocabulary; supply it to get the
    known/unknown breakdown.  ``morph=True`` scores against the Part 3 tags
    (``NOUN-Fem-Sg``) instead of the plain POS tags -- a strictly harder task,
    so the two numbers are *not* comparable; project the morphology-aware
    predictions with ``coarse_tag`` and score them here with ``morph=False``
    to put both models on the same decision.
    """
    scores = TaggingScores()
    for sentence, tags in zip(sentences, predictions):
        gold = sentence.morph_tags if morph else sentence.tags
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


# --------------------------------------------------------------------------
# Morphological agreement (Part 3)
# --------------------------------------------------------------------------
@dataclass
class AgreementScores:
    """Did the tagger reproduce grammatical agreement, not just the right tag?

    Scored over *adjacent gold token pairs that agree in the gold annotation*
    -- ``la casa``, ``casa roja``.  For each such pair two questions are asked,
    and they are different questions:

    * **reproduced** -- do the model's two tags agree *with each other* on the
      features the gold pair agrees on?  This is the agreement question: a pair
      tagged Masc-Sg/Masc-Sg where the gold is Fem-Sg/Fem-Sg still shows the
      model propagating a consistent gender across the pair.
    * **correct** -- did it also pick the right values?

    Reporting only the second would conflate "does not understand agreement"
    with "guessed the wrong gender for an unknown noun"; reporting only the
    first would let a model that tags everything Masc-Sg look perfect.  Both
    are given, per syntactic context, so the two failure modes stay separate.
    """

    n_pairs: int = 0
    n_reproduced: int = 0
    n_correct: int = 0
    #: pairs lost because a gold span was never recovered (pipeline mode only)
    n_unrecovered: int = 0
    #: context ("DET+NOUN") -> [pairs, reproduced, correct]
    by_context: dict[str, list[int]] = field(default_factory=dict)

    @property
    def reproduced_rate(self) -> float:
        return self.n_reproduced / self.n_pairs if self.n_pairs else 0.0

    @property
    def correct_rate(self) -> float:
        return self.n_correct / self.n_pairs if self.n_pairs else 0.0

    def rows(self, min_pairs: int = 1) -> list[tuple[str, int, float, float]]:
        """Per-context rows, most frequent context first."""
        rows = []
        for context, (pairs, reproduced, correct) in self.by_context.items():
            if pairs < min_pairs:
                continue
            rows.append((context, pairs, reproduced / pairs, correct / pairs))
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows


def _abbreviate(feature: str, value: str) -> str | None:
    return MORPH_ABBREVIATIONS.get(feature, {}).get(value)


def score_agreement(
    sentences: Sequence[Sentence],
    predictions: Sequence[Sequence[str]],
    features: Sequence[str] = MORPH_FEATURES,
    predicted_words: Sequence[Sequence[str]] | None = None,
) -> AgreementScores:
    """Agreement reproduction over adjacent gold pairs that agree in the gold.

    A pair contributes only if the gold tokens genuinely agree on at least one
    feature -- otherwise there is no agreement to reproduce, and scoring it
    would just be measuring tag accuracy again under another name.

    Two modes, and the difference is large enough to matter:

    * **gold segmentation** (``predicted_words=None``) -- ``predictions`` holds
      one tag per gold token.  This isolates the tagger.
    * **pipeline** (``predicted_words`` given) -- ``predictions`` holds one tag
      per *predicted* token, aligned to the gold by character span.  A pair
      whose gold spans the segmenter never recovered stays in the denominator
      and counts as not reproduced, because end to end that agreement really
      was lost.  ``n_unrecovered`` reports how many failures arose that way.
    """
    scores = AgreementScores()
    for index, (sentence, tags) in enumerate(zip(sentences, predictions)):
        tokens = sentence.tokens
        by_span: dict[tuple[int, int], str] | None = None
        if predicted_words is not None:
            by_span = dict(zip(spans_from_words(predicted_words[index]), tags))
        elif len(tags) != len(tokens):
            raise ValueError(
                f"{sentence.sent_id}: {len(tags)} tags for {len(tokens)} gold tokens"
            )
        for i in range(len(tokens) - 1):
            left, right = tokens[i], tokens[i + 1]
            shared: dict[str, str] = {}
            for feature in features:
                left_value = left.feature(feature)
                right_value = right.feature(feature)
                if left_value and left_value == right_value:
                    abbreviation = _abbreviate(feature, left_value)
                    if abbreviation:
                        shared[feature] = abbreviation
            if not shared:
                continue

            unrecovered = False
            if by_span is None:
                left_tag, right_tag = tags[i], tags[i + 1]
            else:
                left_tag = by_span.get(sentence.gold_spans[i])
                right_tag = by_span.get(sentence.gold_spans[i + 1])
                unrecovered = left_tag is None or right_tag is None
                left_tag, right_tag = left_tag or "", right_tag or ""

            predicted_left = tag_features(left_tag)
            predicted_right = tag_features(right_tag)
            reproduced = not unrecovered and all(
                predicted_left.get(feature) is not None
                and predicted_left.get(feature) == predicted_right.get(feature)
                for feature in shared
            )
            correct = reproduced and all(
                predicted_left.get(feature) == value for feature, value in shared.items()
            )

            context = f"{left.upos}+{right.upos}"
            counts = scores.by_context.setdefault(context, [0, 0, 0])
            counts[0] += 1
            counts[1] += int(reproduced)
            counts[2] += int(correct)
            scores.n_pairs += 1
            scores.n_reproduced += int(reproduced)
            scores.n_correct += int(correct)
            scores.n_unrecovered += int(unrecovered)
    return scores


@dataclass
class AgreementBias:
    """One row of "does the transition model itself prefer agreement?".

    Read directly out of ``P(next tag | context)`` rather than inferred from
    output, so it says what the model believes, not merely what it produced.
    """

    context: tuple[str, ...]
    agreeing: float            # next tag marks both features, both match
    clashing: float            # next tag marks both features, at least one differs
    underspecified: float      # next tag does not mark the features at all

    @property
    def ratio(self) -> float:
        return self.agreeing / self.clashing if self.clashing else float("inf")


def agreement_bias(
    tagger,
    head: str = "NOUN",
    dependent: str = "ADJ",
    determiner: str = "DET",
    limit: int = 4,
) -> list[AgreementBias]:
    """How much more likely is an agreeing continuation than a clashing one?

    For each fully specified head tag (``NOUN-Fem-Sg``), the mass over
    ``dependent`` tags is split three ways in the context (matching determiner,
    that head).  The three-way split is the point: an adjective that does not
    mark gender at all (Spanish *grande*, ``ADJ-Sg``) is **underspecified, not
    disagreeing**, and counting it as a clash understates the effect by an order
    of magnitude.
    """
    heads = [t for t in tagger.tags
             if coarse_tag(t) == head and len(tag_features(t)) == 2]
    heads.sort(key=lambda t: tagger.tag_count[t], reverse=True)

    rows: list[AgreementBias] = []
    for head_tag in heads[:limit]:
        gold = tag_features(head_tag)
        context = (f"{determiner}-{gold['Gender']}-{gold['Number']}", head_tag)
        agreeing = clashing = underspecified = 0.0
        for tag in tagger.tags:
            if coarse_tag(tag) != dependent:
                continue
            probability = tagger.transition_prob(tag, context)
            predicted = tag_features(tag)
            if len(predicted) < 2:
                underspecified += probability
            elif predicted == gold:
                agreeing += probability
            else:
                clashing += probability
        rows.append(AgreementBias(context, agreeing, clashing, underspecified))
    return rows
