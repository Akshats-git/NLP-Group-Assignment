"""Correctness tests for the Q1 models.

Run:  .venv/bin/python scripts/test_q1_models.py

These check the properties the Part 1 segmenter and the Part 2 tagger silently
assume -- above all that the models are normalised: the segmenter compares
hypotheses containing different numbers of words and would otherwise acquire a
systematic bias towards long or short words, and the tagger's interpolated
transition model has to remain a distribution once unseen histories start
dropping out of the interpolation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q1.data import (
    Sentence,
    Token,
    brown_to_penn,
    coarse_tag,
    make_morph_tag,
    normalise_brown_tag,
    normalise_form,
    parse_feats,
    tag_features,
)
from q1.evaluate import (
    agreement_bias,
    score_agreement,
    score_pipeline,
    score_segmentation,
    score_tagging,
)
from q1.lm import CharLM, NgramLM
from q1.segment import (
    DecoderConfig,
    decode_segmentation,
    greedy_longest_match,
    spans_from_words,
)
from q1.tagger import (
    BOS_TAG,
    EOS_TAG,
    HMMTagger,
    MostFrequentTagger,
    TaggerConfig,
    tagged_pairs,
)

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(message)
    PASSED += 1
    print(f"  ok  {message}")


def toy_sentences() -> list[Sentence]:
    raw = [
        [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("mat", "NOUN")],
        [("the", "DET"), ("dog", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("log", "NOUN")],
        [("a", "DET"), ("cat", "NOUN"), ("ate", "VERB"), ("the", "DET"), ("mat", "NOUN")],
        [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("ran", "VERB")],
        [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB"), ("quickly", "ADV")],
        [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("quietly", "ADV")],
        [("a", "DET"), ("dog", "NOUN"), ("ate", "VERB"), ("slowly", "ADV")],
    ]
    return [
        Sentence.from_tokens([Token(w, t) for w, t in row], sent_id=f"toy-{i}")
        for i, row in enumerate(raw)
        for _ in range(25)
    ]


# --------------------------------------------------------------------------
def test_tagsets() -> None:
    check(normalise_brown_tag("NN-TL") == "NN", "Brown suffixes are stripped")
    check(normalise_brown_tag("PPSS+BER") == "PPSS", "contraction tags take the first tag")
    check(normalise_brown_tag("*") == "*", "the standalone negator tag survives")
    expected = ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN"]
    actual = [brown_to_penn(t) for t in
              ["AT", "JJ", "JJ", "NN", "VBZ", "IN", "AT", "JJ", "NN"]]
    check(actual == expected, "Brown->Penn reproduces the assignment's English example")


def test_normalisation() -> None:
    check(normalise_form("Don't") == "dont", "non-letters are stripped from inside")
    check(normalise_form("U.S.") == "us", "punctuation-only remainders collapse")
    check(normalise_form("123") == "", "numeric tokens are dropped")
    check(normalise_form("Hincapié") == "hincapié", "accents are preserved")


def test_sentence_invariants() -> None:
    sentence = Sentence.from_tokens([Token("la", "DET"), Token("casa", "NOUN")])
    check(sentence.chars == "lacasa", "chars is the concatenation of forms")
    check(sentence.gold_spans == ((0, 2), (2, 6)), "spans partition chars")
    check(sentence.gold_boundaries == frozenset({2}), "internal boundaries exclude the ends")
    check(spans_from_words(("la", "casa")) == sentence.gold_spans,
          "predicted and gold spans are computed identically")


def test_lm_normalisation() -> None:
    sentences = [s.words for s in toy_sentences()]
    for smoothing in ("witten_bell", "kneser_ney", "addk"):
        lm = NgramLM(3, smoothing=smoothing, unk_threshold=0).fit(sentences)
        for context in ((), ("the",), ("the", "cat"), ("zzz",)):
            total = lm.check_normalised(context)
            check(abs(total - 1.0) < 1e-6,
                  f"{smoothing}: P(.|{context}) sums to 1 ({total:.9f})")


def test_lm_open_vocabulary() -> None:
    lm = NgramLM(3, unk_threshold=0).fit([s.words for s in toy_sentences()])
    plausible = lm.logprob("cot", ["the"])      # English-shaped
    implausible = lm.logprob("xqzptr", ["the"])  # not English-shaped
    check(plausible > implausible,
          "the character model prefers word-shaped unknowns to noise")
    check(math.isfinite(plausible), "unknown words receive a finite score")
    char_lm = CharLM().fit(["cat", "cot", "cut"])
    check(char_lm.logprob("cat") > char_lm.logprob("ttt"),
          "CharLM scores seen character sequences higher")


def test_segmentation() -> None:
    lm = NgramLM(3, unk_threshold=0).fit([s.words for s in toy_sentences()])
    config = DecoderConfig(max_word_len=6, beam_width=None)
    check(decode_segmentation("thecatsatonthemat", lm, config)
          == ("the", "cat", "sat", "on", "the", "mat"),
          "exact DP recovers a familiar sentence")
    beam = decode_segmentation("thecatsatonthemat", lm, config.replace(beam_width=8))
    check(beam == ("the", "cat", "sat", "on", "the", "mat"),
          "beam search agrees with exact DP here")
    check(decode_segmentation("", lm, config) == (), "the empty string segments to ()")
    words = decode_segmentation("thedogran", lm, config)
    check("".join(words) == "thedogran", "output always re-concatenates to the input")


def test_segmentation_scoring() -> None:
    """Token F1 credits a predicted token only for an exactly recovered span."""
    gold = Sentence.from_tokens(
        [Token("the"), Token("quick"), Token("brown"), Token("fox")]
    )
    scores = score_segmentation([gold], [("thequick", "brown", "fox")])
    check(abs(scores.token_recall - 0.5) < 1e-9,
          "token recall counts only exactly recovered spans")
    check(abs(scores.token_precision - 2 / 3) < 1e-9,
          "token precision is over predicted spans")
    check(scores.sentence_accuracy == 0.0,
          "a sentence with any wrong boundary is not exact")


# --------------------------------------------------------------------------
# Part 4 -- baselines
# --------------------------------------------------------------------------
def test_greedy_baseline() -> None:
    vocabulary = {"the", "cat", "sat", "on", "mat", "a", "cats", "at"}
    check(greedy_longest_match("thecatsatonthemat", vocabulary)
          == ("the", "cats", "at", "on", "the", "mat"),
          "greedy takes the longest match, wrongly biting 'cats' out of 'cat sat'")
    check(greedy_longest_match("", vocabulary) == (),
          "the empty string segments to ()")
    words = greedy_longest_match("thexyzcat", vocabulary)
    check(words == ("the", "x", "y", "z", "cat"),
          "unmatched characters fall back to single characters")
    check("".join(words) == "thexyzcat",
          "the output is always a partition of the input")
    check(greedy_longest_match("catsat", vocabulary, max_word_len=3)
          == ("cat", "sat"),
          "max_word_len caps the bite, which here happens to help")


def test_greedy_is_beaten_by_dp() -> None:
    """The comparison Part 4 exists to make.

    The corpus is built so the trap is real: "cats" and "at" are both genuine
    vocabulary words, so longest-match bites "cats" out of "cat sat" -- while
    the trigram model, scoring the whole segmentation, prefers the split it has
    actually seen.
    """
    raw = (
        [["the", "cat", "sat", "on", "the", "mat"]] * 40
        + [["the", "cats", "ran"]] * 4
        + [["a", "cat", "sat", "at", "the", "mat"]] * 4
    )
    sentences = [
        Sentence.from_tokens([Token(w) for w in row]) for row in raw
    ]
    vocabulary = {w for s in sentences for w in s.words}
    check({"cats", "at"} <= vocabulary, "the trap words are in the vocabulary")

    lm = NgramLM(3, unk_threshold=0).fit([s.words for s in sentences])
    config = DecoderConfig(max_word_len=6, beam_width=None)
    gold = ("the", "cat", "sat", "on", "the", "mat")
    greedy = greedy_longest_match("thecatsatonthemat", vocabulary, 6)
    dp = decode_segmentation("thecatsatonthemat", lm, config)
    check(greedy == ("the", "cats", "at", "on", "the", "mat"),
          "greedy takes the locally longest bite and gets the sentence wrong")
    check(dp == gold,
          "the DP decoder scores whole segmentations and gets it right")


# --------------------------------------------------------------------------
# Part 2 -- tagging
# --------------------------------------------------------------------------
def toy_tagger(order: int = 3) -> HMMTagger:
    return HMMTagger(TaggerConfig(order=order)).fit(tagged_pairs(toy_sentences()))


def test_tagger_distributions() -> None:
    tagger = toy_tagger()
    for context in ((), (BOS_TAG, BOS_TAG), ("DET", "NOUN"), ("NOUN", "VERB"),
                    ("never", "seen")):
        total = tagger.check_normalised(context)
        check(abs(total - 1.0) < 1e-6,
              f"transitions sum to 1 after context {context} ({total:.9f})")
    check(abs(sum(tagger.lambdas.values()) - 1.0) < 1e-9,
          "deleted-interpolation weights sum to 1")
    check(tagger.lambdas[3] > tagger.lambdas[1],
          "the trigram order outweighs the unigram on a repetitive corpus")
    emissions = sum(tagger.emission_prob(w, "NOUN") for w in tagger.vocab)
    check(abs(emissions - 1.0) < 1e-9, f"P(word|NOUN) sums to 1 ({emissions:.9f})")


def test_tagger_learns_context() -> None:
    """The point of the transition model: the same word, two tags, by context."""
    tagger = toy_tagger()
    check(tagger.transition_prob("NOUN", ("DET",)) >
          tagger.transition_prob("VERB", ("DET",)),
          "a determiner predicts a noun over a verb")
    check(tagger.transition_prob("NOUN", ("DET", "ADJ")) >
          tagger.transition_prob("ADJ", ("DET", "ADJ")),
          "an adjective after a determiner predicts a noun")
    check(tagger.emission_prob("dog", "NOUN") > tagger.emission_prob("dog", "VERB"),
          "'dog' emits from NOUN more than from VERB")


def test_tagger_viterbi() -> None:
    tagger = toy_tagger()
    check(tagger.tag(["the", "cat", "sat", "on", "the", "mat"])
          == ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"),
          "Viterbi recovers a familiar tagging")
    check(tagger.tag([]) == (), "the empty sentence tags to ()")
    check(len(tagger.tag(["the", "big", "dog"])) == 3,
          "one tag is emitted per word")
    beam = HMMTagger(TaggerConfig(beam_width=4)).fit(tagged_pairs(toy_sentences()))
    check(beam.tag(["the", "cat", "sat", "on", "the", "mat"])
          == tagger.tag(["the", "cat", "sat", "on", "the", "mat"]),
          "a beam of 4 agrees with exact Viterbi here")


def test_tagger_viterbi_is_optimal() -> None:
    """Viterbi must find the best-scoring sequence, not merely a good one."""
    import itertools

    tagger = toy_tagger()
    words = ["the", "dog", "ran"]
    best = max(
        itertools.product(tagger.tags, repeat=len(words)),
        key=lambda tags: tagger.sequence_logprob(words, tags),
    )
    predicted = tagger.tag(words)
    check(abs(tagger.sequence_logprob(words, predicted)
              - tagger.sequence_logprob(words, best)) < 1e-9,
          "Viterbi matches exhaustive search over all tag sequences")


def test_tagger_unknown_words() -> None:
    """Unknown words are tagged from their suffix, not from the tag prior."""
    tagger = toy_tagger()
    check("swiftly" not in tagger.vocab, "'swiftly' is genuinely unseen")
    distribution = tagger.suffix_tag_dist("swiftly")
    check(abs(sum(distribution.values()) - 1.0) < 1e-9,
          "the suffix model is a distribution over tags")
    check(max(distribution, key=distribution.get) == "ADV",
          "an unseen -ly word is guessed ADV from its suffix")
    check(tagger.tag(["the", "dog", "ran", "swiftly"])[-1] == "ADV",
          "the decoder tags the unseen -ly word ADV")
    check(math.isfinite(tagger.emission_logprob("swiftly", "ADV")),
          "unknown words receive a finite emission score")


def test_tagging_baseline() -> None:
    baseline = MostFrequentTagger().fit(tagged_pairs(toy_sentences()))
    check(baseline.tag(["the", "dog"]) == ("DET", "NOUN"),
          "the baseline recovers unambiguous words")
    check(baseline.tag(["zzz"]) == (baseline.default_tag,),
          "unknown words fall back to the rare-word majority tag")


def test_tagging_scoring() -> None:
    gold = Sentence.from_tokens([Token("the", "DET"), Token("dog", "NOUN")])
    scores = score_tagging([gold], [("DET", "VERB")], known_vocab={"the"})
    check(abs(scores.accuracy - 0.5) < 1e-9, "tag accuracy is over gold tokens")
    check(scores.known_accuracy == 1.0 and scores.unknown_accuracy == 0.0,
          "known and unknown accuracy are reported separately")
    check(scores.confusions[("NOUN", "VERB")] == 1, "confusions record (gold, predicted)")


def test_pipeline_scoring() -> None:
    """A tag on a wrongly segmented span cannot count, however plausible."""
    gold = Sentence.from_tokens(
        [Token("the", "DET"), Token("big", "ADJ"), Token("dog", "NOUN")]
    )
    scores = score_pipeline([gold], [("thebig", "dog")], [("ADJ", "NOUN")])
    check(scores.n_seg_induced_errors == 2, "unrecovered gold spans are seg-induced")
    check(scores.n_genuine_errors == 0, "the recovered span was tagged correctly")
    check(abs(scores.accuracy - 1 / 3) < 1e-9, "accuracy is over gold tokens")
    check(scores.tag_accuracy_given_span == 1.0,
          "tag accuracy given a correct span excludes seg-induced errors")
    perfect = score_pipeline([gold], [gold.words], [("DET", "ADJ", "NOUN")])
    check(perfect.accuracy == 1.0 and perfect.sentence_accuracy == 1.0,
          "a fully correct pipeline output scores 1.0")


# --------------------------------------------------------------------------
# Part 3 -- morphology-aware tagging
# --------------------------------------------------------------------------
FEM_SG = "Gender=Fem|Number=Sing"
MASC_SG = "Gender=Masc|Number=Sing"
FEM_PL = "Gender=Fem|Number=Plur"


def spanish_sentences() -> list[Sentence]:
    """A toy Spanish-shaped corpus: agreement is the only regularity in it."""

    def token(form: str, upos: str, feats: str = "") -> Token:
        return Token(form, upos, parse_feats(feats))

    raw = [
        [token("la", "DET", FEM_SG), token("casa", "NOUN", FEM_SG),
         token("roja", "ADJ", FEM_SG)],
        [token("el", "DET", MASC_SG), token("libro", "NOUN", MASC_SG),
         token("rojo", "ADJ", MASC_SG)],
        [token("las", "DET", FEM_PL), token("casas", "NOUN", FEM_PL),
         token("rojas", "ADJ", FEM_PL)],
        [token("la", "DET", FEM_SG), token("mesa", "NOUN", FEM_SG),
         token("blanca", "ADJ", FEM_SG)],
        [token("el", "DET", MASC_SG), token("gato", "NOUN", MASC_SG),
         token("blanco", "ADJ", MASC_SG)],
        # "grande" marks number but not gender -- the underspecified case that
        # must not be counted as a disagreement.
        [token("la", "DET", FEM_SG), token("casa", "NOUN", FEM_SG),
         token("grande", "ADJ", "Number=Sing")],
        [token("el", "DET", MASC_SG), token("libro", "NOUN", MASC_SG),
         token("grande", "ADJ", "Number=Sing")],
    ]
    return [
        Sentence.from_tokens(row, sent_id=f"toy-es-{i}")
        for i, row in enumerate(raw)
        for _ in range(30)
    ]


def test_morph_tags() -> None:
    feats = parse_feats("Gender=Fem|Number=Sing")
    check(feats == (("Gender", "Fem"), ("Number", "Sing")),
          "FEATS parse to a sorted (feature, value) tuple")
    check(parse_feats("_") == (), "an empty FEATS field gives no features")
    check(parse_feats("Gender=Fem,Masc")[0] == ("Gender", "Fem"),
          "an ambiguous FEATS value keeps the first alternative")
    check(Token("casa", "NOUN", feats).morph_tag == "NOUN-Fem-Sg",
          "the tag uses the brief's notation: NOUN-Fem-Sg")
    check(make_morph_tag("ADJ", parse_feats("Gender=Masc|Number=Plur")) == "ADJ-Masc-Pl",
          "ADJ-Masc-Pl is built the same way")
    check(Token("dog", "NOUN").morph_tag == "NOUN",
          "a token with no FEATS keeps its bare POS tag (the Brown case)")
    check(coarse_tag("NOUN-Fem-Sg") == "NOUN" and coarse_tag("NOUN") == "NOUN",
          "projection back to coarse tags is exact")
    check(coarse_tag("-LRB-") == "-LRB-",
          "Penn bracket tags survive projection unchanged")
    check(tag_features("NOUN-Fem-Sg") == {"Gender": "Fem", "Number": "Sg"},
          "features can be read back out of a predicted tag")


def test_morph_corpus_views() -> None:
    sentences = spanish_sentences()
    check(sentences[0].morph_tags == ("DET-Fem-Sg", "NOUN-Fem-Sg", "ADJ-Fem-Sg"),
          "a sentence exposes its morphology-aware tags")
    check(sentences[0].tags == ("DET", "NOUN", "ADJ"),
          "the plain tags are unchanged by Part 3")
    check(sentences[0].has_morphology, "a UD-style sentence reports morphology")
    check(not Sentence.from_tokens([Token("dog", "NOUN")]).has_morphology,
          "a Brown-style sentence reports none")
    check(len({t for s in sentences for t in s.morph_tags})
          > len({t for s in sentences for t in s.tags}),
          "refining the tagset genuinely splits it")


def test_agreement_is_learned() -> None:
    """The brief's claim, checked in the transition table itself."""
    tagger = HMMTagger().fit(tagged_pairs(spanish_sentences(), morph=True))
    context = ("DET-Fem-Sg", "NOUN-Fem-Sg")
    check(tagger.transition_prob("ADJ-Fem-Sg", context) >
          tagger.transition_prob("ADJ-Masc-Sg", context),
          "a feminine noun predicts a feminine adjective over a masculine one")
    check(tagger.transition_prob("ADJ-Fem-Sg", ("DET-Fem-Pl", "NOUN-Fem-Pl")) <
          tagger.transition_prob("ADJ-Fem-Pl", ("DET-Fem-Pl", "NOUN-Fem-Pl")),
          "number agreement is learned as well as gender")
    check(abs(tagger.check_normalised(context) - 1.0) < 1e-6,
          "the refined transition model is still normalised")


def test_agreement_bias() -> None:
    """The transition model must prefer agreement, and by a wide margin."""
    tagger = HMMTagger().fit(tagged_pairs(spanish_sentences(), morph=True))
    rows = agreement_bias(tagger)
    check(bool(rows), "a bias row is produced per fully specified noun tag")
    for row in rows:
        check(row.agreeing > row.clashing,
              f"{' '.join(row.context)}: agreeing beats clashing "
              f"({row.agreeing:.4f} vs {row.clashing:.4f})")
        check(row.ratio > 5.0,
              f"{' '.join(row.context)}: the margin is large, not marginal "
              f"({row.ratio:.0f}x)")
    check(any(row.underspecified > row.clashing for row in rows),
          "a gender-unmarked adjective ('grande') is counted as underspecified, "
          "not as a disagreement")


def test_agreement_scoring() -> None:
    sentences = spanish_sentences()[:1]          # la casa roja, all Fem Sg
    perfect = score_agreement(sentences, [("DET-Fem-Sg", "NOUN-Fem-Sg", "ADJ-Fem-Sg")])
    check(perfect.n_pairs == 2, "both adjacent pairs agree in the gold")
    check(perfect.reproduced_rate == 1.0 and perfect.correct_rate == 1.0,
          "a correct tagging reproduces agreement")

    broken = score_agreement(sentences, [("DET-Fem-Sg", "NOUN-Fem-Sg", "ADJ-Masc-Sg")])
    check(broken.n_reproduced == 1,
          "a gender clash on NOUN+ADJ is counted as agreement not reproduced")

    consistent = score_agreement(
        sentences, [("DET-Masc-Sg", "NOUN-Masc-Sg", "ADJ-Masc-Sg")]
    )
    check(consistent.reproduced_rate == 1.0 and consistent.correct_rate == 0.0,
          "consistently wrong gender still reproduces agreement, but is not correct")

    stripped = score_agreement(sentences, [("DET", "NOUN", "ADJ")])
    check(stripped.n_reproduced == 0,
          "a tagger that cannot express gender reproduces no agreement")


def test_agreement_pipeline_mode() -> None:
    """A pair the segmenter destroyed lost its agreement, and must be counted."""
    sentences = spanish_sentences()[:1]           # la casa roja, all Fem Sg
    gold_tags = ("DET-Fem-Sg", "NOUN-Fem-Sg", "ADJ-Fem-Sg")
    perfect = score_agreement(
        sentences, [gold_tags], predicted_words=[("la", "casa", "roja")]
    )
    check(perfect.n_pairs == 2 and perfect.reproduced_rate == 1.0,
          "a perfectly segmented sentence scores as it does on gold words")
    check(perfect.n_unrecovered == 0, "nothing was lost to segmentation")

    merged = score_agreement(
        sentences, [("DET-Fem-Sg", "NOUN-Fem-Sg")],
        predicted_words=[("la", "casaroja")],
    )
    check(merged.n_pairs == 2, "the gold pairs stay in the denominator")
    check(merged.n_unrecovered == 2 and merged.n_reproduced == 0,
          "a merged token loses both pairs it took part in")
    check(merged.by_context["DET+NOUN"][1] == 0,
          "the loss is attributed to the context it happened in")


def test_agreement_ignores_non_agreeing_pairs() -> None:
    """Only pairs that agree in the gold can be scored for agreement."""
    tokens = [
        Token("la", "DET", parse_feats(FEM_SG)),
        Token("casa", "NOUN", parse_feats(FEM_SG)),
        Token("rojos", "ADJ", parse_feats("Gender=Masc|Number=Plur")),
    ]
    sentence = Sentence.from_tokens(tokens)
    scores = score_agreement([sentence], [("DET-Fem-Sg", "NOUN-Fem-Sg", "ADJ-Masc-Pl")])
    check(scores.n_pairs == 1,
          "the non-agreeing NOUN+ADJ pair is excluded from the denominator")
    check(scores.by_context["DET+NOUN"][0] == 1, "contexts are labelled by POS pair")


def main() -> None:
    for test in (
        test_tagsets,
        test_normalisation,
        test_sentence_invariants,
        test_lm_normalisation,
        test_lm_open_vocabulary,
        test_segmentation,
        test_segmentation_scoring,
        test_greedy_baseline,
        test_greedy_is_beaten_by_dp,
        test_tagger_distributions,
        test_tagger_learns_context,
        test_tagger_viterbi,
        test_tagger_viterbi_is_optimal,
        test_tagger_unknown_words,
        test_tagging_baseline,
        test_tagging_scoring,
        test_pipeline_scoring,
        test_morph_tags,
        test_morph_corpus_views,
        test_agreement_is_learned,
        test_agreement_bias,
        test_agreement_scoring,
        test_agreement_pipeline_mode,
        test_agreement_ignores_non_agreeing_pairs,
    ):
        print(f"\n{test.__name__}")
        test()
    print(f"\n{PASSED} checks passed")


if __name__ == "__main__":
    main()
