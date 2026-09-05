"""Correctness tests for the Q1 models.

Run:  .venv/bin/python scripts/test_q1_models.py

These check the properties the Part 1 segmenter silently assumes -- above all
that the language models are normalised, since the decoder compares hypotheses
containing different numbers of words and would otherwise acquire a systematic
bias towards long or short words.
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
    normalise_brown_tag,
    normalise_form,
)
from q1.evaluate import score_segmentation
from q1.lm import CharLM, NgramLM
from q1.segment import DecoderConfig, decode_segmentation, spans_from_words

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


def main() -> None:
    for test in (
        test_tagsets,
        test_normalisation,
        test_sentence_invariants,
        test_lm_normalisation,
        test_lm_open_vocabulary,
        test_segmentation,
        test_segmentation_scoring,
    ):
        print(f"\n{test.__name__}")
        test()
    print(f"\n{PASSED} checks passed")


if __name__ == "__main__":
    main()
