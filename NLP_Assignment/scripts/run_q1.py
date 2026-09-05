"""Question 1, Parts 1 and 2 -- word segmentation, then POS tagging.

Part 1 trains a trigram word language model on each corpus and recovers word
boundaries from unspaced text with a dynamic-programming (Viterbi) decoder
over character positions.

Part 2 trains a trigram HMM tagger -- emission P(word | tag), transition
P(tag | previous two tags) -- and runs the same kind of dynamic program over
tag sequences, both on gold words (to measure tagging alone) and on the
segmenter's own output (to measure the pipeline).

Part 3 refines the tagset with gender and number (NOUN-Fem-Sg, ADJ-Masc-Pl) and
asks the question that tagset makes possible: did the model reproduce
grammatical *agreement*?

Part 4 asks whether any of it was worth the work, by pitting each component
against a deliberately simple baseline -- greedy longest-match segmentation and
most-frequent-tag tagging -- and reporting the improvement as both an absolute
gain and a share of the baseline's errors removed.  Run:

    .venv/bin/python scripts/run_q1.py                 # full run
    .venv/bin/python scripts/run_q1.py --fast          # small samples, for iteration

The trained LM, the trained tagger and the tuned configurations are persisted
to ``models/q1_<language>.pkl`` so later work can reload this pipeline rather
than retraining it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q1.data import (  # noqa: E402
    Corpus,
    coarse_tag,
    load_brown,
    load_german,
    load_spanish,
    max_word_length,
    vocabulary,
)
from q1.evaluate import (  # noqa: E402
    agreement_bias,
    score_agreement,
    score_pipeline,
    score_segmentation,
    score_tagging,
)
from q1.lm import NgramLM  # noqa: E402
from q1.segment import (  # noqa: E402
    DecoderConfig,
    decode_segmentation,
    greedy_corpus,
)
from q1.tagger import (  # noqa: E402
    HMMTagger,
    MostFrequentTagger,
    TaggerConfig,
    tagged_pairs,
)

RULE = "=" * 78
SEED = 42

# Set by _init_worker in each forked process so the LM is not pickled across
# the process boundary once per sentence.
_WORKER: dict = {}


def _init_worker(payload: dict) -> None:
    _WORKER.update(payload)


def _decode_one(job):
    """Segment one sentence inside a worker process."""
    chars, config = job
    return decode_segmentation(chars, _WORKER["lm"], config)


class Runner:
    """Parallel map over sentences, falling back to serial if fork is unavailable."""

    def __init__(self, payload: dict, workers: int) -> None:
        self.payload = payload
        self.workers = workers
        self.pool: ProcessPoolExecutor | None = None
        if workers > 1:
            try:
                context = mp.get_context("fork")
                self.pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_init_worker,
                    initargs=(payload,),
                )
            except (ValueError, OSError):
                self.pool = None
        if self.pool is None:
            _init_worker(payload)

    def map(self, jobs):
        if self.pool is None:
            return [_decode_one(job) for job in jobs]
        return list(self.pool.map(_decode_one, jobs, chunksize=4))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
def tune_segmentation(runner: Runner, sentences, base: DecoderConfig) -> DecoderConfig:
    """Grid-search the unknown-word penalty and beam width on the dev set.

    ``unk_penalty`` is the decisive knob: it sets how much evidence the decoder
    needs before inventing a word it has never seen.  Too lenient and it
    hallucinates long unknown words; too harsh and it shatters every OOV word
    into known fragments.
    """
    print("\n  tuning segmentation on dev ...")
    best, best_f1 = base, -1.0
    for penalty in (0.0, -2.0, -4.0, -6.0, -8.0):
        config = base.replace(unk_penalty=penalty, beam_width=8)
        predictions = runner.map([(s.chars, config) for s in sentences])
        f1 = score_segmentation(sentences, predictions).token_f1
        print(f"    unk_penalty={penalty:>5}  token F1={f1:.4f}")
        if f1 > best_f1:
            best, best_f1 = config, f1

    for width in (4, 8, 16):
        config = best.replace(beam_width=width)
        start = time.perf_counter()
        predictions = runner.map([(s.chars, config) for s in sentences])
        elapsed = (time.perf_counter() - start) / max(1, len(sentences)) * 1000
        f1 = score_segmentation(sentences, predictions).token_f1
        print(f"    beam={width:>5}  token F1={f1:.4f}  {elapsed:.0f} ms/sentence")
    return best


def tune_tagger(sentences, train_pairs) -> tuple[HMMTagger, dict[int, float]]:
    """Pick the transition order on dev: bigram history vs the brief's trigram.

    The brief specifies conditioning on the previous *two* tags; this fits both
    and reports them, so the trigram's contribution is measured rather than
    assumed.  Selection is on dev only -- test is never consulted.
    """
    print("\n  tuning tagger on dev ...")
    scores: dict[int, float] = {}
    best, best_accuracy = None, -1.0
    for order in (2, 3):
        tagger = HMMTagger(TaggerConfig(order=order)).fit(train_pairs)
        predictions = [tagger.tag(s.words) for s in sentences]
        accuracy = score_tagging(sentences, predictions).accuracy
        scores[order] = accuracy
        print(f"    order={order} (history of {order - 1} tags)  accuracy={accuracy:.4f}")
        # ">=" so a tie goes to the higher order: the trigram is the model the
        # brief specifies, and a tie is no evidence against it.
        if accuracy >= best_accuracy:
            best, best_accuracy = tagger, accuracy
    return best, scores


# --------------------------------------------------------------------------
# Part 3 -- morphology-aware tagging
# --------------------------------------------------------------------------
def run_morphology(corpus: Corpus, test_sample, train_vocab, order: int,
                   plain_accuracy: float, baseline_accuracy: float,
                   segmented=None) -> tuple:
    """Train and evaluate the Part 3 tagger; returns (tagger, results dict).

    The order is inherited from Part 2 rather than re-tuned, so the only thing
    that differs between the two taggers is the tagset -- which is the whole
    point of the comparison.
    """
    plain_tagset = {t for s in corpus.train for t in s.tags}
    morph_tagset = {t for s in corpus.train for t in s.morph_tags}

    # -- the English case: no FEATS in the annotation ----------------------
    if not corpus.has_morphology or morph_tagset == plain_tagset:
        print(f"\n  PART 3 -- {corpus.name} carries no morphological features")
        print(f"    plain tagset={len(plain_tagset)}   "
              f"morphology-aware tagset={len(morph_tagset)}  (identical)")
        print("    With no gender or number to attach, the refined tagset IS the")
        print("    plain one, so the two taggers are the same model and agreement")
        print("    cannot be measured.  This is a fact about the corpus, not about")
        print("    the language: a UD English treebank would carry FEATS.")
        return None, {
            "available": False,
            "reason": "corpus has no FEATS annotation",
            "plain_tagset_size": len(plain_tagset),
            "morph_tagset_size": len(morph_tagset),
        }

    print(f"\n  training morphology-aware tagger (order={order}) ...")
    start = time.perf_counter()
    train_pairs = tagged_pairs(corpus.train, morph=True)
    tagger = HMMTagger(TaggerConfig(order=order)).fit(train_pairs)
    baseline = MostFrequentTagger().fit(train_pairs)
    print(f"    trained in {time.perf_counter() - start:.1f}s   "
          f"plain tagset={len(plain_tagset)}   "
          f"morphology-aware tagset={len(morph_tagset)}")

    predictions = [tagger.tag(s.words) for s in test_sample]
    baseline_predictions = [baseline.tag(s.words) for s in test_sample]
    own_scores = score_tagging(test_sample, predictions, train_vocab, morph=True)

    # Projecting back to coarse tags puts both taggers on the identical
    # decision.  Without this the comparison is rigged: the morphology-aware
    # tagger is choosing among many more labels.
    projected = [tuple(coarse_tag(t) for t in tags) for tags in predictions]
    baseline_projected = [
        tuple(coarse_tag(t) for t in tags) for tags in baseline_predictions
    ]
    projected_scores = score_tagging(test_sample, projected, train_vocab)
    baseline_projected_scores = score_tagging(
        test_sample, baseline_projected, train_vocab
    )

    print_table(
        "PART 3 -- Morphology-aware tagging, on its own label set",
        [
            ("most-frequent-tag (baseline)", len(morph_tagset),
             pct(score_tagging(test_sample, baseline_predictions,
                               train_vocab, morph=True).accuracy), "-"),
            (f"HMM, order={order} (Viterbi)", len(morph_tagset),
             pct(own_scores.accuracy), pct(own_scores.unknown_accuracy)),
        ],
        ("model", "labels", "accuracy", "unknown"),
    )
    print("    Not comparable with Part 2: this is a harder decision, over more")
    print("    labels.  The like-for-like comparison is the next table.")

    print_table(
        "PART 3 -- Same decision: everything projected to coarse tags",
        [
            ("most-frequent-tag, plain tagset", pct(baseline_accuracy), "-"),
            ("most-frequent-tag, morph tagset -> coarse",
             pct(baseline_projected_scores.accuracy),
             f"{100 * (baseline_projected_scores.accuracy - baseline_accuracy):+.2f} pp"),
            ("HMM, plain tagset", pct(plain_accuracy), "-"),
            ("HMM, morph tagset -> coarse", pct(projected_scores.accuracy),
             f"{100 * (projected_scores.accuracy - plain_accuracy):+.2f} pp"),
        ],
        ("model", "coarse-tag accuracy", "vs its plain counterpart"),
    )

    agreement = score_agreement(test_sample, predictions)
    print_table(
        f"PART 3 -- Agreement reproduced ({agreement.n_pairs:,} gold-agreeing "
        "adjacent pairs)",
        [("ALL CONTEXTS", agreement.n_pairs, pct(agreement.reproduced_rate),
          pct(agreement.correct_rate))]
        + [(context, pairs, pct(reproduced), pct(correct))
           for context, pairs, reproduced, correct in agreement.rows(min_pairs=10)],
        ("context", "pairs", "agreement reproduced", "and values correct"),
    )

    # The same question end to end: the segmenter must recover both words
    # before their agreement can survive at all.
    pipeline_agreement = None
    if segmented is not None:
        pipeline_predictions = [tagger.tag(words) for words in segmented]
        pipeline_agreement = score_agreement(
            test_sample, pipeline_predictions, predicted_words=segmented
        )
        print_table(
            "PART 3 -- Agreement end to end (DP segmentation, then morph tagging)",
            [
                ("on gold words", agreement.n_pairs,
                 pct(agreement.reproduced_rate), "-"),
                ("on predicted words", pipeline_agreement.n_pairs,
                 pct(pipeline_agreement.reproduced_rate),
                 pipeline_agreement.n_unrecovered),
            ],
            ("scored", "pairs", "agreement reproduced",
             "pairs lost to segmentation"),
        )

    bias = agreement_bias(tagger)
    if bias:
        print_table(
            "PART 3 -- The learned pattern, read out of the transition model",
            [(f"{' '.join(row.context)} __", f"{row.agreeing:.4f}",
              f"{row.clashing:.4f}", f"{row.ratio:.0f}x",
              f"{row.underspecified:.4f}") for row in bias],
            ("context", "P(agreeing ADJ)", "P(clashing ADJ)", "ratio",
             "P(gender-unmarked ADJ)"),
        )

    return tagger, {
        "available": True,
        "plain_tagset_size": len(plain_tagset),
        "morph_tagset_size": len(morph_tagset),
        "accuracy_own_labels": own_scores.accuracy,
        "unknown_accuracy_own_labels": own_scores.unknown_accuracy,
        "coarse_accuracy_plain_hmm": plain_accuracy,
        "coarse_accuracy_morph_hmm": projected_scores.accuracy,
        "coarse_accuracy_plain_baseline": baseline_accuracy,
        "coarse_accuracy_morph_baseline": baseline_projected_scores.accuracy,
        "agreement_pairs": agreement.n_pairs,
        "agreement_reproduced": agreement.reproduced_rate,
        "agreement_correct": agreement.correct_rate,
        "agreement_reproduced_pipeline": (
            pipeline_agreement.reproduced_rate if pipeline_agreement else None
        ),
        "agreement_pairs_lost_to_segmentation": (
            pipeline_agreement.n_unrecovered if pipeline_agreement else None
        ),
        "agreement_by_context": {
            context: {"pairs": pairs, "reproduced": reproduced, "correct": correct}
            for context, pairs, reproduced, correct in agreement.rows()
        },
    }


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def print_table(title: str, rows: list[tuple], headers: tuple) -> None:
    print(f"\n  {title}")
    widths = [
        max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0)) + 2
        for i in range(len(headers))
    ]
    header = "".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print("    " + header)
    print("    " + "-" * sum(widths))
    for row in rows:
        print("    " + "".join(str(c).ljust(w) for c, w in zip(row, widths)))


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# --------------------------------------------------------------------------
# Main experiment for one language
# --------------------------------------------------------------------------
def run_language(corpus: Corpus, args, results: dict) -> None:
    print("\n" + RULE)
    print(f"{corpus.language}  --  {corpus.name}")
    print(RULE)

    rng = random.Random(SEED)
    dev_sample = rng.sample(list(corpus.dev), min(args.dev_size, len(corpus.dev)))
    test_sample = rng.sample(list(corpus.test), min(args.test_size, len(corpus.test)))
    print(f"  train={len(corpus.train):,} sentences   "
          f"dev sample={len(dev_sample)}   test sample={len(test_sample)}")

    # -- training ----------------------------------------------------------
    start = time.perf_counter()
    lm = NgramLM(order=3, smoothing="witten_bell").fit(s.words for s in corpus.train)
    train_vocab = vocabulary(corpus.train)
    print(f"  trained in {time.perf_counter() - start:.1f}s   "
          f"LM vocab={len(lm.vocab):,}   word types={len(train_vocab):,}")
    print(f"  dev perplexity (trigram word LM): "
          f"{lm.perplexity([s.words for s in dev_sample]):.1f}")

    max_len = min(20, max_word_length(corpus.train, 0.999) + 4)
    base_config = DecoderConfig(max_word_len=max_len, beam_width=8)
    runner = Runner({"lm": lm}, args.workers)

    try:
        # -- tuning --------------------------------------------------------
        tuned = tune_segmentation(runner, dev_sample, base_config)
        print(f"\n  selected config: {tuned}")

        # -- evaluation ----------------------------------------------------
        # Exact (unpruned) decoding is O(n L^3) -- far costlier than the beam --
        # so it runs on a subsample, purely to show that the beam loses nothing.
        # The headline numbers use the beam.
        exact_sample = test_sample[: args.exact_size]
        exact_cfg = tuned.replace(beam_width=None)
        start = time.perf_counter()
        dp_exact = runner.map([(s.chars, exact_cfg) for s in exact_sample])
        exact_ms = (time.perf_counter() - start) / max(1, len(exact_sample)) * 1000
        dp_beam_on_exact = runner.map([(s.chars, tuned) for s in exact_sample])
        start = time.perf_counter()
        dp_beam = runner.map([(s.chars, tuned) for s in test_sample])
        beam_ms = (time.perf_counter() - start) / len(test_sample) * 1000

        seg_rows = []
        for name, prediction, reference, ms in (
            (f"trigram DP, beam={tuned.beam_width}", dp_beam, test_sample, beam_ms),
            (f"trigram DP, exact  [n={len(exact_sample)}]", dp_exact,
             exact_sample, exact_ms),
            ("trigram DP, beam   [same n]", dp_beam_on_exact,
             exact_sample, beam_ms),
        ):
            s = score_segmentation(reference, prediction)
            seg_rows.append((name, pct(s.token_f1), pct(s.boundary_f1),
                             pct(s.sentence_accuracy), f"{ms:.0f}"))
        print_table(
            "PART 1 -- Segmentation (test sample)",
            seg_rows,
            ("model", "token F1", "boundary F1", "exact sent", "ms/sent"),
        )

        # -- PART 2: tagging -----------------------------------------------
        start = time.perf_counter()
        train_pairs = tagged_pairs(corpus.train)
        baseline_tagger = MostFrequentTagger().fit(train_pairs)
        tagger, order_scores = tune_tagger(dev_sample, train_pairs)
        print(f"\n  tagger trained in {time.perf_counter() - start:.1f}s   "
              f"order={tagger.order}   tags={len(tagger.tags)}   "
              f"lambdas={ {n: round(w, 3) for n, w in tagger.lambdas.items()} }")

        # (a) Tagging alone: gold words in, tags out.  This isolates the two
        # distributions Part 2 learns from any segmentation error.
        gold_words = [s.words for s in test_sample]
        start = time.perf_counter()
        hmm_tags = [tagger.tag(words) for words in gold_words]
        tag_ms = (time.perf_counter() - start) / len(test_sample) * 1000
        baseline_tags = [baseline_tagger.tag(words) for words in gold_words]

        hmm_scores = score_tagging(test_sample, hmm_tags, train_vocab)
        baseline_scores = score_tagging(test_sample, baseline_tags, train_vocab)
        print_table(
            "PART 2 -- Tagging on gold segmentation (test sample)",
            [
                ("most-frequent-tag (baseline)", pct(baseline_scores.accuracy),
                 pct(baseline_scores.known_accuracy),
                 pct(baseline_scores.unknown_accuracy),
                 pct(baseline_scores.sentence_accuracy), "-"),
                (f"HMM, order={tagger.order} (Viterbi)", pct(hmm_scores.accuracy),
                 pct(hmm_scores.known_accuracy),
                 pct(hmm_scores.unknown_accuracy),
                 pct(hmm_scores.sentence_accuracy), f"{tag_ms:.1f}"),
            ],
            ("model", "accuracy", "known", "unknown", "exact sent", "ms/sent"),
        )
        print(f"    OOV rate on the test sample: {pct(hmm_scores.oov_rate)}")

        print_table(
            "PART 2 -- Most confused tag pairs (HMM)",
            [(gold, predicted, count)
             for (gold, predicted), count in hmm_scores.top_confusions(6)],
            ("gold", "predicted", "count"),
        )

        # (b) The actual Part 2 task: tag the *segmented* words.  Scored over
        # gold tokens, so a token the segmenter never recovered counts as an
        # error even though the tagger was never asked about it.
        pipeline_tags = [tagger.tag(words) for words in dp_beam]
        baseline_pipeline_tags = [baseline_tagger.tag(words) for words in dp_beam]
        pipeline = score_pipeline(test_sample, dp_beam, pipeline_tags)
        baseline_pipeline = score_pipeline(test_sample, dp_beam, baseline_pipeline_tags)
        print_table(
            "PART 1 + 2 -- End to end: DP segmentation then HMM tagging",
            [
                ("DP segment -> most-frequent tag", pct(baseline_pipeline.accuracy),
                 pct(baseline_pipeline.tag_accuracy_given_span),
                 baseline_pipeline.n_seg_induced_errors,
                 baseline_pipeline.n_genuine_errors,
                 pct(baseline_pipeline.share_segmentation_induced)),
                ("DP segment -> HMM tag", pct(pipeline.accuracy),
                 pct(pipeline.tag_accuracy_given_span),
                 pipeline.n_seg_induced_errors,
                 pipeline.n_genuine_errors,
                 pct(pipeline.share_segmentation_induced)),
            ],
            ("system", "accuracy", "tag acc | correct span",
             "seg-induced errors", "genuine errors", "% from segmentation"),
        )

        # -- PART 3: morphology-aware tagging ------------------------------
        morph_tagger, morph_results = run_morphology(
            corpus, test_sample, train_vocab, tagger.order,
            plain_accuracy=hmm_scores.accuracy,
            baseline_accuracy=baseline_scores.accuracy,
            segmented=dp_beam,
        )

        # -- PART 4: is any of this worth it? ------------------------------
        # Greedy longest-match gets the *full* training vocabulary, hapax words
        # included -- a more generous lexicon than the LM keeps for itself.
        start = time.perf_counter()
        greedy = greedy_corpus(test_sample, train_vocab, tuned.max_word_len)
        greedy_ms = (time.perf_counter() - start) / len(test_sample) * 1000
        greedy_scores = score_segmentation(test_sample, greedy)
        model_seg = score_segmentation(test_sample, dp_beam)

        # The end-to-end baseline is both simple components together, which is
        # the honest thing to compare the full pipeline against.
        greedy_tags = [baseline_tagger.tag(words) for words in greedy]
        baseline_end_to_end = score_pipeline(test_sample, greedy, greedy_tags)

        comparisons = [
            ("Segmentation (token F1)",
             greedy_scores.token_f1, model_seg.token_f1),
            ("Segmentation (exact sentences)",
             greedy_scores.sentence_accuracy, model_seg.sentence_accuracy),
            ("Tagging, gold segmentation",
             baseline_scores.accuracy, hmm_scores.accuracy),
            ("End to end (segment + tag)",
             baseline_end_to_end.accuracy, pipeline.accuracy),
        ]
        rows = []
        for task, baseline_value, model_value in comparisons:
            gain = model_value - baseline_value
            headroom = 1.0 - baseline_value
            # Error reduction: the share of the baseline's mistakes removed.
            # A +2 pp gain on a 94% baseline is a third of the remaining
            # errors; the same +2 pp on a 50% baseline is a twenty-fifth.
            reduction = gain / headroom if headroom > 1e-12 else float("nan")
            rows.append((task, pct(baseline_value), pct(model_value),
                         f"{100 * gain:+.2f} pp", pct(reduction)))
        print_table(
            "PART 4 -- Improvement over the simple baselines (test sample)",
            rows,
            ("task", "baseline", "model", "absolute gain", "error reduction"),
        )
        print(f"    baselines: greedy longest-match ({greedy_ms:.2f} ms/sent, "
              f"vs {beam_ms:.0f} for the DP decoder) and most-frequent-tag")

        # -- persist -------------------------------------------------------
        model_path = ROOT / "models" / f"q1_{corpus.language.lower()}.pkl"
        model_path.parent.mkdir(exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "language": corpus.language,
                    "corpus": corpus.name,
                    "tagset": corpus.tagset,
                    "lm": lm,
                    "config": tuned,
                    "tagger": tagger,
                    "tagger_config": tagger.config,
                    "morph_tagger": morph_tagger,
                    "baseline_tagger": baseline_tagger,
                    "train_vocab": train_vocab,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"\n  models saved -> {model_path.relative_to(ROOT)} "
              f"({model_path.stat().st_size / 1e6:.1f} MB)")

        # -- sample strings -------------------------------------------------
        print(f"\n  Sample outputs ({corpus.language})")
        for string in SAMPLES.get(corpus.language, []):
            words = decode_segmentation(string, lm, tuned)
            tags = tagger.tag(words)
            pairs = ", ".join(f"({w}, {t})" for w, t in zip(words, tags))
            print(f"    input : {string}")
            print(f"    words : {' '.join(words)}")
            print(f"    tagged: [{pairs}]")
            if morph_tagger is not None:
                morph = morph_tagger.tag(words)
                print("    morph : "
                      + " ".join(f"{w}/{t}" for w, t in zip(words, morph)))
            print()

        scores = score_segmentation(test_sample, dp_beam)
        results[corpus.language] = {
            "segmentation": {
                "token_f1": scores.token_f1,
                "boundary_f1": scores.boundary_f1,
                "sentence_accuracy": scores.sentence_accuracy,
            },
            "tagging": {
                "baseline_accuracy": baseline_scores.accuracy,
                "hmm_accuracy": hmm_scores.accuracy,
                "hmm_known_accuracy": hmm_scores.known_accuracy,
                "hmm_unknown_accuracy": hmm_scores.unknown_accuracy,
                "oov_rate": hmm_scores.oov_rate,
                "dev_accuracy_by_order": order_scores,
                "n_tags": len(tagger.tags),
            },
            "end_to_end": {
                "baseline_accuracy": baseline_pipeline.accuracy,
                "pipeline_accuracy": pipeline.accuracy,
                "tag_accuracy_given_span": pipeline.tag_accuracy_given_span,
                "seg_induced_errors": pipeline.n_seg_induced_errors,
                "genuine_errors": pipeline.n_genuine_errors,
                "share_segmentation_induced": pipeline.share_segmentation_induced,
            },
            "baselines": {
                "greedy_token_f1": greedy_scores.token_f1,
                "greedy_boundary_f1": greedy_scores.boundary_f1,
                "greedy_sentence_accuracy": greedy_scores.sentence_accuracy,
                "greedy_latency_ms": greedy_ms,
                "most_frequent_tag_accuracy": baseline_scores.accuracy,
                "end_to_end_accuracy": baseline_end_to_end.accuracy,
                "improvement": {
                    task: {
                        "baseline": baseline_value,
                        "model": model_value,
                        "absolute_gain": model_value - baseline_value,
                        "error_reduction": (
                            (model_value - baseline_value) / (1.0 - baseline_value)
                            if 1.0 - baseline_value > 1e-12 else None
                        ),
                    }
                    for task, baseline_value, model_value in comparisons
                },
            },
            "morphology": morph_results,
            "config": asdict(tuned),
            "tagger_config": asdict(tagger.config),
            "latency_ms": {"exact": exact_ms, "beam": beam_ms, "tag": tag_ms},
            "test_sample_size": len(test_sample),
        }
    finally:
        runner.close()


SAMPLES = {
    "English": [
        "thequickbrownfoxjumpsoverthelazydog",
        "thegovernmentannouncedanewplanlastweek",
        "shesaidthatthebookwasonthetable",
    ],
    "Spanish": [
        "mispadrespuedenviajar",
        "elcielodespejadoesazul",
        "lacasarojaesgrande",
        "lascasasrojassongrandes",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="small samples")
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--exact-size", type=int, default=40,
                        help="sentences decoded without beam pruning")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--languages", default="English,Spanish")
    args = parser.parse_args()
    if args.fast:
        args.dev_size, args.test_size, args.exact_size = 40, 60, 8

    results: dict = {}
    wanted = {name.strip() for name in args.languages.split(",")}

    if "English" in wanted:
        print("\nloading English (Brown, universal tagset) ...")
        run_language(load_brown(tagset="universal"), args, results)
    if "Spanish" in wanted:
        print("\nloading Spanish (UD Spanish-GSD) ...")
        run_language(load_spanish(), args, results)
    if "German" in wanted:
        # Not run by default: clone UD_German-GSD into data/ first (see README).
        # German adds case to the agreement picture, so Part 3 is the reason to
        # bother with a third language.
        print("\nloading German (UD German-GSD) ...")
        run_language(load_german(), args, results)

    out = ROOT / "models" / "q1_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults written -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
