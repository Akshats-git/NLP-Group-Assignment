"""Question 1, Parts 1 and 2 -- word segmentation, then POS tagging.

Part 1 trains a trigram word language model on each corpus and recovers word
boundaries from unspaced text with a dynamic-programming (Viterbi) decoder
over character positions.

Part 2 trains a trigram HMM tagger -- emission P(word | tag), transition
P(tag | previous two tags) -- and runs the same kind of dynamic program over
tag sequences, both on gold words (to measure tagging alone) and on the
segmenter's own output (to measure the pipeline).  Run:

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

from q1.data import Corpus, load_brown, load_spanish, max_word_length, vocabulary  # noqa: E402
from q1.evaluate import score_pipeline, score_segmentation, score_tagging  # noqa: E402
from q1.lm import NgramLM  # noqa: E402
from q1.segment import DecoderConfig, decode_segmentation  # noqa: E402
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

    out = ROOT / "models" / "q1_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults written -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
