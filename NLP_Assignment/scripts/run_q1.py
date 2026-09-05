"""Question 1, Part 1 -- word segmentation.

Trains a trigram word language model on each corpus and recovers word
boundaries from unspaced text with a dynamic-programming (Viterbi) decoder
over character positions.  Run:

    .venv/bin/python scripts/run_q1.py                 # full run
    .venv/bin/python scripts/run_q1.py --fast          # small samples, for iteration

The trained LM and the tuned DecoderConfig are persisted to
``models/q1_<language>.pkl`` so later work can reload this decoder rather than
retraining it.
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
from q1.evaluate import score_segmentation  # noqa: E402
from q1.lm import NgramLM  # noqa: E402
from q1.segment import DecoderConfig, decode_segmentation  # noqa: E402

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
            print(f"    input : {string}")
            print(f"    output: {' '.join(words)}")
            print()

        scores = score_segmentation(test_sample, dp_beam)
        results[corpus.language] = {
            "segmentation": {
                "token_f1": scores.token_f1,
                "boundary_f1": scores.boundary_f1,
                "sentence_accuracy": scores.sentence_accuracy,
            },
            "config": asdict(tuned),
            "latency_ms": {"exact": exact_ms, "beam": beam_ms},
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
