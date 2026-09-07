"""
speed_demon.py - Speed Demon benchmark for the live checks (Q4 Part 5)

Times the two layers of the editor separately on the same batch of 1,000
corrupted words: the per-token segmentation and spelling check that runs on
every word typed, and the grammar trigger that only runs once every N words.
The gap between the two is what the segmentation and spelling layer costs on
top of the grammar layer.

The batch itself comes from Question 3's corrupted-word generator, so the words
are built exactly the way they were for the Q3 benchmark, and the timed calls go
through q4.pipeline, so this measures the same code the editor runs.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

Q3_DIR = Path(__file__).resolve().parents[2] / "Question-3" / "q3_spelling_corrector"
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from benchmark import generate_benchmark_words

from q4.grammar import TRIGGER_N
from q4.pipeline import check_token, check_window

BATCH_SIZE = 1000
DEFAULT_SEED = 123


def build_batch(models: dict[str, Any], seed: int = DEFAULT_SEED, count: int = BATCH_SIZE) -> list[str]:
    """A batch of exactly `count` misspelled words, built before any timing starts."""
    rng = random.Random(seed)
    return generate_benchmark_words(models["q3_vocab"], rng, count=count)


def build_clean_batch(
    models: dict[str, Any], seed: int = DEFAULT_SEED, count: int = BATCH_SIZE
) -> list[str]:
    """The same size batch of ordinary vocabulary words.

    Every word in the corrupted batch is out of vocabulary, which is the worst
    case for the segmentation check and the best case for the grammar check,
    since the real-word pass skips unknown words outright. Timing a clean batch
    as well is the only way to see what each layer costs on ordinary typing.
    """
    rng = random.Random(seed)
    vocabulary = sorted(models["q3_vocab"])
    return [rng.choice(vocabulary) for _ in range(count)]


def _time_layers(
    models: dict[str, Any], batch: list[str], trigger_n: int
) -> tuple[float, float, int]:
    """Total seconds for the per-token layer and for the grammar layer."""
    started = time.perf_counter()
    for word in batch:
        check_token(word, models)
    token_total = time.perf_counter() - started

    windows = [batch[i : i + trigger_n] for i in range(0, len(batch), trigger_n)]
    windows = [w for w in windows if len(w) >= 2]

    started = time.perf_counter()
    for window in windows:
        check_window(window, models)
    grammar_total = time.perf_counter() - started

    return token_total, grammar_total, len(windows)


def run_speed_demon(
    models: dict[str, Any],
    batch: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = BATCH_SIZE,
    trigger_n: int = TRIGGER_N,
) -> dict[str, Any]:
    """Time the per-token layer and the grammar-trigger layer on the same batch.

    Model loading, the SymDel index and the batch itself are all prepared before
    the clock starts, so the numbers are the cost of the checks alone. The
    grammar layer is timed over the windows the editor would actually have run,
    which is one check per `trigger_n` words rather than one per word.
    """
    if batch is None:
        batch = build_batch(models, seed=seed, count=batch_size)
    batch_size = len(batch)
    clean_batch = build_clean_batch(models, seed=seed, count=batch_size)

    token_total, grammar_total, n_windows = _time_layers(models, batch, trigger_n)
    clean_token, clean_grammar, _ = _time_layers(models, clean_batch, trigger_n)

    return {
        "batch_size": batch_size,
        "trigger_n": trigger_n,
        "n_windows": n_windows,
        "token_total_seconds": token_total,
        "token_avg_ms": token_total / batch_size * 1000,
        "grammar_total_seconds": grammar_total,
        "grammar_avg_ms_per_window": grammar_total / n_windows * 1000 if n_windows else 0.0,
        "grammar_avg_ms_per_word": grammar_total / batch_size * 1000,
        "added_ms_per_word": (token_total - grammar_total) / batch_size * 1000,
        "ratio": token_total / grammar_total if grammar_total else float("inf"),
        "clean_token_avg_ms": clean_token / batch_size * 1000,
        "clean_grammar_avg_ms_per_window": clean_grammar / n_windows * 1000 if n_windows else 0.0,
        "clean_grammar_avg_ms_per_word": clean_grammar / batch_size * 1000,
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the benchmark as the block of text the report quotes."""
    return "\n".join(
        [
            "Q4 Speed Demon benchmark",
            "=" * 46,
            f"Batch                       : {report['batch_size']} corrupted words",
            f"Grammar trigger interval    : every {report['trigger_n']} words "
            f"({report['n_windows']} windows)",
            "",
            f"Segmentation + spelling     : {report['token_total_seconds']:.3f} s total, "
            f"{report['token_avg_ms']:.3f} ms per word",
            f"Grammar trigger only        : {report['grammar_total_seconds']:.3f} s total, "
            f"{report['grammar_avg_ms_per_window']:.3f} ms per window, "
            f"{report['grammar_avg_ms_per_word']:.3f} ms per word",
            "",
            f"Added by the per-token layer: {report['added_ms_per_word']:.3f} ms per word "
            f"({report['ratio']:.1f}x the grammar layer)",
            "",
            "Control batch of ordinary vocabulary words",
            f"Segmentation + spelling     : {report['clean_token_avg_ms']:.3f} ms per word",
            f"Grammar trigger only        : {report['clean_grammar_avg_ms_per_window']:.3f} ms "
            f"per window, {report['clean_grammar_avg_ms_per_word']:.3f} ms per word",
        ]
    )
