"""
benchmark.py — Q3 Phase 5 (part 2: Speed Demon benchmark)

Times Method A vs Method B on the EXACT SAME batch of 1,000 misspelled
(non-word) tokens, isolating non-word correction as the assignment
specifies. Reuses:
    - evaluation.corrupt_word_single_edit for batch generation (not
      duplicated here)
    - corrector.SpellingCorrector.correct_nonword for the timed logic
      itself (not duplicated/reimplemented here)

Model loading and SymDel index construction happen BEFORE timing starts
(outside the timed block) — see run_speed_demon's docstring.

This module does NOT implement accuracy evaluation, a CLI, or Q4
integration.
"""

import random
import time
from typing import Dict, List

from corrector import SpellingCorrector
from evaluation import corrupt_word_single_edit

DEFAULT_BATCH_SIZE = 1000


def generate_benchmark_words(
    vocab: set, rng: random.Random, count: int = DEFAULT_BATCH_SIZE
) -> List[str]:
    """
    Build a batch of exactly `count` non-word misspelled tokens by
    corrupting randomly chosen vocabulary words with a single edit.

    Reuses evaluation.corrupt_word_single_edit (want_nonword=True) rather
    than reimplementing corruption logic, so the benchmark's notion of
    "a misspelled word" is identical to the one used for accuracy
    evaluation.

    `vocab` is sorted once into a list so rng.choice() draws from a fixed,
    reproducible ordering (sets have no stable iteration order across
    runs/interpreters, which would otherwise break reproducibility even
    with a fixed seed).

    Raises
    ------
    RuntimeError
        If `count` valid words cannot be generated within a generous
        attempt budget — surfaced rather than silently returning a
        short/invalid batch, since the assignment requires EXACTLY 1,000.
    """
    vocab_list = sorted(vocab)
    words: List[str] = []
    attempts = 0
    max_total_attempts = count * 50  # generous safety cap

    while len(words) < count and attempts < max_total_attempts:
        attempts += 1
        base_word = rng.choice(vocab_list)
        result = corrupt_word_single_edit(base_word, vocab, rng, want_nonword=True)
        if result is None:
            continue
        corrupted_word, _edit_type = result
        words.append(corrupted_word)

    if len(words) < count:
        raise RuntimeError(
            f"Only generated {len(words)} of {count} required benchmark "
            "words within the attempt budget — vocabulary may be too "
            "small or too dense for further corruptions to miss it."
        )
    return words


def run_speed_demon(
    vocab: set,
    unigram_counts,
    bigram_counts,
    vocab_size: int,
    symdel_index: Dict[str, List[str]],
    seed: int = 123,
    batch_size: int = DEFAULT_BATCH_SIZE,
    real_word_threshold: float = 2.0,
    k: float = 1.0,
) -> Dict:
    """
    Time Method A vs Method B on the same 1,000-word batch, running each
    through the FULL non-word correction pipeline
    (SpellingCorrector.correct_nonword), which internally calls the
    respective candidate generator and then unigram-ranks the result —
    this matches the assignment's "non-word error correction logic",
    not just raw candidate generation.

    What is excluded from the timed sections (per the assignment):
        - loading/building vocab, unigram_counts, bigram_counts,
          vocab_size (Phase 2 — must already be loaded before this
          function is called)
        - building symdel_index (Phase 3's build_symdel_index — must
          already be built and passed in; this function never calls it)
        - generating the benchmark batch itself (done once, before either
          timed loop, so batch generation cost isn't attributed to either
          method)

    What IS timed: exactly the loop of `count` calls to
    `corrector.correct_nonword(word)`, separately for method="A" and
    method="B", using `time.perf_counter()` (monotonic, sub-microsecond
    resolution — appropriate for micro-benchmarking).

    Fairness
    --------
    - Same batch, same order, for both methods (batch is generated once
      and reused, not regenerated per method).
    - Both correctors share the same underlying vocab/unigram/bigram/
      symdel_index objects — only `method` differs, so any timing
      difference is attributable to candidate generation strategy alone,
      not to different models or reference data.
    - Each corrector is only used for its OWN timed loop, but both are
      constructed before either loop starts, so object-construction cost
      (trivial here — no training happens in __init__) isn't lopsided.

    Returns
    -------
    dict with total/average times for each method, which method was
    faster, and a speedup ratio.
    """
    rng = random.Random(seed)
    batch = generate_benchmark_words(vocab, rng, count=batch_size)

    corrector_a = SpellingCorrector(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        method="A", real_word_threshold=real_word_threshold, k=k,
    )
    corrector_b = SpellingCorrector(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        method="B", real_word_threshold=real_word_threshold, k=k,
    )

    start_a = time.perf_counter()
    for word in batch:
        corrector_a.correct_nonword(word)
    total_a = time.perf_counter() - start_a

    start_b = time.perf_counter()
    for word in batch:
        corrector_b.correct_nonword(word)
    total_b = time.perf_counter() - start_b

    avg_a = total_a / batch_size
    avg_b = total_b / batch_size

    if total_a < total_b:
        faster = "A"
    elif total_b < total_a:
        faster = "B"
    else:
        faster = "tie"

    speedup_a_over_b = (total_a / total_b) if total_b > 0 else float("inf")

    return {
        "batch_size": batch_size,
        "method_a_total_seconds": total_a,
        "method_b_total_seconds": total_b,
        "method_a_avg_seconds": avg_a,
        "method_b_avg_seconds": avg_b,
        "faster_method": faster,
        "speedup_a_over_b": speedup_a_over_b,
    }


def format_speed_demon_report(report: Dict) -> str:
    """Render the Speed Demon results as a short human-readable summary."""
    lines = [
        "Q3 Speed Demon Benchmark",
        "=" * 40,
        f"Batch size: {report['batch_size']} words",
        f"Method A total: {report['method_a_total_seconds']:.4f} s "
        f"(avg {report['method_a_avg_seconds'] * 1000:.4f} ms/word)",
        f"Method B total: {report['method_b_total_seconds']:.4f} s "
        f"(avg {report['method_b_avg_seconds'] * 1000:.4f} ms/word)",
        f"Faster method: {report['faster_method']} "
        f"(Method A took {report['speedup_a_over_b']:.2f}x Method B's total time)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manual orchestration entry point (NOT executed by Claude — run yourself)
# ---------------------------------------------------------------------------

def run_benchmark_from_scratch(seed: int = 123, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict:
    """
    Convenience wrapper: load Phase 2 artifacts, build the Phase 3 SymDel
    index (outside the timed region), then run the Speed Demon benchmark.

    Intended to be run manually — see the commands listed in the Phase 5
    write-up.
    """
    from candidates import build_symdel_index
    from corpus_models import load_models

    models = load_models()
    vocab = models["vocab"]
    unigram_counts = models["unigram_counts"]
    bigram_counts = models["bigram_counts"]
    vocab_size = models["vocab_size"]
    k = models["k"]

    symdel_index = build_symdel_index(vocab)  # built BEFORE timing starts

    return run_speed_demon(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        seed=seed, batch_size=batch_size, k=k,
    )


if __name__ == "__main__":
    report = run_benchmark_from_scratch()
    print(format_speed_demon_report(report))
