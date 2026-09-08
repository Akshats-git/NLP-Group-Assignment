"""
run_experiments.py - The measurements the Q4 report is built on

Four experiments, all printed as one block:

1. A sweep over many sampled passages, aggregating alerts, chosen methods,
   verdicts, the live-versus-final agreement and latency.
2. What the merge probability p does to the number of segmentation alerts, with
   p = 0 giving the segmentation false-alert rate on text that was never merged.
3. What the trigger interval N does to the grammar false-alert rate, measured on
   clean Brown dev text where every alert is by definition a false one.
4. What the tagset reconciliation costs, by parsing treebank sentences with the
   mapped Q1 tags and with the treebank's own tags.

Run from Question-4/:  venv/bin/python scripts/run_experiments.py
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from nltk.corpus import treebank

from q4.analysis import agreement_summary, analyse_document, load_floors
from q4.grammar import TRIGGER_N
from q4.model_loader import load_all_models
from q4.ngram_lm import _windows
from q4.passage import MERGE_PROB, passage_token_stream, sample_passage
from q4.pcfg import cky_parse, train_pcfg
from q4.pipeline import check_window, clean_word
from q4.runner import run_passage
from q4.tagset import reconcile_tag_sequence


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def passage_sweep(models, grammar, floors, n_passages: int) -> None:
    methods: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    alerts: Counter[str] = Counter()
    totals = Counter()
    token_latencies: list[float] = []
    trigger_latencies: list[float] = []
    parsed = 0
    sentences = 0

    for seed in range(n_passages):
        passage, _ = sample_passage(seed=seed)
        tokens = list(passage_token_stream(passage, p=MERGE_PROB, seed=seed))
        run = run_passage(tokens, models)
        analyses = analyse_document(run.document.sentences, grammar, models, floors)

        live_flagged = list(run.live_flagged)
        while len(live_flagged) < len(analyses):
            live_flagged.append(False)
        summary = agreement_summary(analyses, live_flagged)
        for key in ("both_flagged", "live_only", "final_only", "neither"):
            totals[key] += summary[key]

        alerts.update(run.counts)
        token_latencies.extend(run.token_latencies)
        trigger_latencies.extend(run.trigger_latencies)

        for item in analyses:
            sentences += 1
            methods[item.chosen_method] += 1
            verdicts[item.verdict] += 1
            parsed += int(item.pcfg_parseable)

    agreed = totals["both_flagged"] + totals["neither"]

    print(f"1. Passage sweep over {n_passages} sampled passages")
    print(f"   sentences                : {sentences}")
    print(f"   alerts                   : {dict(alerts)}")
    print(f"   sentences that parsed    : {parsed / sentences:.1%}")
    print(f"   chosen method            : {dict(methods)}")
    print(f"   verdicts                 : {dict(verdicts)}")
    print(
        f"   live vs final            : both {totals['both_flagged']}, "
        f"live only {totals['live_only']}, final only {totals['final_only']}, "
        f"clean {totals['neither']}, agreement {agreed / sentences:.0%}"
    )
    print(
        f"   per-token latency        : mean {sum(token_latencies) / len(token_latencies):.2f} ms, "
        f"p95 {percentile(token_latencies, 0.95):.2f} ms, "
        f"max {max(token_latencies):.2f} ms"
    )
    print(
        f"   per-trigger latency      : mean {sum(trigger_latencies) / len(trigger_latencies):.2f} ms, "
        f"p95 {percentile(trigger_latencies, 0.95):.2f} ms"
    )
    print()


def merge_probability_sweep(models, n_passages: int) -> None:
    print("2. Merge probability against segmentation alerts")
    print("   p      segment alerts  per 100 words")
    for p in (0.0, 0.04, 0.08, 0.16):
        segment_alerts = 0
        words = 0
        for seed in range(n_passages):
            passage, _ = sample_passage(seed=seed)
            tokens = list(passage_token_stream(passage, p=p, seed=seed))
            run = run_passage(tokens, models)
            segment_alerts += run.counts["SEGMENT"]
            words += run.document.word_count
        print(f"   {p:<6.2f} {segment_alerts:<15} {segment_alerts / words * 100:.2f}")
    print("   at p = 0 no token was ever merged, so every alert there is a false one")
    print()


def trigger_interval_sweep(models, floors, n_windows: int = 400) -> None:
    from q1.data import load_brown

    corpus = load_brown(tagset="universal")
    dev = [list(s.words) for s in corpus.dev[:600]]

    print("3. Trigger interval against the grammar false-alert rate on clean Brown text")
    print("   N     windows  fired  perplexity only  real-word only  rate")
    for trigger_n in (5, 10, 20):
        windows = list(_windows(dev, trigger_n))[:n_windows]
        fired = perplexity_only = realword_only = 0
        for window in windows:
            result = check_window(window, models, ppl_threshold=floors["window_ppl_p95"])
            if not result.fired:
                continue
            fired += 1
            over_threshold = result.ppl > floors["window_ppl_p95"]
            if over_threshold and not result.real_word_fixes:
                perplexity_only += 1
            elif result.real_word_fixes and not over_threshold:
                realword_only += 1
        print(
            f"   {trigger_n:<5} {len(windows):<8} {fired:<6} {perplexity_only:<16} "
            f"{realword_only:<15} {fired / len(windows):.0%}"
        )
    print("   every window here is real Brown text, so all of these are false alerts")
    print()


def tagset_cost(models, grammar, sample_size: int = 200) -> None:
    rng = random.Random(42)
    usable = []
    for tagged in treebank.tagged_sents():
        pairs = [
            (clean_word(word), tag)
            for word, tag in tagged
            if tag != "-NONE-" and clean_word(word)
        ]
        if 3 <= len(pairs) <= 25:
            usable.append(pairs)
    rng.shuffle(usable)
    sample = usable[:sample_size]

    mapped_parsed = gold_parsed = matched = total = 0
    for pairs in sample:
        words = tuple(word for word, _ in pairs) + (".",)
        gold_tags = tuple(tag for _, tag in pairs) + (".",)
        mapped_tags = reconcile_tag_sequence(models["q1_tagger"].tag(words[:-1])) + (".",)

        total += len(gold_tags)
        matched += sum(1 for a, b in zip(mapped_tags, gold_tags) if a == b)
        mapped_parsed += cky_parse(words, mapped_tags, grammar)[0] is not None
        gold_parsed += cky_parse(words, gold_tags, grammar)[0] is not None

    print(f"4. Tagset reconciliation cost over {len(sample)} treebank sentences")
    print(f"   mapped tags match the treebank tag : {matched / total:.1%}")
    print(f"   parse rate with mapped tags        : {mapped_parsed / len(sample):.1%}")
    print(f"   parse rate with the treebank tags  : {gold_parsed / len(sample):.1%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=int, default=20)
    args = parser.parse_args()

    models = load_all_models()
    grammar = train_pcfg()
    floors = load_floors(models, grammar, models["q1_tagger"])

    print("Q4 experiments")
    print(f"trigger N = {TRIGGER_N}, merge probability p = {MERGE_PROB}, "
          f"perplexity threshold = {floors['window_ppl_p95']:.0f}")
    print("=" * 78)
    print()

    passage_sweep(models, grammar, floors, args.passages)
    merge_probability_sweep(models, args.passages)
    trigger_interval_sweep(models, floors)
    tagset_cost(models, grammar)


if __name__ == "__main__":
    main()
