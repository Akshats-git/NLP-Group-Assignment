"""
run_passage_demo.py - One full pass of the editor over a sampled passage

Streams a randomly sampled passage through the same checks the Streamlit app
runs, prints every alert as it fires, and then prints the Part 4 summary table,
the live-versus-final agreement counts and the latency figures. The transcripts
quoted in the report come straight out of this script.

Run from Question-4/:
    venv/bin/python scripts/run_passage_demo.py --seed 7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from q4.analysis import agreement_summary, analyse_document, load_floors, render_table, summary_rows
from q4.grammar import TRIGGER_N
from q4.model_loader import load_all_models
from q4.passage import MERGE_PROB, passage_token_stream, sample_passage
from q4.pcfg import train_pcfg
from q4.runner import run_passage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="fix the passage and the merges")
    parser.add_argument("--merge-prob", type=float, default=MERGE_PROB, help="space-drop probability p")
    parser.add_argument("--trigger-n", type=int, default=TRIGGER_N, help="words between grammar checks")
    parser.add_argument("--sentences", type=int, default=6, help="passage length in sentences")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = load_all_models()
    grammar = train_pcfg()
    floors = load_floors(models, grammar, models["q1_tagger"])

    passage, corpus_name = sample_passage(seed=args.seed, n_sentences=args.sentences)
    tokens = list(passage_token_stream(passage, p=args.merge_prob, seed=args.seed))

    print("=" * 78)
    print("Q4 live editor run")
    print(f"corpus {corpus_name}, {len(passage)} sentences, {len(tokens)} tokens typed")
    print(f"merge probability p = {args.merge_prob}, grammar trigger N = {args.trigger_n}")
    print(
        f"perplexity threshold = {floors['window_ppl_p95']:.0f} "
        "(95th percentile of clean Brown windows)"
    )
    print("=" * 78)
    print()
    print("Typed text as it arrives")
    print("  " + " ".join(token for token, _ in tokens))
    print()
    print("Live alerts")

    def show(kind: str, detail: str, latency_ms: float) -> None:
        print(f"  [{kind}-ALERT] {detail}  ({latency_ms:.2f} ms)")

    run = run_passage(
        tokens,
        models,
        trigger_n=args.trigger_n,
        delay=args.delay,
        on_alert=show,
    )

    started = time.perf_counter()
    analyses = analyse_document(run.document.sentences, grammar, models, floors)
    analysis_ms = (time.perf_counter() - started) * 1000

    live_flagged = list(run.live_flagged)
    while len(live_flagged) < len(analyses):
        live_flagged.append(False)

    print()
    print("Corrected text")
    for item in analyses:
        print(f"  {item.index}. {item.text}")

    print()
    print("Part 4 summary table")
    print(render_table(summary_rows(analyses)))

    print()
    print("Live alerts against the final verdict")
    agreement = agreement_summary(analyses, live_flagged)
    print(f"  sentences               : {agreement['sentences']}")
    print(f"  flagged by both layers  : {agreement['both_flagged']}")
    print(f"  live alert only         : {agreement['live_only']}")
    print(f"  final verdict only      : {agreement['final_only']}")
    print(f"  clean for both          : {agreement['neither']}")
    print(f"  agreement               : {agreement['agreement']:.0%}")

    print()
    print("Latency")
    print(f"  alerts fired            : {run.counts}")
    print(
        f"  per-token check         : {run.average_token_ms:.2f} ms average "
        f"over {len(run.token_latencies)} tokens"
    )
    print(
        f"  per-trigger check       : {run.average_trigger_ms:.2f} ms average "
        f"over {len(run.trigger_latencies)} triggers"
    )
    print(f"  end-of-passage analysis : {analysis_ms:.1f} ms for {len(analyses)} sentences")
    print(f"  total                   : {run.wall_ms + analysis_ms:.1f} ms")


if __name__ == "__main__":
    main()
