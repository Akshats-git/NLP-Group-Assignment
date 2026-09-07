"""
calibrate_pcfg.py - Measure what a normal parse probability looks like

Parses a sample of Penn Treebank sentences through the same path a live sentence
takes, which means Q1 tags mapped onto Penn Treebank tags, and writes the
resulting floors to models/q4_pcfg_floors.json. Part 4 reads that file to decide
whether a parse is good enough to trust or is an outlier.

The same pass reports how often the mapped tags match the treebank's own tags,
which is the price of the tagset reconciliation.

Run from Question-4/:  venv/bin/python scripts/calibrate_pcfg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from q4.analysis import PCFG_FLOOR_PATH, calibrate_pcfg_floors
from q4.model_loader import load_all_models
from q4.pcfg import train_pcfg


def main() -> None:
    models = load_all_models()
    grammar = train_pcfg()
    floors = calibrate_pcfg_floors(grammar, models["q1_tagger"])

    print("PCFG calibration on the Penn Treebank sample")
    print(
        f"  sentences sampled : {int(floors['n_sampled'])} "
        f"(3 to {int(floors['max_words'])} words)"
    )
    print(f"  parsed            : {floors['parse_rate']:.1%}")
    print(f"  tag agreement     : {floors['tag_agreement']:.1%} against the treebank tags")
    print(
        f"  per-word log prob : median {floors['pcfg_median']:.2f}, "
        f"p10 {floors['pcfg_p10']:.2f}, p02 {floors['pcfg_p02']:.2f}"
    )
    print(f"  written to        : {PCFG_FLOOR_PATH}")


if __name__ == "__main__":
    main()
