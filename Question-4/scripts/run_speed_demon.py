"""
run_speed_demon.py - Part 5 Speed Demon benchmark

Builds a batch of exactly 1,000 corrupted words with Question 3's generator and
pushes it through the two live layers separately, reporting what each costs.

Run from Question-4/:  venv/bin/python scripts/run_speed_demon.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from q4.grammar import TRIGGER_N
from q4.model_loader import load_all_models
from q4.speed_demon import BATCH_SIZE, DEFAULT_SEED, format_report, run_speed_demon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--trigger-n", type=int, default=TRIGGER_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_all_models()
    report = run_speed_demon(
        models,
        seed=args.seed,
        batch_size=args.batch_size,
        trigger_n=args.trigger_n,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
