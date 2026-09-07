"""
tune_lm_k.py - Sweep the add-k constant for Q4's grammar models

Fits the bigram and the trigram model at each candidate k on the Brown train
split and reports perplexity on the dev split, which is how the k in
q4/ngram_lm.py was chosen.

Run from Question-4/:  venv/bin/python scripts/tune_lm_k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from q4.ngram_lm import tune_k


def main() -> None:
    rows = tune_k()

    print("Add-k sweep on the Brown dev split")
    print(f"{'k':>8}  {'bigram dev PPL':>16}  {'trigram dev PPL':>16}")
    for row in rows:
        print(
            f"{row['k']:>8}  {row['bigram_dev_ppl']:>16.1f}  {row['trigram_dev_ppl']:>16.1f}"
        )

    best_bigram = min(rows, key=lambda r: r["bigram_dev_ppl"])
    best_trigram = min(rows, key=lambda r: r["trigram_dev_ppl"])
    print()
    print(f"Best bigram  k = {best_bigram['k']} at PPL {best_bigram['bigram_dev_ppl']:.1f}")
    print(f"Best trigram k = {best_trigram['k']} at PPL {best_trigram['trigram_dev_ppl']:.1f}")


if __name__ == "__main__":
    main()
