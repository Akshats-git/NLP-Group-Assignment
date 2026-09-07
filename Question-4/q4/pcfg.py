"""
pcfg.py — Q4 Part 2: PCFG Constituency Parser

Trains a Probabilistic Context-Free Grammar from the Penn Treebank sample
bundled with NLTK (nltk.corpus.treebank), then implements a
Viterbi/CKY-based most-probable-parse function.

Training
---------
  Source  : nltk.corpus.treebank (~3,900 parsed sentences)
  Method  : nltk.induce_pcfg over productions collected from CNF-binarised trees.
  Key fix : Productions MUST be collected AFTER chomsky_normal_form() so the
            grammar contains only binary + lexical rules, which is what the CKY
            algorithm requires.  The original trees are deep-copied first so
            the NLTK corpus cache is not mutated.

CKY / Viterbi Parser
---------------------
  Standard probabilistic bottom-up CKY chart.
  table[i][j] maps symbol → (log_prob, back_pointer).
  Unknown surface words fall back to their (reconciled) PTB POS tag as the
  pre-terminal, which keeps the chart alive without crashing.
  UNPARSEABLE sentences return None gracefully.

Tagset reconciliation
---------------------
  Q1 tags are in the 12-class universal set; the PCFG uses Penn Treebank tags.
  reconcile_tag_sequence() (q4/tagset.py) provides the mapping; see that
  module for the documented accuracy trade-off.
"""
from __future__ import annotations

import copy
import math
import pickle
import sys
from pathlib import Path
from typing import Optional

import nltk
from nltk import induce_pcfg, Nonterminal
from nltk.corpus import treebank
from nltk.grammar import PCFG

_Q4_ROOT = Path(__file__).resolve().parents[1]
if str(_Q4_ROOT) not in sys.path:
    sys.path.insert(0, str(_Q4_ROOT))

from q4.tagset import reconcile_tag_sequence

PCFG_CACHE_PATH = Path(__file__).parent.parent / "models" / "q4_pcfg.pkl"


# ---------------------------------------------------------------------------
# NLTK data guard
# ---------------------------------------------------------------------------

def _ensure_treebank() -> None:
    try:
        _ = treebank.fileids()
    except LookupError:
        nltk.download("treebank", quiet=True)


# ---------------------------------------------------------------------------
# PCFG Training
# ---------------------------------------------------------------------------

def train_pcfg(cache: bool = True) -> PCFG:
    """Induce a CNF PCFG from the Penn Treebank sample.

    Productions are extracted from trees AFTER binarisation so the resulting
    grammar is purely binary + lexical — exactly what CKY needs.

    Parameters
    ----------
    cache : Pickle the grammar after induction so subsequent calls are instant.
    """
    if cache and PCFG_CACHE_PATH.exists():
        with PCFG_CACHE_PATH.open("rb") as fh:
            return pickle.load(fh)

    _ensure_treebank()
    print("[Q4-PCFG] Inducing PCFG from Penn Treebank … (one-time, ~10 s)")

    raw_productions: list = []
    skipped = 0
    for fileid in treebank.fileids():
        for tree in treebank.parsed_sents(fileid):
            try:
                t = copy.deepcopy(tree)       # don't mutate the cached corpus
                t.collapse_unary()
                t.chomsky_normal_form(horzMarkov=2)
                raw_productions.extend(t.productions())
            except Exception:
                skipped += 1

    # Filter: keep only productions that are valid (non-empty RHS).
    productions = [p for p in raw_productions if len(p.rhs()) > 0]

    if not productions:
        raise RuntimeError(
            "[Q4-PCFG] No valid productions found — check Penn Treebank download."
        )

    grammar = induce_pcfg(Nonterminal("S"), productions)
    print(f"[Q4-PCFG] Done: {len(grammar.productions())} productions "
          f"({skipped} trees skipped).")

    if cache:
        PCFG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PCFG_CACHE_PATH.open("wb") as fh:
            pickle.dump(grammar, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Q4-PCFG] Grammar cached → {PCFG_CACHE_PATH}")

    return grammar


# ---------------------------------------------------------------------------
# Index builder (binary + lexical rules, indexed by RHS for O(1) lookup)
# ---------------------------------------------------------------------------

_INDEX_CACHE: dict[int, tuple] = {}


def _get_indices(grammar: PCFG) -> tuple[dict, dict]:
    gid = id(grammar)
    if gid in _INDEX_CACHE:
        return _INDEX_CACHE[gid]

    binary_rules: dict[tuple, list] = {}
    lexical_rules: dict[str, list]  = {}

    for prod in grammar.productions():
        prob = prod.prob()
        if prob <= 0:
            continue
        lp  = math.log(prob)
        rhs = prod.rhs()
        lhs = str(prod.lhs())

        if len(rhs) == 2 and isinstance(rhs[0], Nonterminal):
            key = (str(rhs[0]), str(rhs[1]))
            binary_rules.setdefault(key, []).append((lhs, lp))
        elif len(rhs) == 1 and not isinstance(rhs[0], Nonterminal):
            word = str(rhs[0]).lower()
            lexical_rules.setdefault(word, []).append((lhs, lp))

    _INDEX_CACHE[gid] = (binary_rules, lexical_rules)
    return binary_rules, lexical_rules


# ---------------------------------------------------------------------------
# Viterbi CKY Parser
# ---------------------------------------------------------------------------

def cky_parse(
    words: tuple[str, ...],
    tags: tuple[str, ...],
    grammar: PCFG,
) -> Optional[str]:
    """Most-probable-parse via probabilistic CKY.

    Parameters
    ----------
    words   : Lowercase surface tokens.
    tags    : PTB POS tags (already reconciled from universal).
    grammar : CNF PCFG from train_pcfg().

    Returns
    -------
    Bracket-notation parse string, or None (UNPARSEABLE).
    Never raises — failures are caught and return None.
    """
    if not words:
        return None

    binary_rules, lexical_rules = _get_indices(grammar)
    n = len(words)

    # table[i][j]: symbol → (log_prob, back_pointer)
    # back_pointer = None          for lexical leaves
    # back_pointer = (k, B, C)    for binary splits
    table: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]

    # ---- 1. Diagonal: lexical rules ----
    for i, (word, tag) in enumerate(zip(words, tags)):
        entries = lexical_rules.get(word.lower(), [])
        if not entries:
            # Unknown word — use the reconciled PTB tag as pre-terminal (p=1)
            entries = [(tag, 0.0)]
        for sym, lp in entries:
            if sym not in table[i][i] or lp > table[i][i][sym][0]:
                table[i][i][sym] = (lp, None)

    # ---- 2. Upper triangle: binary rules (increasing span) ----
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for B, (lp_B, _) in table[i][k].items():
                    for C, (lp_C, _) in table[k + 1][j].items():
                        matches = binary_rules.get((B, C))
                        if not matches:
                            continue
                        for A, rule_lp in matches:
                            total = rule_lp + lp_B + lp_C
                            cur = table[i][j].get(A)
                            if cur is None or total > cur[0]:
                                table[i][j][A] = (total, (k, B, C))

    # ---- 3. Find best spanning root ----
    top = table[0][n - 1]
    if not top:
        return None  # UNPARSEABLE

    root = "S" if "S" in top else max(top, key=lambda s: top[s][0])

    # ---- 4. Traceback → bracket notation ----
    def _trace(sym: str, i: int, j: int) -> str:
        entry = table[i][j].get(sym)
        if entry is None or entry[1] is None:
            return f"({sym} {words[i]})"
        k, B, C = entry[1]
        return f"({sym} {_trace(B, i, k)} {_trace(C, k + 1, j)})"

    try:
        return _trace(root, 0, n - 1)
    except RecursionError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sentence(
    words: tuple[str, ...],
    universal_tags: tuple[str, ...],
    grammar: PCFG,
) -> dict:
    """Parse a word sequence annotated with Q1 universal tags.

    Performs tagset reconciliation internally (universal → PTB).

    Returns
    -------
    dict:
        parseable : bool
        parse     : str | None  (bracket notation)
        words     : tuple[str, ...]
        ptb_tags  : tuple[str, ...]  (reconciled)
    """
    ptb_tags  = reconcile_tag_sequence(universal_tags)
    parse_str = cky_parse(words, ptb_tags, grammar)
    return {
        "parseable": parse_str is not None,
        "parse":     parse_str,
        "words":     words,
        "ptb_tags":  ptb_tags,
    }
