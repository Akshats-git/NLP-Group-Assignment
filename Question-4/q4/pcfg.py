"""
pcfg.py - PCFG Constituency Parser for Q4 Part 2

Trains a Probabilistic Context-Free Grammar on Penn Treebank data (NLTK sample)
and parses sentences using Viterbi CKY chart parsing.
"""
from __future__ import annotations

import copy
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import nltk
from nltk import Nonterminal, induce_pcfg
from nltk.corpus import treebank
from nltk.grammar import PCFG

from q4.tagset import reconcile_tag_sequence

PCFG_CACHE_PATH = Path(__file__).parent.parent / "models" / "q4_pcfg.pkl"

NEG_INF = float("-inf")


def _ensure_treebank():
    try:
        _ = treebank.fileids()
    except LookupError:
        nltk.download("treebank", quiet=True)


def train_pcfg(cache: bool = True, min_count: int = 2) -> PCFG:
    """
    Induce a CNF PCFG from Penn Treebank.
    Rules appearing less than min_count times are filtered to keep model size reasonable (~18MB).
    """
    if cache and PCFG_CACHE_PATH.exists():
        with PCFG_CACHE_PATH.open("rb") as f:
            return pickle.load(f)

    _ensure_treebank()

    raw_prods = []
    for fileid in treebank.fileids():
        for tree in treebank.parsed_sents(fileid):
            try:
                t = copy.deepcopy(tree)
                t.collapse_unary()
                t.chomsky_normal_form(horzMarkov=2)
                raw_prods.extend(t.productions())
            except Exception:
                continue

    counts = Counter(raw_prods)
    weighted_prods = []
    for prod, count in counts.items():
        if count >= min_count and len(prod.rhs()) > 0:
            weighted_prods.extend([prod] * count)

    if not weighted_prods:
        raise RuntimeError("No valid PCFG productions found.")

    grammar = induce_pcfg(Nonterminal("S"), weighted_prods)

    if cache:
        PCFG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PCFG_CACHE_PATH.open("wb") as f:
            pickle.dump(grammar, f, protocol=pickle.HIGHEST_PROTOCOL)

    return grammar


_INDEX_CACHE: dict[int, tuple] = {}


def _get_indices(grammar: PCFG) -> tuple[dict, dict]:
    gid = id(grammar)
    if gid in _INDEX_CACHE:
        return _INDEX_CACHE[gid]

    binary_rules: dict[tuple, list] = {}
    lexical_rules: dict[str, list] = {}

    for prod in grammar.productions():
        prob = prod.prob()
        if prob <= 0:
            continue
        lp = math.log(prob)
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


def cky_parse(words: tuple[str, ...], tags: tuple[str, ...], grammar: PCFG) -> tuple[Optional[str], float]:
    """
    Probabilistic CKY Viterbi parser.
    Returns the bracketed parse tree and its log probability, or (None, -inf)
    if the sentence cannot be parsed.
    """
    if not words:
        return None, NEG_INF

    binary_rules, lexical_rules = _get_indices(grammar)
    n = len(words)

    table: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]

    # Fill base case (lexical)
    for i, (word, tag) in enumerate(zip(words, tags)):
        entries = lexical_rules.get(word.lower(), [])
        if not entries:
            entries = [(tag, 0.0)]
        for sym, lp in entries:
            if sym not in table[i][i] or lp > table[i][i][sym][0]:
                table[i][i][sym] = (lp, None)

    # Bottom-up chart filling
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

    top = table[0][n - 1]
    if not top:
        return None, NEG_INF

    root = "S" if "S" in top else max(top, key=lambda s: top[s][0])
    log_prob = top[root][0]

    def _trace(sym: str, i: int, j: int) -> str:
        entry = table[i][j].get(sym)
        if entry is None or entry[1] is None:
            return f"({sym} {words[i]})"
        k, B, C = entry[1]
        return f"({sym} {_trace(B, i, k)} {_trace(C, k + 1, j)})"

    try:
        return _trace(root, 0, n - 1), log_prob
    except RecursionError:
        return None, NEG_INF


def parse_sentence(words: tuple[str, ...], universal_tags: tuple[str, ...], grammar: PCFG) -> dict:
    ptb_tags = reconcile_tag_sequence(universal_tags)
    parse_str, log_prob = cky_parse(words, ptb_tags, grammar)
    return {
        "parseable": parse_str is not None,
        "parse": parse_str,
        "log_prob": log_prob,
        "words": words,
        "ptb_tags": ptb_tags,
    }
