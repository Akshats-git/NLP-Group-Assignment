"""
spelling.py - Spelling Alert Wrapper

Wraps Q3's SpellingCorrector for non-word spelling error checks using Method B (SymDel).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from corrector import SpellingCorrector

Q3_DIR = Path(__file__).resolve().parents[2] / "Question-3" / "q3_spelling_corrector"
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))


class SpellResult(NamedTuple):
    fired: bool
    original_token: str
    suggestion: str
    candidates: list[str]
    latency_ms: float


def check_token_spelling(token: str, corrector: "SpellingCorrector", vocab: set[str]) -> SpellResult:
    """Run non-word spelling check on token."""
    t0 = time.perf_counter()

    cleaned = token.lower().strip()
    alpha = "".join(c for c in cleaned if c.isalpha())

    if not alpha or alpha in vocab:
        return SpellResult(False, token, token, [], (time.perf_counter() - t0) * 1000)

    res = corrector.correct_nonword(alpha)
    return SpellResult(
        fired=res["changed"],
        original_token=token,
        suggestion=res["corrected"],
        candidates=res["candidates"],
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
