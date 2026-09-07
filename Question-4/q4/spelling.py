"""
spelling.py — Q4 Spelling Alert Wrapper

Wraps Q3's SpellingCorrector for per-token SPELL-ALERT checks.

SPELL-ALERT logic:
    For any token that is still NOT in the Q3 vocabulary after the
    segmentation check has passed, run Q3's candidate generation
    (Method B — SymDel) and suggest the highest-unigram-frequency
    candidate as the non-word-error correction.

Method B (SymDel) is chosen because:
    - Q3's Phase 5 Speed Demon benchmark showed it is 10–50× faster
      than Method A (brute-force edit enumeration) at query time.
    - The pre-computation (build_symdel_index) is done once at startup.
    - Coverage is identical to Method A (both reach full edit-distance-1).
    - Low latency is critical here — each token arrives during simulated
      live typing, so any per-token cost must be << the inter-word delay.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from corrector import SpellingCorrector

_Q3_SRC = Path(__file__).resolve().parents[2] / "Question-3" / "q3_spelling_corrector"
if str(_Q3_SRC) not in sys.path:
    sys.path.insert(0, str(_Q3_SRC))


class SpellResult(NamedTuple):
    fired: bool
    original_token: str
    suggestion: str        # best candidate (or same as original if no candidates)
    candidates: list[str]  # all candidates, ranked by unigram frequency
    latency_ms: float


def check_token_spelling(
    token: str,
    corrector: "SpellingCorrector",
    vocab: set[str],
) -> SpellResult:
    """Run the SPELL-ALERT check for ``token``.

    Called only on tokens that are NOT in the vocabulary after the
    segmentation check.  If the token is in vocab, returns immediately
    with ``fired=False``.

    Parameters
    ----------
    token     : Lowercase surface form (alphabetic characters only).
    corrector : Q3 SpellingCorrector instance (already initialised).
    vocab     : Q3 vocabulary set for the fast in-vocab check.
    """
    t0 = time.perf_counter()

    cleaned = token.lower().strip()
    alpha = "".join(ch for ch in cleaned if ch.isalpha())

    if not alpha or alpha in vocab:
        return SpellResult(
            fired=False,
            original_token=token,
            suggestion=token,
            candidates=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    result = corrector.correct_nonword(alpha)
    latency = (time.perf_counter() - t0) * 1000

    return SpellResult(
        fired=result["changed"],
        original_token=token,
        suggestion=result["corrected"],
        candidates=result["candidates"],
        latency_ms=latency,
    )
