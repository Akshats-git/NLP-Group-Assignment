"""
cli.py — Q3 Phase 6

Continuous terminal interface for the spelling corrector. This module
contains NO correction logic of its own — it only:
    1. loads the already-trained Phase 2 artifacts (models/q3_language_models.pkl)
    2. builds the Phase 3 SymDel index once, at startup
    3. constructs a corrector.SpellingCorrector (Phase 4)
    4. loops: read a sentence, call corrector.correct_sentence(), print
       the result, measure latency, repeat until "exit"

Nothing here retrains or rebuilds the Brown language model, duplicates
candidate generation, or reimplements correction decisions — all of that
is reused as-is from corpus_models.py, candidates.py, and corrector.py.

Not implemented here (out of scope for Phase 6): evaluation, benchmarking,
Streamlit/Q4 integration, POS tagging, parsing.
"""

import time
from typing import Callable, Dict, List

from candidates import build_symdel_index
from corpus_models import load_models
from corrector import SpellingCorrector

EXIT_COMMAND = "exit"


# ---------------------------------------------------------------------------
# Startup: model loading + corrector construction (kept separate from the
# input loop, per the Phase 6 architecture requirement, so Q4 can later
# call create_corrector() on its own without pulling in the CLI loop).
# ---------------------------------------------------------------------------

def create_corrector(
    method: str = "B",
    real_word_threshold: float = 1.1,
) -> SpellingCorrector:
    """
    Load the trained Phase 2 artifacts, build the Phase 3 SymDel index,
    and construct a SpellingCorrector — nothing is trained or rebuilt
    here, only loaded/assembled.

    Parameters
    ----------
    method : str
        Candidate-generation method for the corrector. Defaults to "B"
        (Symmetric Delete) since it was demonstrated to be the faster
        method in the Phase 5 Speed Demon benchmark — appropriate for a
        live, interactive CLI where per-sentence latency is visible to
        the user. "A" and "both" remain fully supported by
        SpellingCorrector and can be passed here unchanged; this default
        does not remove or restrict either option.
    real_word_threshold : float
        Passed straight through to SpellingCorrector (see corrector.py
        for the log-probability-margin explanation).

    Returns
    -------
    SpellingCorrector
        Ready to use — callers should reuse this single instance across
        every sentence in a session rather than rebuilding it per call.

    Notes
    -----
    load_models() returns a dict with keys "vocab", "unigram_counts",
    "bigram_counts", "vocab_size", "k" (see corpus_models.save_models);
    build_symdel_index(vocab) takes just the vocab set and returns the
    deletion-index dict SpellingCorrector expects. Both integration
    points match SpellingCorrector's constructor signature from Phase 4
    without any changes needed on either side.
    """
    models = load_models()
    vocab = models["vocab"]
    unigram_counts = models["unigram_counts"]
    bigram_counts = models["bigram_counts"]
    vocab_size = models["vocab_size"]
    k = models["k"]

    # Built once, here, at startup — never inside the per-sentence timed
    # region in run_cli(), per the latency-measurement requirement.
    symdel_index = build_symdel_index(vocab)

    return SpellingCorrector(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        method=method, real_word_threshold=real_word_threshold, k=k,
    )


# ---------------------------------------------------------------------------
# Change highlighting
# ---------------------------------------------------------------------------

def render_highlighted(corrected_sentence: str, metadata: List[Dict]) -> str:
    """
    Rebuild the corrected sentence with each changed word wrapped in
    **asterisks**, per the assignment's Part 5 requirement to visually
    highlight changed words.

    `corrected_sentence.split()` and `metadata` are guaranteed to line up
    one-to-one and in order: correct_sentence() builds both from the same
    whitespace-split `raw_tokens` list, substituting only the token core
    (punctuation/casing preserved), so no re-tokenization or fuzzy
    matching is needed here — position i in one is position i in the
    other.

    Returns
    -------
    str
        The sentence with every entry whose `changed` flag is set
        rendered as `**token**` in place.
    """
    tokens = corrected_sentence.split()
    if len(tokens) != len(metadata):
        # Defensive fallback (should not happen given the 1:1 contract
        # above) — return the sentence unhighlighted rather than
        # mis-wrap the wrong token.
        return corrected_sentence
    return " ".join(
        f"**{tok}**" if entry.get("changed") else tok
        for tok, entry in zip(tokens, metadata)
    )


def highlight_changes(metadata: List[Dict]) -> str:
    """
    Render a one-line summary of which words changed, from the metadata
    list returned by SpellingCorrector.correct_sentence().

    Each metadata entry already carries "original", "corrected", and
    "changed" (set by correct_nonword/correct_realword, or the
    pass-through record for non-alphabetic tokens) — this function reads
    those fields directly rather than re-deriving what changed, so no
    change to correct_sentence()'s output format was necessary to
    support highlighting (see integration notes in the Phase 6 write-up).

    Returns
    -------
    str
        e.g. "Changes: hav -> have, sentnce -> sentence", or
        "Changes: none" if nothing was changed.
    """
    changed_pairs = [
        f"{entry['original']} -> {entry['corrected']}"
        for entry in metadata
        if entry.get("changed")
    ]
    if not changed_pairs:
        return "Changes: none"
    return "Changes: " + ", ".join(changed_pairs)


# ---------------------------------------------------------------------------
# Input loop
# ---------------------------------------------------------------------------

def run_cli(
    corrector: SpellingCorrector,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> None:
    """
    Continuously read a sentence, correct it, and print the result, until
    the input is exactly "exit".

    `input_func` and `output_func` are injectable (defaulting to the
    real `input`/`print`) so this loop has no hidden global state and can
    be driven programmatically — e.g. by a test harness, or later by a
    Q4 wrapper that wants the same loop behaviour with a different I/O
    surface — without modifying this function.

    Per-sentence latency is measured with time.perf_counter() around
    exactly the `corrector.correct_sentence(sentence)` call — nothing
    else (not the print statements, not the input read) is included in
    the timed region, and model/index construction already happened
    before run_cli was ever called (see create_corrector), so startup
    cost never leaks into a per-sentence number.
    """
    while True:
        try:
            raw = input_func("> ")
        except EOFError:
            # Stdin closed (e.g. piped input ran out) — exit cleanly
            # rather than crashing.
            output_func("")
            output_func("End of input. Goodbye.")
            break

        sentence = raw.strip()

        if sentence == EXIT_COMMAND:
            output_func("Goodbye.")
            break

        if sentence == "":
            output_func("(empty input — nothing to correct)")
            continue

        start = time.perf_counter()
        corrected_sentence, metadata = corrector.correct_sentence(sentence)
        elapsed_seconds = time.perf_counter() - start

        output_func(f"Original:  {sentence}")
        output_func(f"Corrected: {render_highlighted(corrected_sentence, metadata)}")
        output_func(highlight_changes(metadata))
        output_func(f"Latency: {elapsed_seconds * 1000:.3f} ms")
        output_func("")  # blank line between turns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point. Wrapped in its own function (rather than running at
    import time) so `import cli` — as Q4 will do — never starts the
    interactive loop as a side effect; only `python cli.py` does.
    """
    print("Q3 Spelling Corrector")
    print("Type a sentence to correct.")
    print("Type 'exit' to quit.")
    print()

    corrector = create_corrector(method="B")
    run_cli(corrector)


if __name__ == "__main__":
    main()
