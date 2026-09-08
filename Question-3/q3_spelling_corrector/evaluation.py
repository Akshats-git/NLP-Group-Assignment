"""
Accuracy evaluation for the spelling corrector.

Builds a reproducible non-word and real-word test set from the Brown
Corpus, then evaluates Method A and Method B on the exact same test
cases via the existing SpellingCorrector. No candidate-generation or
correction logic is duplicated here.

This module does not:
    - build/train the vocabulary, unigram model, bigram model
      (corpus_models.py)
    - implement candidate generation (candidates.py)
    - implement correction decisions (corrector.py)
    - implement the Speed Demon benchmark (that's benchmark.py)
    - implement a CLI or Q4 integration

Reused, not reimplemented:
    - corpus_models.load_brown_sentences() for corpus access
    - candidates.ALPHABET for corruption generation
    - corrector.SpellingCorrector for all correction decisions
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from candidates import ALPHABET
from corrector import SpellingCorrector

# ---------------------------------------------------------------------------
# Test-case data structure
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """
    One evaluation instance: a sentence with exactly one word corrupted by
    exactly one edit operation.

    sentence_words : the ORIGINAL (uncorrupted) sentence, as a list of
        normalized word tokens. Callers substitute `corrupted_word` at
        `index` themselves when they need the corrupted sentence (kept
        this way so the same TestCase can serve as ground truth for
        multiple purposes without mutating shared state).
    index : position of the corrupted word within sentence_words.
    original_word : the ground-truth word (what a correct system should
        restore).
    corrupted_word : the single-edit-corrupted word actually fed to the
        corrector.
    edit_type : one of "deletion", "insertion", "replacement",
        "transposition"; which operation produced corrupted_word.
    """

    sentence_words: List[str]
    index: int
    original_word: str
    corrupted_word: str
    edit_type: str


# ---------------------------------------------------------------------------
# Single-edit corruption (reused by benchmark.py too, not duplicated there)
# ---------------------------------------------------------------------------

def _apply_deletion(word: str, rng: random.Random) -> Optional[str]:
    """Delete one random character. Needs len(word) >= 2 to still be non-empty and different."""
    if len(word) < 2:
        return None
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1:]


def _apply_insertion(word: str, rng: random.Random) -> Optional[str]:
    """Insert one random letter at a random position."""
    if len(word) == 0:
        return None
    i = rng.randrange(len(word) + 1)
    c = rng.choice(ALPHABET)
    return word[:i] + c + word[i:]


def _apply_replacement(word: str, rng: random.Random) -> Optional[str]:
    """Replace one random character with a different random letter."""
    if len(word) == 0:
        return None
    i = rng.randrange(len(word))
    original_char = word[i]
    choices = [c for c in ALPHABET if c != original_char]
    if not choices:
        return None
    c = rng.choice(choices)
    return word[:i] + c + word[i + 1:]


def _apply_transposition(word: str, rng: random.Random) -> Optional[str]:
    """Swap two adjacent characters. Skipped if they're identical (no-op edit)."""
    if len(word) < 2:
        return None
    i = rng.randrange(len(word) - 1)
    if word[i] == word[i + 1]:
        return None
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


_EDIT_FUNCS = {
    "deletion": _apply_deletion,
    "insertion": _apply_insertion,
    "replacement": _apply_replacement,
    "transposition": _apply_transposition,
}


def corrupt_word_single_edit(
    word: str,
    vocab: set,
    rng: random.Random,
    want_nonword: bool,
    max_attempts: int = 20,
) -> Optional[Tuple[str, str]]:
    """
    Attempt to produce a single-edit corruption of `word` satisfying the
    requested category:
        want_nonword=True  -> corrupted word must not be in vocab
        want_nonword=False -> corrupted word must be in vocab and != word
                               (a valid real-word error)

    Randomly tries edit types (deletion/insertion/replacement/
    transposition) and positions, up to `max_attempts` times, using the
    supplied `rng` (so the whole process is reproducible given a fixed
    seed and a fixed sequence of calls). Returns None if no valid
    corruption is found within the attempt budget; callers are expected
    to then try a different word (this function never invents an invalid
    case).

    Shared by evaluation.py (test-case generation) and benchmark.py
    (Speed Demon batch generation) so the corruption logic exists in
    exactly one place.
    """
    edit_types = list(_EDIT_FUNCS.keys())
    for _ in range(max_attempts):
        edit_type = rng.choice(edit_types)
        corrupted = _EDIT_FUNCS[edit_type](word, rng)
        if corrupted is None or corrupted == word:
            continue
        in_vocab = corrupted in vocab
        if want_nonword and not in_vocab:
            return corrupted, edit_type
        if (not want_nonword) and in_vocab:
            return corrupted, edit_type
    return None


# ---------------------------------------------------------------------------
# Test-set generation
# ---------------------------------------------------------------------------

def _build_case(
    sent_words: List[str],
    vocab: set,
    rng: random.Random,
    want_nonword: bool,
    need_context: bool,
) -> Optional[TestCase]:
    """
    Try to build one TestCase from a single (already-cleaned) sentence.

    Tries each word position in a randomized (but rng-reproducible) order
    until one produces a valid corruption of the requested type. Returns
    None if no word in the sentence can produce a valid case (caller then
    falls back to a different sentence; see generate_test_sets).

    need_context=True (real-word case) additionally requires the sentence
    to have at least 2 words, since real-word correction needs at least
    one neighbor to compute a bigram score.
    """
    if not sent_words:
        return None
    if need_context and len(sent_words) < 2:
        return None

    positions = list(range(len(sent_words)))
    rng.shuffle(positions)

    for idx in positions:
        original_word = sent_words[idx]
        if original_word not in vocab:
            # Defensive skip: sent_words come from the same cleaned corpus
            # the vocab was built from, so this shouldn't normally happen.
            continue
        corruption = corrupt_word_single_edit(original_word, vocab, rng, want_nonword)
        if corruption is None:
            continue
        corrupted_word, edit_type = corruption
        return TestCase(
            sentence_words=list(sent_words),
            index=idx,
            original_word=original_word,
            corrupted_word=corrupted_word,
            edit_type=edit_type,
        )
    return None


def generate_test_sets(
    cleaned_sentences: List[List[str]],
    vocab: set,
    fraction: float = 0.1,
    seed: int = 42,
    target_count: Optional[int] = None,
) -> Tuple[List[TestCase], List[TestCase]]:
    """
    Generate reproducible non-word and real-word test sets from Brown
    Corpus sentences.

    Parameters
    ----------
    cleaned_sentences : list[list[str]]
        Sentences already cleaned to lowercased alphabetic tokens. Pass
        the output of corpus_models.load_brown_sentences() so tokens
        match how the vocab/unigram/bigram models were built. (This
        module does not re-clean sentences itself, to avoid duplicating
        utils.clean_sentence's logic.)
    vocab : set[str]
        Trained vocabulary, from corpus_models.
    fraction : float
        Target fraction of sentences to draw a test case from (default
        0.1 = 10%, per the assignment). Ignored if target_count is given.
    seed : int
        Fixed RNG seed for full reproducibility of test-set generation.
    target_count : int, optional
        Explicit target case count, overriding `fraction`. Pass 5734 to
        match the assignment's literal "10% of 57,340 sentences" figure
        exactly, regardless of the exact sentence count NLTK reports in
        your environment (see README.md section 7 for why this count is
        environment-dependent).

    Returns
    -------
    (nonword_cases, realword_cases) : (list[TestCase], list[TestCase])
        Each list has up to `target_count` entries. If the corpus is
        exhausted before reaching the target (unlikely for Brown), the
        lists are simply shorter; no invalid cases are ever included.

    Reproducibility
    ----------------
    Sentence selection order is a single rng.shuffle() over all sentence
    indices, seeded once. Sentences are then consumed in that fixed order,
    attempting a non-word case and a real-word case from each in turn,
    continuing to the next sentence when one is exhausted. This is the
    concrete mechanism behind the assignment's "if a word cannot produce
    a valid case, choose another word/sentence" instruction: word choice
    within a sentence is retried (via _build_case's shuffled position
    list), and sentence choice falls through naturally to the next
    shuffled sentence when a whole sentence yields nothing usable.
    Everything is driven by one seeded `random.Random(seed)` instance, so
    re-running with the same seed reproduces identical test sets.
    """
    rng = random.Random(seed)
    n_total = len(cleaned_sentences)
    n_select = target_count if target_count is not None else round(n_total * fraction)

    order = list(range(n_total))
    rng.shuffle(order)

    nonword_cases: List[TestCase] = []
    realword_cases: List[TestCase] = []

    for idx in order:
        if len(nonword_cases) >= n_select and len(realword_cases) >= n_select:
            break

        sent = cleaned_sentences[idx]
        if not sent:
            continue  # empty/invalid sentence after cleaning -> skip

        if len(nonword_cases) < n_select:
            case = _build_case(sent, vocab, rng, want_nonword=True, need_context=False)
            if case is not None:
                nonword_cases.append(case)

        if len(realword_cases) < n_select:
            case = _build_case(sent, vocab, rng, want_nonword=False, need_context=True)
            if case is not None:
                realword_cases.append(case)

    return nonword_cases, realword_cases


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_nonword(
    test_cases: List[TestCase], corrector: SpellingCorrector
) -> Dict:
    """
    Run non-word correction on every test case and score whether the
    corrector restored the original ground-truth word.

    Accuracy = (# cases where corrector.correct_nonword(corrupted_word)
                 returns the original word) / total cases.

    Reuses corrector.correct_nonword() directly; no correction logic is
    reimplemented here.
    """
    correct = 0
    total = len(test_cases)
    for case in test_cases:
        result = corrector.correct_nonword(case.corrupted_word)
        if result["corrected"] == case.original_word:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def evaluate_realword(
    test_cases: List[TestCase], corrector: SpellingCorrector
) -> Dict:
    """
    Run real-word correction on every test case and score whether the
    corrector restored the original ground-truth word.

    For each case, the corrupted word is substituted into a COPY of the
    original sentence at `index` (the stored sentence_words is never
    mutated, so the same TestCase can be reused across Method A and
    Method B evaluation without cross-contamination), and
    corrector.correct_realword() is called on that copy.

    Accuracy = (# cases where the corrector's suggestion equals the
                 original word) / total cases.
    """
    correct = 0
    total = len(test_cases)
    for case in test_cases:
        sentence_copy = list(case.sentence_words)
        sentence_copy[case.index] = case.corrupted_word
        result = corrector.correct_realword(sentence_copy, case.index)
        if result["corrected"] == case.original_word:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def format_evaluation_report(report: Dict[str, Dict]) -> str:
    """
    Render the four accuracy reports (Method A/B x non-word/real-word)
    as a short human-readable summary.
    """
    lines = ["Q3 Evaluation Report", "=" * 40]
    for key, label in [
        ("method_a_nonword", "Method A — Non-word accuracy"),
        ("method_b_nonword", "Method B — Non-word accuracy"),
        ("method_a_realword", "Method A — Real-word accuracy"),
        ("method_b_realword", "Method B — Real-word accuracy"),
    ]:
        r = report[key]
        lines.append(
            f"{label}: {r['correct']}/{r['total']} = {r['accuracy']:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manual orchestration entry point. Not run automatically; run yourself.
# ---------------------------------------------------------------------------

def run_full_evaluation(
    seed: int = 42,
    fraction: float = 0.1,
    target_count: Optional[int] = None,
    real_word_threshold: float = 1.1,
) -> Dict[str, Dict]:
    """
    Full accuracy pipeline: load the trained corpus_models artifacts,
    build the candidates.py SymDel index, generate the test sets, and
    evaluate both methods on both error types.

    Intended to be run manually by the user; see README.md section 5 for
    the exact command.
    """
    from candidates import build_symdel_index
    from corpus_models import load_brown_sentences, load_models

    models = load_models()
    vocab = models["vocab"]
    unigram_counts = models["unigram_counts"]
    bigram_counts = models["bigram_counts"]
    vocab_size = models["vocab_size"]
    k = models["k"]

    symdel_index = build_symdel_index(vocab)  # built once, before any timing/eval

    cleaned_sentences = load_brown_sentences()
    nonword_cases, realword_cases = generate_test_sets(
        cleaned_sentences, vocab, fraction=fraction, seed=seed, target_count=target_count
    )

    corrector_a = SpellingCorrector(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        method="A", real_word_threshold=real_word_threshold, k=k,
    )
    corrector_b = SpellingCorrector(
        vocab, unigram_counts, bigram_counts, vocab_size, symdel_index,
        method="B", real_word_threshold=real_word_threshold, k=k,
    )

    report = {
        "method_a_nonword": evaluate_nonword(nonword_cases, corrector_a),
        "method_b_nonword": evaluate_nonword(nonword_cases, corrector_b),
        "method_a_realword": evaluate_realword(realword_cases, corrector_a),
        "method_b_realword": evaluate_realword(realword_cases, corrector_b),
    }
    return report


if __name__ == "__main__":
    report = run_full_evaluation()
    print(format_evaluation_report(report))
