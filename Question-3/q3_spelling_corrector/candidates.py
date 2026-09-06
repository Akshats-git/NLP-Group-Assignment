"""
candidates.py — Q3 Phase 3 (revised)

Candidate generation only. No ranking, no correction decisions, no
evaluation — this module's job is purely: given a (possibly misspelled)
word, return the set of vocabulary words that could be the intended word,
at edit distance <= 1 (Damerau-Levenshtein: deletion, insertion,
replacement, or adjacent transposition).

Two independent methods are implemented, per the assignment:

    Method A — brute-force edit-distance-1 generation
        Generate every string reachable from `word` by one deletion,
        transposition, replacement, or insertion, then keep only the ones
        that are real vocabulary words. Candidate STRINGS are generated
        first, filtered against the vocabulary second.

    Method B — Symmetric Delete (SymDel)
        Precompute, once, a mapping from every one-character deletion of
        every vocabulary word back to that vocabulary word (the "index").
        At query time, only the (much smaller) one-character deletions of
        the misspelled word are generated and looked up in that index —
        this never enumerates the full edit space (no explicit
        replacement/insertion string generation) at query time, which is
        the source of its speed advantage (measured in Phase 5).

        Deletion-based matching alone directly guarantees an UPPER BOUND
        on edit distance: if a string D is obtained by deleting dA
        characters from word A and dB characters from word B, then
        edit_distance(A, B) <= dA + dB. For maxEditDistance = 1:
          - dA + dB <= 1 (i.e. one side deletes 0, the other deletes 1)
            guarantees true edit distance is EXACTLY 1 (no verification
            needed) — this captures pure insertion/deletion errors.
          - dA = dB = 1 (sum = 2) only guarantees edit distance <= 2, not
            exactly 1. This is the combination that surfaces substitution
            and transposition matches (a single differing/swapped position
            deletes to the same string from both sides), so those
            candidates ARE generated this way, but must be verified with a
            direct, cheap edit-distance-<=1 check before being accepted.
            This verification step is standard in symmetric-delete
            implementations (e.g. SymSpell) and is what lets Method B
            reach full edit-distance-1 coverage, including substitution
            and transposition, without ever falling back to Method A's
            brute-force generation.

Both methods return the SAME kind of result — a set[str] of vocabulary
words at edit distance <= 1 from the input — but reach it very
differently. Ranking one candidate as "the" correction (by unigram
frequency, or by bigram context) is Phase 4's job, not this module's.
"""

from typing import Dict, List, Set

from utils import normalize_token

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Method A — brute-force edit-distance-1 generation
# ---------------------------------------------------------------------------

def _edits1(word: str) -> Set[str]:
    """
    Generate every string at edit distance exactly 1 from `word`, using the
    four classic single-character edit operations:

        deletion      : remove one character
        transposition : swap two adjacent characters
        replacement   : substitute one character with another letter
        insertion     : insert one letter at any position

    This produces candidate STRINGS (not necessarily real words) — filtering
    against the vocabulary happens separately in `edit_distance_1_candidates`.

    Deterministic: splits are generated left-to-right, and the alphabet is
    iterated in fixed a-z order.
    """
    if word == "":
        return set()

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = {L + R[1:] for L, R in splits if R}
    transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
    replaces = {L + c + R[1:] for L, R in splits if R for c in ALPHABET}
    inserts = {L + c + R for L, R in splits for c in ALPHABET}

    return deletes | transposes | replaces | inserts


def edit_distance_1_candidates(word: str, vocab: Set[str]) -> Set[str]:
    """
    Method A: return every vocabulary word at edit distance 1 from `word`.

    Parameters
    ----------
    word : str
        The (possibly misspelled) input word. Normalized (lowercased) via
        utils.normalize_token before generating edits.
    vocab : set[str]
        The trained vocabulary (from corpus_models.build_vocab_and_unigram).

    Returns
    -------
    set[str]
        Vocabulary words reachable from `word` by exactly one edit.
        Empty set if none exist.

    Edge cases
    ----------
    - Empty string input -> returns an empty set immediately.
    - One-character words still work: deletion produces "", which won't
      match any real word and is naturally filtered out by the vocab check.
    """
    word = normalize_token(word)
    if word == "":
        return set()

    candidate_strings = _edits1(word)
    return {c for c in candidate_strings if c in vocab}


# ---------------------------------------------------------------------------
# Method B — Symmetric Delete (SymDel)
# ---------------------------------------------------------------------------

def _deletions(word: str) -> Set[str]:
    """
    Generate all one-character deletions of `word` (strictly shorter
    variants only — `word` itself is not included).
    For a 1-character word, this returns {""}.
    For an empty word, this returns an empty set (no deletion is possible).
    """
    if word == "":
        return set()
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def build_symdel_index(vocab: Set[str]) -> Dict[str, List[str]]:
    """
    Preprocessing step for Method B (run once, offline, after the
    vocabulary is built — NOT at query time).

    For every vocabulary word, compute its one-character deletions and map
    each deletion back to the list of vocabulary word(s) that produced it,
    e.g. 'ello' -> ['hello'], 'hllo' -> ['hello'], ...

    A single deletion key can map to multiple original words, so values
    are lists, not single strings.

    Parameters
    ----------
    vocab : set[str]
        The trained vocabulary.

    Returns
    -------
    dict[str, list[str]]
        Deletion-variant -> list of original vocabulary words.
    """
    index: Dict[str, List[str]] = {}
    for vocab_word in vocab:
        for deletion in _deletions(vocab_word):
            index.setdefault(deletion, []).append(vocab_word)
    return index


def _is_edit_distance_le_1(a: str, b: str) -> bool:
    """
    Direct (non-DP) check for whether `a` and `b` are at Damerau-Levenshtein
    edit distance <= 1 — i.e. identical, or reachable from one another by
    exactly one deletion, insertion, replacement, or adjacent transposition.

    Implemented directly rather than with a full Levenshtein DP table
    because the tolerance is fixed at <= 1: this keeps the check
    O(len(a) + len(b)) instead of O(len(a) * len(b)), which matters since
    it runs once per coarse candidate inside symdel_candidates.

    This is only needed to verify the "sum = 2" deletion matches (see
    module docstring) — matches found with sum <= 1 are exact by
    construction and never need this check.
    """
    if a == b:
        return True

    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        diff_positions = [i for i in range(la) if a[i] != b[i]]
        if len(diff_positions) <= 1:
            return True  # zero or one substitution
        if len(diff_positions) == 2:
            i, j = diff_positions
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return True  # adjacent transposition
        return False

    # Lengths differ by exactly 1 -> check single deletion/insertion.
    longer, shorter = (a, b) if la > lb else (b, a)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def symdel_candidates(
    word: str, symdel_index: Dict[str, List[str]], vocab: Set[str]
) -> Set[str]:
    """
    Method B: return candidate vocabulary words for a misspelled `word`
    using the precomputed symdel_index (from build_symdel_index).

    Three lookups are combined, corresponding to the deletion-depth
    combinations that can produce a true edit distance of 1 (see module
    docstring for the dA/dB reasoning):

    1. dA=0, dB=1 — `word` itself is looked up directly in the index.
       Catches the case where `word` is missing one character relative to
       the intended (longer) vocabulary word. Exact match, no verification
       needed.
    2. dA=1, dB=0 — each one-character deletion of `word` is checked
       directly against the vocabulary SET (not the index). Catches the
       case where `word` has one extra character relative to the intended
       (shorter) vocabulary word. Exact match, no verification needed.
    3. dA=1, dB=1 — each one-character deletion of `word` is looked up in
       the index (which stores one-character deletions of vocab words).
       A shared deletion here only guarantees edit distance <= 2, so every
       match from this step is verified with `_is_edit_distance_le_1`
       before being accepted. This is the step that recovers substitution
       and transposition errors (same-length words), since a single
       differing/swapped character position deletes to an identical string
       from both sides.

    All three steps only ever generate deletions of `word` (never of
    vocabulary words at query time — those were precomputed once in
    build_symdel_index) and use O(1) average-case dict/set lookups, which
    is what keeps this method fast relative to Method A's full string
    enumeration.

    Parameters
    ----------
    word : str
        The (possibly misspelled) input word. Normalized before lookup.
    symdel_index : dict[str, list[str]]
        Precomputed index from build_symdel_index.
    vocab : set[str]
        The trained vocabulary.

    Returns
    -------
    set[str]
        Vocabulary words at true edit distance <= 1 from `word`.

    Edge cases
    ----------
    - Empty string input -> returns an empty set.
    - One-character words: their only deletion is "", handled like any
      other deletion key/set-membership check.
    - Repeated characters (e.g. "letter"): multiple deletion positions can
      produce the same deletion string (e.g. deleting either 't' in
      "letter" gives "leter") — this is naturally deduplicated since
      `_deletions` returns a set.
    - Duplicate candidates across the three steps are naturally
      deduplicated by using a `set` as the accumulator.
    """
    word = normalize_token(word)
    if word == "":
        return set()

    word_deletions = _deletions(word)  # dA = 1 deletion set, computed once

    candidates: Set[str] = set()

    # Step 1 (dA=0, dB=1): word is missing a character vs. a vocab word.
    candidates.update(symdel_index.get(word, []))

    # Step 2 (dA=1, dB=0): word has an extra character vs. a vocab word.
    for d in word_deletions:
        if d in vocab:
            candidates.add(d)

    # Step 3 (dA=1, dB=1): substitution / transposition candidates —
    # coarse matches must be verified before acceptance.
    coarse_candidates: Set[str] = set()
    for d in word_deletions:
        coarse_candidates.update(symdel_index.get(d, []))

    for cand in coarse_candidates:
        if _is_edit_distance_le_1(word, cand):
            candidates.add(cand)

    # Safety net: restrict to actual vocab membership (steps 1 and 3 are
    # already built from vocab, but this guards against a stale index).
    return {c for c in candidates if c in vocab}
