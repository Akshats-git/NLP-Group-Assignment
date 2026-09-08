"""
Correction logic only: given already-trained models (vocab, unigram
counts, bigram counts, a SymDel index) and the candidate-generation
functions from candidates.py, decide what, if anything, to correct.

This module does not:
    - build/train the vocabulary, unigram model, bigram model, or SymDel
      index (corpus_models.py / candidates.py already did that;
      everything here is injected via the constructor)
    - evaluate accuracy or run the Speed Demon benchmark (evaluation.py,
      benchmark.py)
    - provide a CLI (cli.py)
    - know anything about Q4

Two correction tasks, per the assignment:

    Non-word correction   - the word is not in the vocabulary at all.
                             Rank candidates by raw unigram frequency,
                             take the best one.
    Real-word correction  - the word is in the vocabulary, but may be the
                             wrong word for this context (e.g. "sea" vs
                             "see"). Rank candidates (including the
                             original) by a bigram-context score and only
                             switch if a candidate clears a configurable
                             margin over the original.
"""

import re
from typing import Dict, List, Optional, Tuple

from candidates import edit_distance_1_candidates, symdel_candidates
from corpus_models import DEFAULT_K, bigram_log_prob
from utils import normalize_token

# Matches: optional leading non-letters, an alphabetic core, optional
# trailing non-letters. Used to strip/reattach punctuation around a word
# (e.g. "dog." -> leading="", core="dog", trailing=".").
_TOKEN_RE = re.compile(r"^(\W*)([A-Za-z]+)(\W*)$")


def _split_token(token: str) -> Tuple[str, Optional[str], str]:
    """
    Split a raw whitespace-delimited token into (leading, core, trailing).

    If the token has no alphabetic core at all (pure punctuation, numbers,
    empty string), returns (token, None, "") so the caller can pass it
    through untouched. Correction only ever applies to the alphabetic
    core of a token.
    """
    match = _TOKEN_RE.match(token)
    if not match:
        return token, None, ""
    return match.group(1), match.group(2), match.group(3)


def _match_case(reference: str, word: str) -> str:
    """
    Reapply `reference`'s casing pattern to `word` (assumed lowercase):
      - reference is ALL CAPS      -> word is upper-cased
      - reference is Capitalized   -> word is capitalized
      - otherwise (lowercase/mixed) -> word is left as-is (lowercase)

    This is a simple, "reasonable" heuristic for punctuation/casing
    handling, not a full grammar-aware capitalizer.
    """
    if reference.isupper():
        return word.upper()
    if reference[:1].isupper():
        return word.capitalize()
    return word


class SpellingCorrector:
    """
    Wraps already-trained Q3 models and exposes correction methods.

    Nothing is trained here. `vocab`, `unigram_counts`, `bigram_counts`,
    `vocab_size`, and `symdel_index` are all expected to come from
    `corpus_models.load_models()` / `candidates.build_symdel_index()`,
    so the same instance's artifacts can be reused verbatim by Q4
    without retraining.
    """

    def __init__(
        self,
        vocab: set,
        unigram_counts,
        bigram_counts,
        vocab_size: int,
        symdel_index: Dict[str, List[str]],
        method: str = "B",
        real_word_threshold: float = 1.1,
        k: float = DEFAULT_K,
    ):
        """
        Parameters
        ----------
        vocab, unigram_counts, bigram_counts, vocab_size
            Trained artifacts from corpus_models.
        symdel_index
            Trained artifact from candidates.build_symdel_index.
        method
            Which candidate generator(s) to use: "A", "B", or "both".
            Exposed so evaluation.py and benchmark.py can instantiate one
            corrector per method for comparison without touching this
            class.
        real_word_threshold
            Minimum required improvement, in natural-log-probability
            "nats", for a real-word candidate to replace the original
            word. Not specified by the assignment; an explicit, tunable
            implementation decision. Chosen empirically by sweeping
            thresholds against the real-word test set: 1.1 nats raises
            real-word accuracy from 62.31% (at the original default, 2.0)
            to 74.75%, at the cost of the false-positive rate on
            already-correct in-vocabulary words rising from 1.93% to
            4.23%. A lower threshold (1.0) scores marginally higher
            (76.07%) but was rejected, since at that margin the corrector
            also "corrects" the assignment's own "Please meat me at the
            station" example to "beat", a confidently wrong answer. See
            REPORT_Q3.md section 3.1 for the full sweep and the reasoning
            behind this trade-off.
        k
            Add-k smoothing constant, passed through to
            corpus_models.bigram_log_prob. Defaults to the same constant
            used when the bigram model was trained; exposed here only so
            a corrector could experiment with a different k against the
            same trained counts, without retraining.

        Note on this signature
        -----------------------
        `k` was added to the original API sketch because
        `bigram_log_prob` requires it and the assignment explicitly says
        "reuse bigram_log_prob / smoothing logic" rather than re-deriving
        smoothing here. Everything else matches the requested signature.
        """
        if method not in ("A", "B", "both"):
            raise ValueError(f"method must be 'A', 'B', or 'both', got {method!r}")

        self.vocab = vocab
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.vocab_size = vocab_size
        self.symdel_index = symdel_index
        self.method = method
        self.real_word_threshold = real_word_threshold
        self.k = k

    # ------------------------------------------------------------------
    # Candidate generation (delegates to candidates.py, does not duplicate it)
    # ------------------------------------------------------------------

    def _generate_candidates(self, word: str) -> set:
        """
        Generate candidates for `word` using whichever method(s) this
        corrector was configured with. `word` is assumed already
        normalized (lowercased) by the caller.
        """
        candidates = set()
        if self.method in ("A", "both"):
            candidates |= edit_distance_1_candidates(word, self.vocab)
        if self.method in ("B", "both"):
            candidates |= symdel_candidates(word, self.symdel_index, self.vocab)
        return candidates

    # ------------------------------------------------------------------
    # Non-word correction
    # ------------------------------------------------------------------

    def correct_nonword(self, word: str) -> Dict:
        """
        Correct `word` assuming it is a non-word error (not in vocab).

        Returns a metadata dict (not just a bare string) so callers,
        including correct_sentence and evaluation.py/cli.py, can inspect
        what happened without recomputing candidates:

            {
                "original": <input word>,
                "corrected": <chosen word, or unchanged>,
                "changed": bool,
                "error_type": "nonword" | None,
                "candidates": [candidate words, ranked best-first],
            }

        Ranking: highest unigram frequency wins; ties are broken
        alphabetically for determinism (so re-running produces identical
        output, which matters for evaluation/debugging).

        If `word` is empty or already in the vocabulary, this is a no-op
        (returns it unchanged). Callers are expected to route only true
        non-word tokens here, but this guard keeps the method safe to
        call standalone too.
        """
        word_norm = normalize_token(word)

        result = {
            "original": word,
            "corrected": word,
            "changed": False,
            "error_type": None,
            "candidates": [],
        }

        if word_norm == "" or word_norm in self.vocab:
            return result

        candidates = self._generate_candidates(word_norm)
        if not candidates:
            return result  # no candidates -> leave unchanged

        ranked = sorted(
            candidates,
            key=lambda w: (-self.unigram_counts.get(w, 0), w),
        )
        result["candidates"] = ranked
        result["corrected"] = ranked[0]
        result["changed"] = True
        result["error_type"] = "nonword"
        return result

    # ------------------------------------------------------------------
    # Real-word correction
    # ------------------------------------------------------------------

    def _score_word_in_context(
        self, sentence_words: List[str], index: int, word: str
    ) -> Optional[float]:
        """
        score(word) = log P(word | previous) + log P(next | word)

        Uses whichever term(s) are available at this position:
          - both neighbors exist -> both terms, summed
          - only a previous word exists (word is last in sentence) ->
            only log P(word | previous)
          - only a next word exists (word is first in sentence) ->
            only log P(next | word)
          - neither exists (single-word sentence) -> returns None,
            since no bigram-based score can be computed at all

        `sentence_words` must already be normalized (lowercased) word
        tokens; `index` is `word`'s position within it. `word` itself is
        passed separately (rather than read from sentence_words[index])
        so this same method can score both the original word and any
        candidate substitution at that position.
        """
        n = len(sentence_words)
        has_prev = index > 0
        has_next = index < n - 1

        if not has_prev and not has_next:
            return None

        score = 0.0
        if has_prev:
            score += bigram_log_prob(
                sentence_words[index - 1], word,
                self.bigram_counts, self.unigram_counts, self.vocab_size, self.k,
            )
        if has_next:
            score += bigram_log_prob(
                word, sentence_words[index + 1],
                self.bigram_counts, self.unigram_counts, self.vocab_size, self.k,
            )
        return score

    def correct_realword(self, sentence_words: List[str], index: int) -> Dict:
        """
        Check whether the in-vocabulary word at `sentence_words[index]`
        is a real-word error given its context, and correct it if a
        candidate clears `self.real_word_threshold`.

        Parameters
        ----------
        sentence_words : list[str]
            Normalized (lowercased) word tokens for the whole sentence.
            Read-only; this method does not mutate the list.
        index : int
            Position of the word to check within sentence_words.

        Returns
        -------
        dict
            {
                "original": <word at this position>,
                "corrected": <chosen word, or unchanged>,
                "changed": bool,
                "error_type": "realword" | None,
                "candidates": [(score, candidate), ...] ranked best-first,
                "original_score": float | None,
                "best_candidate_score": float | None,
            }

        No-ops (returns unchanged, with reasons implicit in the empty/None
        fields) when:
          - the word is empty or NOT in the vocabulary (this method only
            handles real-word errors; non-word errors go through
            correct_nonword instead)
          - no context is available at all (single-word sentence)
          - candidate generation finds nothing
          - the best candidate does not beat the original by more than
            `real_word_threshold`
        """
        word = sentence_words[index]
        word_norm = normalize_token(word)

        result = {
            "original": word,
            "corrected": word,
            "changed": False,
            "error_type": None,
            "candidates": [],
            "original_score": None,
            "best_candidate_score": None,
        }

        if word_norm == "" or word_norm not in self.vocab:
            return result

        original_score = self._score_word_in_context(sentence_words, index, word_norm)
        if original_score is None:
            return result  # no context to score against (single-word sentence)
        result["original_score"] = original_score

        candidates = self._generate_candidates(word_norm) - {word_norm}
        if not candidates:
            return result

        scored = []
        for cand in candidates:
            cand_score = self._score_word_in_context(sentence_words, index, cand)
            if cand_score is not None:
                scored.append((cand_score, cand))

        if not scored:
            return result

        # Deterministic ranking: highest score first, alphabetical tiebreak.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        result["candidates"] = scored

        best_score, best_word = scored[0]
        result["best_candidate_score"] = best_score

        if best_score - original_score > self.real_word_threshold:
            result["corrected"] = best_word
            result["changed"] = True
            result["error_type"] = "realword"

        return result

    # ------------------------------------------------------------------
    # Sentence-level orchestration
    # ------------------------------------------------------------------

    def correct_sentence(self, sentence: str) -> Tuple[str, List[Dict]]:
        """
        Correct a full sentence: tokenize, run non-word correction, then
        real-word correction (using the non-word-corrected sequence as
        context), and reconstruct the sentence text with punctuation and
        casing reapplied.

        Two-pass design (documented, not hidden): non-word correction
        runs first because a token that is a non-word error can't
        meaningfully participate as "known context" for its neighbors'
        real-word checks until it's been resolved to a real word (or
        left as an unresolved unknown token if no candidates exist).
        Real-word correction then runs using that updated sequence, so a
        neighbor's bigram context reflects the best available word at
        each position rather than the raw misspelled input.

        Returns
        -------
        (corrected_sentence, metadata) : (str, list[dict])
            corrected_sentence : the reconstructed sentence text.
            metadata : one dict per whitespace-delimited token, in order,
                each either a non-word result, a real-word result, or a
                pass-through record for non-alphabetic tokens
                (error_type=None, changed=False).

        Edge cases handled
        -------------------
        - Empty / whitespace-only input -> returns (sentence, []).
        - Single-word sentence -> non-word correction still applies;
          real-word correction is a no-op (no context), per
          _score_word_in_context.
        - Repeated words -> each occurrence is corrected independently
          by its own index, so "the the dog" checks both "the"s
          separately (no shared/cached state across positions).
        - Already-correct words -> in-vocab, no candidate clears the
          threshold -> unchanged, error_type=None.
        - Punctuation -> leading/trailing non-letters are stripped before
          lookup and reattached afterward; tokens with no alphabetic core
          (pure punctuation/numbers) pass through untouched.
        """
        if not sentence or not sentence.strip():
            return sentence, []

        raw_tokens = sentence.split()
        n = len(raw_tokens)

        leading: List[str] = [""] * n
        trailing: List[str] = [""] * n
        original_cores: List[Optional[str]] = [None] * n
        cores: List[Optional[str]] = [None] * n  # normalized core, or None

        for i, tok in enumerate(raw_tokens):
            lead, core, trail = _split_token(tok)
            leading[i] = lead
            trailing[i] = trail
            original_cores[i] = core
            cores[i] = normalize_token(core) if core is not None else None

        metadata: List[Optional[Dict]] = [None] * n
        working: List[Optional[str]] = list(cores)  # updated as corrections apply

        # Pass 1 - non-word correction.
        for i in range(n):
            word_norm = cores[i]
            if word_norm is None:
                metadata[i] = {
                    "original": raw_tokens[i],
                    "corrected": raw_tokens[i],
                    "changed": False,
                    "error_type": None,
                    "candidates": [],
                }
                continue

            if word_norm not in self.vocab:
                res = self.correct_nonword(word_norm)
                metadata[i] = res
                if res["changed"]:
                    working[i] = res["corrected"]
                # else: leave the unresolved out-of-vocab token in
                # `working` as-is; it still participates as (weak,
                # smoothed) bigram context for its neighbors.

        # Pass 2 - real-word correction, using the (possibly non-word-
        # corrected) `working` sequence as context.
        for i in range(n):
            if metadata[i] is not None:
                continue  # already resolved as a non-word / pass-through token
            res = self.correct_realword(working, i)
            metadata[i] = res
            if res["changed"]:
                working[i] = res["corrected"]

        # Reconstruct sentence text.
        out_tokens = []
        for i in range(n):
            if cores[i] is None:
                out_tokens.append(raw_tokens[i])
                continue
            final_core = working[i]
            cased = _match_case(original_cores[i], final_core)
            out_tokens.append(f"{leading[i]}{cased}{trailing[i]}")

        corrected_sentence = " ".join(out_tokens)
        return corrected_sentence, metadata
