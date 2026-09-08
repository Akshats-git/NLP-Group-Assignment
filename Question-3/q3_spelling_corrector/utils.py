"""
utils.py
Shared preprocessing helpers for the Q3 spelling corrector.

Design decisions (stated explicitly, per assignment ambiguity):
- All vocabulary/model lookups are case-insensitive: every token is lowercased
  before entering the vocabulary, unigram, or bigram model. Original casing is
  not tracked here; restoring casing on output is the CLI's concern, not a
  language-model concern.
- Tokens are restricted to purely alphabetic strings (`str.isalpha()`).
  Punctuation, digits, and Brown's own POS-tag-bearing tokens are dropped
  when building the vocabulary/models. This keeps the correction problem
  focused on real word spelling, matching the assignment's word-level scope
  (Method A/B, unigram/bigram) rather than tokenization of punctuation.
- No stemming/lemmatization is applied. Brown corpus surface forms are used
  as-is, since the assignment's edit-distance-1 framing operates on surface
  forms (e.g., "sentnce" -> "sentence"), not lemmas.
"""

from typing import Iterable, List


def normalize_token(token: str) -> str:
    """Lowercase a token. Casing restoration (if any) happens at the CLI layer."""
    return token.lower()


def is_valid_word_token(token: str) -> bool:
    """
    A token counts as a 'word' for vocabulary/model purposes if it is
    purely alphabetic (no digits, punctuation, or symbols).
    """
    return token.isalpha()


def clean_sentence(tokens: Iterable[str]) -> List[str]:
    """
    Filter a raw token list down to lowercased alphabetic word tokens.
    Non-word tokens (punctuation, numbers) are dropped, not replaced, so
    this changes sentence length. That's fine for LM training, but
    callers needing positional alignment (e.g., Q4's live editor) should
    do their own punctuation-preserving tokenization and only pass word
    tokens through the corrector.
    """
    cleaned = []
    for tok in tokens:
        if is_valid_word_token(tok):
            cleaned.append(normalize_token(tok))
    return cleaned
