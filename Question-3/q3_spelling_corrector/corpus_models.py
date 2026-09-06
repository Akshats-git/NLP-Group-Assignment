"""
corpus_models.py — Q3 Phase 2

Builds and persists the language resources shared by the rest of Q3
(candidate generation, correction logic, evaluation) and reused as-is by Q4:

    - vocabulary            : set[str]
    - unigram frequency     : Counter[str, int]
    - bigram counts + a smoothed bigram probability function (add-k smoothing)

Nothing here performs correction, candidate generation, or evaluation —
those are later phases. This module only trains and saves/loads models.
"""

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import brown

from utils import clean_sentence

# ---------------------------------------------------------------------------
# Assumptions (stated explicitly)
# ---------------------------------------------------------------------------
# 1. NLTK's 'brown' corpus resource is assumed to already be downloaded
#    (nltk.download('brown')). We do NOT call nltk.download() here, since
#    Phase 2 must not perform network/download actions itself; that is a
#    one-time environment setup step for whoever runs this.
# 2. Sentence boundaries come from brown.sents() (untagged, using Brown's
#    own sentence segmentation) — no custom sentence splitting is done.
# 3. Add-k (Laplace-style) smoothing is used for the bigram model, with a
#    default k = 1.0. This is an implementation decision (the assignment
#    doesn't fix k) and is exposed as a parameter so it can be tuned later
#    without retraining from scratch.
# 4. Bigrams are computed within each sentence only (no bigrams spanning a
#    sentence boundary), which is the standard convention and avoids
#    spurious cross-sentence dependencies.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_K = 1.0  # add-k smoothing constant


def load_brown_sentences() -> List[List[str]]:
    """
    Load Brown Corpus sentences and clean each one down to lowercased
    alphabetic word tokens (see utils.clean_sentence).
    """
    raw_sents = brown.sents()
    return [clean_sentence(sent) for sent in raw_sents]


def build_vocab_and_unigram(sentences: List[List[str]]) -> Tuple[set, Counter]:
    """
    Build the vocabulary and unigram frequency distribution from a list of
    already-cleaned sentences (each a list of lowercased word tokens).

    Returns
    -------
    vocab : set[str]
        The set of unique words seen in the corpus.
    unigram_counts : Counter[str, int]
        Raw frequency count of each word — used later for non-word
        correction ranking (highest-frequency candidate wins).
    """
    unigram_counts: Counter = Counter()
    for sent in sentences:
        unigram_counts.update(sent)

    vocab = set(unigram_counts.keys())
    return vocab, unigram_counts


def build_bigram_model(sentences: List[List[str]]) -> Counter:
    """
    Build raw bigram counts from cleaned sentences. Bigrams are counted
    only within a sentence (no wraparound across sentence boundaries).

    Returns
    -------
    bigram_counts : Counter[(str, str), int]
        Raw co-occurrence counts of consecutive word pairs.
    """
    bigram_counts: Counter = Counter()
    for sent in sentences:
        for w1, w2 in zip(sent, sent[1:]):
            bigram_counts[(w1, w2)] += 1
    return bigram_counts


def bigram_log_prob(
    w1: str,
    w2: str,
    bigram_counts: Counter,
    unigram_counts: Counter,
    vocab_size: int,
    k: float = DEFAULT_K,
) -> float:
    """
    Add-k smoothed bigram probability, returned as a natural log-probability
    (log-space avoids underflow when phrase probabilities are multiplied
    together in later correction/evaluation phases).

        P(w2 | w1) = (count(w1, w2) + k) / (count(w1) + k * vocab_size)

    Unseen bigrams (count = 0) and unseen w1 (count(w1) = 0) are handled
    gracefully by the +k smoothing term instead of returning zero probability.
    """
    import math

    pair_count = bigram_counts.get((w1, w2), 0)
    w1_count = unigram_counts.get(w1, 0)

    numerator = pair_count + k
    denominator = w1_count + k * vocab_size
    prob = numerator / denominator
    return math.log(prob)


# ---------------------------------------------------------------------------
# Persistence — so Q4 (and later Q3 phases) can load trained artifacts
# without retraining.
# ---------------------------------------------------------------------------

def save_models(
    vocab: set,
    unigram_counts: Counter,
    bigram_counts: Counter,
    model_dir: Path = DEFAULT_MODEL_DIR,
    k: float = DEFAULT_K,
) -> None:
    """
    Persist vocab, unigram counts, bigram counts, and the smoothing constant
    k to a single pickle file under model_dir. vocab_size is derived from
    vocab and stored alongside so bigram_log_prob can be reproduced exactly
    at load time.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "vocab": vocab,
        "unigram_counts": unigram_counts,
        "bigram_counts": bigram_counts,
        "vocab_size": len(vocab),
        "k": k,
    }
    out_path = model_dir / "q3_language_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)


def load_models(model_dir: Path = DEFAULT_MODEL_DIR) -> Dict:
    """
    Load the pickled artifact dict. Returns a dict with keys:
        'vocab', 'unigram_counts', 'bigram_counts', 'vocab_size', 'k'

    This is the function Q4 (and Q3 Phases 3-6) should call at startup to
    reuse the trained models without retraining:

        models = load_models()
        vocab = models["vocab"]
        unigram_counts = models["unigram_counts"]
        bigram_counts = models["bigram_counts"]
        vocab_size = models["vocab_size"]
        k = models["k"]
    """
    in_path = model_dir / "q3_language_models.pkl"
    with open(in_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


# ---------------------------------------------------------------------------
# Orchestration entry point for this phase (build + save everything).
# Not executed by Claude — provided for the user to run themselves.
# ---------------------------------------------------------------------------

def build_and_save_all(model_dir: Path = DEFAULT_MODEL_DIR, k: float = DEFAULT_K) -> None:
    """
    Full Phase 2 pipeline: load corpus -> build vocab/unigram/bigram ->
    save artifacts. Intended to be run once, manually, by the user
    (e.g., `python -c "from corpus_models import build_and_save_all; build_and_save_all()"`).
    """
    sentences = load_brown_sentences()
    vocab, unigram_counts = build_vocab_and_unigram(sentences)
    bigram_counts = build_bigram_model(sentences)
    save_models(vocab, unigram_counts, bigram_counts, model_dir=model_dir, k=k)


if __name__ == "__main__":
    # Left as a manual entry point — the user runs this file themselves.
    build_and_save_all()
