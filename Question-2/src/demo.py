"""
Demo Script

Demonstrates the trained transition-based dependency parser on three
example sentences from the assignment:

  1. "The cat sat on the mat."
  2. "She eats a green salad."
  3. "I saw the man with a telescope."

POS tags are manually provided for each sentence since the parser
requires POS-tagged input (it does not perform POS tagging itself).

Usage:
    python demo.py
"""

import sys
import joblib

from transition_parser import parse_sentence
from utils import MODEL_FILE, VECTORIZER_FILE


# Example sentences with manually assigned Universal POS tags.
# POS tags follow the Universal Dependencies UPOS tagset.
EXAMPLE_SENTENCES = [
    {
        "text": "The cat sat on the mat.",
        "words": ["The", "cat", "sat", "on", "the", "mat", "."],
        "pos_tags": ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
    },
    {
        "text": "She eats a green salad.",
        "words": ["She", "eats", "a", "green", "salad", "."],
        "pos_tags": ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
    },
    {
        "text": "I saw the man with a telescope.",
        "words": ["I", "saw", "the", "man", "with", "a", "telescope", "."],
        "pos_tags": ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
    },
]


def print_parse_result(example: dict, arcs: list) -> None:
    """Print the parse result for an example sentence in a readable format.
    
    Args:
        example: Dictionary with 'text', 'words', 'pos_tags'.
        arcs: List of (head_id, dep_id, label) tuples from the parser.
    """
    words = example["words"]
    pos_tags = example["pos_tags"]

    print(f"\nSentence: \"{example['text']}\"", flush=True)
    print("-" * 50, flush=True)

    # Print tokens and POS tags
    print(f"  {'ID':<4} {'Word':<15} {'POS':<8}", flush=True)
    print(f"  {'--':<4} {'----':<15} {'---':<8}", flush=True)
    for i, (w, p) in enumerate(zip(words, pos_tags), start=1):
        print(f"  {i:<4} {w:<15} {p:<8}", flush=True)

    # Build arc lookup: dep_id -> (head_id, label)
    arc_lookup = {}
    for head, dep, label in arcs:
        arc_lookup[dep] = (head, label)

    # Print dependency arcs
    print(f"\n  Predicted Dependency Arcs:", flush=True)
    print(f"  {'ID':<4} {'Word':<15} {'Head':<6} {'Head Word':<15} {'Relation':<12}", flush=True)
    print(f"  {'--':<4} {'----':<15} {'----':<6} {'---------':<15} {'--------':<12}", flush=True)
    for i, word in enumerate(words, start=1):
        if i in arc_lookup:
            head_id, label = arc_lookup[i]
            head_word = words[head_id - 1] if head_id > 0 else "ROOT"
        else:
            head_id = 0
            head_word = "ROOT"
            label = "?"
        print(f"  {i:<4} {word:<15} {head_id:<6} {head_word:<15} {label:<12}", flush=True)


def demo():
    """Run the parser on three example sentences and display results."""
    print("=" * 60, flush=True)
    print("Transition-Based Dependency Parser — Demo", flush=True)
    print("=" * 60, flush=True)

    # Load trained model
    print(f"\nLoading trained model...", flush=True)
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print(f"  Model loaded successfully.", flush=True)
    except FileNotFoundError:
        print("ERROR: Trained model not found. Run train.py first.", flush=True)
        sys.exit(1)

    # Parse each example sentence
    for example in EXAMPLE_SENTENCES:
        arcs = parse_sentence(
            words=example["words"],
            pos_tags=example["pos_tags"],
            model=model,
            vectorizer=vectorizer
        )
        print_parse_result(example, arcs)

    print(f"\n{'=' * 60}", flush=True)
    print("Demo complete!", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    demo()
