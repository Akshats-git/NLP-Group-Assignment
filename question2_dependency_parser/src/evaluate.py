"""
Evaluation Module

Evaluates the trained transition-based dependency parser on the dev set.
Calculates Labeled Attachment Score (LAS):
  - For each token (excluding ROOT), check if BOTH:
    1. Predicted head == gold head
    2. Predicted dependency label == gold label
  - LAS = (correct tokens / total tokens) * 100

Usage:
    python evaluate.py
"""

import sys
import time
import joblib

from conllu_parser import parse_conllu
from transition_parser import parse_conllu_sentence
from utils import DEV_FILE, MODEL_FILE, VECTORIZER_FILE, OUTPUTS_DIR, ensure_dir


def evaluate():
    """Main evaluation pipeline.
    
    Steps:
      1. Load trained model and vectorizer
      2. Load dev CoNLL-U data
      3. Parse each sentence using the trained parser
      4. Compare predicted arcs with gold-standard arcs
      5. Calculate and print LAS
    """
    print("=" * 60, flush=True)
    print("Transition-Based Dependency Parser — Evaluation", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load model and vectorizer
    print(f"\n[Step 1] Loading trained model...", flush=True)
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print(f"  Classifier loaded from: {MODEL_FILE}", flush=True)
        print(f"  Vectorizer loaded from: {VECTORIZER_FILE}", flush=True)
    except FileNotFoundError:
        print("ERROR: Trained model not found. Run train.py first.", flush=True)
        sys.exit(1)

    # Step 2: Load dev data
    print(f"\n[Step 2] Loading dev data from:\n  {DEV_FILE}", flush=True)
    sentences = parse_conllu(DEV_FILE)
    print(f"  Loaded {len(sentences)} sentences", flush=True)

    # Step 3 & 4: Parse each sentence and compare with gold
    print(f"\n[Step 3] Parsing dev sentences and computing LAS...", flush=True)
    start = time.time()

    total_tokens = 0
    correct_tokens = 0
    sentences_processed = 0

    for i, sentence in enumerate(sentences):
        # Skip empty sentences
        if len(sentence) == 0:
            continue

        # Parse the sentence using the trained model
        predicted_arcs = parse_conllu_sentence(sentence, model, vectorizer)

        # Build a lookup: dependent_id -> (head_id, label)
        pred_lookup = {}
        for head, dep, label in predicted_arcs:
            pred_lookup[dep] = (head, label)

        # Compare with gold standard for each token (skip ROOT at index 0)
        for token in sentence.tokens[1:]:
            total_tokens += 1
            gold_head = token.head
            gold_label = token.deprel

            if token.id in pred_lookup:
                pred_head, pred_label = pred_lookup[token.id]
                if pred_head == gold_head and pred_label == gold_label:
                    correct_tokens += 1

        sentences_processed += 1

        # Progress reporting
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  Processed {i + 1}/{len(sentences)} sentences "
                  f"({elapsed:.1f}s elapsed)", flush=True)

    elapsed = time.time() - start

    # Step 5: Calculate and print LAS
    las = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0.0

    print(f"\n{'=' * 60}", flush=True)
    print("EVALUATION RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Sentences processed: {sentences_processed}", flush=True)
    print(f"  Total tokens evaluated: {total_tokens}", flush=True)
    print(f"  Correct tokens (head + label): {correct_tokens}", flush=True)
    print(f"  Labeled Attachment Score (LAS): {las:.2f}%", flush=True)
    print(f"  Evaluation time: {elapsed:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Save results to output file
    ensure_dir(OUTPUTS_DIR)
    output_file = f"{OUTPUTS_DIR}/evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write("Transition-Based Dependency Parser — Evaluation Results\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dev file: {DEV_FILE}\n")
        f.write(f"Sentences processed: {sentences_processed}\n")
        f.write(f"Total tokens evaluated: {total_tokens}\n")
        f.write(f"Correct tokens (head + label): {correct_tokens}\n")
        f.write(f"Labeled Attachment Score (LAS): {las:.2f}%\n")
        f.write(f"Evaluation time: {elapsed:.1f}s\n")
    print(f"\nResults saved to: {output_file}", flush=True)


if __name__ == "__main__":
    evaluate()
