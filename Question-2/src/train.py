"""
Training Pipeline

Loads the CoNLL-U training data, runs the Arc-Standard oracle to generate
training instances, extracts features, and trains a scikit-learn classifier.

Classifier choice: Logistic Regression (multinomial). It handles the
88-way multi-class problem efficiently, trains fast on a large sparse
one-hot feature matrix, and stays interpretable, which matters more here
than squeezing out extra accuracy. It was picked over SVM (too slow with
this many classes) and Random Forest (slower to train at this scale).

Usage:
    python train.py
"""

import sys
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from conllu_parser import parse_conllu
from oracle import generate_training_data
from features import features_to_dict
from utils import TRAIN_FILE, MODELS_DIR, MODEL_FILE, VECTORIZER_FILE, ensure_dir


def train():
    """Main training pipeline.
    
    Steps:
      1. Load CoNLL-U training data
      2. Run oracle to generate (features, transition) pairs
      3. Encode features using DictVectorizer (one-hot encoding)
      4. Train Logistic Regression classifier
      5. Save model and vectorizer to disk
    """
    print("=" * 60, flush=True)
    print("Transition-Based Dependency Parser — Training Pipeline", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load training data
    print(f"\n[Step 1] Loading training data from:\n  {TRAIN_FILE}", flush=True)
    start = time.time()
    sentences = parse_conllu(TRAIN_FILE)
    print(f"  Loaded {len(sentences)} sentences in {time.time() - start:.1f}s", flush=True)

    # Step 2: Run oracle to generate training instances
    print(f"\n[Step 2] Running Arc-Standard oracle...", flush=True)
    start = time.time()
    all_features, all_labels = generate_training_data(sentences)
    print(f"  Generated {len(all_features)} training instances in {time.time() - start:.1f}s", flush=True)

    # Print label distribution summary
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"  Unique transition labels: {len(label_counts)}", flush=True)
    
    # Show top 10 most common labels
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
    print("  Top 10 transitions:", flush=True)
    for label, count in sorted_labels[:10]:
        pct = count / len(all_labels) * 100
        print(f"    {label}: {count} ({pct:.1f}%)", flush=True)

    # Step 3: Encode features using DictVectorizer
    print(f"\n[Step 3] Encoding features with DictVectorizer...", flush=True)
    start = time.time()
    feature_dicts = [features_to_dict(f) for f in all_features]
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(feature_dicts)
    print(f"  Feature matrix shape: {X_train.shape}", flush=True)
    print(f"  Encoding done in {time.time() - start:.1f}s", flush=True)

    # Step 4: Train Logistic Regression classifier
    print(f"\n[Step 4] Training Logistic Regression classifier...", flush=True)
    print("  (multi-class, max_iter=1000, solver=lbfgs)", flush=True)
    start = time.time()
    classifier = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        n_jobs=-1,      # Use all available cores
        verbose=0
    )
    classifier.fit(X_train, all_labels)
    
    # Report training accuracy
    train_acc = classifier.score(X_train, all_labels)
    print(f"  Training accuracy: {train_acc * 100:.2f}%", flush=True)
    print(f"  Training done in {time.time() - start:.1f}s", flush=True)

    # Step 5: Save model and vectorizer
    print(f"\n[Step 5] Saving model and vectorizer...", flush=True)
    ensure_dir(MODELS_DIR)
    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"  Classifier saved to: {MODEL_FILE}", flush=True)
    print(f"  Vectorizer saved to: {VECTORIZER_FILE}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print("Training complete!", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    train()
