"""
Utility Module

Shared constants and helper functions for the dependency parser project.
Provides centralized path definitions for data, models, and outputs.
"""

import os

# Project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "en_ewt-ud-train.conllu")
DEV_FILE = os.path.join(DATA_DIR, "en_ewt-ud-dev.conllu")

# Model directory and file paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "classifier.joblib")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.joblib")

# Output directory
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
