# Question 2: Transition-Based Dependency Parser

A data-driven transition-based dependency parser using the **Arc-Standard** transition system, trained on the Universal Dependencies English-EWT corpus.

## Overview

The pipeline has six pieces, each in its own module. A CoNLL-U parser reads the Universal Dependencies `.conllu` files. An oracle simulator walks each gold-standard tree with the Arc-Standard transition system and records what a perfect parser would have done at every step, which is how the training data gets built. A feature extractor pulls 4 POS-tag features out of each parser configuration, and those features feed a Logistic Regression classifier (scikit-learn) that learns to predict transitions. The parser itself applies the predicted transitions with validity checking so it can't crash on an illegal move, and an evaluator computes Labeled Attachment Score (LAS) on the dev set at the end.

## Dataset

- **Training**: `en_ewt-ud-train.conllu` (Universal Dependencies English-EWT)
- **Evaluation**: `en_ewt-ud-dev.conllu`
- Source: https://github.com/UniversalDependencies/UD_English-EWT

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# The UD English-EWT data should be in data/
# If not already cloned:
git clone https://github.com/UniversalDependencies/UD_English-EWT.git data/UD_English-EWT
ln -sf UD_English-EWT/en_ewt-ud-train.conllu data/en_ewt-ud-train.conllu
ln -sf UD_English-EWT/en_ewt-ud-dev.conllu data/en_ewt-ud-dev.conllu
```

Note: the symlink target is resolved relative to the symlink's own directory
(`data/`), not your shell's working directory, so the target must be
`UD_English-EWT/...`, not `data/UD_English-EWT/...`, or you'll get a dangling
link.

## Usage

### Train the model

```bash
cd src
python train.py
```

### Evaluate on dev set

```bash
cd src
python evaluate.py
```

### Run demo with example sentences

```bash
cd src
python demo.py
```

## Project Structure

```
Question-2/
├── data/
│   ├── en_ewt-ud-train.conllu     # Training data (symlink)
│   ├── en_ewt-ud-dev.conllu       # Dev data (symlink)
│   └── UD_English-EWT/            # Cloned UD repository
├── src/
│   ├── conllu_parser.py           # CoNLL-U file parser
│   ├── oracle.py                  # Arc-Standard oracle simulator
│   ├── features.py                # Feature extraction (4 POS features)
│   ├── transition_parser.py       # Transition-based parser
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # LAS evaluation
│   ├── demo.py                    # Example sentence demos
│   └── utils.py                   # Shared utilities/paths
├── models/                        # Saved trained model
├── outputs/                       # Evaluation results
├── requirements.txt
├── README.md
└── REPORT_Q2.md
```

## Features

The parser uses exactly 4 features per configuration:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Stack top POS | POS tag of the top word on the stack |
| 2 | Stack second POS | POS tag of the second word on the stack |
| 3 | Buffer first POS | POS tag of the first word in the buffer |
| 4 | Buffer second POS | POS tag of the second word in the buffer |

Missing positions use the sentinel value `NONE`.

## Arc-Standard Transitions

SHIFT moves the first word from the buffer to the top of the stack. LEFT-ARC(label) makes the top of the stack the head of the second item on the stack, then pops that second item. RIGHT-ARC(label) does the mirror image: the second item on the stack becomes the head of the top item, and the top item gets popped.
