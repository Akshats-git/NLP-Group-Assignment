# Question 2: Transition-Based Dependency Parser

A data-driven transition-based dependency parser using the **Arc-Standard** transition system, trained on the Universal Dependencies English-EWT corpus.

## Overview

This project implements a complete pipeline for transition-based dependency parsing:

1. **CoNLL-U Parser** — reads Universal Dependencies `.conllu` files
2. **Oracle Simulator** — generates training data from gold-standard trees using the Arc-Standard transition system
3. **Feature Extractor** — extracts 4 POS-tag features from parser configurations
4. **Classifier** — Logistic Regression (scikit-learn) trained to predict transitions
5. **Parser** — applies predicted transitions with validity checking
6. **Evaluator** — computes Labeled Attachment Score (LAS) on the dev set

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
(`data/`), not your shell's working directory — so the target must be
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

- **SHIFT**: Move the first word from the buffer to the top of the stack
- **LEFT-ARC(label)**: Top of stack becomes head of second on stack; second is popped
- **RIGHT-ARC(label)**: Second on stack becomes head of top; top is popped
