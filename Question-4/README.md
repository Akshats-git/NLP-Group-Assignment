# Question 4 — Live NLP Editor

A Streamlit application implementing a live-typing NLP editor with three
real-time alert systems and a PCFG constituency parser.

**This module builds directly on Questions 1 and 3 — it reuses their trained
models without retraining.**

---

## Architecture

```
Question-4/
├── app.py                   # Streamlit entry point
├── requirements.txt         # streamlit, nltk
├── models/                  # Auto-created: PCFG cache
└── q4/
    ├── __init__.py
    ├── model_loader.py      # Loads Q1 + Q3 models (trains Q1 if needed)
    ├── segmentation.py      # SEGMENT-ALERT: Q1 beam-search decoder wrapper
    ├── spelling.py          # SPELL-ALERT:   Q3 SymDel corrector wrapper
    ├── grammar.py           # GRAMMAR-ALERT: trigram PPL + real-word check
    ├── pcfg.py              # PCFG training (Penn Treebank) + CKY parser
    ├── tagset.py            # Universal → PTB tagset reconciliation
    └── passage.py           # Random passage sampler + merge-token generator
```

---

## Running

```bash
cd Question-4
pip install -r requirements.txt
streamlit run app.py
```

The first launch will:
1. Train Q1 English models from the Brown corpus (~30s, saved to `Question-1/models/q1_english.pkl`)
2. Train the PCFG from the Penn Treebank sample (~10s, saved to `models/q4_pcfg.pkl`)

All subsequent launches load from the saved files instantly.

---

## Design Decisions

### Part 1: Simulated Typing

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Merge probability `p` | 0.08 | ~1.5 merges/sentence; 78% of sentences have ≥1 merge |
| Grammar trigger `N` | 10 | Aligns with sentence length; fires ~once/sentence |
| Candidate method | B (SymDel) | 10–50× faster than Method A (Q3 Phase 5 benchmark) |
| PPL threshold | 300 | Tuned against Brown corpus held-out; configurable in UI |
| Real-word margin | 2.0 nats | Same default as Q3 `SpellingCorrector` |

### Part 2: PCFG Parser

- **Training corpus**: Penn Treebank sample (`nltk.corpus.treebank`, ~3,900 sentences)
- **Induction**: `nltk.induce_pcfg` over CNF-converted productions
- **Parsing**: Probabilistic CKY (Viterbi) bottom-up chart parser
- **Failure handling**: Returns `None` / "UNPARSEABLE" — no crashes

### Tagset Reconciliation

Q1's HMM uses the **universal tagset** (12 tags). The PCFG uses **Penn Treebank tags** (~36 tags).

**Approach**: Static lookup table `UNIVERSAL_TO_PTB` in `q4/tagset.py`:
```
NOUN → NN,  VERB → VBD,  ADJ → JJ,  ADV → RB,  PRON → PRP,
DET  → DT,  ADP  → IN,   CONJ → CC, PRT  → RP,  NUM  → CD
```

**Accuracy loss**: The PCFG receives coarser pre-terminals (e.g., all verbs
become VBD regardless of tense). This reduces parse accuracy vs. using full
PTB tags but requires zero retraining of Q1 models — the correct trade-off
given the assignment constraint "reuse Q1's trained decoder as-is".

---

## Alert Types

| Alert | Trigger | Source models |
|-------|---------|---------------|
| `[SEGMENT-ALERT]` | Token not in Q1 vocab or unusually long | Q1 NgramLM + DecoderConfig |
| `[SPELL-ALERT]` | Token still OOV after segmentation | Q3 SymDel index + unigram counts |
| `[GRAMMAR-ALERT]` | Every 10 words: PPL > 300 or real-word score jump | Q1 trigram LM + Q3 bigram model |

All three alerts display their processing latency in milliseconds.
