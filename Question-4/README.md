# Question 4: Live NLP Text Editor with Integrated Pipeline

This directory contains the Question 4 implementation for the NLP Group Assignment. It combines three sub-systems into a real-time Streamlit text editor:

1. **Joint Word Segmentation & POS Tagging** (reused from Question 1)
2. **Spelling Correction** (Method B / SymDel, reused from Question 3)
3. **PCFG Constituency Parser & Language Model Grammar Checker** (Question 4)

---

## Project Structure

```
Question-4/
├── app.py                # Main Streamlit dashboard
├── requirements.txt      # Python dependencies
├── models/
│   └── q4_pcfg.pkl       # Pre-trained/cached PCFG model binary (~19MB)
└── q4/
    ├── model_loader.py   # Loads Q1 and Q3 models
    ├── passage.py        # Passage sampler and token merger (simulates missing space errors)
    ├── segmentation.py   # Q1 DP segmentation check
    ├── spelling.py       # Q3 SymDel spelling corrector check
    ├── grammar.py        # Sliding window perplexity and real-word error check
    ├── pcfg.py           # Penn Treebank PCFG trainer & Viterbi CKY parser
    └── tagset.py         # Universal -> Penn Treebank POS tag reconciliation
```

---

## Setup & Running

1. **Virtual Environment Setup**:
   ```bash
   cd Question-4
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

---

## Technical Notes

- **Tagset Reconciliation (`q4/tagset.py`)**: Maps Q1's 12-class Universal POS tags to Penn Treebank (PTB) tags required by the PCFG grammar.
- **PCFG Model Size (`q4/pcfg.py`)**: Induced on `nltk.corpus.treebank` with rare rule filtering (`min_count=2`), producing a 19MB pickle model suitable for version control.
- **Real-Time Latency**:
  - Segmentation + Spelling Check: ~0.1 - 2.5 ms / token
  - Grammar Check (N=10 window): ~0.9 ms / window
