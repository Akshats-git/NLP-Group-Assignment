# Question 4: Live NLP Editor & Integrated Pipeline

This folder contains the complete implementation for **Question 4 (Part 1 and Part 2)** of the NLP Group Assignment.

It builds a background-running, live-typing NLP editor using **Streamlit** that brings together three core sub-systems onto a single live text stream:
1. **Joint Word Segmentation & POS Decoder** (reused from Question 1)
2. **Spelling Corrector** (SymDel method, reused from Question 3)
3. **PCFG Constituency Parser & LM Grammar Engine** (Question 4)

---

## File Structure & Module Overview

```
Question-4/
├── app.py                # Main Streamlit web application
├── requirements.txt      # Python dependencies with exact versions
├── README.md             # This documentation file
├── models/
│   └── q4_pcfg.pkl       # Pre-trained PCFG grammar pickle (~19 MB)
└── q4/
    ├── __init__.py       # Package initializer
    ├── model_loader.py   # Centralized loader for Q1 and Q3 models
    ├── passage.py        # Passage sampler & merge-error generator
    ├── segmentation.py   # Q1 DP word segmentation alert wrapper
    ├── spelling.py       # Q3 SymDel non-word spelling alert wrapper
    ├── grammar.py        # Sliding window perplexity & real-word error check
    ├── pcfg.py           # Penn Treebank PCFG trainer & Viterbi CKY parser
    └── tagset.py         # Universal -> Penn Treebank POS tag reconciliation
```

---

## Detailed System Architecture

### 1. Model Loading (`q4/model_loader.py`)
- **Shared LM**: Uses Q1's Witten-Bell trigram Language Model (LM) trained on the Brown corpus as the primary LM across segmentation and grammar perplexity checks.
- **Auto-Train Fallback**: If `Question-1/models/q1_english.pkl` does not exist on disk, `model_loader.py` automatically trains the Q1 LM and HMM tagger on the Brown corpus and caches the result.
- **Q3 Integration**: Loads Q3's pre-trained unigram/bigram counts and builds the SymDel index (`q3_symdel`) in memory.

### 2. Passage Sampler & Live Simulator (`q4/passage.py`)
- Samples contiguous 5–8 sentence passages from NLTK corpora (`gutenberg`, `brown`, or `reuters`).
- **Merge Generator (`inject_merges`)**: Concatenates adjacent tokens with probability $p = 0.08$ to simulate missing space-bar typing errors (producing merged tokens like `thequick` or `cattable`).
- Yields a token stream for real-time typing simulation in `app.py`.

### 3. Real-Time Alert Pipeline

#### A. Segmentation Alert (`q4/segmentation.py`)
- Evaluates incoming tokens using Q1's beam-search segmentation decoder.
- Compares the sentence log-likelihood of splitting the token into multiple words versus keeping it as a single token.
- If splitting improves likelihood, it fires **`[SEGMENT-ALERT]`** and outputs the split words with their Q1 Universal POS tags.

#### B. Spelling Alert (`q4/spelling.py`)
- Evaluates tokens not found in the vocabulary using Q3's `SpellingCorrector` configured with **Method B (SymDel)**.
- Method B allows sub-millisecond candidate lookup using symmetric delete pre-indexing.
- Fires **`[SPELL-ALERT]`** and provides top candidate suggestions for non-word errors.

#### C. Grammar & Real-Word Error Alert (`q4/grammar.py`)
- Runs every $N = 10$ words (configured via `TRIGGER_N`).
- **Perplexity Check**: Computes window perplexity using the shared Q1 LM. If perplexity exceeds 300 (configurable via slider), it fires a grammar alert.
- **Real-Word Error Check**: Evaluates edit-distance-1 replacement candidates for valid dictionary words using Q3's context bigram model. If a candidate improves local bigram score by more than 2.0 nats, it flags a real-word error (e.g., `from` -> `form`).

### 4. PCFG Constituency Parser (`q4/pcfg.py`)
- **Grammar Induction**: Induces a Chomsky Normal Form (CNF) PCFG from `nltk.corpus.treebank` (~3,900 parsed sentences).
- **Rule Pruning (`min_count=2`)**: Filters single-occurrence productions to keep the saved model pickle size at ~19 MB (ensuring it can be pushed to GitHub under the 100 MB file limit).
- **Viterbi CKY Parser**: Implements standard probabilistic bottom-up CKY parsing. Unknown words fall back to using their POS tag as the pre-terminal. Sentences that cannot be parsed return `None` gracefully without crashing.

### 5. Tagset Reconciliation (`q4/tagset.py`)
- Q1 tags tokens using the **12-class Universal Tagset** (`NOUN`, `VERB`, `DET`, `ADJ`, etc.).
- The PCFG parser expects **Penn Treebank (PTB) Tags** (`NN`, `VBD`, `DT`, `JJ`, etc.).
- `tagset.py` maps Universal tags to PTB tags via a deterministic mapping table (`NOUN -> NN`, `VERB -> VBD`, `DET -> DT`, etc.), enabling direct compatibility between Q1 and the PCFG parser without retraining Q1's tagger.

---

## How to Set Up and Run

### 1. Environment Activation
Using the dedicated virtual environment in `Question-4/`:

```bash
cd Question-4
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the Streamlit App
Run the following command from the `Question-4/` directory:

```bash
streamlit run app.py
```

The Streamlit UI will open in your browser, offering two modes:
- **Simulated Live Typing**: Streams sampled passage text token-by-token with configurable speed and merge probability.
- **Manual Typing**: Lets you type or paste custom text to observe alerts and parse trees live.

---

## Guidance for Team Members & Future Work

If you are extending Question 4 or building the next part:

1. **Reusing Loaded Models**:
   Call `from q4.model_loader import load_all_models` to retrieve the cached model dictionary (`q1_lm`, `q1_tagger`, `corrector`, `q3_vocab`, etc.).

2. **Adding New Alert Types**:
   To add a new alert, implement a check function in a new module under `q4/` and call it inside `process_token()` or `process_grammar()` in `app.py`.

3. **Modifying Grammar / Parser**:
   `q4/pcfg.py` caches the trained grammar in `models/q4_pcfg.pkl`. If you change grammar induction logic, delete `models/q4_pcfg.pkl` so it retrains on the next run.
