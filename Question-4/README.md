# Question 4: Live NLP Editor & Integrated Pipeline

This folder contains the complete implementation for **Question 4** of the NLP Group Assignment.

It builds a background-running, live-typing NLP editor using **Streamlit** that brings together three core sub-systems onto a single live text stream:
1. **Joint Word Segmentation & POS Decoder** (reused from Question 1)
2. **Spelling Corrector** (SymDel method, reused from Question 3)
3. **PCFG Constituency Parser & LM Grammar Engine** (Question 4)

Once the passage is finished the editor scores every sentence with the parser and with both
n-gram models, applies a documented rule to pick which score to trust, and reports the result
as a per-sentence table. The written analysis is in [REPORT_Q4.md](REPORT_Q4.md).

---

## File Structure & Module Overview

```
Question-4/
├── app.py                # Main Streamlit web application
├── requirements.txt      # Python dependencies with exact versions
├── README.md             # This documentation file
├── REPORT_Q4.md          # Written report with all measured results
├── models/
│   ├── q4_pcfg.pkl       # Pre-trained PCFG grammar pickle (~19 MB)
│   ├── q4_pcfg_floors.json   # Calibrated parse-probability floors
│   └── q4_grammar_lms.pkl    # Add-k bigram and trigram cache (built on first run)
├── reports/              # Transcripts, benchmark output and app screenshots
├── scripts/
│   ├── tune_lm_k.py      # Add-k sweep on the Brown dev split
│   ├── calibrate_pcfg.py # Measures the parse floors and the tagset cost
│   ├── run_passage_demo.py   # One full passage end to end
│   ├── run_experiments.py    # The measurements behind the report
│   └── run_speed_demon.py    # Part 5 benchmark
└── q4/
    ├── __init__.py       # Package initializer
    ├── model_loader.py   # Centralized loader for Q1, Q3 and Q4 models
    ├── passage.py        # Passage sampler & merge-error generator
    ├── segmentation.py   # Q1 DP word segmentation alert wrapper
    ├── spelling.py       # Q3 SymDel non-word spelling alert wrapper
    ├── grammar.py        # Sliding window perplexity & real-word error check
    ├── pcfg.py           # Penn Treebank PCFG trainer & Viterbi CKY parser
    ├── tagset.py         # Universal -> Penn Treebank POS tag reconciliation
    ├── ngram_lm.py       # Shared add-k bigram and trigram grammar models
    ├── pipeline.py       # Per-token checks and the running corrected document
    ├── runner.py         # Streams a whole passage through the checks
    ├── analysis.py       # End-of-passage scoring, decision rule and table
    └── speed_demon.py    # Timing harness for the two live layers
```

---

## Detailed System Architecture

### 1. Model Loading (`q4/model_loader.py`)
- **Q1 Integration**: Loads Q1's Witten-Bell trigram LM and HMM tagger. If `Question-1/models/q1_english.pkl` is missing, it trains them once on Brown and caches the result.
- **Q3 Integration**: Loads Q3's pre-trained unigram/bigram counts and builds the SymDel index (`q3_symdel`) in memory.
- **Q4 Models**: Loads the add-k bigram and trigram models built by `ngram_lm.py`.
- Everything is loaded once and shared, so no Q1 or Q3 model is ever retrained while Q4 runs.

### 2. Passage Sampler & Live Simulator (`q4/passage.py`)
- Samples contiguous 5-8 sentence passages from NLTK corpora (`gutenberg`, `brown`, or `reuters`), shuffling the corpus order so runs are not always drawn from the same one.
- **Merge Generator (`inject_merges`)**: Concatenates adjacent tokens with probability p = 0.08 to simulate missing space-bar typing errors (producing merged tokens like `thequick` or `cattable`).
- Yields a token stream for real-time typing simulation in `app.py`.

### 3. Real-Time Alert Pipeline (`q4/pipeline.py`)
`check_token` runs the two per-token checks in order and returns the corrected words, their
tags, the alerts and the latency. `LiveDocument` accumulates the result into sentences and
keeps the merge and correction counts that Part 4 reports.

#### A. Segmentation Alert (`q4/segmentation.py`)
- Evaluates incoming tokens using Q1's beam-search decoder with the settings Q1 selected: maximum word length 19, beam width 8, alpha and beta 1.0.
- Compares the log-likelihood of splitting the token against keeping it whole.
- If splitting wins it fires **`[SEGMENT-ALERT]`** and outputs the split words with their Q1 Universal POS tags.

#### B. Spelling Alert (`q4/spelling.py`)
- Runs on every word the segmentation step produced that is still not in the vocabulary.
- Uses Q3's `SpellingCorrector` with **Method B (SymDel)**, which gives sub-millisecond candidate lookup.
- Fires **`[SPELL-ALERT]`** with the highest-frequency candidate. Words are retagged after a correction, since the original tags were predicted from the misspelled form.

#### C. Grammar & Real-Word Error Alert (`q4/grammar.py`)
- Runs every N = 10 words (`TRIGGER_N`).
- **Perplexity Check**: Computes window perplexity with Q4's add-k trigram. The threshold is not a fixed constant: it is the 95th percentile of perplexity over clean Brown dev windows, which works out at about 17,344 and is adjustable from the sidebar.
- **Real-Word Check**: Scores edit-distance-1 replacements for in-vocabulary words with Q3's bigram model, and flags a substitution that improves the local score by more than 2.0 nats.

### 4. PCFG Constituency Parser (`q4/pcfg.py`)
- **Grammar Induction**: Induces a Chomsky Normal Form PCFG from `nltk.corpus.treebank`.
- **Rule Pruning (`min_count=2`)**: Keeps the pickle at ~19 MB and parsing at ~34 ms per sentence. Keeping every rule reaches a higher parse rate but produces a 116 MB pickle and takes about a second per sentence, which is measured in the report.
- **Viterbi CKY Parser**: Standard probabilistic bottom-up parsing, returning both the bracketed tree and its log probability. A chart only counts as parsed when a sentence symbol spans the whole input, so a bare noun phrase is not mistaken for a sentence. Sentences that cannot be parsed come back as unparseable rather than crashing.

### 5. Tagset Reconciliation (`q4/tagset.py`)
- Q1 tags with the 12-class Universal tagset and the PCFG expects Penn Treebank tags, so each Universal tag is mapped to its most common Penn equivalent.
- The map agrees with the treebank's own tags 58.2% of the time and costs 3.5 points of parse rate, both measured by `scripts/calibrate_pcfg.py`.

### 6. Shared N-gram Grammar Models (`q4/ngram_lm.py`)
- Trains one bigram and one trigram model on the Brown train split with add-k smoothing, using k = 0.01 and k = 0.001 respectively, chosen by the sweep in `scripts/tune_lm_k.py`.
- Calibrates the score floors and the live perplexity threshold on held-out Brown text, so every threshold downstream is a measured percentile rather than a guess.
- Cached in `models/q4_grammar_lms.pkl`, rebuilt in about 13 seconds when missing.

### 7. End-of-Passage Analysis (`q4/analysis.py`)
Scores each finished sentence three ways and picks one:
1. The PCFG when the sentence parses, is at most 25 words, and the parse is not a probability outlier.
2. Otherwise the trigram, when at least 80 percent of the sentence is in its vocabulary.
3. Otherwise the bigram.

The verdict is the chosen score against the calibrated floors, and the summary table carries
the sentence, all three scores, the chosen method, the verdict, and how many merges and
spelling fixes landed in that sentence.

---

## How to Set Up and Run

### 1. Environment

```bash
cd Question-4
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(d) for d in ['brown','universal_tagset','treebank','gutenberg','reuters','punkt']]"
```

### 2. Launching the Streamlit App

```bash
streamlit run app.py
```

The first launch trains anything that is missing, which takes under a minute, and caches it.
The UI offers two modes:
- **Simulated Live Typing**: streams a sampled passage token by token with configurable speed and merge probability, then runs the final analysis when the stream ends.
- **Manual Typing**: processes each word as it is entered, with an **Analyse Text** button for the final table.

### 3. Running the scripts

```bash
venv/bin/python scripts/run_passage_demo.py --seed 1   # one passage end to end
venv/bin/python scripts/run_speed_demon.py             # Part 5 benchmark
venv/bin/python scripts/run_experiments.py             # every number in the report
venv/bin/python scripts/tune_lm_k.py                   # add-k sweep
venv/bin/python scripts/calibrate_pcfg.py              # rebuild the parse floors
```

---

## Guidance for Team Members & Future Work

1. **Reusing Loaded Models**:
   Call `from q4.model_loader import load_all_models` to retrieve the cached model dictionary (`q1_lm`, `q1_tagger`, `corrector`, `q3_vocab`, `q4_bigram`, `q4_trigram`, `q4_floors`).

2. **Adding New Alert Types**:
   Add the check to `check_token` in `q4/pipeline.py` rather than to `app.py`, so the scripts and the benchmark pick it up as well.

3. **Modifying Grammar / Parser**:
   `q4/pcfg.py` caches the trained grammar in `models/q4_pcfg.pkl`. If you change grammar induction, delete that file so it retrains, then rerun `scripts/calibrate_pcfg.py` because the floors depend on it.

4. **Changing the n-gram models**:
   Delete `models/q4_grammar_lms.pkl` after editing `q4/ngram_lm.py`. The floors and the live perplexity threshold are stored inside that pickle and are recomputed with it.
