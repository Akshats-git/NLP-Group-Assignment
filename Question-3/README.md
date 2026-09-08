# Q3 — Spelling Corrector (Grading README)

## 1. Overview

Q3 implements a Brown-Corpus-based spelling corrector: vocabulary +
unigram + bigram language models, two candidate-generation methods
(brute-force edit distance and Symmetric Delete), non-word and
real-word/context correction, reproducible accuracy evaluation, a
1,000-word Speed Demon timing benchmark, and an interactive terminal
CLI. This README is written for grading — for full technical detail and
evidence, see the accompanying report ([REPORT_Q3.md](REPORT_Q3.md)).

## 2. Project Structure

```
Q3/
├── nltk_data/                          # local Brown Corpus data (~13 MB) — see Setup
├── requirements.txt                     # Python dependencies (submitted)
├── README.md                            # this file
├── REPORT_Q3.md                         # full technical report (design choices, error analysis, sample runs)
└── q3_spelling_corrector/
    ├── corpus_models.py                 # Part 1 — vocab, unigram, bigram models
    ├── candidates.py                     # Part 2 — Method A + Method B
    ├── corrector.py                      # Part 3 — non-word + real-word correction
    ├── evaluation.py                      # Part 4 — accuracy evaluation
    ├── benchmark.py                       # Part 4 — Speed Demon benchmark
    ├── cli.py                             # Part 5 — interactive terminal CLI
    ├── utils.py                           # shared preprocessing
    └── models/
        └── q3_language_models.pkl        # persisted vocab/unigram/bigram artifact
```

## 3. Requirements

- Python 3
- Dependencies listed in `requirements.txt` (includes `nltk`)
- The Brown Corpus, provided locally under `Q3/nltk_data/corpora/brown/`
  (no download should be necessary — see Setup)

## 4. Setup

Run from the `Q3/` directory:

```bash
cd Q3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NLTK_DATA="$PWD/nltk_data"
```

- `.venv` is a local virtual environment created by the command above —
  it is **not** part of the submission.
- `requirements.txt` **is** part of the submission and is what
  recreates the environment via `pip install -r requirements.txt`.
- `NLTK_DATA` points NLTK at the **submitted, local** copy of the Brown
  Corpus so no download is needed. Export it once per shell session,
  from the `Q3/` directory, before running any commands below.
- The code itself does **not** automatically download the Brown Corpus.
  Since `nltk_data/corpora/brown` is included in this submission, no
  download should normally be required.

**Verify the corpus is visible to NLTK** (run after exporting
`NLTK_DATA`):

```bash
python3 -c "from nltk.corpus import brown; print(len(brown.sents()))"
```

Expected output: `57340`

**Optional fallback** — only needed if the above fails because the
local corpus is missing for some reason (this is not a required step
for grading this submission):

```bash
python3 -c "import nltk; nltk.download('brown')"
```

## 5. How to Run

⚠️ **Working directory matters.** The Python modules in
`q3_spelling_corrector/` import each other directly (e.g. `from utils
import ...`), so they must be run with `q3_spelling_corrector/` as the
working directory. After the `Q3/`-level setup in Section 4 (which
exports `NLTK_DATA`), change into the subfolder before running anything:

```bash
cd Q3/q3_spelling_corrector
```

All commands below assume you are in `Q3/q3_spelling_corrector/` with
`NLTK_DATA` already exported.

| # | Command | What it does |
|---|---|---|
| 1 | `python3 corpus_models.py` | Builds vocabulary, unigram counts, and bigram counts from Brown, and saves them to `models/q3_language_models.pkl`. **Optional** — the artifact is already included in this submission; every other command loads it rather than retraining. Only run this to rebuild it from scratch. |
| 2 | `python3 evaluation.py` | Generates the reproducible test set and reports non-word/real-word accuracy for Method A and Method B. |
| 3 | `python3 benchmark.py` | Runs the 1,000-word Speed Demon benchmark, timing Method A vs Method B. |
| 4 | `python3 cli.py` | Starts the interactive terminal corrector. Type a sentence, see it corrected, type `exit` to quit. |

## 6. What Each Command Tests 

| Part | Marks | Command(s) / Source | What the grader should look for |
|---|---:|---|---|
| Part 1 — Corpus & Model Prep | 6 | `python3 corpus_models.py`; source: `corpus_models.py` | Vocabulary (`set`), unigram `Counter`, sentence-bounded bigram counts, add-k smoothed `bigram_log_prob`, saved to `models/q3_language_models.pkl` |
| Part 2 — Candidate Generation (Method A: 5, Method B: 5) | 10 | Source inspection: `candidates.py`; exercised indirectly via `evaluation.py`'s per-method results | `edit_distance_1_candidates` (Method A: deletion/insertion/replacement/transposition, filtered against vocab); `build_symdel_index` + `symdel_candidates` (Method B: one-character-deletion index, query-time deletion lookup, edit-distance verification). No standalone script exists for Part 2 in isolation — it is exercised through Parts 3–4. |
| Part 3 — Correction Logic (non-word: 4, real-word: 4) | 8 | Source inspection: `corrector.py`; exercised live via `python3 cli.py` and via `evaluation.py` | `SpellingCorrector.correct_nonword` (unigram-frequency ranking) and `correct_realword` (bigram-context scoring against a threshold). No standalone script exists for Part 3 in isolation. |
| Part 4 — Evaluation & Speed Demon (4 + 4) | 8 | `python3 evaluation.py`; `python3 benchmark.py` | Reproducible test set (5,734 cases in the verified environment, where Brown reported 57,340 sentences — see Section 7), separate non-word/real-word accuracy for A and B on identical cases; 1,000-word timed comparison with model/index construction excluded from timing |
| Part 5 — Live Interactive CLI | 8 | `python3 cli.py` | Continuous loop, corrected sentence output, changed-word highlighting, per-sentence latency, clean exit on `exit` |
| **Total** | **40** | | |

## 7. Expected / Verified Results

**Accuracy** (test set: 5,734 non-word cases, 5,734 real-word cases,
fixed random seed, identical test cases for both methods). The verified
environment reported 57,340 Brown sentences, so 10% = 5,734 test cases;
this count reflects that environment's NLTK/Brown installation and is
not guaranteed to be exactly 5,734 in every environment:

| Method | Non-word Accuracy | Real-word Accuracy |
|---|---:|---:|
| Method A | 4679/5734 = 81.60% | 4286/5734 = 74.75% |
| Method B | 4679/5734 = 81.60% | 4286/5734 = 74.75% |

Method A and Method B report identical accuracy because both ultimately
identify the same edit-distance-1 candidate universe from the
vocabulary — they differ in how candidates are generated/retrieved, not
in which candidates exist. Real-word accuracy reflects `real_word_threshold
= 1.1` (see §9 and [REPORT_Q3.md](REPORT_Q3.md) for how this value was
chosen empirically, including a false-positive-rate check on already-correct
text and why a lower, higher-scoring threshold was rejected).

**Speed Demon** (1,000-word batch, same batch/order for both methods,
model/index construction excluded from timing):

| Run | Method A | Method B | Speedup |
|---|---:|---:|---:|
| Run 1 | 0.1130 s | 0.0108 s | ≈ 10.43× |
| Run 2 | 0.0835 s | 0.0089 s | ≈ 9.40× |

Method A explicitly enumerates possible
deletion/insertion/replacement/transposition strings for each query
word. Method B does generate one-character deletions of the input word
at query time, but uses the precomputed Symmetric Delete index to look
them up instead of generating all insertion/replacement/transposition
candidates the way Method A does — it does not skip query-time
generation entirely, only the much larger enumeration Method A
performs.

## 8. CLI Usage and Examples

```bash
python3 cli.py
```

Verified example session (changed words are wrapped in `**asterisks**` in
the `Corrected:` line, per the Part 5 highlighting requirement):

```
> Original:  I hav a good feeling about this.
Corrected: I **had** a good feeling about this.
Changes: hav -> had
Latency: 0.410 ms

> Original:  This is a test sentnce.
Corrected: This is a test **sentence.**
Changes: sentnce -> sentence
Latency: 0.307 ms

> Original:  I would like to sea the world.
Corrected: I would like to **see** the world.
Changes: sea -> see
Latency: 0.390 ms

> Original:  Please meat me at the station.
Corrected: Please meat me at the station.
Changes: none
Latency: 0.348 ms

> Goodbye.
```

Each turn also prints which words changed (`Changes:` line) and the
correction latency. The "meat" example is intentionally left unchanged —
see [REPORT_Q3.md](REPORT_Q3.md) for why: the local bigram context ranks
"beat" above the intended "meet" for this exact phrase regardless of
threshold, so firing here would swap in a different wrong word rather
than the right one; leaving it unchanged was the deliberate choice. These
examples illustrate behavior — they do not represent every possible
spelling error the corrector can or cannot handle.

## 9. Implementation Summary

- **Vocabulary/unigram**: Brown Corpus tokens filtered with
  `str.isalpha()`, lowercased; unigram counts via `Counter`.
- **Bigram model**: counted within sentences only (no cross-sentence
  bigrams); add-k smoothing, `k = 1.0`:
  `P(w2|w1) = (count(w1,w2)+k) / (count(w1)+k·|V|)`.
- **Method A**: brute-force generation of all deletion/insertion/
  replacement/transposition strings, filtered against the vocabulary.
- **Method B**: precomputed one-character-deletion index over the
  vocabulary; query-time deletions of the input word are looked up in
  it, with an edit-distance verification step before acceptance.
- **Non-word correction**: highest-unigram-frequency candidate wins,
  alphabetical tie-break.
- **Real-word correction**: `score(word) = log P(word|previous) + log
  P(next|word)`; a candidate replaces the original only if it scores
  more than 1.1 nats higher. The assignment does not specify an exact
  threshold; 1.1 was chosen empirically by sweeping thresholds against
  the Part 4 real-word test set and a false-positive check on
  already-correct text — see [REPORT_Q3.md](REPORT_Q3.md) for the full
  sweep and why a slightly higher-scoring value (1.0) was rejected.
- **Sentence correction**: non-word correction runs first, then
  real-word correction, using the updated word sequence as context.

Full technical detail is in [REPORT_Q3.md](REPORT_Q3.md).

## 10. Q4 Reuse

Q4 reuses the Q3 vocabulary, unigram model, bigram model, Method A,
Method B/Symmetric Delete, and `SpellingCorrector` without retraining —
see `Question-4/q4/spelling.py` and `Question-4/q4/model_loader.py`.

## 11. Limitations

- Coverage is limited to the Brown Corpus vocabulary.
- Contractions and hyphenated forms (e.g. "don't", "well-known") are not
  handled like normal alphabetic words, since `str.isalpha()` excludes
  tokens containing apostrophes or hyphens.
- Real-word correction depends on local bigram context and the fixed
  threshold.
- Only edit-distance-1 corrections are generated.
