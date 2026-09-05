# NLP Group Assignment 1 — Question 1, Part 1 (Word Segmentation)

Trains a trigram word language model on each corpus and recovers word
boundaries from unspaced text with a dynamic-programming (Viterbi) decoder.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['brown','universal_tagset','punkt','treebank','gutenberg']]"
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-GSD.git data/UD_Spanish-GSD
```

Verify the data layer and print the corpus statistics used in the report:

```bash
.venv/bin/python scripts/inspect_q1_data.py
```

## Layout

| Path | Purpose |
| --- | --- |
| [q1/data.py](q1/data.py) | Corpus loading, normalisation, splits, Brown→Penn map — the `Token`/`Sentence`/`Corpus` types |
| [q1/lm.py](q1/lm.py) | Trigram word LM (Witten-Bell / Kneser-Ney / add-k) + character LM for unknowns |
| [q1/segment.py](q1/segment.py) | `DecoderConfig` and the DP/Viterbi segmentation decoder |
| [q1/evaluate.py](q1/evaluate.py) | Segmentation scoring — token F1, boundary F1, exact-sentence rate |
| [scripts/inspect_q1_data.py](scripts/inspect_q1_data.py) | Invariant checks + corpus statistics |
| [scripts/test_q1_models.py](scripts/test_q1_models.py) | Correctness tests (LM normalisation, decoder, scoring) |
| [scripts/run_q1.py](scripts/run_q1.py) | Train → tune on dev → evaluate on test → persist |
| `data/` | Cloned treebanks (git-ignored) |
| `models/` | Serialised trained models and run results (git-ignored) |

## Q1 data-layer decisions

* **Segmentation input.** `Sentence.chars` is the token forms concatenated with no
  spaces; `Sentence.gold_spans` is the exact character partition. Because the gold
  is a true partition, the DP decoder can work over character offsets and a
  predicted token counts only when its exact character span is a gold span.
* **Normalisation.** Lowercase, NFC, non-letters stripped from inside tokens
  (`don't`→`dont`), empty results dropped. Accents kept. Tokens over 25 characters
  and `SYM` tokens are discarded (flattened URLs).
* **Spanish MWTs.** UD range lines (`del` → `de` + `el`) are expanded to the
  syntactic words. 8,236 occurrences.
* **Splits.** Spanish uses UD's own files. Brown is split 80/10/10, genre-stratified
  and seeded (`seed=42`) — Brown is ordered by its 15 genres, so a contiguous split
  would measure domain shift rather than model quality.
* **Tagsets.** `universal` (12-tag) for the English/Spanish comparison, `penn` for
  the assignment's English sample output, `brown` for the raw tags.

## Running

```bash
.venv/bin/python scripts/inspect_q1_data.py   # data-layer checks + corpus statistics
.venv/bin/python scripts/test_q1_models.py    # model correctness tests
.venv/bin/python scripts/run_q1.py            # full Part 1 run, both languages
.venv/bin/python scripts/run_q1.py --fast     # small samples, for iteration
```

Results land in [models/q1_results.json](models/q1_results.json). The
comparative report is [REPORT_Q1.md](REPORT_Q1.md); note that it was written
against the full Q1 pipeline and its sections on tagging, morphological
agreement and end-to-end error attribution describe code no longer in this
tree (recover it from git commit `ed7c498` if needed).
