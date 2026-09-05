# NLP Group Assignment 1 — Question 1, Parts 1–2

**Part 1 — Word segmentation.** A trigram word language model per corpus, plus a
dynamic-programming (Viterbi) decoder that recovers word boundaries from
unspaced text.

**Part 2 — POS tagging.** A trigram HMM that learns emission probabilities
`P(word | tag)` and transition probabilities `P(tag | previous two tags)`, and
tags with the same kind of dynamic program — a Viterbi lattice whose state is
the last two tags. It is evaluated twice: on gold words (tagging alone) and on
the Part 1 segmenter's own output (the pipeline the brief asks for).

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
| [q1/tagger.py](q1/tagger.py) | Part 2: trigram HMM tagger (emissions, deleted-interpolation transitions, suffix model for unknowns) + most-frequent-tag baseline |
| [q1/evaluate.py](q1/evaluate.py) | Scoring — segmentation (token/boundary F1), tagging (known/unknown accuracy), end-to-end with error attribution |
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

## Q1 Part 2 modelling decisions

* **Transitions.** `P(t_i | t_{i-2}, t_{i-1})` interpolated with the bigram and
  unigram estimates. The weights come from **deleted interpolation** (Brants 2000):
  every observed trigram votes for the order that predicts it best with itself
  held out. An unseen history drops out of the interpolation and its weight is
  redistributed, so the transition model stays a proper distribution —
  `HMMTagger.check_normalised` asserts this.
* **Emissions.** `P(w | t) = c(w, t) / c(t)`, and candidate tags for a known word
  are restricted to the tags it actually occurred with. That is both the largest
  decoder speedup and an accuracy gain, since it makes an unattested word/tag
  pairing impossible rather than merely improbable.
* **Unknown words.** Words seen ≤ 10 times train a **suffix model**
  `P(tag | last k characters)` by successive abstraction; at decode time it is
  Bayes-inverted (`log P(t|w) − log P(t)`, dropping the tag-independent `P(w)`).
  This is what lets an unseen `-mente` be tagged ADV — it matters most in Spanish,
  where inflection guarantees a steady OOV stream.
* **Decoding.** Viterbi over states = the last two tags, with an optional beam.
  `test_tagger_viterbi_is_optimal` checks it against exhaustive search over all
  tag sequences on a toy corpus.
* **Error attribution.** End-to-end errors are split into *segmentation-induced*
  (the gold character span was never recovered, so no tag decision was made) and
  *genuine* (right span, wrong tag) — the two halves of the pipeline can then be
  judged separately.

## Running

```bash
.venv/bin/python scripts/inspect_q1_data.py   # data-layer checks + corpus statistics
.venv/bin/python scripts/test_q1_models.py    # model correctness tests
.venv/bin/python scripts/run_q1.py            # full Parts 1-2 run, both languages
.venv/bin/python scripts/run_q1.py --fast     # small samples, for iteration
```

Results land in [models/q1_results.json](models/q1_results.json). The
comparative report is [REPORT_Q1.md](REPORT_Q1.md); it was written against the
full Q1 pipeline, so its sections on segmentation, tagging and end-to-end error
attribution match what is here, while its sections on morphology-aware tagsets
and the joint segment-and-tag decoder describe work not yet re-added to this
tree.
