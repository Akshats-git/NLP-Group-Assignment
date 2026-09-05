# NLP Group Assignment 1 — Question 1, Parts 1–3

**Part 1 — Word segmentation.** A trigram word language model per corpus, plus a
dynamic-programming (Viterbi) decoder that recovers word boundaries from
unspaced text.

**Part 2 — POS tagging.** A trigram HMM that learns emission probabilities
`P(word | tag)` and transition probabilities `P(tag | previous two tags)`, and
tags with the same kind of dynamic program — a Viterbi lattice whose state is
the last two tags. It is evaluated twice: on gold words (tagging alone) and on
the Part 1 segmenter's own output (the pipeline the brief asks for).

**Part 3 — Morphology-aware tagging.** The same tagger over a tagset refined
with gender and number (`NOUN-Fem-Sg`, `ADJ-Masc-Pl`), which lets the transition
model learn agreement — and lets us ask *"did the model reproduce grammatical
agreement?"* rather than only *"was the tag right?"*. Spanish has the FEATS to
support this; Brown does not, which makes English an explicit null result.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['brown','universal_tagset','punkt','treebank','gutenberg']]"
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-GSD.git data/UD_Spanish-GSD
# optional, for the Part 3 German comparison:
# git clone --depth 1 https://github.com/UniversalDependencies/UD_German-GSD.git data/UD_German-GSD
```

Verify the data layer and print the corpus statistics used in the report:

```bash
.venv/bin/python scripts/inspect_q1_data.py
```

## Layout

| Path | Purpose |
| --- | --- |
| [q1/data.py](q1/data.py) | Corpus loading, normalisation, splits, Brown→Penn map, UD FEATS → `NOUN-Fem-Sg` tags — the `Token`/`Sentence`/`Corpus` types |
| [q1/lm.py](q1/lm.py) | Trigram word LM (Witten-Bell / Kneser-Ney / add-k) + character LM for unknowns |
| [q1/segment.py](q1/segment.py) | `DecoderConfig` and the DP/Viterbi segmentation decoder |
| [q1/tagger.py](q1/tagger.py) | Part 2: trigram HMM tagger (emissions, deleted-interpolation transitions, suffix model for unknowns) + most-frequent-tag baseline |
| [q1/evaluate.py](q1/evaluate.py) | Scoring — segmentation (token/boundary F1), tagging (known/unknown accuracy), end-to-end with error attribution, agreement reproduction |
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

## Q1 Part 3 modelling decisions

* **The tagset.** POS plus gender and number, in the brief's notation:
  `NOUN-Fem-Sg`, `ADJ-Masc-Pl`. Only the two agreement-carrying features are used
  by default (`MORPH_FEATURES`) — every extra feature multiplies the tagset and
  divides the counts. A token with no relevant FEATS keeps its bare tag.
* **No model changes.** The tagger is agnostic about what a tag means, so Part 3
  is entirely a data decision. Agreement is learned by the ordinary transition
  model: `P(ADJ-Fem-Sg | DET-Fem-Sg, NOUN-Fem-Sg)` is just estimated from more
  specific counts than `P(ADJ | DET, NOUN)`. The run prints that probability
  against its disagreeing counterpart, so the learned pattern is read out of the
  model rather than inferred from its output.
* **Fair comparison.** Accuracy on the refined label set is *not* comparable with
  Part 2 — it is a harder decision over more labels. Both taggers are therefore
  also projected to coarse tags (`coarse_tag`) and scored on that identical
  decision. The morphology-aware tagger inherits Part 2's selected order, so the
  tagset is the only thing that differs.
* **Agreement is scored separately from accuracy.** Over adjacent gold pairs that
  agree in the gold, `score_agreement` reports both *reproduced* (the model's two
  tags agree with each other) and *correct* (they also carry the right values).
  A pair tagged Masc/Masc where the gold is Fem/Fem did propagate a consistent
  gender, and collapsing that into plain accuracy would hide it; reporting only
  the first would let a tag-everything-Masc-Sg model look perfect. On this corpus
  the two columns come out identical — the emission model pins the value from the
  word form, so the model never propagates a *consistently wrong* gender.
* **Agreement is reported on gold words and end to end.** Passing
  `predicted_words` aligns the tags to the gold by character span and keeps a
  pair whose words the segmenter never recovered *in the denominator*, counted as
  not reproduced — end to end that agreement really was lost. The gap between the
  two numbers is the segmenter's contribution, and it is large.
* **English is a null result about the corpus, not the language.** Brown carries
  no FEATS, so the refined tagset is identical to the plain one and there is no
  agreement to measure. The run says so explicitly instead of reporting a
  meaningless comparison. A UD English treebank would be needed to ask the
  question.
* **German.** `--languages German` runs the same Part 3 analysis on UD German-GSD
  (clone it into `data/` first); German adds case to the agreement picture.

## Running

```bash
.venv/bin/python scripts/inspect_q1_data.py   # data-layer checks + corpus statistics
.venv/bin/python scripts/test_q1_models.py    # model correctness tests
.venv/bin/python scripts/run_q1.py            # full Parts 1-3 run, both languages
.venv/bin/python scripts/run_q1.py --fast     # small samples, for iteration
.venv/bin/python scripts/run_q1.py --languages German   # needs data/UD_German-GSD
```

Results land in [models/q1_results.json](models/q1_results.json). The
comparative report is [REPORT_Q1.md](REPORT_Q1.md); it was written against the
full Q1 pipeline. Its segmentation, tagging and error-attribution numbers
reproduce here to within ~0.2 pp. Its agreement section is measured on the
*pipeline* and on gender alone (88.37% over 1,479 pairs); scored the same way,
this tree gives 88.22% over 1,511 pairs. Scored on gold words and on both
features — the run's headline — it is 96.46%, because segmentation errors, not
tagging errors, account for most of the difference. The run prints both modes so
the two are never confused. Its sections on the joint segment-and-tag decoder
describe work not yet re-added to this tree.
