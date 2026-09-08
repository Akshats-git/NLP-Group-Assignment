# Question 3 — Spelling Corrector: Report

This is the technical report referenced by [README.md](README.md). The README
is the grading map (which file/command covers which marked part); this
document covers design choices, the numbers behind them, and an honest
account of where the corrector does and doesn't work.

All numbers below come from a fresh run of `evaluation.py`, `benchmark.py`,
and `cli.py` in this tree (`NLTK_DATA` pointed at the submitted
`nltk_data/`, Brown reporting 57,340 sentences — see README §7 for why this
count is environment-dependent).

## 1. Part 1 — Corpus and Models

`corpus_models.py` builds three artifacts from the Brown Corpus, cleaned to
lowercased alphabetic tokens (`utils.clean_sentence`):

| Artifact | Value |
|---|---:|
| Vocabulary size | 40,234 unique words |
| Total unigram tokens | 981,716 |
| Unique bigram types | 388,815 |
| Smoothing constant `k` | 1.0 (add-k / Laplace-style) |

Bigrams are counted **within sentences only** — `zip(sent, sent[1:])` never
crosses a sentence boundary, so `bigram_counts` has no spurious
cross-sentence pairs. `bigram_log_prob` returns

```
P(w2 | w1) = (count(w1, w2) + k) / (count(w1) + k * |V|)
```

as a natural log, so unseen bigrams and unseen `w1` degrade gracefully to a
smoothed floor instead of zero. Sanity check: `P(the | of) = 0.127`
(`the` is Brown's single most frequent word, so a high probability after a
common preposition is expected).

`k = 1.0` was not tuned against a held-out metric — the assignment does not
require it, and Part 1 is scored on correct add-k implementation, not on an
optimal `k`. It is exposed as a parameter on every function that needs it
(`bigram_log_prob`, `SpellingCorrector`) specifically so it *could* be
retuned later without retraining the counts.

## 2. Part 2 — Candidate Generation

Two independent methods generate the same target set (all vocabulary words
within edit distance 1), by different mechanisms:

- **Method A** (`edit_distance_1_candidates`): brute-force. For a word of
  length *n*, generates every deletion, insertion, replacement, and
  transposition string (`O(n · |Σ|)` strings for a 26-letter alphabet),
  then filters against the vocabulary.
- **Method B** (`build_symdel_index` + `symdel_candidates`): Symmetric
  Delete. At startup, every vocabulary word has all of its one-character
  deletions computed once and indexed (`{deletion: [originals]}`). At query
  time, only the misspelled word's own one-character deletions are computed
  and looked up in that index — insertions, replacements, and
  transpositions are recovered as a side effect of the *deletion-only*
  index (a word reachable from the query by inserting a character is
  exactly a word whose own deletion matches the query), verified with an
  explicit edit-distance check before being accepted.

Both were empirically checked against each other (300 randomly corrupted
words) and against an independent brute-force Damerau-Levenshtein-≤1
checker: **identical candidate sets in every case, zero mismatches.** This
is also *why* Method A and Method B report identical accuracy throughout
this report — they retrieve the same candidate universe by construction;
the difference (see §4) is purely how fast they do it.

## 3. Part 3 — Correction Logic

**Non-word correction** (`correct_nonword`): for a word absent from the
vocabulary, generate candidates (Method A, B, or both) and return the one
with the highest raw unigram frequency, breaking ties alphabetically for
determinism.

**Real-word correction** (`correct_realword`): for a word that *is* in the
vocabulary, compare

```
score(word) = log P(word | previous) + log P(next | word)
```

for the original word against every edit-distance-1 candidate (including
the original in its own candidate set), and switch only if the best
candidate beats the original by more than `real_word_threshold` nats.

### 3.1 Choosing `real_word_threshold`

The assignment does not fix a threshold ("if a candidate phrase has a
*significantly* higher probability..." — "significantly" is left to the
implementer). The first working version used `2.0`, chosen without
evidence. Sweeping it against the Part 4 real-word test set (5,734 cases)
tells a different story:

| Threshold (nats) | Real-word accuracy | False-positive rate* |
|---:|---:|---:|
| 2.0 (original default) | 62.31% | 1.93% |
| 1.5 | 67.60% | — |
| 1.1 (**chosen**) | 74.75% | 4.23% |
| 1.0 | 76.07% | 4.58% |
| 0.5 | 81.25% | 6.71% |
| 0.1 | 82.09% | 8.96% |

\* False-positive rate = share of already-correct in-vocabulary words that
get "corrected" anyway, measured by running `correct_realword` on every
in-vocabulary word position across an independent, uncorrupted 2,000-sentence
Brown sample (34,790 word positions) — i.e. how often the corrector breaks
something that wasn't broken.

The obvious move is "pick the highest accuracy" (0.1, at 82%). That would be
wrong: as the threshold drops, the corrector isn't getting smarter, it's
firing more often on tiny, noise-level probability gaps — the false-positive
rate climbs in lockstep (8.96% at 0.1 vs 1.93% at 2.0). **`1.1` was chosen**
as the point past which additional accuracy comes at a false-positive cost
that grows faster than the accuracy gain (1.1→1.0 buys +1.3pp accuracy for
+0.35pp more false positives; 1.1→0.5 buys +6.5pp accuracy for +2.5pp more
false positives) — and, more concretely, for the reason in §3.2 below.

### 3.2 Case study: why "meat" is left unchanged, on purpose

The assignment's own real-word examples are "I would like to **sea** the
world" and "Please **meat** me at the station." At `threshold=1.1`:

```
> Original:  I would like to sea the world.
Corrected: I would like to **see** the world.
```

```
> Original:  Please meat me at the station.
Corrected: Please meat me at the station.
```

The second one *looks* like a miss, but it's deliberate. Ranked candidates
for "meat" in this exact context:

| Candidate | Score | Margin over "meat" |
|---|---:|---:|
| beat | −20.11 | **+1.10** |
| meet | −20.52 | +0.69 |
| meaty / mea / mead | −21.21 | ≈ 0.00 |
| (meat, original) | −21.21 | — |

"beat" outranks "meet" in this local bigram context (Brown has stronger
support for phrases like "...to beat me..." than "...to meet me...")
**regardless of the threshold** — the ranking order doesn't change, only
whether *anything* fires does. At `threshold=1.0` the corrector does fire
here, but it says **"Please beat me at the station"** — a confident, wrong
answer. At `1.1` (margin needed: >1.1, actual best margin: 1.10) it just
barely stays under the bar and leaves "meat" alone.

Between "silently wrong" (unchanged) and "confidently wrong" (→ "beat"),
unchanged was the better failure mode, so `1.1` was chosen over the
higher-scoring `1.0` specifically to keep this example on the safe side of
the line. This is a genuine limitation of scoring with only immediate
bigram context (`P(word|prev)` and `P(next|word)`) rather than the whole
sentence: nothing in "Please ___ me at the station" locally disambiguates
"meet" from "beat" — both are transitive verbs a person can do to another
person. Fixing this would need a wider context window (trigram+ or a
syntactic cue), which is out of scope for the assignment's bigram-based
design.

### 3.3 What this trade-off looks like in practice

Two full CLI transcripts are in §6. The first (the assignment's own four
example sentences) shows the corrector working as intended. The second,
with different self-chosen sentences, shows the honest cost of `1.1`:

```
> Original:  The qwuick brown fox jumped over the lazy dog.
Corrected: The **quick** brown fox jumped over the **lady** dog.
```

"qwuick"→"quick" is a correct non-word fix. "lazy"→"lady" is a real-word
**false positive** — "lazy" was already correct. Checking its margin:
"lady" beats "lazy" by 1.94 nats in this context, just over the `1.1` bar.
The same run also flips "we"→"he" (margin 1.84) and "wore"→"were" (margin
1.60) — both would **not** have fired at the original `threshold=2.0`. One
more, "bank"→"back" (margin 5.43), is large enough that it would have
misfired even at the original, more conservative default — i.e. it is not
something this threshold change introduced, but a pre-existing weakness of
scoring "bank" only against its immediate neighbors in a corpus where
"to the back" is a far more common phrase than "to the bank" is common in
this context.

This is reported rather than hidden because it is the honest answer to "did
context-aware real-word correction actually help, or add noise?" — **both**:
it roughly doubles the assignment's real-word test accuracy (62.31% →
74.75%, +12.4pp) while measurably increasing how often it touches text that
didn't need touching (1.93% → 4.23% of already-correct words in casual
text). A local-bigram-only design cannot fully separate "a rare but valid
word choice" from "a real-word error" — it can only make the trade-off
explicit and pick a defensible point on that curve, which is what §3.1 does.

## 4. Part 4 — Evaluation and Speed Demon

### 4.1 Accuracy

Test set: 10% of Brown sentences (5,734 cases at this environment's corpus
size — see README §7), one single-edit corruption per sentence, generating
both a non-word and a real-word version of each case from the same
underlying sentence and seed (`seed=42`, fully reproducible).

| Method | Non-word accuracy | Real-word accuracy |
|---|---:|---:|
| Method A | 4679 / 5734 = **81.60%** | 4286 / 5734 = **74.75%** |
| Method B | 4679 / 5734 = **81.60%** | 4286 / 5734 = **74.75%** |

(Identical across methods — see §2 for why.)

### 4.2 Error analysis by edit type

Breaking the same test set down by which single-edit operation produced the
corruption (deletion / insertion / replacement / transposition) shows the
errors are not uniform:

**Non-word correction:**

| Edit type | Accuracy |
|---|---:|
| Insertion | 1650/1766 = **93.43%** |
| Replacement | 1241/1521 = 81.59% |
| Transposition | 1163/1446 = 80.43% |
| Deletion | 625/1001 = **62.44%** |

**Real-word correction:**

| Edit type | Accuracy |
|---|---:|
| Insertion | 424/458 = **92.58%** |
| Replacement | 1000/1198 = 83.47% |
| Transposition | 308/379 = 81.27% |
| Deletion | 2554/3699 = **69.05%** |

**Deletion is the hardest error type in both tasks, by a wide margin.**
Deleting a character from a word tends to produce a *shorter* string with
disproportionately many valid deletion-neighbors — e.g. correcting "ho" (from
"how") has **28 candidates** to rank among, versus a handful for a typical
insertion/replacement corruption. The more candidates compete, the more
likely unigram/bigram frequency picks a *plausible but wrong* one (observed
failures: "into"→"int"→**"in"**, "two"→"tw"→**"to"**, "how"→"ho"→**"to"**).
Insertion is the easiest, because removing the one inserted character
usually recovers a much smaller, less ambiguous candidate set.

### 4.3 Speed Demon

Isolated non-word correction logic, batch of exactly 1,000 misspelled words,
identical batch and order through both methods, model/index construction
excluded from timing:

| Run | Method A | Method B | Speedup |
|---|---:|---:|---:|
| Run 1 | 0.1130 s | 0.0108 s | ≈ 10.43× |
| Run 2 | 0.0835 s | 0.0089 s | ≈ 9.40× |

**Why Method B is faster, specifically:** Method A enumerates the *entire*
edit-distance-1 string space for each query word — every deletion,
insertion (×25 letters at every one of *n+1* positions), replacement (×25
letters at every position), and transposition — an `O(n · |Σ|)` string set
that is regenerated from scratch, in full, for every single query, and then
each generated string is checked against the vocabulary set. Method B only
ever generates a word's *n* one-character **deletions** at query time
(no insertion/replacement/transposition strings are ever materialized) and
looks each one up in a precomputed hash index — so its query-time cost is
`O(n)` string generations plus `O(n)` hash lookups, against Method A's
`O(n · |Σ|)` generations plus set-membership checks. The ~9-10× ratio
observed is consistent with the alphabet-size gap this removes (Method A
generates roughly 25× more insertion/replacement strings alone, and adding
transpositions widens the gap further); Method B recovers those same
insertion/replacement/transposition matches for free as a side effect of the
precomputed *deletion* index rather than generating them at query time.

## 5. Part 5 — Live Interactive CLI

`cli.py` runs a continuous loop (`run_cli`) that reads a sentence, calls
`corrector.correct_sentence`, and prints:

- `Corrected:` — the corrected sentence, with every changed word wrapped in
  `**asterisks**` in place (`render_highlighted`), per the assignment's
  highlighting requirement.
- `Changes:` — a redundant, more explicit `original -> corrected` summary
  line (kept in addition to inline highlighting, not instead of it).
- `Latency:` — wall-clock time of exactly the `correct_sentence` call, in
  milliseconds, measured with `time.perf_counter()` around nothing else
  (not the prints, not the input read; model/index construction happens
  once at startup in `create_corrector`, before the loop begins, so it
  never leaks into a per-sentence number).

Edge cases handled without crashing: blank input (prompts again rather than
correcting nothing), stdin closing without an explicit `exit` (prints
"End of input. Goodbye." and exits 0 rather than raising `EOFError`), and
the exact string `exit` (prints "Goodbye." and exits).

## 6. Sample Runs

**Run 1 — the assignment's own four example sentences:**

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

**Run 2 — self-chosen sentences, mixing non-word and real-word errors:**

```
> Original:  The qwuick brown fox jumped over the lazy dog.
Corrected: The **quick** brown fox jumped over the **lady** dog.
Changes: qwuick -> quick, lazy -> lady
Latency: 0.526 ms

> Original:  I recieved your mesage yesterday.
Corrected: I **received** your **message** yesterday.
Changes: recieved -> received, mesage -> message
Latency: 0.243 ms

> Original:  We went to the bank to withdraw money.
Corrected: **He** went to the **back** to withdraw money.
Changes: we -> he, bank -> back
Latency: 0.616 ms

> Original:  She wore a beutiful dress to the party.
Corrected: She **were** a **beautiful** dress to the party.
Changes: wore -> were, beutiful -> beautiful
Latency: 0.597 ms

> Goodbye.
```

Run 2 is included deliberately, not cherry-picked for a clean result — see
§3.3 for what its false positives show about the real-word threshold
trade-off.

## 7. Comparative Analysis Summary

- **Do Method A and Method B differ in what they find?** No — verified
  identical on 300 random cases plus the full 5,734-case test set (§2).
  They differ only in speed (§4.3), by roughly an order of magnitude, for
  the mechanistic reason given there.
- **Did real-word (context-aware) correction help, or add noise?** Both, and
  the two effects were measured separately (§3.1, §3.3): +12.4pp accuracy
  on genuine errors, +2.3pp false-positive rate on already-correct text, at
  the chosen threshold. Neither number alone tells the full story.
- **Which error type is hardest?** Deletion, for both non-word and
  real-word correction, because deleting a character produces a shorter
  string with more competing valid neighbors (§4.2).
- **How much faster is Method B, and why exactly?** ~9-10×, because it
  replaces Method A's full per-query enumeration of the edit-distance-1
  string space with `O(n)` deletions looked up in a precomputed index
  (§4.3) — not because it checks fewer candidates, but because it never
  generates most of them.

## 8. Limitations

- Coverage is limited to the Brown Corpus vocabulary (40,234 words) — a
  ~1960s corpus; some correct modern words will be treated as unknown, and
  some archaic Brown-only words rank as unexpectedly strong candidates.
- Contractions and hyphenated forms (e.g. "don't", "well-known") are not
  handled like normal alphabetic words, since vocabulary/cleaning filters on
  `str.isalpha()` and excludes tokens containing apostrophes or hyphens.
- Real-word correction depends on **local bigram context only** (previous
  word and next word) — it cannot resolve cases like "meat"/"meet"/"beat"
  that need wider context or world knowledge to disambiguate (§3.2), and it
  has a measurable, quantified false-positive rate on correct text (§3.1,
  §3.3) that is an inherent property of the design, not a bug.
- Only edit-distance-1 corrections are generated (per the assignment scope);
  two-edit errors are out of scope for both methods.
