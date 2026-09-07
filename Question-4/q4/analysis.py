"""
analysis.py - End-of-passage sentence analysis (Q4 Part 4)

Once the passage has been typed, every sentence of the corrected stream is
scored three ways: a PCFG parse, the shared add-k bigram model and the shared
add-k trigram model. A documented rule then picks which of the three verdicts to
trust for that particular sentence, and the whole thing is rendered as the
summary table the report quotes.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from nltk.corpus import treebank

from q4.ngram_lm import coverage, per_word_logprob
from q4.pcfg import parse_sentence
from q4.pipeline import DocumentSentence, clean_word
from q4.tagset import reconcile_tag_sequence

PCFG_FLOOR_PATH = Path(__file__).parent.parent / "models" / "q4_pcfg_floors.json"

# CKY cost grows with the cube of the sentence length, and treebank sentences
# this long are rare enough that the grammar has little to say about them.
MAX_PCFG_WORDS = 25

# How much of a sentence the trigram model must have seen before its opinion is
# worth more than the bigram's, since a sparse trigram backs off anyway.
COVERAGE_MIN = 0.80

# Almost every Penn Treebank S rule ends on the sentence-final punctuation, so a
# sentence handed to the parser without it has no derivation at all. The live
# token stream keeps only alphabetic words, so the full stop is put back here.
SENTENCE_TERMINATOR = "."

NEG_INF = float("-inf")


@dataclass
class SentenceAnalysis:
    """Every number the summary table needs for one sentence."""

    index: int
    text: str
    n_words: int
    pcfg_parseable: bool
    pcfg_log_prob: float
    pcfg_per_word: float
    pcfg_parse: str | None
    bigram_log_prob: float
    bigram_per_word: float
    trigram_log_prob: float
    trigram_per_word: float
    trigram_coverage: float
    chosen_method: str
    chosen_reason: str
    verdict: str
    merges_resolved: int
    spelling_fixes: int
    ptb_tags: tuple[str, ...] = ()


def _band(score: float, low: float, mid: float) -> str:
    """Turn a length-normalised score into a three-way grammaticality verdict."""
    if score <= NEG_INF or score < low:
        return "ungrammatical"
    if score < mid:
        return "borderline"
    return "grammatical"


def choose_method(
    parseable: bool,
    n_words: int,
    pcfg_per_word: float,
    trigram_coverage: float,
    floors: dict[str, float],
) -> tuple[str, str]:
    """The Part 4 decision rule, kept in one function so it stays documented.

    A parse is preferred whenever there is one and it is not a probability
    outlier, because it is the only score that says anything about structure.
    Otherwise the sentence falls back to the trigram if the trigram has actually
    seen enough of it, and to the bigram if it has not.
    """
    if parseable and n_words <= MAX_PCFG_WORDS and pcfg_per_word >= floors["pcfg_p02"]:
        return "pcfg", "parsed with a parse probability in the normal range"

    if parseable:
        reason = "parse probability is an outlier"
    elif n_words > MAX_PCFG_WORDS:
        reason = "too long to parse"
    else:
        reason = "no parse found"

    if trigram_coverage >= COVERAGE_MIN:
        return "trigram", f"{reason}, trigram coverage {trigram_coverage:.0%}"
    return "bigram", f"{reason}, trigram coverage only {trigram_coverage:.0%}"


def analyse_sentence(
    sentence: DocumentSentence,
    index: int,
    grammar,
    models: dict[str, Any],
    floors: dict[str, float],
) -> SentenceAnalysis:
    """Score one corrected sentence and apply the decision rule to it."""
    words = tuple(sentence.words)
    n_words = len(words)

    parse_words = words + (SENTENCE_TERMINATOR,)
    parse_tags = tuple(sentence.tags) + (SENTENCE_TERMINATOR,)

    if n_words <= MAX_PCFG_WORDS:
        parse = parse_sentence(parse_words, parse_tags, grammar)
    else:
        parse = {
            "parseable": False,
            "parse": None,
            "log_prob": NEG_INF,
            "ptb_tags": reconcile_tag_sequence(parse_tags),
        }

    pcfg_log_prob = parse["log_prob"]
    pcfg_per_word = (
        pcfg_log_prob / len(parse_words) if parse["parseable"] else NEG_INF
    )

    bigram, trigram = models["q4_bigram"], models["q4_trigram"]
    bigram_log_prob = bigram.sentence_logprob(words) if n_words else NEG_INF
    trigram_log_prob = trigram.sentence_logprob(words) if n_words else NEG_INF
    trigram_coverage = coverage(trigram, words)

    method, reason = choose_method(
        parse["parseable"], n_words, pcfg_per_word, trigram_coverage, floors
    )

    if method == "pcfg":
        verdict = _band(pcfg_per_word, floors["pcfg_p02"], floors["pcfg_p10"])
    elif method == "trigram":
        verdict = _band(
            per_word_logprob(trigram, words), floors["trigram_p02"], floors["trigram_p10"]
        )
    else:
        verdict = _band(
            per_word_logprob(bigram, words), floors["bigram_p02"], floors["bigram_p10"]
        )

    return SentenceAnalysis(
        index=index,
        text=sentence.text,
        n_words=n_words,
        pcfg_parseable=bool(parse["parseable"]),
        pcfg_log_prob=pcfg_log_prob,
        pcfg_per_word=pcfg_per_word,
        pcfg_parse=parse["parse"],
        bigram_log_prob=bigram_log_prob,
        bigram_per_word=per_word_logprob(bigram, words),
        trigram_log_prob=trigram_log_prob,
        trigram_per_word=per_word_logprob(trigram, words),
        trigram_coverage=trigram_coverage,
        chosen_method=method,
        chosen_reason=reason,
        verdict=verdict,
        merges_resolved=sentence.merges_resolved,
        spelling_fixes=sentence.spelling_fixes,
        ptb_tags=tuple(parse["ptb_tags"]),
    )


def analyse_document(
    sentences: Sequence[DocumentSentence],
    grammar,
    models: dict[str, Any],
    floors: dict[str, float],
) -> list[SentenceAnalysis]:
    """Run the Part 4 analysis over every sentence of a finished passage."""
    return [
        analyse_sentence(sentence, i + 1, grammar, models, floors)
        for i, sentence in enumerate(sentences)
        if sentence.words
    ]


def summary_rows(analyses: Sequence[SentenceAnalysis]) -> list[dict[str, Any]]:
    """The Part 4 table, one dict per sentence, in the column order the brief asks for."""
    rows = []
    for item in analyses:
        rows.append(
            {
                "#": item.index,
                "sentence": item.text,
                "pcfg": (
                    f"{item.pcfg_per_word:.2f}" if item.pcfg_parseable else "unparseable"
                ),
                "bigram": round(item.bigram_per_word, 2),
                "trigram": round(item.trigram_per_word, 2),
                "chosen method": item.chosen_method,
                "verdict": item.verdict,
                "merges resolved": item.merges_resolved,
                "spelling fixes": item.spelling_fixes,
            }
        )
    return rows


def render_table(rows: Sequence[dict[str, Any]], sentence_width: int = 46) -> str:
    """Fixed-width rendering of the summary table for terminal output."""
    if not rows:
        return "No sentences to report."

    headers = list(rows[0].keys())
    shown = []
    for row in rows:
        cells = dict(row)
        text = str(cells["sentence"])
        if len(text) > sentence_width:
            text = text[: sentence_width - 3] + "..."
        cells["sentence"] = text
        shown.append({h: str(cells[h]) for h in headers})

    widths = {h: max(len(h), *(len(r[h]) for r in shown)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    out = [line, "-" * len(line)]
    for row in shown:
        out.append("  ".join(row[h].ljust(widths[h]) for h in headers))
    return "\n".join(out)


def agreement_summary(
    analyses: Sequence[SentenceAnalysis], live_flagged: Sequence[bool]
) -> dict[str, Any]:
    """Compare the live alerts against the end-of-passage verdict, sentence by sentence.

    A sentence counts as flagged live if any alert fired while it was being
    typed, and as flagged at the end if its final verdict is anything other than
    grammatical. The two layers are looking at different things, so the cases
    where only one of them fires are the interesting ones for the report.
    """
    both = live_only = final_only = neither = 0

    for item, live in zip(analyses, live_flagged):
        final = item.verdict != "grammatical"
        if live and final:
            both += 1
        elif live:
            live_only += 1
        elif final:
            final_only += 1
        else:
            neither += 1

    total = both + live_only + final_only + neither
    return {
        "sentences": total,
        "both_flagged": both,
        "live_only": live_only,
        "final_only": final_only,
        "neither": neither,
        "agreement": (both + neither) / total if total else 0.0,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return NEG_INF
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def calibrate_pcfg_floors(
    grammar,
    tagger,
    sample_size: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """Find out what parse probability a normal sentence gets.

    The sample is parsed through the same path a live sentence takes, that is
    Q1 tags mapped to Penn Treebank tags, so the floors carry whatever cost the
    tagset reconciliation adds. The agreement between the mapped tags and the
    treebank's own tags is measured in the same pass and reported with them.
    """
    rng = random.Random(seed)

    usable = []
    for tagged in treebank.tagged_sents():
        pairs = [
            (clean_word(word), tag)
            for word, tag in tagged
            if tag != "-NONE-" and clean_word(word)
        ]
        if 3 <= len(pairs) <= MAX_PCFG_WORDS:
            usable.append(pairs)

    rng.shuffle(usable)
    sample = usable[:sample_size]

    per_word_scores: list[float] = []
    parsed = 0
    tags_matched = 0
    tags_total = 0

    for pairs in sample:
        words = tuple(word for word, _ in pairs)
        gold_tags = tuple(tag for _, tag in pairs)
        universal_tags = tagger.tag(words)
        mapped_tags = reconcile_tag_sequence(universal_tags)

        tags_total += len(gold_tags)
        tags_matched += sum(1 for a, b in zip(mapped_tags, gold_tags) if a == b)

        # Same shape as a live sentence: alphabetic words plus a full stop.
        parse_words = words + (SENTENCE_TERMINATOR,)
        result = parse_sentence(parse_words, universal_tags + (SENTENCE_TERMINATOR,), grammar)
        if result["parseable"]:
            parsed += 1
            per_word_scores.append(result["log_prob"] / len(parse_words))

    floors = {
        "pcfg_p02": _percentile(per_word_scores, 0.02),
        "pcfg_p10": _percentile(per_word_scores, 0.10),
        "pcfg_median": _percentile(per_word_scores, 0.50),
        "parse_rate": parsed / len(sample) if sample else 0.0,
        "tag_agreement": tags_matched / tags_total if tags_total else 0.0,
        "n_sampled": float(len(sample)),
        "max_words": float(MAX_PCFG_WORDS),
    }

    PCFG_FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PCFG_FLOOR_PATH.open("w", encoding="utf-8") as f:
        json.dump(floors, f, indent=2)
    return floors


def load_floors(models: dict[str, Any], grammar=None, tagger=None) -> dict[str, float]:
    """Combine the n-gram floors from Part 3 with the cached PCFG floors."""
    floors = dict(models["q4_floors"])

    if PCFG_FLOOR_PATH.exists():
        with PCFG_FLOOR_PATH.open(encoding="utf-8") as f:
            floors.update(json.load(f))
    elif grammar is not None and tagger is not None:
        floors.update(calibrate_pcfg_floors(grammar, tagger))
    else:
        raise RuntimeError(
            "No PCFG floors on disk. Run scripts/calibrate_pcfg.py first, or pass "
            "the grammar and the tagger so they can be measured now."
        )
    return floors
