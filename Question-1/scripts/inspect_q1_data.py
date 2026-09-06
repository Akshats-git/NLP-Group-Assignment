"""Sanity-check the Q1 data layer and print the statistics the report needs.

Run:  .venv/bin/python scripts/inspect_q1_data.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data import (            # noqa: E402
    Corpus,
    CorpusStats,
    load_brown,
    load_spanish,
    max_word_length,
    vocabulary,
)

RULE = "=" * 78


def check_invariants(corpus: Corpus) -> None:
    """Every Sentence must satisfy the partition invariant the decoder assumes."""
    checked = 0
    for split in ("train", "dev", "test"):
        for sentence in corpus.split(split):
            assert sentence.chars == "".join(sentence.words), sentence.sent_id
            assert sentence.gold_spans[0][0] == 0, sentence.sent_id
            assert sentence.gold_spans[-1][1] == len(sentence.chars), sentence.sent_id
            for (_, end), (next_start, _) in zip(
                sentence.gold_spans, sentence.gold_spans[1:]
            ):
                assert end == next_start, sentence.sent_id
            for token, (start, end) in zip(sentence.tokens, sentence.gold_spans):
                assert sentence.chars[start:end] == token.form, sentence.sent_id
            assert sentence.chars.isalpha(), sentence.sent_id
            checked += 1
    print(f"  invariants verified on {checked:,} sentences  [OK]")


def describe(corpus: Corpus, stats: CorpusStats) -> None:
    print(RULE)
    print(f"{corpus.name}  ({corpus.language}, tagset={corpus.tagset})")
    print(RULE)

    for label, value in stats.as_rows():
        print(f"  {label:<34} {value}")

    n_train_tokens = sum(len(s) for s in corpus.train)
    print()
    print(f"  {'split':<10}{'sentences':>12}{'tokens':>12}{'vocab':>10}{'OOV rate':>11}")
    train_vocab = vocabulary(corpus.train)
    for split in ("train", "dev", "test"):
        sentences = corpus.split(split)
        tokens = [t for s in sentences for t in s.tokens]
        vocab = vocabulary(sentences)
        oov = sum(1 for t in tokens if t.form not in train_vocab)
        oov_rate = f"{100 * oov / len(tokens):.2f}%" if tokens else "-"
        print(
            f"  {split:<10}{len(sentences):>12,}{len(tokens):>12,}"
            f"{len(vocab):>10,}{oov_rate:>11}"
        )

    # Word lengths drive the decoder's inner loop (max_word_len hyper-parameter).
    print()
    print("  word length: ", end="")
    for pct in (0.95, 0.99, 0.999, 1.0):
        label = "max" if pct == 1.0 else f"p{pct * 100:g}"
        print(f"{label}={max_word_length(corpus.train, pct)}  ", end="")
    print()

    tags = Counter(t.upos for s in corpus.train for t in s.tokens)
    print(f"  tagset size: {len(tags)}")
    print(f"  avg tokens/sentence: {n_train_tokens / max(1, len(corpus.train)):.1f}")

    print("\n  most frequent tags:")
    for tag, count in tags.most_common(6):
        print(f"    {tag:<8}{count:>10,}")

    example = next(s for s in corpus.dev if 4 <= len(s) <= 9)
    print("\n  example sentence")
    print(f"    input  : {example.chars}")
    print(f"    gold   : {' '.join(example.words)}")

    check_invariants(corpus)


def check_sample_strings(corpus: Corpus, samples: list[tuple[str, list[str]]]) -> None:
    """Are the assignment's sample test strings even reachable from training?

    A word missing from the training vocabulary cannot be produced by the
    decoder, so this is a hard ceiling on the sample-output demo.
    """
    train_vocab = vocabulary(corpus.train)
    print(f"\n  assignment sample strings vs {corpus.language} training vocabulary:")
    for string, gold_words in samples:
        missing = [w for w in gold_words if w not in train_vocab]
        verdict = "reachable" if not missing else f"MISSING {missing}"
        print(f"    {string:<40} {verdict}")


def main() -> None:
    print("\nLoading English (Brown, universal tagset) ...")
    brown_universal, brown_stats = load_brown(tagset="universal", return_stats=True)
    describe(brown_universal, brown_stats)
    check_sample_strings(
        brown_universal,
        [
            (
                "thequickbrownfoxjumpsoverthelazydog",
                ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            )
        ],
    )

    print("\nLoading English (Brown, Penn tagset) ...")
    brown_penn = load_brown(tagset="penn")
    example = next(s for s in brown_penn.dev if 5 <= len(s) <= 9)
    print(f"  Penn-tagged example: {[(t.form, t.upos) for t in example.tokens]}")

    print("\nLoading Spanish (UD Spanish-GSD) ...")
    spanish, spanish_stats = load_spanish(return_stats=True)
    describe(spanish, spanish_stats)
    check_sample_strings(
        spanish,
        [
            ("mispadrespuedenviajar", ["mis", "padres", "pueden", "viajar"]),
            ("elcielodespejadoesazul", ["el", "cielo", "despejado", "es", "azul"]),
        ],
    )


if __name__ == "__main__":
    main()
