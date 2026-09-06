from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Literal, Sequence

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

Smoothing = Literal["witten_bell", "kneser_ney", "addk"]
NEG_INF = float("-inf")


class CharLM:
   
    def __init__(self, k: float = 0.1) -> None:
        self.k = k
        self._bigrams: Counter[tuple[str, str]] = Counter()
        self._contexts: Counter[str] = Counter()
        self._alphabet: set[str] = set()
        self._v = 1

    def fit(self, word_types: Iterable[str]) -> "CharLM":
        for word in word_types:
            previous = BOS
            for char in word:
                self._bigrams[(previous, char)] += 1
                self._contexts[previous] += 1
                self._alphabet.add(char)
                previous = char
            self._bigrams[(previous, EOS)] += 1
            self._contexts[previous] += 1
        # +1 for the end-of-word symbol, which is a legal continuation.
        self._v = len(self._alphabet) + 1
        return self

    def logprob(self, word: str) -> float:
        if not word:
            return NEG_INF
        total = 0.0
        previous = BOS
        for char in list(word) + [EOS]:
            numerator = self._bigrams[(previous, char)] + self.k
            denominator = self._contexts[previous] + self.k * self._v
            total += math.log(numerator / denominator)
            previous = char
        return total


class NgramLM:

    def __init__(
        self,
        order: int = 3,
        smoothing: Smoothing = "witten_bell",
        k: float = 0.01,
        discount: float = 0.75,
        unk_threshold: int = 1,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.smoothing = smoothing
        self.k = k
        self.discount = discount
        self.unk_threshold = unk_threshold

        self.vocab: set[str] = set()
        self.char_lm: CharLM | None = None
        # ngrams[n] maps an n-gram tuple to its count, for n = 1 .. order
        self.ngrams: dict[int, Counter[tuple[str, ...]]] = {}
        # context_total[n][u] = sum over w of c(u + (w,)), for |u| = n
        self.context_total: dict[int, dict[tuple[str, ...], int]] = {}
        # context_types[n][u] = number of distinct w with c(u + (w,)) > 0
        self.context_types: dict[int, dict[tuple[str, ...], int]] = {}
        # Kneser-Ney continuation counts for the lower orders
        self.cont_count: dict[int, dict[tuple[str, ...], int]] = {}
        self.cont_total: dict[int, int] = {}
        self.cont_context_total: dict[int, dict[tuple[str, ...], int]] = {}
        self.cont_context_types: dict[int, dict[tuple[str, ...], int]] = {}
        self._n_tokens = 0
        self._cache: dict[tuple[str, tuple[str, ...]], float] = {}
        self._cache_limit = 2_000_000

    # -- training ----------------------------------------------------------
    def fit(
        self,
        sentences: Iterable[Sequence[str]],
        char_lm: bool = True,
    ) -> "NgramLM":
        sentences = [list(s) for s in sentences]

        raw_counts: Counter[str] = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in raw_counts.items() if c > self.unk_threshold}
        self.vocab.update({UNK, EOS})

        for n in range(1, self.order + 1):
            self.ngrams[n] = Counter()

        for sentence in sentences:
            words = [w if w in self.vocab else UNK for w in sentence]
            padded = [BOS] * (self.order - 1) + words + [EOS]
            self._n_tokens += len(words) + 1
            for n in range(1, self.order + 1):
                counter = self.ngrams[n]
                # Start at index order-1 so that every n-gram ends on a real
                # token; BOS only ever appears inside a context.
                for i in range(self.order - 1, len(padded)):
                    counter[tuple(padded[i - n + 1 : i + 1])] += 1

        self._build_context_tables()
        if self.smoothing == "kneser_ney":
            self._build_continuation_tables()

        if char_lm:
            self.char_lm = CharLM().fit(
                w for w in raw_counts if w not in (UNK, EOS, BOS)
            )
        return self

    def _build_context_tables(self) -> None:
        for n in range(1, self.order + 1):
            totals: dict[tuple[str, ...], int] = defaultdict(int)
            types: dict[tuple[str, ...], int] = defaultdict(int)
            for gram, count in self.ngrams[n].items():
                context = gram[:-1]
                totals[context] += count
                types[context] += 1
            self.context_total[n - 1] = dict(totals)
            self.context_types[n - 1] = dict(types)

    def _build_continuation_tables(self) -> None:
        """Kneser-Ney lower-order counts: how many *distinct* contexts precede.

        The unigram term becomes "how likely is this word to appear in a new
        context", not "how frequent is it" -- the distinction that stops
        high-frequency-but-fixed-context words (``Francisco``) from dominating.
        """
        for n in range(1, self.order):
            counts: dict[tuple[str, ...], int] = defaultdict(int)
            for gram in self.ngrams[n + 1]:
                counts[gram[1:]] += 1
            self.cont_count[n] = dict(counts)
            self.cont_total[n] = sum(counts.values())
            totals: dict[tuple[str, ...], int] = defaultdict(int)
            types: dict[tuple[str, ...], int] = defaultdict(int)
            for gram, count in counts.items():
                totals[gram[:-1]] += count
                types[gram[:-1]] += 1
            self.cont_context_total[n - 1] = dict(totals)
            self.cont_context_types[n - 1] = dict(types)

    # -- scoring -----------------------------------------------------------
    def _prob(self, word: str, context: tuple[str, ...], continuation: bool) -> float:
        """Recursive interpolated probability of ``word`` given ``context``."""
        n = len(context) + 1

        if n == 1:
            if self.smoothing == "kneser_ney" and continuation:
                total = self.cont_total.get(1, 0)
                if total:
                    numerator = self.cont_count[1].get((word,), 0)
                    return (numerator + self.k) / (total + self.k * len(self.vocab))
            count = self.ngrams[1].get((word,), 0)
            total = self.context_total[0].get((), 0)
            return (count + self.k) / (total + self.k * len(self.vocab))

        lower = self._prob(word, context[1:], continuation)

        if self.smoothing == "kneser_ney" and continuation:
            gram_count = self.cont_count.get(n, {}).get(context + (word,), 0)
            total = self.cont_context_total.get(n - 1, {}).get(context, 0)
            types = self.cont_context_types.get(n - 1, {}).get(context, 0)
        else:
            gram_count = self.ngrams[n].get(context + (word,), 0)
            total = self.context_total[n - 1].get(context, 0)
            types = self.context_types[n - 1].get(context, 0)

        if total == 0:
            return lower

        if self.smoothing == "addk":
            return (gram_count + self.k) / (total + self.k * len(self.vocab))
        if self.smoothing == "witten_bell":
            return (gram_count + types * lower) / (total + types)
        # kneser_ney / absolute discounting
        discounted = max(gram_count - self.discount, 0.0) / total
        backoff_weight = self.discount * types / total
        # A context seen only with unseen continuations must still back off.
        return discounted + backoff_weight * lower

    def prob(self, word: str, context: Sequence[str] = ()) -> float:
        """P(word | context), with the context truncated to the model order."""
        context = tuple(context)[-(self.order - 1) :] if self.order > 1 else ()
        known = word if word in self.vocab else UNK
        context = tuple(w if (w in self.vocab or w == BOS) else UNK for w in context)
        key = (known, context)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._prob(known, context, continuation=self.smoothing == "kneser_ney")
        if len(self._cache) < self._cache_limit:
            self._cache[key] = value
        return value

    def __getstate__(self) -> dict:
        """Drop the memo cache when pickling -- it is derived, and large."""
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def logprob(self, word: str, context: Sequence[str] = ()) -> float:
        """log P(word | context), open-vocabulary.

        Unseen words are charged ``log P(<unk>|context)`` plus the character
        model's log-probability of the specific string, so that ``despejado``
        and ``xqzptr`` are not treated as equally likely unknowns.
        """
        probability = self.prob(word, context)
        score = math.log(probability) if probability > 0 else NEG_INF
        if word not in self.vocab and self.char_lm is not None:
            score += self.char_lm.logprob(word)
        return score

    def sentence_logprob(self, words: Sequence[str]) -> float:
        history: list[str] = [BOS] * (self.order - 1)
        total = 0.0
        for word in list(words) + [EOS]:
            total += self.logprob(word, history)
            history = (history + [word])[-(self.order - 1) :] if self.order > 1 else []
        return total

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        total_logprob = 0.0
        n_tokens = 0
        for sentence in sentences:
            total_logprob += self.sentence_logprob(sentence)
            n_tokens += len(sentence) + 1        # +1 for </s>
        if n_tokens == 0:
            return float("nan")
        return math.exp(-total_logprob / n_tokens)

    # -- validation --------------------------------------------------------
    def check_normalised(
        self, context: Sequence[str] = (), tolerance: float = 1e-6
    ) -> float:
        """Sum P(w|context) over the closed vocabulary; should be 1.0.

        The segmenter compares hypotheses with different word counts, so a
        model that leaks or gains probability mass would bias segmentation
        towards longer or shorter words respectively.  This is asserted in the
        model tests rather than merely assumed.
        """
        total = sum(self.prob(w, context) for w in self.vocab)
        if abs(total - 1.0) > tolerance:
            raise AssertionError(
                f"{self.smoothing} not normalised for context {tuple(context)!r}: "
                f"sum = {total:.9f}"
            )
        return total
