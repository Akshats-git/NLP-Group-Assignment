from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

BOS_TAG = "<bos>"
EOS_TAG = "<eos>"
NEG_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class TaggerConfig:
    """Hyper-parameters of the HMM tagger.

    ``order=3`` is the model the brief specifies (condition on the previous two
    tags); ``order=2`` is kept so the trigram's contribution can be measured
    rather than assumed.
    """

    order: int = 3
    #: Words seen at most this many times train the suffix model for unknowns.
    rare_threshold: int = 10
    #: Longest word-final character sequence the suffix model conditions on.
    max_suffix_len: int = 5
    #: Keep only the k most likely tags for an unknown word (None = all).
    max_unknown_tags: int | None = 8
    #: States kept per Viterbi column (None = exact search).
    beam_width: int | None = None
    #: Floor on each interpolation weight, so no order is ever fully switched off.
    lambda_floor: float = 1e-3

    def replace(self, **changes) -> "TaggerConfig":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


class HMMTagger:
    """Trigram HMM POS tagger with deleted interpolation and a suffix model."""

    def __init__(self, config: TaggerConfig = TaggerConfig()) -> None:
        if config.order < 1:
            raise ValueError("order must be >= 1")
        self.config = config
        self.order = config.order

        # -- transitions ----------------------------------------------------
        # ngrams[n] counts tag n-grams; context_total[n-1] sums them over the
        # final tag, so every conditional below is normalised by construction.
        self.ngrams: dict[int, Counter[tuple[str, ...]]] = {}
        self.context_total: dict[int, dict[tuple[str, ...], int]] = {}
        self.lambdas: dict[int, float] = {}

        # -- emissions ------------------------------------------------------
        self.emissions: dict[str, Counter[str]] = {}   # tag -> word -> count
        self.tag_count: Counter[str] = Counter()       # tag -> tokens
        self.word_tags: dict[str, tuple[str, ...]] = {}  # word -> tags seen
        self.tags: tuple[str, ...] = ()                # real tags, no BOS/EOS
        self.tag_prior: dict[str, float] = {}
        self.vocab: set[str] = set()

        # -- unknown-word suffix model --------------------------------------
        self.suffix_counts: dict[str, Counter[str]] = {}
        self.suffix_base: dict[str, float] = {}
        self.theta: float = 0.0
        self.default_tag: str = "NOUN"

        self._suffix_cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, tagged_sentences: Iterable[Sequence[tuple[str, str]]]) -> "HMMTagger":
        """Estimate emissions, transitions and the unknown-word model.

        ``tagged_sentences`` yields sequences of ``(word, tag)`` pairs -- e.g.
        ``zip(sentence.words, sentence.tags)`` for each training sentence.
        """
        sentences = [list(s) for s in tagged_sentences]

        emissions: dict[str, Counter[str]] = defaultdict(Counter)
        word_tag_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for n in range(1, self.order + 1):
            self.ngrams[n] = Counter()

        for sentence in sentences:
            tags = [tag for _, tag in sentence]
            for word, tag in sentence:
                emissions[tag][word] += 1
                word_tag_counts[word][tag] += 1
                self.tag_count[tag] += 1
            # Pad so that sentence-initial and sentence-final transitions are
            # modelled: the first real tag is predicted from (BOS, BOS), and
            # EOS is predicted from the last two real tags.
            padded = [BOS_TAG] * (self.order - 1) + tags + [EOS_TAG]
            for n in range(1, self.order + 1):
                counter = self.ngrams[n]
                for i in range(self.order - 1, len(padded)):
                    counter[tuple(padded[i - n + 1 : i + 1])] += 1

        self.emissions = {tag: counts for tag, counts in emissions.items()}
        self.word_tags = {
            word: tuple(sorted(counts, key=counts.get, reverse=True))
            for word, counts in word_tag_counts.items()
        }
        self.vocab = set(self.word_tags)
        self.tags = tuple(sorted(self.tag_count))
        total_tags = sum(self.tag_count.values())
        self.tag_prior = {t: c / total_tags for t, c in self.tag_count.items()}
        self.default_tag = self.tag_count.most_common(1)[0][0]

        self._build_context_tables()
        self._fit_lambdas()
        self._fit_suffix_model(word_tag_counts)
        return self

    def _build_context_tables(self) -> None:
        for n in range(1, self.order + 1):
            totals: dict[tuple[str, ...], int] = defaultdict(int)
            for gram, count in self.ngrams[n].items():
                totals[gram[:-1]] += count
            self.context_total[n - 1] = dict(totals)

    def _fit_lambdas(self) -> None:
        """Deleted interpolation (Brants 2000, TnT).

        For every observed highest-order tag n-gram, ask which order predicts
        its final tag best *with that n-gram itself deleted* -- the -1 in both
        numerator and denominator -- and give the winning order a vote weighted
        by the n-gram's frequency.  Orders that only ever win on singleton
        contexts therefore earn little weight.
        """
        votes = {n: 0.0 for n in range(1, self.order + 1)}
        unigram_total = self.context_total[0].get((), 0)

        for gram, count in self.ngrams[self.order].items():
            best_order, best_ratio = 1, -1.0
            for n in range(1, self.order + 1):
                sub = gram[self.order - n :]
                context = sub[:-1]
                if n == 1:
                    denominator = unigram_total - 1
                else:
                    denominator = self.context_total[n - 1].get(context, 0) - 1
                if denominator <= 0:
                    continue
                ratio = (self.ngrams[n].get(sub, 0) - 1) / denominator
                # ">" keeps the lowest order on ties, the conservative choice.
                if ratio > best_ratio:
                    best_order, best_ratio = n, ratio
            votes[best_order] += count

        floor = self.config.lambda_floor
        total = sum(votes.values())
        if total <= 0:
            self.lambdas = {n: 1.0 / self.order for n in votes}
            return
        weights = {n: max(v / total, floor) for n, v in votes.items()}
        scale = sum(weights.values())
        self.lambdas = {n: w / scale for n, w in weights.items()}

    def _fit_suffix_model(self, word_tag_counts: dict[str, Counter[str]]) -> None:
        """P(tag | word-final k characters), fitted on rare training words.

        Rare words are the ones whose behaviour resembles a word the tagger has
        never seen, so they -- not the whole vocabulary -- are the right sample
        for guessing an unknown word's tag.
        """
        cfg = self.config
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        base: Counter[str] = Counter()

        rare = [
            (word, tag_counts)
            for word, tag_counts in word_tag_counts.items()
            if sum(tag_counts.values()) <= cfg.rare_threshold
        ]
        # On a corpus small or repetitive enough to contain no rare words at
        # all, fall back to the whole vocabulary: a less well-matched sample
        # for the unknown-word problem, but far better than no suffix evidence.
        if not rare:
            rare = list(word_tag_counts.items())

        for word, tag_counts in rare:
            base.update(tag_counts)
            for length in range(1, min(cfg.max_suffix_len, len(word)) + 1):
                counts[word[-length:]].update(tag_counts)

        # Add-one over the rare-word tag distribution: a tag that never lands
        # on a rare word stays reachable, just very unlikely.
        n_tags = len(self.tags)
        total = sum(base.values()) + n_tags
        self.suffix_base = {t: (base.get(t, 0) + 1) / total for t in self.tags}
        self.suffix_counts = {s: c for s, c in counts.items()}

        # theta = variance of the unconditional tag distribution.  A corpus
        # whose tags are near-uniform gets a large theta and so leans on short
        # suffixes; a peaked one trusts the long suffixes more.
        if n_tags > 1:
            mean = 1.0 / n_tags
            self.theta = sum(
                (p - mean) ** 2 for p in self.suffix_base.values()
            ) / (n_tags - 1)
        else:
            self.theta = 0.0

    # ------------------------------------------------------------------
    # Transition probabilities
    # ------------------------------------------------------------------
    def transition_prob(self, tag: str, context: Sequence[str] = ()) -> float:
        """Interpolated P(tag | context), context truncated to order - 1 tags.

        Unseen contexts contribute nothing instead of contributing zero: their
        weight is redistributed over the orders that do have counts, which is
        what keeps the distribution normalised (see ``check_normalised``).
        """
        context = tuple(context)
        if self.order > 1:
            context = context[-(self.order - 1) :]
        else:
            context = ()

        numerator = 0.0
        weight_sum = 0.0
        for n in range(1, self.order + 1):
            sub_context = context[len(context) - (n - 1) :] if n > 1 else ()
            total = self.context_total[n - 1].get(sub_context, 0)
            if total == 0:
                continue
            count = self.ngrams[n].get(sub_context + (tag,), 0)
            weight = self.lambdas[n]
            numerator += weight * (count / total)
            weight_sum += weight
        if weight_sum == 0.0:
            return 0.0
        return numerator / weight_sum

    def transition_logprob(self, tag: str, context: Sequence[str] = ()) -> float:
        probability = self.transition_prob(tag, context)
        return math.log(probability) if probability > 0 else NEG_INF

    # ------------------------------------------------------------------
    # Emission probabilities
    # ------------------------------------------------------------------
    def emission_prob(self, word: str, tag: str) -> float:
        """P(word | tag) for a known word; 0 if that pairing was never seen."""
        counts = self.emissions.get(tag)
        if not counts:
            return 0.0
        count = counts.get(word, 0)
        return count / self.tag_count[tag] if count else 0.0

    def suffix_tag_dist(self, word: str) -> dict[str, float]:
        """P(tag | word) for an unknown word, by successive abstraction.

        Start from the rare-word tag prior and refine it one character at a
        time from the right: each longer suffix is mixed with the estimate from
        the shorter one, weighted by ``theta``.  Longer suffixes are more
        specific but rest on fewer counts, and this is the standard way of
        trading those off without a separate held-out set.
        """
        cached = self._suffix_cache.get(word)
        if cached is not None:
            return cached

        theta = self.theta
        distribution = dict(self.suffix_base)
        for length in range(1, min(self.config.max_suffix_len, len(word)) + 1):
            counts = self.suffix_counts.get(word[-length:])
            if counts is None:
                break                      # no longer suffix can exist either
            total = sum(counts.values())
            distribution = {
                tag: (counts.get(tag, 0) / total + theta * distribution[tag])
                / (1.0 + theta)
                for tag in self.tags
            }
        if len(self._suffix_cache) < 200_000:
            self._suffix_cache[word] = distribution
        return distribution

    def emission_logprob(self, word: str, tag: str) -> float:
        """log P(word | tag), with unknown words routed through the suffix model.

        For an unknown word the suffix model gives P(tag | word), so Bayes is
        applied in reverse: P(w|t) = P(t|w) P(w) / P(t).  P(w) is the same for
        every tag at a given position, so it is dropped -- the Viterbi argmax
        is unaffected, and only that argmax is used.
        """
        probability = self.emission_prob(word, tag)
        if probability > 0:
            return math.log(probability)
        if word in self.vocab:
            # Known word, unseen with this tag: candidate lists exclude it.
            return NEG_INF
        posterior = self.suffix_tag_dist(word).get(tag, 0.0)
        prior = self.tag_prior.get(tag, 0.0)
        if posterior <= 0 or prior <= 0:
            return NEG_INF
        return math.log(posterior) - math.log(prior)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def candidate_tags(self, word: str) -> tuple[str, ...]:
        """Tags worth considering for ``word``.

        A known word is restricted to the tags it was actually seen with -- the
        single largest speedup in the decoder, and an accuracy gain too, since
        it makes an unattested word/tag pairing impossible rather than merely
        improbable.
        """
        known = self.word_tags.get(word)
        if known:
            return known
        distribution = self.suffix_tag_dist(word)
        ranked = sorted(distribution, key=distribution.get, reverse=True)
        limit = self.config.max_unknown_tags
        return tuple(ranked[:limit] if limit else ranked)

    def tag(self, words: Sequence[str]) -> tuple[str, ...]:
        """Viterbi: the most probable tag sequence for ``words``.

        The DP state is the last ``order - 1`` tags, so column ``i`` holds one
        best path per tag history rather than one per tagging of the prefix --
        the same trick as the Part 1 segmenter, whose state is the last
        ``order - 1`` words.
        """
        n = len(words)
        if n == 0:
            return ()
        order = self.order
        state_len = max(order - 1, 1)
        beam_width = self.config.beam_width

        start = (BOS_TAG,) * state_len
        column: dict[tuple[str, ...], float] = {start: 0.0}
        back: list[dict[tuple[str, ...], tuple[str, ...]]] = []

        for word in words:
            candidates = self.candidate_tags(word)
            emissions = {t: self.emission_logprob(word, t) for t in candidates}
            emissions = {t: e for t, e in emissions.items() if e != NEG_INF}
            if not emissions:
                emissions = {self.default_tag: 0.0}

            next_column: dict[tuple[str, ...], float] = {}
            pointers: dict[tuple[str, ...], tuple[str, ...]] = {}
            for state, score in column.items():
                context = state[len(state) - (order - 1) :] if order > 1 else ()
                for tag, emission in emissions.items():
                    transition = self.transition_logprob(tag, context)
                    if transition == NEG_INF:
                        continue
                    total = score + transition + emission
                    new_state = (state + (tag,))[-state_len:]
                    if total > next_column.get(new_state, NEG_INF):
                        next_column[new_state] = total
                        pointers[new_state] = state
            if not next_column:
                # Every continuation was impossible: restart the history from
                # the best available tag rather than abandoning the sentence.
                tag = max(emissions, key=emissions.get)
                state = max(column, key=column.get)
                new_state = (state + (tag,))[-state_len:]
                next_column = {new_state: column[state] + emissions[tag]}
                pointers = {new_state: state}
            if beam_width is not None and len(next_column) > beam_width:
                kept = sorted(next_column, key=next_column.get, reverse=True)[:beam_width]
                next_column = {s: next_column[s] for s in kept}
                pointers = {s: pointers[s] for s in kept}

            column = next_column
            back.append(pointers)

        best_state = max(
            column,
            key=lambda s: column[s]
            + self.transition_logprob(
                EOS_TAG, s[len(s) - (order - 1) :] if order > 1 else ()
            ),
        )
        tags: list[str] = []
        state = best_state
        for pointers in reversed(back):
            tags.append(state[-1])
            state = pointers[state]
        tags.reverse()
        return tuple(tags)

    def tag_corpus(self, sentences: Iterable[Sequence[str]]) -> list[tuple[str, ...]]:
        return [self.tag(words) for words in sentences]

    def sequence_logprob(self, words: Sequence[str], tags: Sequence[str]) -> float:
        """log P(words, tags) -- the joint score the Viterbi search maximises."""
        history = [BOS_TAG] * max(self.order - 1, 0)
        total = 0.0
        for word, tag in zip(words, tags):
            context = tuple(history)[-(self.order - 1) :] if self.order > 1 else ()
            total += self.transition_logprob(tag, context)
            total += self.emission_logprob(word, tag)
            history = (history + [tag])[-(self.order - 1) :] if self.order > 1 else []
        context = tuple(history)[-(self.order - 1) :] if self.order > 1 else ()
        return total + self.transition_logprob(EOS_TAG, context)

    # ------------------------------------------------------------------
    # Validation / bookkeeping
    # ------------------------------------------------------------------
    def check_normalised(
        self, context: Sequence[str] = (), tolerance: float = 1e-6
    ) -> float:
        """Sum P(t | context) over every tag the model can emit; must be 1.0.

        Viterbi compares paths of equal length, so a leak here would not be as
        visible as it is in the segmenter -- but the same tagger's scores are
        combined with the word LM's downstream, and that comparison is only
        meaningful if both are proper distributions.
        """
        total = sum(
            self.transition_prob(t, context) for t in self.tags + (EOS_TAG,)
        )
        if abs(total - 1.0) > tolerance:
            raise AssertionError(
                f"transitions not normalised for context {tuple(context)!r}: "
                f"sum = {total:.9f}"
            )
        return total

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_suffix_cache"] = {}
        return state


class MostFrequentTagger:
    """Baseline: give every word the tag it carried most often in training.

    Context-free by construction, so the difference between this and the HMM is
    exactly what the transition model buys.  Unknown words take the most
    frequent tag among *rare* training words, which is the fair comparison --
    rare words are the ones an unknown word resembles.
    """

    def __init__(self, rare_threshold: int = 10) -> None:
        self.rare_threshold = rare_threshold
        self.word_tag: dict[str, str] = {}
        self.default_tag: str = "NOUN"
        self.vocab: set[str] = set()

    def fit(self, tagged_sentences: Iterable[Sequence[tuple[str, str]]]) -> "MostFrequentTagger":
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for sentence in tagged_sentences:
            for word, tag in sentence:
                counts[word][tag] += 1
        rare: Counter[str] = Counter()
        for word, tag_counts in counts.items():
            self.word_tag[word] = tag_counts.most_common(1)[0][0]
            if sum(tag_counts.values()) <= self.rare_threshold:
                rare.update(tag_counts)
        self.vocab = set(self.word_tag)
        if rare:
            self.default_tag = rare.most_common(1)[0][0]
        return self

    def tag(self, words: Sequence[str]) -> tuple[str, ...]:
        return tuple(self.word_tag.get(w, self.default_tag) for w in words)

    def tag_corpus(self, sentences: Iterable[Sequence[str]]) -> list[tuple[str, ...]]:
        return [self.tag(words) for words in sentences]


def tagged_pairs(sentences: Iterable, morph: bool = False) -> list[list[tuple[str, str]]]:
    """``Sentence`` objects -> the ``(word, tag)`` lists the taggers train on.

    ``morph=True`` supplies the Part 3 tags (``NOUN-Fem-Sg``) instead of the
    plain POS tags.  Nothing in the model changes: the tagger is agnostic about
    what a tag means, so refining the tagset is entirely a data decision, and
    the agreement patterns the brief describes are learned by the ordinary
    transition model -- P(ADJ-Fem-Sg | ..., NOUN-Fem-Sg) is simply estimated
    from more specific counts than P(ADJ | ..., NOUN).
    """
    return [
        list(zip(s.words, s.morph_tags if morph else s.tags)) for s in sentences
    ]
