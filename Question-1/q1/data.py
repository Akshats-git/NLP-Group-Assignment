from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

# Default 80/10/10 split -- 80/20 train/held-out as instructed, with the
# held-out half split into a tuning set and an untouched final test set.
DEFAULT_SPLIT = (0.8, 0.1, 0.1)
DEFAULT_SEED = 42

NonAlphaPolicy = Literal["strip", "drop"]


# --------------------------------------------------------------------------
# Brown -> Penn Treebank
# --------------------------------------------------------------------------
# Brown tags carry optional suffixes (-HL headline, -TL title, -NC cited) and
# may be joined with '+' for contractions ("isn't" -> BEZ*).  normalise_brown_tag
# strips those before the table below is consulted.  Part 1 never reads a tag --
# the segmenter is purely lexical -- but the loader still offers the three
# tagsets so the corpora are described faithfully.
BROWN_TO_PENN: dict[str, str] = {
    # determiners / articles
    "AT": "DT", "DT": "DT", "DTI": "DT", "DTS": "DT", "DTX": "DT",
    "ABL": "PDT", "ABN": "PDT", "ABX": "PDT",
    "AP": "JJ",
    # conjunctions
    "CC": "CC", "CS": "IN",
    # numerals
    "CD": "CD", "OD": "JJ",
    # "be"
    "BE": "VB", "BED": "VBD", "BEDZ": "VBD", "BEG": "VBG",
    "BEM": "VBP", "BEN": "VBN", "BER": "VBP", "BEZ": "VBZ",
    # "do"
    "DO": "VB", "DOD": "VBD", "DOZ": "VBZ",
    # "have"
    "HV": "VB", "HVD": "VBD", "HVG": "VBG", "HVN": "VBN", "HVZ": "VBZ",
    # existential / foreign / interjection
    "EX": "EX", "FW": "FW", "UH": "UH",
    # prepositions, modals, particles
    "IN": "IN", "MD": "MD", "RP": "RP", "TO": "TO",
    # adjectives
    "JJ": "JJ", "JJR": "JJR", "JJS": "JJS", "JJT": "JJS",
    # nouns
    "NN": "NN", "NN$": "NN", "NNS": "NNS", "NNS$": "NNS",
    "NP": "NNP", "NP$": "NNP", "NPS": "NNPS", "NPS$": "NNPS",
    "NR": "NN", "NR$": "NN", "NRS": "NNS", "NC": "NN",
    # pronouns
    "PN": "NN", "PN$": "NN",
    "PP$": "PRP$", "PP$$": "PRP",
    "PPL": "PRP", "PPLS": "PRP", "PPO": "PRP", "PPS": "PRP", "PPSS": "PRP",
    # adverbs / qualifiers
    "QL": "RB", "QLP": "RB",
    "RB": "RB", "RBR": "RBR", "RBT": "RBS", "RN": "RB",
    "*": "RB",  # Brown's negator ("not")
    # verbs
    "VB": "VB", "VBD": "VBD", "VBG": "VBG", "VBN": "VBN",
    "VBP": "VBP", "VBZ": "VBZ",
    # wh-words
    "WDT": "WDT", "WP$": "WP$", "WPO": "WP", "WPS": "WP",
    "WQL": "WRB", "WRB": "WRB",
    # punctuation (retained for completeness; normalisation usually drops these)
    ".": ".", ",": ",", ":": ":", "(": "-LRB-", ")": "-RRB-",
    "``": "``", "''": "''", "--": ":",
    "NIL": "X",
}

#: Brown suffixes that mark a usage, not a different part of speech.
_BROWN_SUFFIXES = ("-HL", "-TL", "-NC", "-TL-HL", "-NC-TL")


def normalise_brown_tag(tag: str) -> str:
    """Strip Brown's usage suffixes and contraction joins.

    ``NN-TL`` -> ``NN``, ``BEZ*`` -> ``BEZ``, ``PPSS+BER`` -> ``PPSS``.
    The genuine tags ``NN$``/``PP$``/``*`` keep their punctuation, so ``$`` is
    never stripped and a lone ``*`` is passed through unchanged.
    """
    tag = tag.upper().strip()
    if not tag:
        return "X"
    for suffix in _BROWN_SUFFIXES:
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    # Contractions ("isn't" -> BEZ*, "I'm" -> PPSS+BEM): keep the first tag.
    if "+" in tag:
        tag = tag.split("+", 1)[0]
    # Trailing negation marker, but not the standalone negator tag "*".
    if len(tag) > 1 and tag.endswith("*"):
        tag = tag[:-1]
    return tag or "X"


def brown_to_penn(tag: str) -> str:
    """Map one Brown tag to its closest Penn Treebank equivalent."""
    return BROWN_TO_PENN.get(normalise_brown_tag(tag), "X")


# --------------------------------------------------------------------------
# Morphology (Part 3)
# --------------------------------------------------------------------------
# A morphology-aware tag is the POS tag with the agreement-carrying features
# appended, in the notation the brief uses: NOUN-Fem-Sg, ADJ-Masc-Pl.  Only
# gender and number are included by default -- they are the features that
# actually participate in agreement between adjacent words, and every extra
# feature multiplies the tagset (and so divides the counts).
MORPH_FEATURES: tuple[str, ...] = ("Gender", "Number")

#: UD feature value -> the short form used inside a tag.
MORPH_ABBREVIATIONS: dict[str, dict[str, str]] = {
    "Gender": {"Fem": "Fem", "Masc": "Masc", "Neut": "Neut", "Com": "Com"},
    "Number": {"Sing": "Sg", "Plur": "Pl", "Dual": "Du", "Ptan": "Pl", "Coll": "Sg"},
    "Person": {"1": "1", "2": "2", "3": "3"},
    "Case": {"Nom": "Nom", "Acc": "Acc", "Dat": "Dat", "Gen": "Gen"},
}

#: The inverse: short form -> which feature it belongs to.  Used to read the
#: features back out of a predicted tag, so evaluation never has to consult the
#: gold token to find out what the model claimed.
MORPH_VALUES: dict[str, str] = {
    short: feature
    for feature, table in MORPH_ABBREVIATIONS.items()
    for short in table.values()
}


def parse_feats(field: str) -> tuple[tuple[str, str], ...]:
    """CoNLL-U FEATS -> a sorted, hashable tuple of ``(feature, value)``.

    ``_`` and empty fields give ``()``.  Ambiguous values (``Gender=Fem,Masc``)
    keep the first alternative -- the tagset has to be a partition, and the
    first listed value is UD's convention for the more likely reading.
    """
    if not field or field == "_":
        return ()
    pairs: list[tuple[str, str]] = []
    for item in field.split("|"):
        name, separator, value = item.partition("=")
        if not separator:
            continue
        pairs.append((name.strip(), value.split(",")[0].strip()))
    return tuple(sorted(pairs))


def make_morph_tag(
    upos: str,
    feats: Sequence[tuple[str, str]],
    features: Sequence[str] = MORPH_FEATURES,
) -> str:
    """``("NOUN", (("Gender","Fem"),("Number","Sing")))`` -> ``"NOUN-Fem-Sg"``.

    A token with none of the requested features keeps its bare POS tag, so a
    corpus without FEATS (Brown) yields a morphology-aware tagset identical to
    its plain one -- the null result, made explicit rather than crashing.
    """
    lookup = dict(feats)
    parts = [upos]
    for feature in features:
        value = lookup.get(feature)
        if not value:
            continue
        abbreviation = MORPH_ABBREVIATIONS.get(feature, {}).get(value)
        if abbreviation:
            parts.append(abbreviation)
    return "-".join(parts)


def coarse_tag(tag: str) -> str:
    """``"NOUN-Fem-Sg"`` -> ``"NOUN"``.

    Projecting the morphology-aware tagger's output back to coarse tags is what
    makes it comparable with the plain tagger: both are then scored on the same
    decision.  Penn's bracket tags (``-LRB-``) start with the separator and are
    returned unchanged.
    """
    if tag.startswith("-"):
        return tag
    return tag.split("-", 1)[0]


def tag_features(tag: str) -> dict[str, str]:
    """``"NOUN-Fem-Sg"`` -> ``{"Gender": "Fem", "Number": "Sg"}``."""
    features: dict[str, str] = {}
    if tag.startswith("-"):
        return features
    for part in tag.split("-")[1:]:
        feature = MORPH_VALUES.get(part)
        if feature:
            features[feature] = part
    return features


# --------------------------------------------------------------------------
# Core types
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Token:
    """One word of a sentence, after normalisation."""

    form: str                      # lowercased, alphabetic surface form
    upos: str = ""                 # POS tag; carried through, unused by Part 1
    feats: tuple[tuple[str, str], ...] = ()   # UD FEATS, sorted; () if none

    def feature(self, name: str) -> str | None:
        """The raw UD value of one feature, e.g. ``Gender`` -> ``"Fem"``."""
        for feature, value in self.feats:
            if feature == name:
                return value
        return None

    @property
    def morph_tag(self) -> str:
        """The Part 3 tag: POS plus gender/number, e.g. ``NOUN-Fem-Sg``."""
        return make_morph_tag(self.upos, self.feats)

    def __len__(self) -> int:
        return len(self.form)


@dataclass(frozen=True, slots=True)
class Sentence:
    """A normalised sentence plus everything the segmenter needs.

    ``chars`` is the concatenation of the token forms with no spaces -- the
    input to the segmentation model.  ``gold_spans[i]`` is the (start, end)
    half-open character range of ``tokens[i]`` inside ``chars``.
    """

    tokens: tuple[Token, ...]
    chars: str
    gold_spans: tuple[tuple[int, int], ...]
    sent_id: str = ""

    @classmethod
    def from_tokens(cls, tokens: Sequence[Token], sent_id: str = "") -> "Sentence":
        spans: list[tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            spans.append((cursor, cursor + len(token.form)))
            cursor += len(token.form)
        return cls(
            tokens=tuple(tokens),
            chars="".join(t.form for t in tokens),
            gold_spans=tuple(spans),
            sent_id=sent_id,
        )

    # -- convenience views -------------------------------------------------
    @property
    def words(self) -> tuple[str, ...]:
        return tuple(t.form for t in self.tokens)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(t.upos for t in self.tokens)

    @property
    def morph_tags(self) -> tuple[str, ...]:
        """Part 3 tags -- identical to ``tags`` on a corpus without FEATS."""
        return tuple(t.morph_tag for t in self.tokens)

    @property
    def has_morphology(self) -> bool:
        return any(t.feats for t in self.tokens)

    @property
    def text(self) -> str:
        """Space-separated reference rendering (what the model must recover)."""
        return " ".join(self.words)

    @property
    def gold_boundaries(self) -> frozenset[int]:
        """Internal boundary offsets, i.e. every split point except 0 and len.

        Segmentation P/R/F1 is computed over these sets.
        """
        return frozenset(end for _, end in self.gold_spans[:-1])

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True, slots=True)
class Corpus:
    """A named corpus already divided into the three splits."""

    name: str
    language: str
    tagset: str
    train: tuple[Sentence, ...]
    dev: tuple[Sentence, ...]
    test: tuple[Sentence, ...]

    def split(self, which: str) -> tuple[Sentence, ...]:
        return {"train": self.train, "dev": self.dev, "test": self.test}[which]

    @property
    def has_morphology(self) -> bool:
        """Whether the annotation carries FEATS at all.

        False for Brown, which is why the Part 3 comparison is a null result in
        English: with no gender or number to attach, the morphology-aware
        tagset is *identical* to the plain one.  Checked on a prefix of train,
        since the answer is a property of the annotation scheme.
        """
        return any(s.has_morphology for s in self.train[:1000])

    def __iter__(self) -> Iterator[Sentence]:
        yield from self.train
        yield from self.dev
        yield from self.test


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def normalise_form(form: str, policy: NonAlphaPolicy = "strip") -> str:
    """Lowercase, NFC-normalise and remove non-letters from a surface form.

    Returns "" for tokens that should be dropped entirely (punctuation,
    numbers, or any token whose letters are all stripped away).
    """
    form = unicodedata.normalize("NFC", form).lower()
    if policy == "drop":
        return form if form.isalpha() else ""
    return "".join(ch for ch in form if ch.isalpha())


#: Tokens longer than this are discarded -- see the module docstring.
DEFAULT_MAX_TOKEN_LEN = 25

#: POS tags whose tokens are never real words.
DEFAULT_DROP_UPOS: frozenset[str] = frozenset({"SYM"})


# --------------------------------------------------------------------------
# Statistics (used by the data-handling section of the report)
# --------------------------------------------------------------------------
@dataclass
class CorpusStats:
    n_sentences_raw: int = 0
    n_sentences_kept: int = 0
    n_tokens_raw: int = 0
    n_tokens_kept: int = 0
    n_tokens_altered: int = 0     # non-alpha characters stripped from inside
    n_tokens_dropped: int = 0     # emptied by normalisation
    n_mwt_expanded: int = 0       # UD range lines expanded into syntactic words
    n_tokens_too_long: int = 0    # discarded by max_token_len (flattened URLs)
    n_tokens_dropped_upos: int = 0  # discarded by drop_upos (symbols)

    def as_rows(self) -> list[tuple[str, str]]:
        kept_sent = f"{self.n_sentences_kept:,} / {self.n_sentences_raw:,}"
        kept_tok = f"{self.n_tokens_kept:,} / {self.n_tokens_raw:,}"
        return [
            ("sentences kept", kept_sent),
            ("tokens kept", kept_tok),
            ("tokens altered by normalisation", f"{self.n_tokens_altered:,}"),
            ("tokens dropped by normalisation", f"{self.n_tokens_dropped:,}"),
            ("tokens dropped: over length limit", f"{self.n_tokens_too_long:,}"),
            ("tokens dropped: filtered POS", f"{self.n_tokens_dropped_upos:,}"),
            ("multi-word tokens expanded", f"{self.n_mwt_expanded:,}"),
        ]


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(
    groups: dict[str, list[Sentence]],
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    """Shuffle within each group and split each group by the same fractions.

    Splitting per group (Brown genre) rather than over the pooled corpus keeps
    the genre mix identical across train/dev/test, so the held-out score is not
    measuring domain shift.
    """
    rng = random.Random(seed)
    train: list[Sentence] = []
    dev: list[Sentence] = []
    test: list[Sentence] = []
    train_frac, dev_frac, _ = fractions
    for _, sentences in sorted(groups.items()):
        pool = list(sentences)
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(n * train_frac)
        n_dev = int(n * (train_frac + dev_frac)) - n_train
        train.extend(pool[:n_train])
        dev.extend(pool[n_train : n_train + n_dev])
        test.extend(pool[n_train + n_dev :])
    # A final shuffle so that iteration order is not grouped by genre.
    for part in (train, dev, test):
        rng.shuffle(part)
    return train, dev, test


# --------------------------------------------------------------------------
# English: Brown corpus
# --------------------------------------------------------------------------
def load_brown(
    tagset: Literal["universal", "penn", "brown"] = "universal",
    fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    policy: NonAlphaPolicy = "strip",
    max_token_len: int | None = DEFAULT_MAX_TOKEN_LEN,
    drop_upos: frozenset[str] = DEFAULT_DROP_UPOS,
    max_sentences: int | None = None,
    return_stats: bool = False,
) -> Corpus | tuple[Corpus, CorpusStats]:
    """Load the Brown corpus, genre-stratified into train/dev/test.

    ``tagset='universal'`` gives the 12-tag set used for the English/Spanish
    comparison; ``'penn'`` maps Brown's native tags to Penn Treebank tags (the
    tags the assignment's English example prints, and the ones Q4's PCFG
    expects); ``'brown'`` leaves the native tags untouched.
    """
    from nltk.corpus import brown as brown_corpus

    stats = CorpusStats()
    groups: dict[str, list[Sentence]] = {}
    nltk_tagset = "universal" if tagset == "universal" else None

    for category in brown_corpus.categories():
        bucket: list[Sentence] = []
        tagged = brown_corpus.tagged_sents(categories=category, tagset=nltk_tagset)
        for index, raw_sentence in enumerate(tagged):
            stats.n_sentences_raw += 1
            tokens: list[Token] = []
            for word, tag in raw_sentence:
                stats.n_tokens_raw += 1
                form = normalise_form(word, policy)
                if not form:
                    stats.n_tokens_dropped += 1
                    continue
                if max_token_len and len(form) > max_token_len:
                    stats.n_tokens_too_long += 1
                    continue
                if form != word.lower():
                    stats.n_tokens_altered += 1
                upos = brown_to_penn(tag) if tagset == "penn" else tag
                if upos in drop_upos:
                    stats.n_tokens_dropped_upos += 1
                    continue
                tokens.append(Token(form=form, upos=upos))
            if not tokens:
                continue
            stats.n_sentences_kept += 1
            stats.n_tokens_kept += len(tokens)
            bucket.append(
                Sentence.from_tokens(tokens, sent_id=f"brown-{category}-{index}")
            )
            if max_sentences and stats.n_sentences_kept >= max_sentences:
                break
        groups[category] = bucket
        if max_sentences and stats.n_sentences_kept >= max_sentences:
            break

    train, dev, test = stratified_split(groups, fractions, seed)
    corpus = Corpus(
        name="Brown",
        language="English",
        tagset=tagset,
        train=tuple(train),
        dev=tuple(dev),
        test=tuple(test),
    )
    return (corpus, stats) if return_stats else corpus


# --------------------------------------------------------------------------
# Spanish / German: Universal Dependencies .conllu
# --------------------------------------------------------------------------
def iter_conllu(path: Path) -> Iterator[list[list[str]]]:
    """Yield one sentence at a time as a list of CoNLL-U column lists."""
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    yield rows
                    rows = []
                continue
            if line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    if rows:
        yield rows


def _sentences_from_conllu(
    path: Path,
    stats: CorpusStats,
    policy: NonAlphaPolicy,
    max_token_len: int | None,
    drop_upos: frozenset[str],
    max_sentences: int | None = None,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for index, rows in enumerate(iter_conllu(path)):
        stats.n_sentences_raw += 1
        tokens: list[Token] = []
        for row in rows:
            token_id = row[0]
            if "-" in token_id:
                # Range line: the surface multi-word token ("del").  Skipped --
                # the syntactic words that follow carry the POS and FEATS.
                stats.n_mwt_expanded += 1
                continue
            if "." in token_id:
                # Empty node from an enhanced dependency graph; not a real word.
                continue
            stats.n_tokens_raw += 1
            word, upos = row[1], row[3]
            feats = parse_feats(row[5]) if len(row) > 5 else ()
            if upos in drop_upos:
                stats.n_tokens_dropped_upos += 1
                continue
            form = normalise_form(word, policy)
            if not form:
                stats.n_tokens_dropped += 1
                continue
            if max_token_len and len(form) > max_token_len:
                stats.n_tokens_too_long += 1
                continue
            if form != word.lower():
                stats.n_tokens_altered += 1
            tokens.append(Token(form=form, upos=upos, feats=feats))
        if not tokens:
            continue
        stats.n_sentences_kept += 1
        stats.n_tokens_kept += len(tokens)
        sentences.append(
            Sentence.from_tokens(tokens, sent_id=f"{path.stem}-{index}")
        )
        if max_sentences and len(sentences) >= max_sentences:
            break
    return sentences


def load_ud(
    directory: str | Path,
    name: str,
    language: str,
    policy: NonAlphaPolicy = "strip",
    max_token_len: int | None = DEFAULT_MAX_TOKEN_LEN,
    drop_upos: frozenset[str] = DEFAULT_DROP_UPOS,
    max_sentences: int | None = None,
    return_stats: bool = False,
) -> Corpus | tuple[Corpus, CorpusStats]:
    """Load a Universal Dependencies treebank using its own train/dev/test files."""
    directory = Path(directory)
    stats = CorpusStats()
    splits: dict[str, list[Sentence]] = {}
    for split_name in ("train", "dev", "test"):
        matches = sorted(directory.glob(f"*-ud-{split_name}.conllu"))
        if not matches:
            raise FileNotFoundError(
                f"No *-ud-{split_name}.conllu in {directory}. "
                "Clone the treebank first (see README)."
            )
        splits[split_name] = _sentences_from_conllu(
            matches[0], stats, policy, max_token_len, drop_upos, max_sentences,
        )
    corpus = Corpus(
        name=name,
        language=language,
        tagset="upos",
        train=tuple(splits["train"]),
        dev=tuple(splits["dev"]),
        test=tuple(splits["test"]),
    )
    return (corpus, stats) if return_stats else corpus


def load_spanish(
    directory: str | Path = "data/UD_Spanish-GSD", **kwargs
) -> Corpus | tuple[Corpus, CorpusStats]:
    return load_ud(directory, name="UD Spanish-GSD", language="Spanish", **kwargs)


def load_german(
    directory: str | Path = "data/UD_German-GSD", **kwargs
) -> Corpus | tuple[Corpus, CorpusStats]:
    return load_ud(directory, name="UD German-GSD", language="German", **kwargs)


# --------------------------------------------------------------------------
# Small helpers shared by the models
# --------------------------------------------------------------------------
def vocabulary(sentences: Iterable[Sentence]) -> set[str]:
    """The set of word forms occurring in the given sentences."""
    return {token.form for sentence in sentences for token in sentence.tokens}


def max_word_length(sentences: Iterable[Sentence], percentile: float = 1.0) -> int:
    """Longest word form, or the given percentile of the length distribution.

    The decoder's inner loop is O(max_word_length) per character, so this is
    the knob that trades decoding speed against the ability to recover very
    long words (German compounds in particular).
    """
    lengths = sorted(
        len(token.form) for sentence in sentences for token in sentence.tokens
    )
    if not lengths:
        return 0
    if percentile >= 1.0:
        return lengths[-1]
    return lengths[min(len(lengths) - 1, int(len(lengths) * percentile))]
