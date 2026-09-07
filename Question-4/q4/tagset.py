"""
tagset.py - POS Tagset Reconciliation

Maps Q1's 12-class Universal POS tags to Penn Treebank (PTB) tags required by
the PCFG parser.
"""

# Map each Universal tag to the most frequent Penn Treebank equivalent
UNIVERSAL_TO_PTB = {
    "NOUN": "NN",
    "VERB": "VBD",
    "ADJ": "JJ",
    "ADV": "RB",
    "PRON": "PRP",
    "DET": "DT",
    "ADP": "IN",
    "CONJ": "CC",
    "PRT": "RP",
    "NUM": "CD",
    "X": "FW",
    ".": ".",
}


def universal_to_ptb(tag: str) -> str:
    """Convert a Universal POS tag to a Penn Treebank tag."""
    return UNIVERSAL_TO_PTB.get(tag, "NN")


def reconcile_tag_sequence(universal_tags: tuple[str, ...]) -> tuple[str, ...]:
    """Map a sequence of Universal POS tags to Penn Treebank tags for parsing."""
    return tuple(universal_to_ptb(t) for t in universal_tags)
