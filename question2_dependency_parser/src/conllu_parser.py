"""
CoNLL-U Parser Module

Reads CoNLL-U format files from the Universal Dependencies project.
Parses each sentence into a list of Token objects containing:
  - word/token form
  - POS tag (UPOS)
  - gold-standard head
  - dependency relation/label

Handles:
  - Comment lines (starting with #)
  - Multi-word tokens (range IDs like 1-2) — skipped
  - Empty nodes (IDs with '.') — skipped
  - Blank lines as sentence boundaries
  - Virtual ROOT token at index 0
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    """Represents a single token/word in a sentence."""
    id: int          # Token index (1-based; 0 for ROOT)
    form: str        # Word form / token text
    upos: str        # Universal POS tag
    head: int        # Gold-standard head token ID (0 = root of sentence)
    deprel: str      # Dependency relation label (e.g., nsubj, obj, root)


class Sentence:
    """Represents a parsed sentence with its tokens.
    
    Tokens are stored 1-indexed, with a virtual ROOT token at index 0.
    """

    def __init__(self, tokens: List[Token]):
        """Initialize sentence with a list of tokens (without ROOT).
        
        Args:
            tokens: List of Token objects (1-indexed, no ROOT).
        """
        # Add virtual ROOT token at index 0
        root_token = Token(id=0, form="ROOT", upos="ROOT", head=-1, deprel="ROOT")
        self.tokens = [root_token] + tokens

    def __len__(self):
        """Returns the number of tokens excluding ROOT."""
        return len(self.tokens) - 1

    def __repr__(self):
        words = [t.form for t in self.tokens[1:]]  # Exclude ROOT
        return f"Sentence({' '.join(words)})"

    def get_token(self, idx: int) -> Optional[Token]:
        """Safely get a token by its index.
        
        Args:
            idx: Token index (0 for ROOT, 1..n for words).
            
        Returns:
            Token object or None if index is out of range.
        """
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None


def parse_conllu(filepath: str) -> List[Sentence]:
    """Parse a CoNLL-U format file and return a list of Sentence objects.
    
    Each sentence in the CoNLL-U file is separated by blank lines.
    Lines starting with '#' are comments and are skipped.
    Multi-word tokens (range IDs) and empty nodes (decimal IDs) are skipped.
    
    CoNLL-U columns (tab-separated):
        0: ID, 1: FORM, 2: LEMMA, 3: UPOS, 4: XPOS,
        5: FEATS, 6: HEAD, 7: DEPREL, 8: DEPS, 9: MISC
    
    Args:
        filepath: Path to the .conllu file.
        
    Returns:
        List of Sentence objects.
    """
    sentences = []
    current_tokens = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Blank line = sentence boundary
            if not line:
                if current_tokens:
                    sentences.append(Sentence(current_tokens))
                    current_tokens = []
                continue

            # Skip comment lines
            if line.startswith('#'):
                continue

            # Split by tab to get CoNLL-U columns
            columns = line.split('\t')
            if len(columns) < 10:
                continue  # Malformed line, skip

            token_id = columns[0]

            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            if '-' in token_id or '.' in token_id:
                continue

            try:
                tid = int(token_id)
                form = columns[1]
                upos = columns[3]
                head = int(columns[6])
                deprel = columns[7]

                token = Token(id=tid, form=form, upos=upos, head=head, deprel=deprel)
                current_tokens.append(token)
            except (ValueError, IndexError):
                # Skip malformed token lines
                continue

    # Handle last sentence if file doesn't end with blank line
    if current_tokens:
        sentences.append(Sentence(current_tokens))

    return sentences


if __name__ == "__main__":
    # Quick test: parse the training file and print stats
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_file = os.path.join(data_dir, "en_ewt-ud-train.conllu")

    if os.path.exists(train_file):
        sentences = parse_conllu(train_file)
        print(f"Parsed {len(sentences)} sentences from training data.")
        if sentences:
            s = sentences[0]
            print(f"\nFirst sentence: {s}")
            print(f"Number of tokens: {len(s)}")
            for t in s.tokens:
                print(f"  {t.id}\t{t.form}\t{t.upos}\t{t.head}\t{t.deprel}")
    else:
        print(f"Training file not found: {train_file}")
