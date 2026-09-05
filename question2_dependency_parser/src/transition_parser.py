"""
Transition-Based Dependency Parser

Implements the Arc-Standard transition-based parser using a trained
scikit-learn classifier to predict transitions.

Arc-Standard Transitions:
  - SHIFT: Move the first word from the buffer to the top of the stack.
  - LEFT-ARC(label): The top of the stack becomes the head of the second
    word on the stack. The second word is popped from the stack.
  - RIGHT-ARC(label): The second word on the stack becomes the head of
    the top word on the stack. The top word is popped from the stack.

The parser uses validity checking to ensure only legal transitions are
applied, with a fallback mechanism to prevent crashes.
"""

from typing import List, Tuple, Optional
import numpy as np
from conllu_parser import Sentence, Token
from features import extract_features, features_to_dict


def is_valid_transition(transition: str, stack: List[int], buffer: List[int]) -> bool:
    """Check whether a transition is valid given the current configuration.
    
    Args:
        transition: Transition string (e.g., "SHIFT", "LEFT-ARC(nsubj)").
        stack: Current stack of token IDs.
        buffer: Current buffer of token IDs.
        
    Returns:
        True if the transition can be legally applied.
    """
    if transition == "SHIFT":
        # SHIFT requires a non-empty buffer
        return len(buffer) > 0

    elif transition.startswith("LEFT-ARC"):
        # LEFT-ARC requires at least 2 items on the stack
        # and the second item cannot be ROOT (index 0)
        return len(stack) >= 2 and stack[-2] != 0

    elif transition.startswith("RIGHT-ARC"):
        # RIGHT-ARC requires at least 2 items on the stack
        return len(stack) >= 2

    return False


def apply_transition(transition: str, stack: List[int], buffer: List[int],
                     arcs: List[Tuple[int, int, str]]) -> None:
    """Apply a transition to modify the configuration in-place.
    
    Args:
        transition: Transition string to apply.
        stack: Current stack (modified in-place).
        buffer: Current buffer (modified in-place).
        arcs: Current arc set (modified in-place).
    """
    if transition == "SHIFT":
        # Move first buffer element to top of stack
        stack.append(buffer.pop(0))

    elif transition.startswith("LEFT-ARC"):
        # Extract label from "LEFT-ARC(label)"
        label = transition[9:-1]  # Remove "LEFT-ARC(" and ")"
        top = stack[-1]     # Head
        second = stack[-2]  # Dependent
        arcs.append((top, second, label))
        stack.pop(-2)  # Remove the second element (dependent)

    elif transition.startswith("RIGHT-ARC"):
        # Extract label from "RIGHT-ARC(label)"
        label = transition[10:-1]  # Remove "RIGHT-ARC(" and ")"
        second = stack[-2]  # Head
        top = stack[-1]     # Dependent
        arcs.append((second, top, label))
        stack.pop()  # Remove the top element (dependent)


def get_fallback_transition(stack: List[int], buffer: List[int]) -> Optional[str]:
    """Get a fallback transition when the predicted one is invalid.
    
    Priority:
      1. SHIFT if buffer is non-empty
      2. RIGHT-ARC(dep) if stack has >= 2 elements (generic label)
      3. None if no valid transition exists (parsing should stop)
    
    Args:
        stack: Current stack.
        buffer: Current buffer.
        
    Returns:
        A valid transition string, or None if parsing should stop.
    """
    if len(buffer) > 0:
        return "SHIFT"
    elif len(stack) >= 2:
        return "RIGHT-ARC(dep)"
    return None


def parse_sentence(words: List[str], pos_tags: List[str],
                   model, vectorizer) -> List[Tuple[int, int, str]]:
    """Parse a sentence using the trained transition-based parser.
    
    Creates a temporary Sentence object from the input words and POS tags,
    then repeatedly predicts and applies transitions until parsing is complete.
    
    Args:
        words: List of word tokens (without ROOT).
        pos_tags: List of POS tags corresponding to each word.
        model: Trained scikit-learn classifier.
        vectorizer: Fitted DictVectorizer for feature encoding.
        
    Returns:
        List of (head_id, dependent_id, label) arcs.
    """
    # Create a temporary Sentence object for feature extraction
    tokens = []
    for i, (word, pos) in enumerate(zip(words, pos_tags), start=1):
        tokens.append(Token(id=i, form=word, upos=pos, head=0, deprel=""))
    sentence = Sentence(tokens)

    # Initialize parser configuration
    n = len(words)
    stack = [0]                    # Start with ROOT
    buffer = list(range(1, n + 1)) # Token IDs 1..n
    arcs: List[Tuple[int, int, str]] = []

    # Parse loop: continue until buffer is empty and stack has at most 1 element
    max_steps = 4 * n + 10  # Safety limit to prevent infinite loops
    step = 0

    while (buffer or len(stack) > 1) and step < max_steps:
        step += 1

        # Extract features from current configuration
        feats = extract_features(stack, buffer, sentence)
        feat_dict = features_to_dict(feats)

        # Encode features and predict transition
        X = vectorizer.transform([feat_dict])
        
        # Get all class probabilities and try transitions in order of confidence
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        # Sort by probability (descending)
        sorted_indices = np.argsort(-probabilities)

        applied = False
        for idx in sorted_indices:
            candidate = classes[idx]
            if is_valid_transition(candidate, stack, buffer):
                apply_transition(candidate, stack, buffer, arcs)
                applied = True
                break

        # If no predicted transition was valid, use fallback
        if not applied:
            fallback = get_fallback_transition(stack, buffer)
            if fallback:
                apply_transition(fallback, stack, buffer, arcs)
            else:
                break  # No valid transition possible, stop parsing

    return arcs


def parse_conllu_sentence(sentence: Sentence, model, vectorizer) -> List[Tuple[int, int, str]]:
    """Parse a Sentence object (from CoNLL-U) using the trained parser.
    
    Convenience wrapper that extracts words and POS tags from a Sentence
    object and passes them to parse_sentence.
    
    Args:
        sentence: A Sentence object (POS tags are used, gold heads are ignored).
        model: Trained classifier.
        vectorizer: Fitted DictVectorizer.
        
    Returns:
        List of (head_id, dependent_id, label) arcs.
    """
    words = [t.form for t in sentence.tokens[1:]]    # Skip ROOT
    pos_tags = [t.upos for t in sentence.tokens[1:]]  # Skip ROOT
    return parse_sentence(words, pos_tags, model, vectorizer)
