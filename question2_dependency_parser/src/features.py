"""
Feature Extraction Module

Extracts features from a parser configuration (stack, buffer) for the
transition-based dependency parser.

Required Features (from the assignment):
  1. POS tag of the word on top of the stack
  2. POS tag of the second word on the stack (if available)
  3. POS tag of the first word in the buffer (if available)
  4. POS tag of the second word in the buffer (if available)

Missing positions are represented by the special value "NONE".
"""

from typing import List
from conllu_parser import Sentence

# Sentinel value for missing stack/buffer positions
NONE_POS = "NONE"


def extract_features(stack: List[int], buffer: List[int],
                     sentence: Sentence) -> List[str]:
    """Extract 4 POS-tag features from the current parser configuration.
    
    Args:
        stack: Current stack of token IDs (0 = ROOT at bottom).
        buffer: Current buffer of token IDs (remaining words).
        sentence: The Sentence being parsed (provides POS tags).
        
    Returns:
        List of 4 strings: [stack_top_pos, stack_second_pos,
                            buffer_first_pos, buffer_second_pos]
    """
    # Feature 1: POS tag of top of stack (stack[-1])
    if len(stack) >= 1:
        token = sentence.get_token(stack[-1])
        stack_top_pos = token.upos if token else NONE_POS
    else:
        stack_top_pos = NONE_POS

    # Feature 2: POS tag of second on stack (stack[-2])
    if len(stack) >= 2:
        token = sentence.get_token(stack[-2])
        stack_second_pos = token.upos if token else NONE_POS
    else:
        stack_second_pos = NONE_POS

    # Feature 3: POS tag of first in buffer (buffer[0])
    if len(buffer) >= 1:
        token = sentence.get_token(buffer[0])
        buffer_first_pos = token.upos if token else NONE_POS
    else:
        buffer_first_pos = NONE_POS

    # Feature 4: POS tag of second in buffer (buffer[1])
    if len(buffer) >= 2:
        token = sentence.get_token(buffer[1])
        buffer_second_pos = token.upos if token else NONE_POS
    else:
        buffer_second_pos = NONE_POS

    return [stack_top_pos, stack_second_pos, buffer_first_pos, buffer_second_pos]


def features_to_dict(features: List[str]) -> dict:
    """Convert feature list to a dictionary for DictVectorizer.
    
    Args:
        features: List of 4 POS-tag strings.
        
    Returns:
        Dictionary with named feature keys.
    """
    return {
        "stack_top_pos": features[0],
        "stack_second_pos": features[1],
        "buffer_first_pos": features[2],
        "buffer_second_pos": features[3],
    }
