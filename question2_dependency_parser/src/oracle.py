"""
Oracle Simulator Module

Implements the Arc-Standard oracle for generating training data from
gold-standard dependency trees.

Arc-Standard Transition System:
  - SHIFT: Move the first word from the buffer to the top of the stack.
  - LEFT-ARC(label): The word at the top of the stack becomes the head of 
    the second word on the stack. The second word is then popped.
  - RIGHT-ARC(label): The second word on the stack becomes the head of the 
    word at the top of the stack. The top word is then popped.

The oracle determines the correct transition at each configuration by
checking the gold-standard head assignments.

A word can only be reduced (via LEFT-ARC or RIGHT-ARC) when all of its 
dependents have already been collected in the arc set.
"""

from typing import List, Tuple, Set, Dict
from conllu_parser import Sentence, Token
from features import extract_features


def get_dependents(sentence: Sentence) -> Dict[int, Set[int]]:
    """Build a mapping from each token ID to its set of dependent IDs.
    
    Args:
        sentence: A parsed Sentence object with gold-standard annotations.
        
    Returns:
        Dictionary mapping token ID -> set of dependent token IDs.
    """
    deps = {}
    for token in sentence.tokens:
        deps[token.id] = set()

    for token in sentence.tokens[1:]:  # Skip ROOT
        head_id = token.head
        if head_id in deps:
            deps[head_id].add(token.id)

    return deps


def has_all_dependents(token_id: int, gold_deps: Dict[int, Set[int]],
                       collected_arcs: Set[Tuple[int, int, str]]) -> bool:
    """Check whether all gold dependents of a token have been collected.
    
    A token can only be reduced (removed from the stack) after all its
    dependents have been attached to it in the arc set.
    
    Args:
        token_id: The token to check.
        gold_deps: Gold-standard dependent mapping.
        collected_arcs: Set of arcs already collected (head, dep, label).
        
    Returns:
        True if all dependents of token_id are in collected_arcs.
    """
    needed = gold_deps.get(token_id, set())
    collected_dep_ids = {dep for head, dep, _ in collected_arcs if head == token_id}
    return needed.issubset(collected_dep_ids)


def simulate_oracle(sentence: Sentence) -> List[Tuple[List[str], str]]:
    """Run the Arc-Standard oracle on a gold-standard sentence.
    
    Simulates the parsing process using the gold tree and generates
    training instances as (features, transition_label) pairs.
    
    Args:
        sentence: A Sentence object with gold-standard head/deprel annotations.
        
    Returns:
        List of (features, transition) tuples, where:
          - features: list of 4 POS-tag strings
          - transition: string like "SHIFT", "LEFT-ARC(nsubj)", "RIGHT-ARC(obj)"
    """
    # Skip empty sentences
    if len(sentence) == 0:
        return []

    # Build gold dependent mapping
    gold_deps = get_dependents(sentence)

    # Initial configuration
    stack = [0]  # Start with ROOT on the stack
    buffer = list(range(1, len(sentence) + 1))  # Token IDs 1..n
    arcs: Set[Tuple[int, int, str]] = set()  # (head_id, dep_id, label)
    training_instances = []

    while buffer or len(stack) > 1:
        # Extract features for the current configuration
        feats = extract_features(stack, buffer, sentence)

        # Try LEFT-ARC: stack[-2] has head == stack[-1]
        if len(stack) >= 2:
            top = stack[-1]      # Top of stack
            second = stack[-2]   # Second on stack

            # LEFT-ARC: top becomes head of second; second is popped
            # Cannot LEFT-ARC the ROOT (second == 0)
            token_second = sentence.get_token(second)
            if (second != 0 and token_second is not None
                    and token_second.head == top
                    and has_all_dependents(second, gold_deps, arcs)):
                label = token_second.deprel
                transition = f"LEFT-ARC({label})"
                training_instances.append((feats, transition))
                # Apply: add arc and remove second from stack
                arcs.add((top, second, label))
                stack.pop(-2)  # Remove second element
                continue

            # RIGHT-ARC: second becomes head of top; top is popped
            token_top = sentence.get_token(top)
            if (token_top is not None
                    and token_top.head == second
                    and has_all_dependents(top, gold_deps, arcs)):
                label = token_top.deprel
                transition = f"RIGHT-ARC({label})"
                training_instances.append((feats, transition))
                # Apply: add arc and remove top from stack
                arcs.add((second, top, label))
                stack.pop()
                continue

        # SHIFT: move first buffer element to stack
        if buffer:
            transition = "SHIFT"
            training_instances.append((feats, transition))
            stack.append(buffer.pop(0))
        else:
            # Fallback: if buffer is empty and no arc was possible,
            # force a RIGHT-ARC to avoid infinite loop
            if len(stack) >= 2:
                top = stack[-1]
                second = stack[-2]
                token_top = sentence.get_token(top)
                label = token_top.deprel if token_top else "dep"
                transition = f"RIGHT-ARC({label})"
                training_instances.append((feats, transition))
                arcs.add((second, top, label))
                stack.pop()
            else:
                break  # Nothing left to do

    return training_instances


def generate_training_data(sentences: List[Sentence]) -> Tuple[List[List[str]], List[str]]:
    """Generate training features and labels from a list of gold sentences.
    
    Runs the oracle on each sentence and collects all training instances.
    
    Args:
        sentences: List of gold-standard Sentence objects.
        
    Returns:
        Tuple of (all_features, all_labels) where:
          - all_features: list of feature lists (each has 4 POS strings)
          - all_labels: list of transition label strings
    """
    all_features = []
    all_labels = []

    for i, sentence in enumerate(sentences):
        instances = simulate_oracle(sentence)
        for feats, label in instances:
            all_features.append(feats)
            all_labels.append(label)

        # Progress reporting every 2000 sentences
        if (i + 1) % 2000 == 0:
            print(f"  Oracle processed {i + 1}/{len(sentences)} sentences "
                  f"({len(all_features)} instances so far)")

    return all_features, all_labels


if __name__ == "__main__":
    # Quick test: run oracle on first few sentences
    import os
    from conllu_parser import parse_conllu

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_file = os.path.join(data_dir, "en_ewt-ud-train.conllu")

    sentences = parse_conllu(train_file)
    print(f"Loaded {len(sentences)} sentences.")

    # Test oracle on first sentence
    if sentences:
        s = sentences[0]
        print(f"\nSentence: {s}")
        instances = simulate_oracle(s)
        print(f"Generated {len(instances)} training instances:")
        for feats, label in instances:
            print(f"  Features: {feats} -> {label}")
