# Report: Transition-Based Dependency Parser

## 1. Objective

The goal of this project is to build a simple, data-driven dependency parser from scratch. The parser uses a transition-based approach with the Arc-Standard transition system. A scikit-learn classifier, trained on a treebank, is used to predict parsing transitions.

## 2. Dataset

The **Universal Dependencies English-EWT** (English Web Treebank) corpus is used: `en_ewt-ud-train.conllu` for training (12,544 sentences) and `en_ewt-ud-dev.conllu` for evaluation (2,001 sentences).

The dataset was obtained by cloning the official repository:
`https://github.com/UniversalDependencies/UD_English-EWT.git`

## 3. CoNLL-U Data Representation

Each sentence in the CoNLL-U file is parsed into a `Sentence` object containing a list of `Token` objects. Each `Token` stores:

| Field    | Description                              |
|----------|------------------------------------------|
| `id`     | Token index (1-based; 0 for ROOT)        |
| `form`   | Word form / token text                   |
| `upos`   | Universal POS tag                        |
| `head`   | Gold-standard head token ID (0 = root)   |
| `deprel`  | Dependency relation label               |

A virtual ROOT token (id=0, POS="ROOT") is added at index 0 of every sentence. Multi-word tokens and empty nodes are skipped during parsing as they are not relevant for dependency tree construction.

## 4. Arc-Standard Transition System

The parser uses the **Arc-Standard** transition system with three transitions:

1. **SHIFT**: Move the first word from the buffer to the top of the stack.

2. **LEFT-ARC(label)**: The word at the top of the stack becomes the head of the second word on the stack. The second word is then popped from the stack. An arc `(top → second, label)` is created.

3. **RIGHT-ARC(label)**: The second word on the stack becomes the head of the word at the top of the stack. The top word is then popped from the stack. An arc `(second → top, label)` is created.

**Initial configuration**: Stack = [ROOT], Buffer = [word_1, word_2, ..., word_n], Arcs = ∅

**Terminal condition**: Buffer is empty and stack contains only ROOT.

## 5. Oracle Simulation

The oracle simulates the parsing process on gold-standard trees to generate training data. At each configuration, it determines the correct transition:

1. **LEFT-ARC(label)**: Applied when the second item on the stack has its gold head equal to the top of the stack, AND all dependents of the second item have already been collected.

2. **RIGHT-ARC(label)**: Applied when the top of the stack has its gold head equal to the second item, AND all dependents of the top item have already been collected.

3. **SHIFT**: Applied as the default when neither arc condition is met.

The dependent-completeness check is what makes this work: a word can only be removed from the stack after all of its children in the gold tree have been attached to it. That's what guarantees the oracle produces transitions that reconstruct the exact gold dependency tree.

The oracle generated **409,156 training instances** across **88 unique transition labels** from the 12,544 training sentences.

## 6. Feature Extraction

For each parser configuration, exactly **4 features** are extracted:

| # | Feature              | Description                                    |
|---|----------------------|------------------------------------------------|
| 1 | Stack top POS        | POS tag of the word on top of the stack         |
| 2 | Stack second POS     | POS tag of the second word on the stack          |
| 3 | Buffer first POS     | POS tag of the first word in the buffer          |
| 4 | Buffer second POS    | POS tag of the second word in the buffer         |

When a position is unavailable (e.g., stack has fewer than 2 elements), the special sentinel value `NONE` is used. Features are encoded as categorical variables using scikit-learn's `DictVectorizer`, which creates one-hot encoded representations. This resulted in a feature matrix of shape **(409,156 × 73)**.

## 7. Classifier

**Model**: Logistic Regression (scikit-learn `LogisticRegression`)

**Configuration**: `max_iter=1000` (enough iterations to reach convergence), `solver='lbfgs'` (works well for multinomial classification), `C=1.0` (the default regularization strength, left untouched), and `n_jobs=-1` so training uses all available CPU cores.

**Why Logistic Regression**: it trains fast even on a large dataset like this one (409K instances, done in about 80 seconds), and it handles the 88 transition labels naturally through multinomial softmax rather than needing a one-vs-rest wrapper. The one-hot encoded POS features are sparse, which logistic regression is efficient with, and the model stays interpretable, which matters for an assignment where the point is understanding the parser rather than squeezing out accuracy. It also outputs class probabilities instead of just a single label, which the parser leans on to fall back to the next most likely transition when the top prediction turns out to be invalid.

**Training accuracy**: 80.33%

## 8. Parser Implementation

The parser takes a sentence (words + POS tags) as input and produces dependency arcs:

1. Initialize: stack = [ROOT], buffer = [1..n], arcs = []
2. At each step:
   - Extract the 4 POS features from the current configuration
   - Use the trained classifier to predict transition probabilities
   - Select the highest-probability **valid** transition
   - Apply the transition to update the configuration
3. Continue until the buffer is empty and the stack has at most 1 element

Validity checking is what keeps the parser from crashing: SHIFT is only valid if the buffer is non-empty, LEFT-ARC needs the stack to have at least 2 elements with the second one not being ROOT, and RIGHT-ARC just needs the stack to have at least 2 elements.

If no predicted transition is valid, a fallback mechanism applies:
1. SHIFT if the buffer has elements
2. RIGHT-ARC(dep) if the stack has ≥ 2 elements
3. Otherwise, parsing terminates

A safety limit of `4n + 10` steps prevents infinite loops.

## 9. LAS Calculation

**Labeled Attachment Score (LAS)** measures the percentage of tokens for which **both**:
1. The predicted head is correct (matches gold head)
2. The predicted dependency label is correct (matches gold label)

Formula: `LAS = (correct tokens / total tokens) × 100`

Only non-ROOT tokens are evaluated. The ROOT token (id=0) is excluded from the count.

## 10. Final LAS Score

| Metric                    | Value      |
|---------------------------|------------|
| Sentences processed       | 2,001      |
| Total tokens evaluated    | 25,148     |
| Correct tokens            | 14,254     |
| **Labeled Attachment Score** | **56.68%** |
| Evaluation time           | 5.8s       |

This LAS of **56.68%** is consistent with expectations for a simple transition-based parser using only 4 POS-tag features and Logistic Regression. More sophisticated parsers use additional features (word forms, lemmas, morphological features, contextual embeddings) and more powerful models (neural networks) to achieve higher accuracy.

## 11. Example Outputs

### Sentence 1: "The cat sat on the mat."

| ID | Word | Head | Head Word | Relation |
|----|------|------|-----------|----------|
| 1  | The  | 2    | cat       | det      |
| 2  | cat  | 0    | ROOT      | root     |
| 3  | sat  | 2    | cat       | acl      |
| 4  | on   | 6    | mat       | case     |
| 5  | the  | 6    | mat       | det      |
| 6  | mat  | 3    | sat       | obj      |
| 7  | .    | 2    | cat       | punct    |

### Sentence 2: "She eats a green salad."

| ID | Word  | Head | Head Word | Relation |
|----|-------|------|-----------|----------|
| 1  | She   | 2    | eats      | nsubj    |
| 2  | eats  | 0    | ROOT      | root     |
| 3  | a     | 5    | salad     | det      |
| 4  | green | 5    | salad     | amod     |
| 5  | salad | 2    | eats      | obj      |
| 6  | .     | 2    | eats      | punct    |

### Sentence 3: "I saw the man with a telescope."

| ID | Word      | Head | Head Word | Relation |
|----|-----------|------|-----------|----------|
| 1  | I         | 2    | saw       | nsubj    |
| 2  | saw       | 0    | ROOT      | root     |
| 3  | the       | 4    | man       | det      |
| 4  | man       | 2    | saw       | obj      |
| 5  | with      | 7    | telescope | case     |
| 6  | a         | 7    | telescope | det      |
| 7  | telescope | 4    | man       | nmod     |
| 8  | .         | 2    | saw       | punct    |

## 12. Design Choices

Logistic Regression was picked over SVM or Random Forest mainly for speed with this many classes, plus it gives natural probability outputs and plays well with sparse one-hot features. Feature encoding uses `DictVectorizer`, which gives clean one-hot encoding of the categorical POS features without having to hand-roll label encoding. For transition selection, the parser doesn't just take the top prediction; it walks down the ranked list of transitions by probability and applies the first one that's actually valid, which cuts down parse failures a lot compared to always trusting the top-1 prediction. The oracle's dependent-completeness check is there to keep the training data correct, since a word can only be reduced once all of its children have been attached. And the virtual ROOT token exists just to simplify the transition system, so there's always a root anchor sitting on the stack instead of having to special-case an empty stack.

## 13. Limitations

The parser only uses 4 POS-tag features, which caps how context-sensitive its decisions can be; adding word form features, dependency relation features from children already attached, or a bit more buffer lookahead would likely help. Related to that, it has no lexical knowledge at all, it only sees POS tags, so it can't pick up on word-specific patterns. Parsing is also greedy: the model commits to a transition at each step with no search, so beam search or some form of global optimization would probably raise accuracy further. The oracle itself is static rather than dynamic, meaning the model has no way to recover from its own prediction errors during training the way dynamic-oracle techniques allow. Finally, Arc-Standard can only produce projective trees, so non-projective dependencies in the gold data fall back to the generic fallback mechanism and aren't always parsed correctly.

## 14. Conclusion

This project builds a working transition-based dependency parser on the Arc-Standard system, trained on the Universal Dependencies English-EWT treebank and evaluated on the dev set, where it reaches a Labeled Attachment Score of **56.68%**. That's a modest number next to state-of-the-art parsers, but it still covers the core ideas of data-driven dependency parsing: generating training data through an oracle, extracting features from parser configurations, predicting transitions with a classifier, and evaluating the result with a standard metric. The code is split into clear modules and handles its edge cases without crashing.
