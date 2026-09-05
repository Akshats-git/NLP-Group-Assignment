# Report: Transition-Based Dependency Parser

## 1. Objective

The goal of this project is to build a simple, data-driven dependency parser from scratch. The parser uses a transition-based approach with the Arc-Standard transition system. A scikit-learn classifier, trained on a treebank, is used to predict parsing transitions.

## 2. Dataset

The **Universal Dependencies English-EWT** (English Web Treebank) corpus is used:

- **Training**: `en_ewt-ud-train.conllu` — 12,544 sentences
- **Evaluation**: `en_ewt-ud-dev.conllu` — 2,001 sentences

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

The dependent-completeness check is critical — a word can only be removed from the stack after all of its children in the gold tree have been attached to it. This ensures the oracle produces transitions that reconstruct the exact gold dependency tree.

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

**Configuration**:
- `max_iter=1000` — sufficient iterations for convergence
- `solver='lbfgs'` — efficient for multinomial classification
- `C=1.0` — default regularization strength
- `n_jobs=-1` — parallel training using all CPU cores

**Why Logistic Regression was chosen**:
- **Speed**: Fast training on large datasets (409K instances), trained in ~80 seconds
- **Multi-class**: Naturally handles 88 different transition labels via multinomial softmax
- **Sparse features**: Works efficiently with one-hot encoded categorical features
- **Interpretable**: Suitable for an academic assignment where understanding the model is important
- **Probability outputs**: Provides class probabilities, enabling the parser to try alternative transitions when the top prediction is invalid

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

**Validity checking** ensures the parser never crashes:
- **SHIFT**: only valid if buffer is non-empty
- **LEFT-ARC**: only valid if stack has ≥ 2 elements AND second element is not ROOT
- **RIGHT-ARC**: only valid if stack has ≥ 2 elements

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

1. **Logistic Regression over SVM or Random Forest**: Chosen for its speed with many classes, natural probability outputs, and compatibility with sparse one-hot features.

2. **DictVectorizer for feature encoding**: Provides clean one-hot encoding of categorical POS features without manual label encoding.

3. **Probability-ranked transition selection**: Rather than just taking the top prediction, the parser tries all transitions in order of decreasing probability, selecting the first valid one. This significantly reduces parse failures.

4. **Dependent-completeness check in oracle**: Ensures training data is correct by only allowing reduction when all of a word's children have been attached.

5. **Virtual ROOT token**: Simplifies the transition system by always having a root anchor on the stack.

## 13. Limitations

1. **Limited features**: Only 4 POS-tag features are used, which limits the parser's ability to make context-sensitive decisions. Adding word form features, dependency relation features of already-attached children, or buffer lookahead would improve accuracy.

2. **No lexical features**: The parser has no knowledge of individual words, only their POS tags. This means it cannot learn word-specific patterns.

3. **Greedy parsing**: The parser makes locally optimal decisions without any search. Beam search or global optimization would improve accuracy.

4. **No dynamic oracle**: The oracle is static — errors during prediction cannot be recovered from using oracle-guided training techniques like dynamic oracles.

5. **No handling of non-projective trees**: The Arc-Standard system can only produce projective trees. Non-projective dependencies in the gold data are handled via fallback but may not be correctly parsed.

## 14. Conclusion

This project successfully implements a complete transition-based dependency parser using the Arc-Standard transition system. The parser was trained on the Universal Dependencies English-EWT treebank and evaluated on the dev set, achieving a Labeled Attachment Score of **56.68%**. While this score is modest compared to state-of-the-art parsers, it demonstrates the core principles of data-driven dependency parsing: oracle-guided training data generation, feature extraction from parser configurations, classifier-based transition prediction, and systematic evaluation using standard metrics. The implementation is modular, well-documented, and handles edge cases safely.
