# CIT-24-01-0251 — SVM and LSTM Model Improvement Summary

## 1. Overview

This document summarizes the model development, controlled experiments, model-selection process, final evaluation results, and production integration completed for the source-code vulnerability detection component developed by **CIT-24-01-0251**.

Two Natural Language Processing based models were implemented:

- **Machine Learning Model:** Linear Support Vector Machine (LinearSVC)
- **Deep Learning Model:** Long Short-Term Memory Network (LSTM)

The objective of the improvement stage was to investigate whether different source-code representations, model capacities, sequence lengths, vocabulary settings, and class-weighting strategies could improve the detection of vulnerable source-code functions.

Because the dataset is highly imbalanced, overall accuracy alone was not considered sufficient for evaluating model quality. Greater attention was given to:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC
- Confusion Matrix

For model selection during the improvement stage, **vulnerable-class validation F1-score** was treated as the main decision metric, while the other metrics were used as supporting evidence.

---

## 2. Dataset

The project uses the **DiverseVul** source-code vulnerability dataset.

A reproducible stratified subset containing **10,000 records** was created for model development.

### Dataset Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Non-vulnerable | 9,409 | 94.09% |
| Vulnerable | 591 | 5.91% |
| **Total** | **10,000** | **100%** |

The dataset is therefore highly imbalanced, with vulnerable samples representing only approximately **5.91%** of the selected data.

---

## 3. Reproducible Data Split

A fixed random seed of:

```text
42
```

was used throughout the experiments.

The 10,000-record subset was divided using stratified sampling.

| Dataset Split | Number of Records |
|---|---:|
| Training | 7,000 |
| Validation | 1,500 |
| Test | 1,500 |

The held-out test set contains:

- **1,412 non-vulnerable samples**
- **88 vulnerable samples**

The training set contains:

- **6,586 non-vulnerable samples**
- **414 vulnerable samples**

The validation set contains:

- **1,411 non-vulnerable samples**
- **89 vulnerable samples**

Using stratification ensured that the original class imbalance was maintained across the training, validation, and test sets.

---

## 4. Experimental Methodology

The model-improvement process followed a controlled experimental approach.

The following principles were maintained:

1. The same dataset subset was used throughout the experiments.
2. The same train, validation, and test split was maintained.
3. The random seed remained fixed at 42.
4. Tokenizers and vectorizers were fitted only using training data.
5. Hyperparameter decisions during the improvement stage were made using validation results.
6. Decision thresholds were optimized using validation predictions.
7. Final selected configurations were evaluated on the held-out test set for reporting.
8. No additional hyperparameter tuning was performed using the final test results.
9. Rejected experiments were retained for reproducibility and documentation.

Earlier baseline held-out results were available from the initial model audit. However, the later improvement-stage configurations were selected using validation results rather than selecting configurations based on their final test scores.

---

## 5. Why Accuracy Is Not Enough

The dataset contains approximately:

```text
94.09% non-vulnerable samples
5.91% vulnerable samples
```

A model predicting almost everything as non-vulnerable could therefore obtain high accuracy while still failing to detect most vulnerable functions.

For this reason, the main metrics used in this project are:

### Precision

Precision measures how many samples predicted as vulnerable were actually vulnerable.

Higher precision indicates fewer false-positive vulnerability alerts.

### Recall

Recall measures how many of the actual vulnerable samples were successfully detected.

Higher recall indicates fewer vulnerable functions were missed.

### F1-score

F1-score balances precision and recall.

Because the vulnerable class is significantly underrepresented, F1-score was used as the main validation-based model-selection metric.

### Average Precision

Average Precision summarizes the precision-recall relationship across different decision thresholds.

It is especially useful for imbalanced classification tasks.

### ROC-AUC

ROC-AUC measures the ability of the model to rank vulnerable samples above non-vulnerable samples across different thresholds.

---

## 6. Baseline SVM Model

The original classical machine-learning model used **LinearSVC** with token-based TF-IDF features.

### Source-Code Tokenizer

A programming-language-aware tokenizer was used instead of a standard natural-language tokenizer.

The tokenizer preserves source-code elements such as:

- Identifiers
- Numeric values
- Hexadecimal values
- Brackets
- Operators
- Comparison operators
- Logical operators
- Pointer operators
- Punctuation

Examples include:

```text
==
!=
<=
>=
->
++
--
&&
||
<<
>>
```

This helps retain programming-language information that may be important for vulnerability detection.

---

## 7. Baseline SVM Feature Extraction

The baseline model used token TF-IDF with the following configuration:

```text
Analyzer         : Token based
N-gram range     : (1, 2)
Minimum DF       : 2
Maximum DF       : 0.98
Maximum features : 50,000
Sublinear TF     : True
```

The TF-IDF vectorizer was fitted only on the training dataset.

---

## 8. Baseline SVM Hyperparameter Selection

The following LinearSVC `C` values were evaluated:

```text
0.01
0.05
0.10
0.25
0.50
1.00
2.00
5.00
```

The candidate models were compared using validation F1-score.

The selected baseline value was:

```text
C = 0.05
```

The validation decision threshold was then optimized separately.

The selected baseline SVM threshold was approximately:

```text
-0.0693
```

---

## 9. Baseline SVM Validation Performance

The original token-only SVM achieved approximately:

| Metric | Validation Result |
|---|---:|
| Precision | 0.1509 |
| Recall | 0.5393 |
| F1-score | 0.2359 |
| Average Precision | 0.1843 |

Validation confusion matrix:

```text
[[1141, 270],
 [  41,  48]]
```

The model achieved relatively high recall but produced a large number of false-positive vulnerability predictions.

This motivated further investigation into source-code feature representations.

---

## 10. SVM Character N-gram Experiment

The first major SVM improvement experiment replaced token features with character-level TF-IDF features.

Character-level representations can capture:

- Programming syntax
- Operator combinations
- Identifier fragments
- Local source-code structures
- Repeated unsafe coding patterns

### Character TF-IDF Configuration

```text
Analyzer         : Character
N-gram range     : (3, 5)
Minimum DF       : 2
Maximum features : 50,000
Sublinear TF     : True
Data type        : float32
```

The resulting training matrix contained:

```text
7,000 rows
50,000 character features
```

---

## 11. Character SVM C-Value Experiment

The following validation results were obtained before threshold optimization.

| C | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.01 | 0.1339 | 0.5506 | 0.2154 |
| 0.05 | 0.1646 | 0.4494 | 0.2410 |
| 0.10 | 0.1784 | 0.3708 | 0.2409 |
| 0.25 | 0.2033 | 0.2809 | 0.2358 |
| **0.50** | **0.2500** | **0.2472** | **0.2486** |
| 1.00 | 0.2453 | 0.1461 | 0.1831 |
| 2.00 | 0.2391 | 0.1236 | 0.1630 |
| 5.00 | 0.2632 | 0.1124 | 0.1575 |

The best default-threshold F1-score was obtained with:

```text
C = 0.50
```

---

## 12. Character SVM Threshold Optimization

After selecting `C = 0.50`, the validation threshold was optimized.

Final validation results:

| Metric | Result |
|---|---:|
| Precision | 0.2838 |
| Recall | 0.2360 |
| F1-score | **0.2577** |
| Average Precision | 0.1687 |
| ROC-AUC | 0.6988 |

Selected threshold:

```text
0.0494
```

Validation confusion matrix:

```text
[[1358, 53],
 [  68, 21]]
```

### Observation

Compared with the token-only baseline, character features:

- Increased precision considerably
- Improved F1-score
- Reduced false positives
- Reduced recall

This experiment demonstrated that character-level source-code patterns contained useful vulnerability-related information.

---

## 13. Token + Character Hybrid SVM

Because token and character representations captured different aspects of source code, the next experiment combined them.

The hybrid representation consisted of:

```text
50,000 token TF-IDF features
+
50,000 character TF-IDF features
=
100,000 sparse features
```

The two sparse matrices were combined using horizontal sparse stacking.

---

## 14. Equal Hybrid SVM

The first hybrid experiment used equal weighting:

```text
Token weight     = 1.00
Character weight = 1.00
```

The best validation result was obtained using:

```text
C = 0.05
```

After threshold optimization, the equal hybrid achieved:

| Metric | Validation Result |
|---|---:|
| Precision | 0.1856 |
| Recall | 0.4045 |
| F1-score | **0.2544** |
| Average Precision | 0.1866 |
| ROC-AUC | **0.7301** |

Selected threshold:

```text
-0.0400
```

Validation confusion matrix:

```text
[[1253, 158],
 [  53,  36]]
```

### Observation

The equal hybrid provided a better balance between:

- Token-level semantic information
- Character-level structural information

This justified experimenting with different relative weights.

---

## 15. Weighted Hybrid SVM Experiments

The LinearSVC parameter was fixed at:

```text
C = 0.05
```

Different feature-weight combinations were then evaluated.

| Token Weight | Character Weight | Precision | Recall | F1 | Average Precision | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 0.1856 | 0.4045 | 0.2544 | 0.1866 | 0.7301 |
| 1.00 | 0.75 | 0.1925 | 0.4045 | 0.2609 | 0.1867 | **0.7337** |
| **0.75** | **1.00** | **0.2000** | 0.3820 | **0.2625** | **0.1897** | 0.7265 |
| 1.00 | 1.25 | 0.1814 | **0.4157** | 0.2526 | 0.1868 | 0.7259 |

The highest validation F1-score was obtained using:

```text
Token weight     = 0.75
Character weight = 1.00
```

Therefore this configuration was selected as the final SVM candidate.

---

## 16. Final Selected SVM Configuration

The final validation-selected SVM configuration is:

```text
Model             : LinearSVC
Class weighting   : Balanced
Random seed       : 42

Token TF-IDF:
Maximum features  : 50,000

Character TF-IDF:
Maximum features  : 50,000
Character n-grams : 3-5

Token weight      : 0.75
Character weight  : 1.00

C                 : 0.05
Decision threshold: approximately 0.0278
```

---

## 17. Final SVM Validation Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8727 |
| Precision | 0.2000 |
| Recall | 0.3820 |
| F1-score | **0.2625** |
| Average Precision | 0.1897 |
| ROC-AUC | 0.7265 |

Validation confusion matrix:

```text
[[1275, 136],
 [  55,  34]]
```

---

## 18. Final SVM Held-Out Test Results

After selecting the SVM using validation performance, the same trained configuration and validation-selected threshold were used for final held-out test evaluation.

| Metric | Final Test Result |
|---|---:|
| Accuracy | 0.8680 |
| Precision | 0.1726 |
| Recall | 0.3295 |
| F1-score | **0.2266** |
| Average Precision | **0.1949** |
| ROC-AUC | **0.7232** |

Test confusion matrix:

```text
[[1273, 139],
 [  59,  29]]
```

This means that among the 88 vulnerable test samples:

```text
29 were correctly detected
59 were missed
```

---

## 19. Final SVM Generalization

The validation and final test results remained relatively consistent.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8727 | 0.8680 |
| Precision | 0.2000 | 0.1726 |
| Recall | 0.3820 | 0.3295 |
| F1-score | 0.2625 | 0.2266 |
| Average Precision | 0.1897 | **0.1949** |
| ROC-AUC | 0.7265 | **0.7232** |

The ROC-AUC changed only slightly:

```text
0.7265 → 0.7232
```

Average Precision slightly increased:

```text
0.1897 → 0.1949
```

This indicates that the weighted token + character representation generalized reasonably well to unseen samples.

---

## 20. Final SVM Improvement Over Original Model

The final weighted SVM can also be compared with the original SVM held-out audit results.

| Metric | Original SVM | Final Weighted SVM |
|---|---:|---:|
| Precision | 0.1262 | **0.1726** |
| Recall | **0.4545** | 0.3295 |
| F1-score | 0.1975 | **0.2266** |
| Average Precision | 0.1937 | **0.1949** |
| ROC-AUC | 0.7139 | **0.7232** |

The vulnerable-class F1-score improved from approximately:

```text
0.1975
```

to:

```text
0.2266
```

The relative improvement is approximately:

```text
14.7%
```

The final SVM therefore achieved:

- Higher precision
- Higher F1-score
- Slightly higher Average Precision
- Higher ROC-AUC

The trade-off was a reduction in recall.

---

## 21. Baseline LSTM Model

The deep-learning component uses a Long Short-Term Memory neural network.

Source code is first tokenized using the programming-language-aware tokenizer.

Tokens are then converted into a space-separated representation and transformed into integer sequences using a Keras tokenizer.

---

## 22. Baseline LSTM Tokenizer

The tokenizer configuration was:

```text
Maximum vocabulary size : 30,000
OOV token               : <OOV>
Lowercase               : False
Filters                 : Empty
Split                   : Space
```

The tokenizer was fitted only on the training data.

The complete training vocabulary contained approximately:

```text
106,343 unique tokens
```

---

## 23. Source-Code Sequence Length Analysis

The training source-code sequence-length distribution was analyzed.

| Percentile | Sequence Length |
|---:|---:|
| 50th | 117 |
| 75th | 274 |
| 90th | 599 |
| 95th | 1,000 |
| 99th | 2,352 |
| Maximum | 29,855 |

There were no empty source-code sequences.

The baseline sequence length was selected as:

```text
600
```

This covers approximately the first 90% of the training sequence-length distribution.

Sequences longer than 600 tokens are truncated using post-truncation.

Shorter sequences are padded using post-padding.

---

## 24. Baseline LSTM Class Weighting

Because the training data is heavily imbalanced, balanced class weights were calculated.

Approximate class weights:

```text
Class 0: 0.5314
Class 1: 8.4541
```

This gives the vulnerable class approximately:

```text
15.9 times
```

the relative importance of the non-vulnerable class during optimization.

---

## 25. Baseline LSTM Architecture

The baseline model used:

```text
Input
↓
Embedding
↓
LSTM
↓
Dropout
↓
Dense ReLU
↓
Dropout
↓
Sigmoid Output
```

Configuration:

```text
Maximum vocabulary size : 30,000
Sequence length         : 600
Embedding dimension     : 64
LSTM units              : 64
LSTM dropout            : 0.20
Dense hidden units      : 32
Dense activation        : ReLU
Dropout                 : 0.30 / 0.20
Output units            : 1
Output activation       : Sigmoid
```

---

## 26. Baseline LSTM Training

The model was compiled using:

```text
Optimizer       : Adam
Learning rate   : 0.001
Loss            : Binary Cross-Entropy
Batch size      : 64
Maximum epochs  : 8
Random seed     : 42
```

Training metrics included:

- Accuracy
- Precision
- Recall
- ROC-AUC
- PR-AUC

Early stopping used:

```text
Monitor              : val_pr_auc
Mode                 : max
Patience             : 2
Restore best weights : True
```

---

## 27. Baseline LSTM Training Behaviour

The baseline model reached its strongest validation PR-AUC during the first epoch.

Approximate validation PR-AUC values:

```text
Epoch 1: 0.1383
Epoch 2: 0.1268
Epoch 3: 0.0798
```

Training stopped after validation performance failed to improve.

The weights from the best epoch were automatically restored.

---

## 28. Baseline LSTM Validation Performance

After validation threshold optimization, the baseline LSTM produced approximately:

| Metric | Result |
|---|---:|
| Precision | 0.1695 |
| Recall | 0.3371 |
| F1-score | **0.2256** |
| ROC-AUC | approximately 0.6653 |

Validation confusion matrix:

```text
[[1264, 147],
 [  59,  30]]
```

This baseline became the reference configuration for controlled LSTM experiments.

---

## 29. LSTM Experiment 1 — Sequence Length 1000

The first LSTM experiment increased the maximum sequence length from:

```text
600
```

to:

```text
1000
```

All other major configuration settings remained unchanged.

The purpose was to determine whether providing more source-code context would improve vulnerability detection.

---

## 30. Sequence Length 1000 Validation Results

| Metric | Result |
|---|---:|
| Accuracy | 0.7787 |
| Precision | 0.1118 |
| Recall | 0.3933 |
| F1-score | 0.1741 |
| Average Precision | 0.1056 |
| ROC-AUC | 0.6242 |

Selected validation threshold:

```text
0.4682
```

Confusion matrix:

```text
[[1133, 278],
 [  54,  35]]
```

### Decision

```text
REJECTED
```

Although recall increased, false positives increased significantly.

F1-score decreased from approximately:

```text
0.2256
```

to:

```text
0.1741
```

The original maximum sequence length of:

```text
600
```

was therefore retained.

---

## 31. LSTM Experiment 2 — Vocabulary Audit

Before increasing vocabulary size, a training and validation vocabulary-coverage audit was performed.

The discovered training vocabulary contained:

```text
106,343 unique tokens
```

Two vocabulary limits were compared:

```text
30,000
50,000
```

---

## 32. Training Vocabulary Coverage

### 30,000-Token Vocabulary

```text
Total token occurrences : 1,955,108
Represented             : 1,852,679
OOV                     : 102,429
Coverage                : 94.76%
OOV rate                : 5.24%
```

### 50,000-Token Vocabulary

```text
Represented : 1,896,427
OOV         : 58,681
Coverage    : 97.00%
OOV rate    : 3.00%
```

---

## 33. Validation Vocabulary Coverage

### 30,000-Token Vocabulary

```text
Total occurrences : 413,003
Represented       : 368,174
OOV               : 44,829
Coverage          : 89.15%
OOV rate          : 10.85%
```

### 50,000-Token Vocabulary

```text
Represented : 372,389
OOV         : 40,614
Coverage    : 90.17%
OOV rate    : 9.83%
```

Increasing the vocabulary by 20,000 entries recovered only approximately:

```text
1.02 percentage points
```

of validation token coverage.

---

## 34. Vulnerable Validation Vocabulary Coverage

The vulnerable validation subset was analyzed separately.

### 30,000-Token Vocabulary

```text
Coverage : 88.04%
OOV      : 11.96%
```

### 50,000-Token Vocabulary

```text
Coverage : 89.03%
OOV      : 10.97%
```

Only approximately:

```text
0.99 percentage points
```

of additional vulnerable validation token coverage would be gained.

---

## 35. Vocabulary Audit Decision

Increasing the vocabulary from:

```text
30,000 → 50,000
```

would increase the embedding vocabulary size by approximately:

```text
66.7%
```

while providing only around one additional percentage point of validation coverage.

### Decision

```text
50,000 vocabulary NOT JUSTIFIED
```

The final model retained:

```text
MAX_VOCAB_SIZE = 30,000
```

---

## 36. LSTM Experiment 3 — Reduced Class Weighting

The original balanced class weights created an approximate vulnerable/non-vulnerable weighting ratio of:

```text
15.9 : 1
```

A milder class-weight experiment was performed using an approximate ratio of:

```text
8 : 1
```

Approximate weights:

```text
Class 0: 0.7072
Class 1: 5.6577
```

---

## 37. Mild Class-Weight Validation Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8587 |
| Precision | 0.1639 |
| Recall | 0.3371 |
| F1-score | 0.2206 |
| Average Precision | 0.1423 |
| ROC-AUC | 0.6524 |

Selected threshold:

```text
0.3540
```

Validation confusion matrix:

```text
[[1258, 153],
 [  59,  30]]
```

### Decision

```text
REJECTED
```

The number of true-positive vulnerable predictions remained the same as the baseline while false positives increased.

The original balanced class weighting was therefore retained.

---

## 38. LSTM Experiment 4 — Embedding Dimension 32

The next controlled architecture experiment reduced the embedding dimension from:

```text
64
```

to:

```text
32
```

All other major model settings remained unchanged:

```text
Vocabulary       : 30,000
Sequence length  : 600
LSTM units       : 64
Balanced weights : Yes
Learning rate    : 0.001
Batch size       : 64
```

---

## 39. Embedding-32 Validation Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8880 |
| Precision | **0.1985** |
| Recall | 0.2921 |
| F1-score | **0.2364** |
| Average Precision | 0.1350 |
| ROC-AUC | 0.6186 |

Selected validation threshold:

```text
0.4743
```

Validation confusion matrix:

```text
[[1306, 105],
 [  63,  26]]
```

The F1-score increased compared with the baseline:

```text
Baseline F1      ≈ 0.2256
Embedding-32 F1  = 0.2364
```

This represented approximately a:

```text
4.8% relative validation F1 improvement
```

Therefore the Embedding-32 configuration became the strongest LSTM candidate according to the predefined validation F1 selection criterion.

---

## 40. LSTM Experiment 5 — LSTM Units 32

A final controlled architecture experiment reduced the recurrent layer from:

```text
64 LSTM units
```

to:

```text
32 LSTM units
```

while restoring the embedding dimension to:

```text
64
```

Other settings remained unchanged.

---

## 41. LSTM-Units-32 Validation Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8327 |
| Precision | 0.1447 |
| Recall | **0.3708** |
| F1-score | 0.2082 |
| Average Precision | 0.1314 |
| ROC-AUC | 0.6446 |

Selected threshold:

```text
0.4875
```

Validation confusion matrix:

```text
[[1216, 195],
 [  56,  33]]
```

### Decision

```text
REJECTED
```

Although recall increased, precision and F1-score decreased.

The Embedding-32 / LSTM-64 model therefore remained the validation-selected candidate.

---

## 42. LSTM Experiment Comparison

| Configuration | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline Embedding-64 / LSTM-64 | ~0.1695 | ~0.3371 | ~0.2256 | **~0.6653** |
| Sequence length 1000 | 0.1118 | 0.3933 | 0.1741 | 0.6242 |
| Mild class weighting | 0.1639 | 0.3371 | 0.2206 | 0.6524 |
| **Embedding-32 / LSTM-64** | **0.1985** | 0.2921 | **0.2364** | 0.6186 |
| Embedding-64 / LSTM-32 | 0.1447 | **0.3708** | 0.2082 | 0.6446 |

The highest validation F1-score was achieved by:

```text
Embedding dimension = 32
LSTM units          = 64
```

Therefore this configuration was selected before the final held-out evaluation.

---

## 43. Final Selected LSTM Configuration

The final validation-selected LSTM candidate uses:

```text
Maximum vocabulary size : 30,000
Maximum sequence length : 600
Embedding dimension     : 32
LSTM units              : 64

LSTM dropout            : 0.20
Dense hidden layer      : 32 units
Dense activation        : ReLU

Optimizer               : Adam
Learning rate           : 0.001
Loss                    : Binary Cross-Entropy
Batch size              : 64
Maximum epochs          : 8

Class weighting         : Balanced
Early stopping monitor  : val_pr_auc
Early stopping patience : 2
Restore best weights    : True

Decision threshold      : approximately 0.4743
Random seed             : 42
```

---

## 44. Final LSTM Held-Out Test Results

The validation-selected Embedding-32 LSTM was then evaluated on the held-out test set.

| Metric | Final Test Result |
|---|---:|
| Accuracy | **0.8833** |
| Precision | 0.1520 |
| Recall | 0.2159 |
| F1-score | **0.1784** |
| Average Precision | 0.1181 |
| ROC-AUC | 0.5901 |

Test confusion matrix:

```text
[[1306, 106],
 [  69,  19]]
```

Among the 88 vulnerable test samples:

```text
19 were correctly detected
69 were missed
```

---

## 45. LSTM Validation vs Test

| Metric | Validation | Final Test |
|---|---:|---:|
| Accuracy | 0.8880 | 0.8833 |
| Precision | 0.1985 | 0.1520 |
| Recall | 0.2921 | 0.2159 |
| F1-score | **0.2364** | **0.1784** |
| Average Precision | 0.1350 | 0.1181 |
| ROC-AUC | 0.6186 | 0.5901 |

The Embedding-32 configuration improved validation F1-score but the same improvement did not fully generalize to the held-out test data.

This suggests that the LSTM remains sensitive to:

- Limited vulnerable training examples
- Severe class imbalance
- Large source-code vocabulary
- Long source-code sequences
- Project-specific identifiers
- Differences between training and unseen code patterns

No further LSTM tuning was performed based on the final test result.

---

## 46. Original LSTM Held-Out Audit

For comparison, the earlier baseline LSTM held-out audit produced approximately:

| Metric | Original LSTM |
|---|---:|
| Accuracy | 0.8620 |
| Precision | 0.1520 |
| Recall | 0.2955 |
| F1-score | 0.2008 |
| Average Precision | 0.1097 |
| ROC-AUC | 0.6220 |

The later Embedding-32 configuration was **not switched back** based on these final test comparisons because model-selection decisions had already been made using validation results.

Doing otherwise would effectively turn the test set into another validation set.

---

## 47. Final SVM vs Final LSTM

The final held-out results of the validation-selected candidates are:

| Metric | Final SVM | Final LSTM |
|---|---:|---:|
| Accuracy | 0.8680 | **0.8833** |
| Precision | **0.1726** | 0.1520 |
| Recall | **0.3295** | 0.2159 |
| F1-score | **0.2266** | 0.1784 |
| Average Precision | **0.1949** | 0.1181 |
| ROC-AUC | **0.7232** | 0.5901 |

Although the LSTM achieved slightly higher accuracy, accuracy is misleading for this problem because the dataset is approximately 94% non-vulnerable.

The weighted SVM achieved stronger:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC

Therefore the SVM is the stronger individual vulnerability-detection model under the current experimental setup.

---

## 48. Why the SVM Performed Better

Several factors may explain why the SVM outperformed the LSTM.

### 48.1 Limited Training Data

Only:

```text
7,000 samples
```

were available for training in the selected subset.

Deep-learning sequence models normally benefit from much larger datasets.

### 48.2 Limited Vulnerable Samples

Only:

```text
414 vulnerable samples
```

were available in the training split.

This gives the LSTM relatively few positive examples from which to learn vulnerability patterns.

### 48.3 Severe Class Imbalance

The dataset contains approximately:

```text
94.09% non-vulnerable
5.91% vulnerable
```

This makes deep-learning optimization difficult even when balanced class weights are used.

### 48.4 Large Vocabulary

The training data contained approximately:

```text
106,343 unique tokens
```

Many identifiers are project-specific and may appear rarely.

The neural model therefore has difficulty learning reliable embeddings for many rare tokens.

### 48.5 Long Source-Code Functions

Some source-code functions contained thousands of tokens.

The maximum observed training sequence length was approximately:

```text
29,855 tokens
```

The LSTM processes only the first:

```text
600 tokens
```

in the selected configuration.

Important vulnerability information may sometimes appear beyond this limit.

### 48.6 TF-IDF Works Well for Sparse Code Patterns

The SVM directly uses high-dimensional sparse representations.

The final SVM combines:

```text
Token TF-IDF
+
Character TF-IDF
```

This allows the model to detect:

- Vulnerability-related tokens
- Function names
- API patterns
- Operators
- Syntax fragments
- Character-level code structures

without requiring the amount of training data normally required by a neural sequence model.

---

## 49. Final Model Selection Summary

### Final SVM

```text
Model              : LinearSVC

Representation:
Weighted Token TF-IDF
+
Character TF-IDF

Token features     : 50,000
Character features : 50,000

Token weight       : 0.75
Character weight   : 1.00

C                  : 0.05
Threshold          : approximately 0.0278

Validation F1      : 0.2625

Final Test:
Accuracy           : 0.8680
Precision          : 0.1726
Recall             : 0.3295
F1-score           : 0.2266
Average Precision  : 0.1949
ROC-AUC            : 0.7232
```

### Final LSTM

```text
Vocabulary size     : 30,000
Sequence length     : 600

Embedding dimension : 32
LSTM units          : 64

Class weighting     : Balanced
Learning rate       : 0.001
Batch size          : 64

Threshold           : approximately 0.4743

Validation F1       : 0.2364

Final Test:
Accuracy            : 0.8833
Precision           : 0.1520
Recall              : 0.2159
F1-score            : 0.1784
Average Precision   : 0.1181
ROC-AUC             : 0.5901
```

---

## 50. Production SVM Integration

The production SVM pipeline was updated to use the final weighted-hybrid model.

The final bundle contains:

```text
model
token_vectorizer
char_vectorizer
token_weight
char_weight
C
decision_threshold
random_seed
```

The runtime prediction pipeline:

1. Accepts source code.
2. Validates that the input is not empty.
3. Applies the token TF-IDF vectorizer.
4. Applies the character TF-IDF vectorizer.
5. Multiplies token features by 0.75.
6. Multiplies character features by 1.00.
7. Combines the two sparse representations.
8. Calculates the LinearSVC decision score.
9. Applies the saved validation-selected threshold.
10. Returns the vulnerability prediction.

The pipeline maintains support for the previous single-vectorizer SVM bundle for backward compatibility.

---

## 51. Production LSTM Integration

The production LSTM pipeline was updated to load:

```text
CIT-24-01-0251_lstm_embedding32_final.keras
```

and:

```text
CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

The runtime pipeline:

1. Accepts source code.
2. Applies the programming-language-aware tokenizer.
3. Converts tokens into a space-separated sequence.
4. Uses the saved training tokenizer.
5. Converts tokens into integer IDs.
6. Pads or truncates the sequence to 600 tokens.
7. Loads the final LSTM.
8. Calculates vulnerability probability.
9. Applies the saved validation-selected threshold.
10. Returns the final prediction.

The production pipeline also supports the earlier preprocessing-bundle format.

---

## 52. Production SVM Runtime Verification

The final SVM pipeline was tested using:

```text
examples/safe_sample.c
```

The model returned:

```text
Model          : Linear Support Vector Machine
Representation : Weighted token + character TF-IDF
Prediction     : Non-vulnerable
Decision score : approximately -0.5407
Threshold      : approximately 0.0278
Token count    : 22
```

Because:

```text
-0.5407 < 0.0278
```

the resulting class was correctly determined as:

```text
Non-vulnerable
```

---

## 53. Production LSTM Runtime Verification

The final LSTM pipeline was tested using the same safe sample.

The model returned approximately:

```text
Prediction                : Non-vulnerable
Vulnerability probability : 0.4356
Decision threshold        : 0.4743
Sequence length setting   : 600
Embedding dimension       : 32
LSTM units                : 64
Token count               : 22
```

Because:

```text
0.4356 < 0.4743
```

the resulting class was:

```text
Non-vulnerable
```

---

## 54. Combined SVM + LSTM Pipeline

A combined prediction pipeline is also available.

The combined pipeline independently executes both models.

The process is:

```text
Input Source Code
        |
        +-------------------+
        |                   |
        v                   v
   Final SVM            Final LSTM
        |                   |
        v                   v
    Prediction          Prediction
        |                   |
        +---------+---------+
                  |
                  v
          Compare Results
                  |
        +---------+---------+
        |                   |
        v                   v
      Agree              Disagree
        |                   |
        v                   v
  Agreed Result       Review Required
```

---

## 55. Combined Decision Strategy

The combined model intentionally does not force a vulnerability prediction when the models disagree.

If:

```text
SVM prediction == LSTM prediction
```

then:

```text
models_agree = True
```

and the common label can be reported.

If:

```text
SVM prediction != LSTM prediction
```

then:

```text
models_agree = False
```

and the result is marked for review.

This provides a conservative integration strategy instead of automatically trusting one model when there is disagreement.

---

## 56. Combined Pipeline Runtime Verification

The safe sample produced:

```text
SVM prediction  : Non-vulnerable
LSTM prediction : Non-vulnerable
```

Therefore:

```text
models_agree     : True
agreed_label     : Non-vulnerable
vulnerable_votes : 0
```

The combined pipeline completed successfully.

---

## 57. Model Artifacts

The selected final model artifacts are stored locally under the `models` directory.

### Final SVM

```text
models/CIT-24-01-0251_svm_weighted_hybrid_final.joblib
```

### Final LSTM

```text
models/CIT-24-01-0251_lstm_embedding32_final.keras
```

### Final LSTM Preprocessing

```text
models/CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

Large trained model artifacts are excluded from Git where required by the repository `.gitignore`.

---

## 58. Final Metric Reports

Final metrics are stored as JSON files.

### SVM

```text
reports/CIT-24-01-0251_svm_weighted_hybrid_final_metrics.json
```

### LSTM

```text
reports/CIT-24-01-0251_lstm_embedding32_final_metrics.json
```

These reports contain:

- Selected configuration
- Validation metrics
- Final held-out test metrics
- Confusion matrices

---

## 59. Experiment Scripts

Controlled experiment scripts are retained under:

```text
experiments/CIT-24-01-0251/
```

The experiment directory contains:

```text
cit_24_01_0251_lstm_1000_experiment.py
cit_24_01_0251_lstm_embedding32_experiment.py
cit_24_01_0251_lstm_mild_weight_experiment.py
cit_24_01_0251_lstm_units32_experiment.py
cit_24_01_0251_lstm_vocab_audit.py

cit_24_01_0251_svm_char_experiment.py
cit_24_01_0251_svm_hybrid_experiment.py
cit_24_01_0251_svm_weighted_hybrid_experiment.py
```

These files provide evidence of the controlled model-development process and allow individual experiments to be reviewed later.

---

## 60. Final Training and Evaluation Scripts

The selected models also have dedicated final scripts.

### Final SVM

```text
src/cit_24_01_0251_svm_weighted_hybrid_final.py
```

### Final LSTM

```text
src/cit_24_01_0251_lstm_embedding32_final.py
```

These scripts reproduce the selected configurations and save the associated artifacts and final metric reports.

---

## 61. Production Prediction Scripts

The reusable prediction pipelines are:

```text
src/cit_24_01_0251_svm_pipeline.py
```

```text
src/cit_24_01_0251_lstm_pipeline.py
```

```text
src/cit_24_01_0251_combined_pipeline.py
```

The project also contains an adapter layer for integration with the wider group application.

---

## 62. Automated Testing

After the final production integration, the full automated test suite was executed using:

```text
pytest
```

Final result:

```text
14 passed
```

The tests included checks for:

- Individual prediction pipelines
- Input validation
- Model loading
- Adapter behaviour
- Combined model integration

Ten dependency/runtime warnings were displayed during the tests, mainly related to NumPy/joblib compatibility.

These warnings did not cause test failures.

---

## 63. Syntax Validation

The following files were successfully checked using Python compilation:

```text
SVM production pipeline
LSTM production pipeline
Combined production pipeline
Final SVM script
Final LSTM script
All experiment scripts
```

The command used was based on:

```text
python -m py_compile
```

No Python syntax errors were detected.

---

## 64. Git and Reproducibility

The completed improvement work was committed to the individual feature branch:

```text
features/CIT-24-01-0251-SVM+LSTM
```

A major model-finalization commit was created with the message:

```text
Finalize SVM and LSTM model pipelines and experiments
```

A later small compatibility cleanup was committed as:

```text
Fix LSTM pad_sequences import
```

The local feature branch was pushed successfully and verified to be synchronized with the remote GitHub branch.

---

## 65. Key Findings

The major findings from the model-development process are:

1. Token-only TF-IDF provided strong vulnerability recall but produced many false-positive alerts.
2. Character n-gram TF-IDF improved precision and vulnerable-class F1-score.
3. Character-level information captures useful syntax and source-code structure that token-only representations may miss.
4. Combining token and character TF-IDF produced a better overall SVM representation.
5. Slightly favoring character features with a token weight of 0.75 and character weight of 1.00 produced the highest SVM validation F1-score.
6. The final weighted SVM achieved better held-out F1 and ROC-AUC than the original token-only SVM.
7. Increasing the LSTM sequence length from 600 to 1000 did not improve performance.
8. The longer sequence experiment produced more false positives and a substantially lower F1-score.
9. Increasing the vocabulary from 30,000 to 50,000 was not justified by the small increase in validation token coverage.
10. Using milder class weights did not improve the LSTM.
11. Reducing the embedding dimension from 64 to 32 improved validation F1-score.
12. Reducing LSTM units from 64 to 32 did not improve performance.
13. The validation improvement from Embedding-32 did not fully generalize to the held-out test set.
14. The final weighted SVM clearly outperformed the final LSTM on vulnerable-class precision, recall, F1, Average Precision, and ROC-AUC.
15. High overall accuracy should not be interpreted as strong vulnerability-detection performance because of the highly imbalanced dataset.
16. Classical sparse NLP representations can remain highly competitive for source-code analysis when training data is limited.

---

## 66. Limitations

Several limitations should be considered when interpreting the current results.

### Dataset Subset

The experiments were performed using a 10,000-record stratified subset rather than the complete DiverseVul dataset.

This was necessary to keep model development computationally manageable within the available project resources.

### Class Imbalance

Only approximately 5.91% of selected records were vulnerable.

The limited number of positive examples reduces the amount of vulnerability information available to the models.

### LSTM Sequence Truncation

The selected LSTM sequence length is 600.

Functions longer than 600 tokens are truncated.

A vulnerability-related pattern appearing after the first 600 tokens may therefore be unavailable to the model.

### Vocabulary Limit

The LSTM vocabulary is limited to 30,000 token IDs.

Rare tokens and identifiers outside this vocabulary are represented using the OOV token.

### Source-Code Context

The models classify individual source-code functions.

They do not currently model:

- Complete program execution flow
- Interprocedural dependencies
- Runtime data-flow information
- External library behaviour
- Full project context

### Vulnerability Type

The current models mainly perform binary classification:

```text
0 = Non-vulnerable
1 = Vulnerable
```

The current production output does not attempt to identify an exact vulnerability category for every prediction.

---

## 67. Future Improvement Opportunities

Future versions of the project could investigate:

- Training on a larger portion of DiverseVul
- Full-dataset model training
- Advanced class-imbalance strategies
- Focal loss for deep learning
- CNN-LSTM architectures
- Bidirectional LSTM
- Attention mechanisms
- Transformer-based code models
- CodeBERT-style representations
- Graph-based source-code representations
- Abstract Syntax Tree features
- Control-flow information
- Data-flow information
- Vulnerability-specific multi-class classification
- Calibration of prediction probabilities
- Ensemble methods
- Additional external vulnerability datasets

These are potential future directions rather than changes to the currently finalized models.

---

## 68. Final Conclusion

The controlled experimental process successfully improved the original classical machine-learning vulnerability detector and provided a systematic comparison with a deep-learning approach.

The strongest final model was the:

```text
Weighted Token + Character TF-IDF LinearSVC
```

with:

```text
Token weight     = 0.75
Character weight = 1.00
C                = 0.05
Threshold        ≈ 0.0278
```

Its final held-out performance was:

```text
Accuracy          = 0.8680
Precision         = 0.1726
Recall            = 0.3295
F1-score          = 0.2266
Average Precision = 0.1949
ROC-AUC           = 0.7232
```

The final validation-selected LSTM used:

```text
Vocabulary        = 30,000
Sequence length   = 600
Embedding         = 32
LSTM units        = 64
Threshold         ≈ 0.4743
```

Its final held-out performance was:

```text
Accuracy          = 0.8833
Precision         = 0.1520
Recall            = 0.2159
F1-score          = 0.1784
Average Precision = 0.1181
ROC-AUC           = 0.5901
```

Although the LSTM achieved higher overall accuracy, the weighted SVM was substantially stronger on the metrics that are more meaningful for vulnerable-class detection.

The experiment therefore demonstrates that, under the current dataset size and class distribution, combining programming-token TF-IDF with character-level TF-IDF provides a strong and computationally efficient approach for NLP-based source-code vulnerability detection.

Both final models have been integrated into reusable prediction pipelines, verified through automated tests, documented through experiment scripts and metric reports, and version-controlled through the individual Git feature branch.