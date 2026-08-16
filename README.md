# SVM and LSTM Vulnerability Detection Pipeline

## Individual Contribution

**Student:** Ravishka Rathnayake  
**Student ID:** CIT-24-01-0251  
**Branch:** `features/CIT-24-01-0251-SVM+LSTM`

This branch contains my individual contribution to the group project **NLP-Based Source Code Vulnerability Detection**.

My assigned models were:

- **Machine Learning Model:** Support Vector Machine
- **Deep Learning Model:** Long Short-Term Memory Network

The branch includes the complete workflow for preprocessing, model experimentation, training, evaluation, reusable prediction pipelines, automated testing and final SVM–LSTM integration.

---

## Project Objective

The objective is to classify source-code functions into two classes:

- `0` — Non-vulnerable
- `1` — Vulnerable

Source code is processed as textual and sequential data so that Machine Learning and Deep Learning models can learn patterns associated with software vulnerabilities.

---

## Dataset

The project uses the **DiverseVul dataset**, which contains more than 320,000 labelled source-code functions collected from real-world open-source projects.

A reproducible stratified subset of **10,000 records** was used for model development.

### Class Distribution

| Class | Records |
|---|---:|
| Non-vulnerable | 9,409 |
| Vulnerable | 591 |
| **Total** | **10,000** |

### Common Dataset Split

| Dataset | Records |
|---|---:|
| Training set | 7,000 |
| Validation set | 1,500 |
| Test set | 1,500 |

The common test set contains:

- 1,412 non-vulnerable functions
- 88 vulnerable functions

The split was created using stratified sampling with `random_state = 42`.

---

## Support Vector Machine

The final SVM uses a weighted combination of token-level and character-level TF-IDF features.

Token-level TF-IDF captures programming keywords, identifiers, operators and token sequences, while character-level TF-IDF captures local syntax, identifier fragments and character patterns.

### Final SVM Configuration

```text
Classifier: LinearSVC
Token TF-IDF features: 50,000
Character TF-IDF features: 50,000
Token n-gram range: (1, 2)
Character n-gram range: (3, 5)
Token feature weight: 0.75
Character feature weight: 1.00
C value: 0.05
Class weighting: Balanced
Maximum iterations: 10,000
Decision threshold: Approximately 0.0278
Random seed: 42
```

The prediction threshold was selected using validation-set F1-score rather than using the default SVM decision boundary.

---

## Long Short-Term Memory Network

The LSTM processes source code as a sequence of programming-related tokens. A code-aware tokenizer preserves elements such as identifiers, keywords, operators, brackets and numerical values.

Each function is converted into an integer sequence and padded or truncated to a maximum length of 600 tokens.

### Final LSTM Configuration

```text
Maximum vocabulary size: 30,000
Maximum sequence length: 600
Embedding dimension: 32
LSTM layers: 1
LSTM units: 64
LSTM dropout: 0.20
Intermediate dropout: 0.30
Dense hidden units: 32
Dense activation: ReLU
Dense dropout: 0.20
Output activation: Sigmoid
Optimizer: Adam
Learning rate: 0.001
Batch size: 64
Maximum epochs: 8
Class weighting: Balanced
Decision threshold: Approximately 0.4743
Random seed: 42
```

Early stopping monitored validation PR-AUC and restored the best model weights.

---

## Final Evaluation Results

Both models were evaluated using the same common test set containing 1,500 source-code functions.

| Metric | SVM | LSTM |
|---|---:|---:|
| Accuracy | 0.8680 | 0.8833 |
| Precision | 0.1726 | 0.1520 |
| Recall | 0.3295 | 0.2159 |
| F1-score | 0.2266 | 0.1784 |
| Average Precision | 0.1949 | 0.1181 |
| ROC-AUC | 0.7232 | 0.5901 |

### SVM Confusion Matrix

```text
[[1273, 139],
 [  59,  29]]
```

### LSTM Confusion Matrix

```text
[[1306, 106],
 [  69,  19]]
```

### Results Interpretation

The LSTM achieved the higher overall accuracy of **88.33%**, while the SVM achieved an accuracy of **86.80%**.

However, the SVM achieved better vulnerable-class precision, recall, F1-score, Average Precision and ROC-AUC. The SVM correctly detected 29 vulnerable functions, while the LSTM correctly detected 19.

Therefore:

- **LSTM provided the highest overall accuracy.**
- **SVM provided stronger vulnerable-class detection.**

Both models were retained for the final integrated pipeline.

---

## Combined SVM–LSTM Pipeline

The combined pipeline runs both final models on the same source-code input.

### Decision Logic

- If both models predict **Vulnerable**, the final result is Vulnerable.
- If both models predict **Non-vulnerable**, the final result is Non-vulnerable.
- If the models disagree, the sample is marked as **Review Required**.

The pipeline returns:

- SVM prediction
- SVM decision score
- SVM threshold
- LSTM prediction
- LSTM vulnerability probability
- LSTM threshold
- Model-agreement status
- Final classification

---

## Main Files

### Production Pipelines

```text
src/cit_24_01_0251_svm_pipeline.py
src/cit_24_01_0251_lstm_pipeline.py
src/cit_24_01_0251_combined_pipeline.py
src/cit_24_01_0251_adapter.py
src/cit_24_01_0251_evaluate.py
```

### Final Training Scripts

```text
src/cit_24_01_0251_svm_weighted_hybrid_final.py
src/cit_24_01_0251_lstm_embedding32_final.py
```

### Common Dataset Evaluation

```text
src/create_common_dataset_split.py
src/evaluate_common_test.py
```

### Experiment Scripts

```text
experiments/CIT-24-01-0251/
```

### Automated Tests

```text
tests/test_cit_24_01_0251_pipelines.py
tests/test_cit_24_01_0251_adapter.py
```

### Documentation

```text
docs/CIT-24-01-0251_PIPELINES.md
docs/CIT-24-01-0251_MODEL_IMPROVEMENT_SUMMARY.md
```

### Evaluation Reports

```text
reports/CIT-24-01-0251_svm_weighted_hybrid_final_metrics.json
reports/CIT-24-01-0251_lstm_embedding32_final_metrics.json
reports/CIT-24-01-0251_common_test_results.json
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/4sTr0J/NLP_Group_02.git
cd NLP_Group_02
```

### 2. Switch to the Individual Branch

```bash
git switch "features/CIT-24-01-0251-SVM+LSTM"
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Local Files

Large dataset and trained-model files are excluded from GitHub through `.gitignore`.

### Dataset Files

```text
data/raw/diversevul_20230702.csv
data/processed/diversevul_stratified_10000.csv
data/processed/common/common_train.csv
data/processed/common/common_validation.csv
data/processed/common/common_test.csv
```

### Trained Model Files

```text
models/CIT-24-01-0251_svm_weighted_hybrid_final.joblib
models/CIT-24-01-0251_lstm_embedding32_final.keras
models/CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

---

## Running the Pipelines

### Run the SVM Pipeline

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --file .\examples\safe_sample.c
```

### Run the LSTM Pipeline

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py --file .\examples\safe_sample.c
```

### Run the Combined Pipeline

```powershell
python .\src\cit_24_01_0251_combined_pipeline.py --file .\examples\safe_sample.c
```

### Evaluate the Common Test Set

```powershell
python .\src\evaluate_common_test.py
```

### Run Automated Tests

```powershell
python -m pytest -q
```

Expected result:

```text
14 passed
```

---

## Limitations

- The dataset is highly imbalanced.
- Vulnerable-class precision and recall remain limited.
- False positives and false negatives may occur.
- LSTM inputs longer than 600 tokens are truncated.
- The models classify vulnerability status but do not always identify the exact vulnerability type.
- The system should not be treated as proof that source code is secure.

The final pipeline is designed as a vulnerability-screening and decision-support tool. Professional static analysis, secure code review and penetration testing are still required.
