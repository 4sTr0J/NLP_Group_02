# NLP-Based Source Code Vulnerability Detection

## Group Members

| Student ID | Student Name | Machine Learning Model | Deep Learning Model |
|---|---|---|---|
| CIT-24-01-0251 | Ravishka Rathnayake | Support Vector Machine | Long Short-Term Memory Network |
| CIT-24-01-0249 | Loshan Mihisara | Random Forest | Convolutional Neural Network |
| CIT-24-01-0475 | Nadil Kularathne | XGBoost | CodeBERT |

Each group member independently developed one Machine Learning model and one Deep Learning model.

A total of six models were trained and evaluated using the same common dataset split to ensure a fair comparison. The best-performing Machine Learning and Deep Learning models were then selected for final system integration.

---

## Problem Statement

Manual source-code vulnerability detection is time-consuming and requires cybersecurity knowledge, secure programming experience and careful code review.

Reviewing a large number of source-code functions manually can be difficult, inefficient and expensive. Existing vulnerability-detection tools may also produce false positives or fail to identify vulnerabilities that depend on contextual source-code patterns.

This project investigates how Natural Language Processing, Machine Learning and Deep Learning techniques can be used to identify patterns associated with vulnerable source code.

The system treats source code as textual and sequential data and classifies each source-code function as:

- **Non-vulnerable**
- **Vulnerable**

The main objectives of the project are to:

- Preprocess source code as textual and sequential data
- Develop three Machine Learning models
- Develop three Deep Learning models
- Evaluate all six models using the same common dataset split
- Compare the models using accuracy and vulnerability-focused evaluation metrics
- Select the best-performing Machine Learning and Deep Learning models
- Integrate the selected models into a reusable vulnerability-detection pipeline

The final system is intended to support vulnerability screening and secure code review. It is not intended to replace professional static analysis, manual security assessment or penetration testing.

---

## Dataset Information

The project uses the **DiverseVul dataset**.

The original DiverseVul dataset contains more than 320,000 labelled source-code functions collected from real-world open-source projects.

The target labels are:

- `0` — Non-vulnerable
- `1` — Vulnerable

For this project, a reproducible stratified subset of **10,000 source-code functions** was used.

### Dataset Distribution

| Class | Number of Records |
|---|---:|
| Non-vulnerable | 9,409 |
| Vulnerable | 591 |
| **Total** | **10,000** |

The dataset is highly imbalanced because the number of non-vulnerable functions is significantly higher than the number of vulnerable functions.

The following techniques were used to address the class imbalance:

- Stratified dataset splitting
- Balanced class weighting
- Validation-selected decision thresholds
- Precision, recall and F1-score evaluation
- Average Precision evaluation
- ROC-AUC evaluation
- Confusion-matrix analysis

### Common Dataset Split

All six models were evaluated using the same common dataset split.

| Dataset | Number of Records |
|---|---:|
| Training set | 7,000 |
| Validation set | 1,500 |
| Test set | 1,500 |

The common test set contains:

- 1,412 non-vulnerable functions
- 88 vulnerable functions

The dataset split configuration is:

```text
Random seed: 42
Training set: 70%
Validation set: 15%
Test set: 15%
Stratified sampling: Enabled
```

The generated common dataset files are:

```text
data/processed/common/common_train.csv
data/processed/common/common_validation.csv
data/processed/common/common_test.csv
```

Each common dataset file contains at least the following columns:

```text
func
target
```

Large dataset files are excluded from GitHub through `.gitignore`.

---

## Setup Instructions

### Prerequisites

The following software is required:

- Python 3.12 or a compatible Python version
- Git
- pip
- A terminal such as PowerShell, Command Prompt, Bash or the VS Code terminal

### 1. Clone the Repository

```bash
git clone https://github.com/4sTr0J/NLP_Group_02.git
cd NLP_Group_02
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the Dataset Files

Place the original DiverseVul dataset inside:

```text
data/raw/
```

Place the prepared 10,000-record dataset inside:

```text
data/processed/
```

The common dataset files should be located at:

```text
data/processed/common/common_train.csv
data/processed/common/common_validation.csv
data/processed/common/common_test.csv
```

### 7. Add the Trained Model Files

The trained model files are excluded from GitHub because of their file sizes.

Place the following final model artifacts inside the `models` directory:

```text
models/CIT-24-01-0251_svm_weighted_hybrid_final.joblib
models/CIT-24-01-0251_lstm_embedding32_final.keras
models/CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

---

## How to Run the Project

All commands should be executed from the project root directory after activating the virtual environment.

### Create the Common Dataset Split

```powershell
python .\src\create_common_dataset_split.py
```

This script creates:

```text
data/processed/common/common_train.csv
data/processed/common/common_validation.csv
data/processed/common/common_test.csv
```

It also displays:

- Number of training, validation and testing records
- Class distributions
- Random seed
- SHA-256 file-verification hashes

### Train the Final Support Vector Machine

```powershell
python .\src\cit_24_01_0251_svm_weighted_hybrid_final.py
```

### Train the Final Long Short-Term Memory Network

```powershell
python .\src\cit_24_01_0251_lstm_embedding32_final.py
```

### Run the SVM Pipeline Using a Source-Code File

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --file .\examples\safe_sample.c
```

### Run the SVM Pipeline Using Source Code Directly

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --code "int add(int a, int b) { return a + b; }"
```

### Run the LSTM Pipeline Using a Source-Code File

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py --file .\examples\safe_sample.c
```

### Run the LSTM Pipeline Using Source Code Directly

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py --code "int add(int a, int b) { return a + b; }"
```

### Run the Combined SVM and LSTM Pipeline

```powershell
python .\src\cit_24_01_0251_combined_pipeline.py --file .\examples\safe_sample.c
```

The combined pipeline returns:

- SVM prediction
- SVM decision score
- SVM decision threshold
- LSTM prediction
- LSTM vulnerability probability
- LSTM decision threshold
- Number of vulnerable votes
- Model-agreement status
- Final agreed prediction

When both models produce the same prediction, the pipeline returns the agreed result.

When the models disagree, the output indicates that further review is required.

### Evaluate the Selected Models on the Common Test Set

```powershell
python .\src\evaluate_common_test.py
```

The evaluation script calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC
- Confusion matrix

The results are saved to:

```text
reports/CIT-24-01-0251_common_test_results.json
```

### Run the Automated Tests

```powershell
python -m pytest -q
```

---

## Model Summary

A total of six models were developed and evaluated during the project.

### Machine Learning Models

| Model | Description |
|---|---|
| Support Vector Machine | Uses weighted token-level and character-level TF-IDF representations with a LinearSVC classifier |
| Random Forest | Uses an ensemble of decision trees to classify source-code functions |
| XGBoost | Uses gradient-boosted decision trees for source-code vulnerability classification |

### Deep Learning Models

| Model | Description |
|---|---|
| Long Short-Term Memory Network | Processes source code as a sequence of programming-related tokens |
| Convolutional Neural Network | Uses convolutional filters to identify local source-code patterns |
| CodeBERT | Uses a pretrained transformer architecture designed for programming-language and natural-language data |

### Final Support Vector Machine Configuration

The final SVM uses a weighted hybrid combination of token-level TF-IDF and character-level TF-IDF features.

```text
Model: LinearSVC
Token TF-IDF features: 50,000
Character TF-IDF features: 50,000
Token weight: 0.75
Character weight: 1.00
C value: 0.05
Class weighting: Balanced
Decision threshold: Approximately 0.0278
Random seed: 42
```

Token-level TF-IDF captures programming-language elements such as:

- Keywords
- Identifiers
- Function names
- Operators
- Data types
- Programming tokens

Character-level TF-IDF captures:

- Local syntax patterns
- Character sequences
- Identifier fragments
- Operator combinations
- Programming structures

The token and character representations are weighted and combined before being passed to the LinearSVC classifier.

### Final Long Short-Term Memory Network Configuration

The final LSTM processes source code as a sequence of programming-related tokens.

```text
Vocabulary size: 30,000
Maximum sequence length: 600
Embedding dimension: 32
LSTM units: 64
Dropout: 0.20
Dense hidden units: 32
Optimizer: Adam
Learning rate: 0.001
Batch size: 64
Class weighting: Balanced
Decision threshold: Approximately 0.4743
Random seed: 42
```

The LSTM pipeline performs the following operations:

1. Source-code preprocessing
2. Tokenization
3. Integer-sequence conversion
4. Sequence padding or truncation
5. Embedding-layer processing
6. LSTM sequence learning
7. Dense-layer classification
8. Validation-selected decision-threshold application

---

## Results Summary

All six models were trained and evaluated using the same common dataset split to ensure a fair comparison.

### Six-Model Accuracy Comparison

| Category | Model | Accuracy |
|---|---|---:|
| Machine Learning | Support Vector Machine | 0.8680 |
| Machine Learning | Random Forest | 0.7451 |
| Machine Learning | XGBoost | 0.5892 |
| Deep Learning | Long Short-Term Memory Network | 0.8833 |
| Deep Learning | Convolutional Neural Network | 0.7748 |
| Deep Learning | CodeBERT | 0.6583 |

### Selected Models

Based on the common-test accuracy comparison:

- The **Support Vector Machine** achieved the highest accuracy among the three Machine Learning models.
- The **Long Short-Term Memory Network** achieved the highest accuracy among the three Deep Learning models.

Therefore, the SVM and LSTM were selected for final system integration.

### Detailed Results of the Selected Models

The final SVM and LSTM models were evaluated using the common test set containing **1,500 source-code functions**.

The test set contains:

- 1,412 non-vulnerable functions
- 88 vulnerable functions

### Performance Comparison

| Metric | SVM | LSTM |
|---|---:|---:|
| Accuracy | 0.8680 | 0.8833 |
| Precision | 0.1726 | 0.1520 |
| Recall | 0.3295 | 0.2159 |
| F1-score | 0.2266 | 0.1784 |
| Average Precision | 0.1949 | 0.1181 |
| ROC-AUC | 0.7232 | 0.5901 |

### Performance Comparison in Percentages

| Metric | SVM | LSTM |
|---|---:|---:|
| Accuracy | 86.80% | 88.33% |
| Precision | 17.26% | 15.20% |
| Recall | 32.95% | 21.59% |
| F1-score | 22.66% | 17.84% |
| Average Precision | 19.49% | 11.81% |
| ROC-AUC | 72.32% | 59.01% |

### SVM Confusion Matrix

```text
[[1273, 139],
 [  59,  29]]
```

The SVM produced:

- 1,273 true negatives
- 139 false positives
- 59 false negatives
- 29 true positives

### LSTM Confusion Matrix

```text
[[1306, 106],
 [  69,  19]]
```

The LSTM produced:

- 1,306 true negatives
- 106 false positives
- 69 false negatives
- 19 true positives

### Results Interpretation

The LSTM achieved the highest overall accuracy of **88.33%**, while the SVM achieved an accuracy of **86.80%**.

However, the SVM performed better in detecting vulnerable source-code functions. It achieved higher:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC

The SVM correctly identified 29 of the 88 vulnerable test functions, while the LSTM correctly identified 19.

The SVM therefore demonstrated stronger vulnerable-class detection performance under the current dataset and experimental configuration.

The accuracy values must be interpreted carefully because the common test set is highly imbalanced. Most test samples are non-vulnerable, which can produce high overall accuracy even when vulnerable functions are missed.

For this reason, the final evaluation considered the following measures in addition to accuracy:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC
- Confusion matrices

The final integrated pipeline runs both selected models on the same source-code input and reports whether their predictions agree.

The developed system should be used as a vulnerability-screening and decision-support tool. It should not be used as the only security-review mechanism or as a replacement for professional static analysis, manual secure code review and penetration testing.
