# NLP-Based Source Code Vulnerability Detection

## Group Members

| Student ID | Student Name | Machine Learning Model | Deep Learning Model |
|---|---|---|---|
| CIT-24-01-0251 | Ravishka Rathnayake | Support Vector Machine | Long Short-Term Memory Network |
| CIT-24-01-0249 | Loshan Mihisara | Random Forest | Convolutional Neural Network |
| CIT-24-01-0475 | Nadil Kularathne | XGBoost | CodeBERT |

Each group member independently developed one machine learning model and one deep learning model. All six models were evaluated using a common dataset split to ensure a fair comparison.

---

## Problem Statement

Manual source-code vulnerability detection is time-consuming and requires experienced cybersecurity professionals. Analysing a large number of source-code functions manually can also be difficult and inefficient.

This project investigates how Natural Language Processing, Machine Learning and Deep Learning techniques can be used to identify patterns associated with software vulnerabilities.

The system treats source code as textual and sequential data and classifies each source-code function as:

- Non-vulnerable
- Vulnerable

The main objective of the project is to develop and compare six different models and integrate the best-performing machine learning and deep learning models into a reusable source-code vulnerability-detection system.

---

## Dataset Information

The project uses the DiverseVul dataset.

The original dataset contains more than 320,000 labelled source-code functions collected from real-world open-source projects.

The target labels are:

- `0` — Non-vulnerable
- `1` — Vulnerable

For the current project, a reproducible stratified subset of 10,000 records was used.

### Dataset Distribution

| Class | Number of Records |
|---|---:|
| Non-vulnerable | 9,409 |
| Vulnerable | 591 |
| Total | 10,000 |

The dataset is highly imbalanced because the number of non-vulnerable functions is much higher than the number of vulnerable functions.

To handle this imbalance, the project uses:

- Stratified dataset splitting
- Balanced class weights
- Validation-based decision thresholds
- Precision, recall, F1-score, Average Precision and ROC-AUC in addition to accuracy

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

The dataset split was created using:

```text
Random seed: 42
Training: 70%
Validation: 15%
Testing: 15%
Stratified sampling: Enabled
```

Large dataset files are excluded from GitHub using `.gitignore`.

---

## Setup Instructions

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

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install the Required Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add the Dataset Files

Place the original and processed dataset files inside:

```text
data/raw/
data/processed/
```

The common dataset files should be placed inside:

```text
data/processed/common/common_train.csv
data/processed/common/common_validation.csv
data/processed/common/common_test.csv
```

Each common dataset file must contain at least the following columns:

```text
func
target
```

### 6. Add the Trained Model Files

The trained model files are stored locally because they are too large to upload directly to GitHub.

Place the following model files inside the `models` folder:

```text
models/CIT-24-01-0251_svm_weighted_hybrid_final.joblib
models/CIT-24-01-0251_lstm_embedding32_final.keras
models/CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

---

## How to Run the Project

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
- LSTM prediction
- LSTM vulnerability probability
- Decision thresholds
- Model agreement status
- Final agreed label

### Create the Common Dataset Split

```powershell
python .\src\create_common_dataset_split.py
```

### Evaluate the Final Models Using the Common Test Set

```powershell
python .\src\evaluate_common_test.py
```

The evaluation results are saved inside:

```text
reports/CIT-24-01-0251_common_test_results.json
```

### Run the Automated Tests

```powershell
python -m pytest -q
```

---

## Model Summary

A total of six models were developed and evaluated.

### Machine Learning Models

| Model | Description |
|---|---|
| Support Vector Machine | Uses weighted token-level and character-level TF-IDF features with a LinearSVC classifier |
| Random Forest | Uses multiple decision trees to classify source-code functions |
| XGBoost | Uses gradient-boosted decision trees for vulnerability classification |

### Deep Learning Models

| Model | Description |
|---|---|
| Long Short-Term Memory Network | Processes source code as a sequence of programming-related tokens |
| Convolutional Neural Network | Uses convolutional filters to identify local source-code patterns |
| CodeBERT | Uses a pretrained transformer model designed for programming-language and natural-language data |

### Final Support Vector Machine Configuration

The final SVM uses a weighted combination of token TF-IDF and character TF-IDF features.

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

Token TF-IDF captures programming-language keywords, identifiers and operators.

Character TF-IDF captures local syntax, character patterns and programming structures.

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

### Six-Model Accuracy Comparison

All six models were trained and evaluated using the same common dataset split to ensure a fair comparison.

| Category | Model | Accuracy |
|---|---|---:|
| Machine Learning | Support Vector Machine | 0.8680 |
| Machine Learning | Random Forest | ADD EXACT COMMON-TEST RESULT |
| Machine Learning | XGBoost | ADD EXACT COMMON-TEST RESULT |
| Deep Learning | Long Short-Term Memory Network | 0.8833 |
| Deep Learning | Convolutional Neural Network | ADD EXACT COMMON-TEST RESULT |
| Deep Learning | CodeBERT | ADD EXACT COMMON-TEST RESULT |

### Selected Models

Based on the common-test accuracy comparison:

- The Support Vector Machine achieved the highest accuracy among the three machine learning models.
- The Long Short-Term Memory Network achieved the highest accuracy among the three deep learning models.

Therefore, the SVM and LSTM were selected for final system integration.

---

## Results Summary

### Results Interpretation

The LSTM achieved the highest overall accuracy of 88.33%, while the SVM achieved an accuracy of 86.80%.

However, the SVM performed better in detecting vulnerable source-code functions. It achieved higher precision, recall, F1-score, Average Precision and ROC-AUC than the LSTM.

The accuracy results must be interpreted carefully because the common test set is highly imbalanced. Most test samples are non-vulnerable, which can result in high accuracy even when some vulnerable functions are not detected.

Therefore, the final evaluation also considered:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC
- Confusion matrices

The final integrated system runs both the SVM and LSTM models on the same source-code input and reports whether their predictions agree.

The developed system should be used as a vulnerability-screening and decision-support tool. It is not intended to replace professional static analysis, manual secure code review or penetration testing.
