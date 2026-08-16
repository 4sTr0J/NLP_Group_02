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

For this project, a reproducible stratified subset of **~20,000 source-code functions** was used.

The dataset is highly imbalanced because the number of non-vulnerable functions is significantly higher than the number of vulnerable functions.

The following techniques were used to address the class imbalance:

- Stratified dataset splitting
- Balanced class weighting
- Validation-selected decision thresholds
- Precision, recall and F1-score evaluation
- Average Precision evaluation
- ROC-AUC evaluation
- Confusion-matrix analysis
  

Large dataset files are excluded from GitHub through `.gitignore`.

---

## Setup Instructions

## Installation

## Requirements

The project requires Python and the following packages:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install joblib
pip install matplotlib
```

Alternatively, install the required packages using a requirements file if one is provided:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Activate the virtual environment:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Step 1 — Data Cleaning

```powershell
python src/data_cleaning.py
```

### Step 2 — Train-Test Split

```powershell
python src/data_split.py
```

This creates:

```text
data/train_data.csv
data/test_data.csv
```

### Step 3 — Tokenization

```powershell
python src/tokenization.py
```

This creates:

```text
data/train_tokenized.csv
data/test_tokenized.csv
```

### Step 4 — Feature Extraction

For Random Forest:

```powershell
python src/feature_extraction_rf.py
```

For CNN:

```powershell
python src/feature_extraction_cnn.py
```

This produces the required preprocessing artifacts:

```text
models/tfidf_vectorizer.pkl
models/tokenizer.pkl
```

### Step 5 — Train Random Forest

```powershell
python src/random_forest.py
```

Output:

```text
models/random_forest.pkl
```

### Step 6 — Train CNN

```powershell
python src/cnn.py
```

Output:

```text
models/cnn_model.keras
```

### Step 7 — Evaluate Models

```powershell
python src/evaluation.py
```

Output:

```text
models/model_comparison.csv
```

---

## Model Comparison

The results show that CNN achieved the highest overall performance across most evaluation metrics.

### Random Forest

* Accuracy: **74.51%**
* Precision: **67.41%**
* Recall: **67.66%**
* F1-Score: **67.53%**

Random Forest achieved a higher recall than CNN, meaning it identified a larger proportion of the actual vulnerable samples.

### CNN

* Accuracy: **77.48%**
* Precision: **74.89%**
* Recall: **63.97%**
* F1-Score: **69.00%**

CNN achieved higher accuracy, precision, and F1-score than Random Forest.

---

The developed system should be used as a vulnerability-screening and decision-support tool. It should not be used as the only security-review mechanism or as a replacement for professional static analysis, manual secure code review and penetration testing.
