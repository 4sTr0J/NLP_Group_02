# CIT-24-01-0251 — SVM and LSTM Vulnerability Detection Pipelines

## 1. Overview

This document describes the reusable prediction pipelines implemented by **CIT-24-01-0251** for the NLP-based source-code vulnerability detection project.

Two independently trained models are available:

- **Machine Learning Model:** Linear Support Vector Machine (LinearSVC)
- **Deep Learning Model:** Long Short-Term Memory Network (LSTM)

The final production pipelines allow source code to be supplied either:

- directly through the command line, or
- from a source-code file.

A combined pipeline is also provided to execute both models on the same source-code function and compare their predictions.

---

## 2. Final Production Components

The main production files are:

```text
src/cit_24_01_0251_svm_pipeline.py
src/cit_24_01_0251_lstm_pipeline.py
src/cit_24_01_0251_combined_pipeline.py
src/cit_24_01_0251_adapter.py
```

The final training and evaluation scripts are:

```text
src/cit_24_01_0251_svm_weighted_hybrid_final.py
src/cit_24_01_0251_lstm_embedding32_final.py
```

Detailed model-development experiments are stored under:

```text
experiments/CIT-24-01-0251/
```

---

## 3. Final SVM Pipeline

### 3.1 Final SVM Architecture

The final SVM uses a weighted combination of:

```text
Token TF-IDF
+
Character TF-IDF
```

The selected configuration is:

```text
Model             : LinearSVC
Token features    : 50,000
Character features: 50,000

Token weight      : 0.75
Character weight  : 1.00

C                 : 0.05
Decision threshold: approximately 0.0278
Class weighting   : Balanced
Random seed       : 42
```

The token representation captures programming-language tokens, while the character representation captures local syntax and character-level source-code structures.

---

### 3.2 Final SVM Model Artifact

The production SVM pipeline loads:

```text
models/CIT-24-01-0251_svm_weighted_hybrid_final.joblib
```

The bundle contains:

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

The model artifact is stored locally and excluded from Git where required by `.gitignore`.

---

### 3.3 SVM Prediction Process

For a source-code function, the SVM pipeline performs the following steps:

```text
Source Code
    |
    v
Input Validation
    |
    +-----------------------+
    |                       |
    v                       v
Token TF-IDF          Character TF-IDF
    |                       |
    v                       v
Weight = 0.75         Weight = 1.00
    |                       |
    +-----------+-----------+
                |
                v
        Sparse Feature Stack
                |
                v
            LinearSVC
                |
                v
          Decision Score
                |
                v
Validation-Selected Threshold
                |
                v
Vulnerable / Non-vulnerable
```

The final classification is determined by:

```text
decision_score >= decision_threshold
```

If the condition is true, the sample is classified as vulnerable.

---

### 3.4 SVM Command-Line Usage

Predict from a file:

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --file .\examples\safe_sample.c
```

Predict source code supplied directly:

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --code "int add(int a, int b) { return a + b; }"
```

A custom compatible model bundle can also be supplied:

```powershell
python .\src\cit_24_01_0251_svm_pipeline.py --file .\examples\safe_sample.c --model .\models\custom_svm.joblib
```

---

### 3.5 Example SVM Runtime Result

Using:

```text
examples/safe_sample.c
```

the final pipeline produced approximately:

```json
{
    "student_id": "CIT-24-01-0251",
    "model": "Linear Support Vector Machine",
    "representation": "Weighted token + character TF-IDF",
    "prediction": 0,
    "label": "Non-vulnerable",
    "decision_score": -0.5407,
    "decision_threshold": 0.0278,
    "token_count": 22
}
```

---

## 4. Final LSTM Pipeline

### 4.1 Final LSTM Architecture

The validation-selected final LSTM configuration is:

```text
Maximum vocabulary size : 30,000
Maximum sequence length : 600
Embedding dimension     : 32
LSTM units              : 64

LSTM dropout            : 0.20
Dense hidden units      : 32
Optimizer               : Adam
Learning rate           : 0.001
Batch size              : 64
Class weighting         : Balanced
Random seed             : 42

Decision threshold      : approximately 0.4743
```

---

### 4.2 Final LSTM Artifacts

The production LSTM pipeline loads:

```text
models/CIT-24-01-0251_lstm_embedding32_final.keras
```

and:

```text
models/CIT-24-01-0251_lstm_embedding32_preprocessing.joblib
```

The preprocessing bundle contains information required to reproduce the training-time input transformation, including:

```text
tokenizer
max_vocabulary_size
vocabulary_size
max_sequence_length
embedding_dimension
lstm_units
decision_threshold
random_seed
```

---

### 4.3 LSTM Prediction Process

The LSTM runtime performs:

```text
Source Code
    |
    v
Input Validation
    |
    v
Programming-Aware Tokenization
    |
    v
Space-Separated Token Sequence
    |
    v
Saved Keras Tokenizer
    |
    v
Integer Sequence
    |
    v
Post Padding / Truncation
Maximum Length = 600
    |
    v
Embedding Layer
    |
    v
LSTM Layer
    |
    v
Sigmoid Probability
    |
    v
Validation-Selected Threshold
    |
    v
Vulnerable / Non-vulnerable
```

The final classification is determined by:

```text
vulnerability_probability >= decision_threshold
```

---

### 4.4 LSTM Command-Line Usage

Predict from a file:

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py --file .\examples\safe_sample.c
```

Predict source code supplied directly:

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py --code "int add(int a, int b) { return a + b; }"
```

Custom compatible artifacts can also be supplied:

```powershell
python .\src\cit_24_01_0251_lstm_pipeline.py `
    --file .\examples\safe_sample.c `
    --model .\models\custom_lstm.keras `
    --preprocessing .\models\custom_lstm_preprocessing.joblib
```

---

### 4.5 Example LSTM Runtime Result

Using:

```text
examples/safe_sample.c
```

the final LSTM produced approximately:

```json
{
    "student_id": "CIT-24-01-0251",
    "model": "Long Short-Term Memory Network",
    "prediction": 0,
    "label": "Non-vulnerable",
    "vulnerability_probability": 0.4356,
    "decision_threshold": 0.4743,
    "token_count": 22,
    "maximum_sequence_length": 600,
    "embedding_dimension": 32,
    "lstm_units": 64
}
```

---

## 5. Combined SVM + LSTM Pipeline

The combined pipeline executes both final production models against the same source-code function.

File:

```text
src/cit_24_01_0251_combined_pipeline.py
```

The pipeline uses:

```text
load_svm_bundle()
load_lstm_artifacts()
```

from the individual pipelines.

Because the individual loaders already point to the selected final artifacts, the combined pipeline automatically uses the final SVM and LSTM models.

---

### 5.1 Combined Decision Process

```text
Input Source Code
        |
        +-----------------------+
        |                       |
        v                       v
   Final SVM                Final LSTM
        |                       |
        v                       v
   Prediction                Prediction
        |                       |
        +-----------+-----------+
                    |
                    v
             Compare Results
                    |
        +-----------+-----------+
        |                       |
        v                       v
      Agree                  Disagree
        |                       |
        v                       v
 Agreed Label            Review Required
```

The two models are not forced into a majority decision.

If both predictions are equal:

```text
models_agree = True
```

and the agreed label is returned.

If the predictions differ:

```text
models_agree = False
```

the result is marked for review instead of being treated as a confident automatic ensemble decision.

---

### 5.2 Combined Pipeline Usage

Run using a source-code file:

```powershell
python .\src\cit_24_01_0251_combined_pipeline.py --file .\examples\safe_sample.c
```

Run using direct source code:

```powershell
python .\src\cit_24_01_0251_combined_pipeline.py --code "int add(int a, int b) { return a + b; }"
```

---

### 5.3 Example Combined Result

For the safe sample:

```text
SVM prediction  : Non-vulnerable
LSTM prediction : Non-vulnerable
```

The comparison result was:

```json
{
    "models_agree": true,
    "agreed_label": "Non-vulnerable",
    "vulnerable_votes": 0,
    "message": "Both models produced the same prediction."
}
```

---

## 6. Adapter Layer

The project includes:

```text
src/cit_24_01_0251_adapter.py
```

The adapter provides an integration-friendly interface for the wider group application.

It allows the individual SVM and LSTM components to be used without requiring the rest of the application to directly manage:

- model loading,
- preprocessing,
- feature transformation,
- sequence padding,
- thresholds,
- or artifact paths.

The adapter also validates source-code input before sending it to the prediction pipelines.

---

## 7. Backward Compatibility

The production pipelines preserve support for the earlier model formats.

### SVM

The final SVM pipeline supports:

```text
Final weighted-hybrid bundle
```

as well as the earlier:

```text
Single token-vectorizer SVM bundle
```

The earlier artifact is:

```text
models/CIT-24-01-0251_svm_model_bundle.joblib
```

This artifact is retained only for compatibility and baseline comparison.

It is **not the default production SVM**.

---

### LSTM

The final LSTM pipeline supports preprocessing bundles containing either:

```text
max_sequence_length
```

or the earlier:

```text
maximum_sequence_length
```

The original artifacts are:

```text
models/CIT-24-01-0251_lstm_model.keras
models/CIT-24-01-0251_lstm_preprocessing.joblib
```

These artifacts are retained for compatibility and baseline comparison.

They are **not the default production LSTM artifacts**.

---

## 8. Final Model Performance Reference

### Final Weighted-Hybrid SVM

| Metric | Held-Out Test Result |
|---|---:|
| Accuracy | 0.8680 |
| Precision | 0.1726 |
| Recall | 0.3295 |
| F1-score | 0.2266 |
| Average Precision | 0.1949 |
| ROC-AUC | 0.7232 |

### Final Validation-Selected LSTM

| Metric | Held-Out Test Result |
|---|---:|
| Accuracy | 0.8833 |
| Precision | 0.1520 |
| Recall | 0.2159 |
| F1-score | 0.1784 |
| Average Precision | 0.1181 |
| ROC-AUC | 0.5901 |

The weighted-hybrid SVM achieved the stronger vulnerable-class performance.

Detailed experiment analysis is available in:

```text
docs/CIT-24-01-0251_MODEL_IMPROVEMENT_SUMMARY.md
```

---

## 9. Final Metric Reports

Final evaluation reports are stored in:

```text
reports/CIT-24-01-0251_svm_weighted_hybrid_final_metrics.json
reports/CIT-24-01-0251_lstm_embedding32_final_metrics.json
```

The earlier baseline reports remain available as:

```text
reports/CIT-24-01-0251_svm_metrics.json
reports/CIT-24-01-0251_lstm_metrics.json
reports/CIT-24-01-0251_combined_evaluation.json
```

---

## 10. Supporting Files

### Example Input

```text
examples/safe_sample.c
```

### Automated Tests

```text
tests/test_cit_24_01_0251_pipelines.py
tests/test_cit_24_01_0251_adapter.py
```

### Notebooks

```text
notebooks/CIT-24-01-0251_01_data_preprocessing.ipynb
notebooks/CIT-24-01-0251_02_svm_model.ipynb
notebooks/CIT-24-01-0251_03_lstm_model.ipynb
```

### Experiment Directory

```text
experiments/CIT-24-01-0251/
```

---

## 11. Testing

The complete automated test suite can be executed using:

```powershell
python -m pytest -q
```

The verified result after final production integration was:

```text
14 passed
```

The warnings displayed during testing were dependency/runtime warnings and did not cause any test failures.

---

## 12. Syntax Verification

Individual production files can be checked using:

```powershell
python -m py_compile .\src\cit_24_01_0251_svm_pipeline.py
python -m py_compile .\src\cit_24_01_0251_lstm_pipeline.py
python -m py_compile .\src\cit_24_01_0251_combined_pipeline.py
```

No output indicates successful Python syntax validation.

---

## 13. Model Artifact Storage

Large generated model artifacts are intentionally excluded from Git through `.gitignore`.

Relevant ignored formats include:

```text
*.joblib
*.keras
*.h5
```

The artifacts must therefore exist locally before prediction pipelines can be executed.

The final artifacts can be regenerated using:

```text
src/cit_24_01_0251_svm_weighted_hybrid_final.py
src/cit_24_01_0251_lstm_embedding32_final.py
```

---

## 14. Final Production Setup

The current default runtime configuration is:

```text
SVM
-----------------------------------------
Model:
CIT-24-01-0251_svm_weighted_hybrid_final.joblib

Representation:
Weighted token + character TF-IDF

Token weight:
0.75

Character weight:
1.00

C:
0.05

Decision threshold:
approximately 0.0278
```

```text
LSTM
-----------------------------------------
Model:
CIT-24-01-0251_lstm_embedding32_final.keras

Preprocessing:
CIT-24-01-0251_lstm_embedding32_preprocessing.joblib

Vocabulary:
30,000

Sequence length:
600

Embedding dimension:
32

LSTM units:
64

Decision threshold:
approximately 0.4743
```

Both models can be run independently or through the combined comparison pipeline.

---

## 15. Summary

The final source-code vulnerability detection implementation provides:

- A weighted token + character TF-IDF LinearSVC pipeline
- A reusable LSTM vulnerability prediction pipeline
- Training-time preprocessing reused during inference
- Validation-selected decision thresholds
- CLI support
- File-based source-code prediction
- Backward compatibility with baseline artifacts
- Combined SVM + LSTM comparison
- An adapter for wider project integration
- Automated tests
- Final metrics reports
- Reproducible experiment scripts
- Detailed model-improvement documentation

The production pipelines have been executed successfully and verified through automated testing.