# SVM and LSTM Vulnerability Detection Pipelines

**Student:** Ravishka Rathnayake  
**Student ID:** CIT-24-01-0251  

This document explains how to run and integrate the SVM and LSTM source-code vulnerability detection pipelines developed for the NLP group project.

---

## 1. Implemented Components

The following components are available:

- Linear SVM vulnerability prediction pipeline
- LSTM vulnerability prediction pipeline
- Combined SVM and LSTM comparison pipeline
- Direct source-code input support
- Source-code file input support
- Automated pipeline tests
- Example C source-code file

---

## 2. Project Files

### Pipeline files

```text
src/cit_24_01_0251_svm_pipeline.py
src/cit_24_01_0251_lstm_pipeline.py
src/cit_24_01_0251_combined_pipeline.py
```

### Model artifacts

```text
models/CIT-24-01-0251_svm_model_bundle.joblib
models/CIT-24-01-0251_lstm_model.keras
models/CIT-24-01-0251_lstm_preprocessing.joblib
```

### Supporting files

```text
examples/safe_sample.c
tests/test_cit_24_01_0251_pipelines.py
reports/CIT-24-01-0251_svm_metrics.json
reports/CIT-24-01-0251_lstm_metrics.json
```