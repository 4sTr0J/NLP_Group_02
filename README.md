# NLP-Based Source Code Vulnerability Detection

## Project Overview

This project develops an NLP-based system for identifying potential vulnerabilities in source-code functions.

The system treats source code as textual and sequential data and applies multiple machine learning and deep learning models to classify each function as:

- Non-vulnerable
- Vulnerable

The final objective is to compare the performance of six different models and integrate the most suitable approach into a complete vulnerability-detection pipeline.

---

## Problem Statement

Manual source-code vulnerability analysis can be time-consuming and requires technical expertise.

This project explores whether Natural Language Processing, Machine Learning and Deep Learning techniques can automatically identify patterns associated with vulnerable source code.

---

## Dataset

The project uses the **DiverseVul** dataset.

The original dataset contains more than 320,000 labelled source-code functions.

For the current development stage, a reproducible stratified dataset of 10,000 records is used:

- Non-vulnerable functions: 9,409
- Vulnerable functions: 591

The dataset is highly imbalanced, so techniques such as stratified splitting, class weighting and decision-threshold optimization are used.

Large dataset files are excluded from GitHub through `.gitignore`.

---

## Team Members and Models

### CIT-24-01-0251 – Ravishka Rathnayake

- Machine Learning Model: Support Vector Machine
- Deep Learning Model: Long Short-Term Memory Network
- Branch: `features/CIT-24-01-0251-SVM+LSTM`

### Team Member 02 – Loshan Mihisara

- Machine Learning Model: Random Forest
- Deep Learning Model: Convolutional Neural Network

### Team Member 03 – Nadil Kularathne

- Machine Learning Model: XGBoost
- Deep Learning Model: CodeBERT

Each member develops and evaluates their models independently before final pipeline integration.

---

## Repository Structure

```text
NLP_Group_02/
│
├── data/
│   ├── raw/              # Original dataset files
│   └── processed/        # Cleaned and sampled datasets
│
├── models/               # Locally saved trained models
├── notebooks/            # Data preprocessing and model notebooks
├── reports/              # Evaluation metrics and reports
├── screenshots/          # Project screenshots and evidence
├── src/                  # Final reusable pipeline source code
├── videos/               # Progress-video-related files
│
├── .gitignore
├── README.md
└── requirements.txt
