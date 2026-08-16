# Automated Source Code Vulnerability Detection Using Machine Learning and Pre-Trained Transformers

## 📌 Problem Statement
Software vulnerabilities in source code pose severe cybersecurity threats, leading to unauthorized access, remote code execution (RCE), data breaches, and system compromise. Traditional static application security testing (SAST) tools rely heavily on rigid pattern matching, manual regex, and rule-based heuristics that struggle to understand context and often produce high false positive or false negative rates.

The objective of this project is to develop an **AI-driven Source Code Vulnerability Detection System** capable of analyzing raw source code snippets across multiple programming languages (C, C++, Java, Python, JavaScript, etc.) and classifying whether code contains security flaws. By pairing **syntactic & lexical structural feature extraction (AST + XGBoost)** with **deep contextual semantic representations (Pre-Trained Transformer - CodeBERT)** into an **Ensemble Union Model**, the system maximizes vulnerability detection recall (≥85%), ensuring safety-critical security flaws are not missed.

---

## 📊 Dataset Information
The system trains and evaluates on curated source code vulnerability benchmarks containing real-world vulnerable and safe code implementations.

*   **Raw Source Data**: `data/vulnerabilities.csv` (~657 MB raw corpus including commit metadata, source code, and binary vulnerability flags).
*   **Preprocessing Pipeline (`src/data cleaning.py`)**:
    *   Missing and null value removal.
    *   Deduplication of identical source code blocks.
    *   Metadata stripping (`commit_id`, `hash`, `project`, `message`).
    *   Size-balanced sampling to avoid length bias between vulnerable (`target=1`) and non-vulnerable (`target=0`) snippets.
    *   Comment stripping to ensure models learn actual logic rather than comment cues.
*   **Cleaned Corpus**: `data/cleaned_data_of_vulnerabilities.csv` (~83.5 MB).
*   **Dataset Splits (`src/data_split.py`)**:
    *   **Training Set (`data/train_data.csv`)**: 25,752 samples (~50.1 MB) used for training XGBoost and fine-tuning CodeBERT.
    *   **Test Set (`data/test_data.csv`)**: 6,439 unseen samples (~16.5 MB) used for benchmark evaluation.

---

## 🛠️ Setup Instructions

### Prerequisites
*   **Operating System**: Windows 10/11 or Linux
*   **Python**: Version 3.11 or 3.12 (Virtual Environment recommended)
*   **Hardware**: NVIDIA GPU with CUDA 12.x support recommended for CodeBERT (CPU fallback supported)
*   **C++ Build Tools**: Microsoft C++ Desktop Development Build Tools (required for tree-sitter/pygments compilation)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/4sTr0J/NLP_Group_02.git
   cd NLP_Group_02
   ```

2. **Create & Activate a Python Virtual Environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r src/Requirements.txt
   ```

---

## 🚀 How to Run the Project

### 1. Interactive Source Code Vulnerability Scanner
To test custom code snippets interactively in your terminal:
```powershell
python src/predict.py
```
> **Usage**: Paste your source code into the shell, type `END` on a new line, and hit `Enter`. Type `EXIT` to quit.

### 2. Full Test Dataset Evaluation
To evaluate all models on the complete 6,439 test samples and output confusion matrices & metrics:
```powershell
python src/evaluate_ensemble.py
```

### 3. Testing Unseen Synthetic Samples
To run the automated test suite across unseen C, C++, and Python cases:
```powershell
python src/test_unseen.py
```

### 4. Retraining the Models
*   **Train XGBoost Pipeline**:
    ```powershell
    python src/xgboost_train.py
    ```
*   **Fine-Tune CodeBERT (GPU accelerated)**:
    ```powershell
    python src/bert-based_train.py
    ```

---

## 🧠 Model Summary

The project implements a multi-tier hybrid architecture combining traditional Machine Learning and Deep Learning Transformers:

```
                          Source Code Input
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
   [AST & Lexical Feature Extractor]      [CodeBERT Tokenizer]
   - Tree-Sitter AST Nodes (50+ types)    - Max token length: 512
   - Lexical & Statistical metrics        - Subword BPE
   - Control Flow & Complexity indicators          │
                 │                                 ▼
                 ▼                      [microsoft/codebert-base]
        [XGBoost Classifier]             Fine-Tuned Transformer
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                     [Ensemble Union Predictor]
         If XGBoost = 1 OR CodeBERT = 1 ──> VULNERABLE (High Recall)
```

1. **Feature Engineering & XGBoost (`src/feature_engineering.py` & `src/xgboost_train.py`)**:
   - **AST Analysis**: Extracts 50+ language-agnostic AST node types across C, C++, Java, Python, and JavaScript via `tree-sitter-language-pack` (with caching and error recovery).
   - **Lexical & Statistical Indicators**: Line length statistics, comment ratios, identifier entropy, character counts, and keyword frequencies.
   - **Classifier**: Tuned XGBoost with gradient-boosted decision trees for tabular AST features.

2. **Pre-Trained Transformer - CodeBERT (`src/bert-based_train.py`)**:
   - Fine-tuned `microsoft/codebert-base` (125M parameters) for sequence classification.
   - Captures contextual token representations and semantic code structures across a 512-token context window.

3. **Ensemble Union Decision Logic**:
   - In safety-critical vulnerability detection, a False Negative (missing a vulnerability) is far more dangerous than a False Positive (requiring manual review).
   - The Ensemble adopts a **Union Strategy**: if either model flags the code as vulnerable, it is flagged for security inspection, achieving an optimal recall rate.

---

## 📈 Results Summary

Evaluated on the full unseen test dataset (**6,439 samples**):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** (AST + Lexical) | 58.92% | 57.02% | 72.44% | 63.81% |
| **CodeBERT** (Fine-Tuned) | **65.23%** | **64.54%** | 67.57% | 66.02% |
| **Ensemble (Union Strategy)** | **61.38%** | **57.72%** | **85.00%** | **68.75%** |

### Confusion Matrices on Test Set (6,439 Samples)

#### 1. XGBoost
```
                Predicted Safe (0)   Predicted Vulnerable (1)
Actual Safe (0)        1462 (TN)              1758 (FP)
Actual Vuln (1)         887 (FN)              2332 (TP)
```

#### 2. CodeBERT
```
                Predicted Safe (0)   Predicted Vulnerable (1)
Actual Safe (0)        2025 (TN)              1195 (FP)
Actual Vuln (1)        1044 (FN)              2175 (TP)
```

#### 3. Ensemble (Union)
```
                Predicted Safe (0)   Predicted Vulnerable (1)
Actual Safe (0)        1216 (TN)              2004 (FP)
Actual Vuln (1)         483 (FN)              2736 (TP)   <-- 85.00% Vulnerability Recall
```

---

## 📁 Repository Structure
```
NLP_Group_02/
├── data/
│   ├── vulnerabilities.csv                 # Raw dataset
│   ├── cleaned_data_of_vulnerabilities.csv # Preprocessed & balanced dataset
│   ├── train_data.csv                      # Training split (25,752 rows)
│   └── test_data.csv                       # Test split (6,439 rows)
├── models/
│   ├── xgboost_model.pkl                   # Trained XGBoost model & feature extractor
│   └── codebert/                           # Fine-tuned CodeBERT weights & tokenizer
├── src/
│   ├── data cleaning.py                    # Data preprocessing & balancing
│   ├── data_split.py                       # Train/test stratification
│   ├── feature_engineering.py              # AST & lexical extraction pipeline
│   ├── xgboost_train.py                    # XGBoost training script
│   ├── bert-based_train.py                 # CodeBERT GPU fine-tuning script
│   ├── evaluate_ensemble.py                # Full test benchmark & evaluation
│   ├── test_unseen.py                      # Unseen synthetic sample testing
│   ├── predict.py                          # Interactive scanning CLI
│   └── Requirements.txt                    # Project Python package requirements
└── README.md                               # Project documentation
```
