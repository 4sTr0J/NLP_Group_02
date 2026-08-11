# Source Code Vulnerability Detection (Ensemble Model)

An advanced source code vulnerability detection application built using an ensemble architecture combining a traditional machine learning classifier (**XGBoost**) and a state-of-the-art pre-trained deep learning transformer (**CodeBERT**).

## 🚀 Key Features
*   **AST & Lexical Feature Engineering**: Extracts lexical, statistical, and structural features from source code to train the XGBoost classifier.
*   **Transformers (CodeBERT)**: Fine-tuned `microsoft/codebert-base` on source code scripts to capture deep semantic dependencies and patterns.
*   **Ensemble Union Model**: Combines both models using a Union strategy to maximize vulnerability detection recall (~84.59%), minimizing false negatives in critical codebases.
*   **Interactive Scanning Shell**: Paste source code directly to classify vulnerabilities with confidence scores in real time.

---

## 📁 Project Structure

```
NLP_Group_02/
├── data/
│   ├── train_data.csv                       # Combined training datasets
│   ├── test_data.csv                        # Evaluation dataset
│   └── cleaned_data_of_vulnerabilities.csv  # Preprocessed corpus
├── models/
│   ├── xgboost_model.pkl                    # Serialized XGBoost model & feature extractor
│   └── codebert/                            # Fine-tuned CodeBERT model weights & config
├── src/
│   ├── bert-based_train.py                  # CodeBERT fine-tuning pipeline (GPU-enabled)
│   ├── xgboost_train.py                     # XGBoost training pipeline
│   ├── feature_engineering.py               # AST and lexical feature extraction logic
│   ├── evaluate_ensemble.py                 # Multi-model evaluation script
│   ├── expand_dataset.py                    # Script to dynamically append diverse safe code
│   └── predict.py                           # Interactive CLI code scanner
├── Requirements.txt                         # Python packages list
└── README.md                                # Project documentation
```

---

## 🛠️ Environment Setup

### Prerequisites
*   Windows / Linux OS
*   Python 3.12 (Supported CUDA virtual environment)
*   **NVIDIA CUDA 12.4 Toolkit** (optional, recommended for GPU CodeBERT training)
*   Microsoft C++ Desktop Development Build Tools (required for Pygments/Word2Vec compilation)

### Installation
Run the following command to install dependencies inside your virtual environment:

```powershell
pip install -r src/Requirements.txt
```

---

## 📊 Model Performance (Full Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 58.63% | 58.98% | 56.60% | 57.77% |
| **CodeBERT** | 64.90% | 62.11% | 76.42% | 68.52% |
| **Ensemble (Union)** | **62.21%** | **58.43%** | **84.59%** | **69.12%** |

---

## 🖥️ Usage

### 1. Run Interactive Vulnerability Scanner
To test snippets or source files interactively, execute the prediction script:

```powershell
.\python_3.12_cuda\python.exe src/predict.py
```
*Paste your source code in the terminal, type `END` on a new line, and hit Enter to get prediction results.*

### 2. Run Test Set Evaluation
To run the full evaluation suite and print performance metrics:

```powershell
.\python_3.12_cuda\python.exe src/evaluate_ensemble.py
```

### 3. Re-train Models
To re-train XGBoost:
```powershell
.\python_3.12_cuda\python.exe src/xgboost_train.py
```

To re-train CodeBERT (GPU accelerated):
```powershell
.\python_3.12_cuda\python.exe src/bert-based_train.py
```
