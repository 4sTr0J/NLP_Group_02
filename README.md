# NLP-Based Source Code Vulnerability Detection

## 1. Project Overview

This project focuses on detecting potential vulnerabilities in source code using **Natural Language Processing (NLP)** and Machine Learning techniques.

The system processes source-code functions, extracts meaningful representations from the code, and classifies them into vulnerable and non-vulnerable categories.

Two different models were implemented and compared:

* **Random Forest** — Machine Learning approach using TF-IDF features.
* **Convolutional Neural Network (CNN)** — Deep Learning approach using tokenized and padded source-code sequences.

The objective is to determine which model provides better performance for source-code vulnerability detection.

---

## 2. Objectives

The main objectives of this project are:

* Process and clean source-code vulnerability data.
* Split the dataset into training and testing sets.
* Tokenize source-code functions into meaningful tokens.
* Extract numerical features suitable for different machine learning models.
* Train a Random Forest classifier using TF-IDF features.
* Train a CNN using token sequences.
* Evaluate both models using standard classification metrics.
* Compare the performance of Random Forest and CNN.
* Select the best-performing model for vulnerability detection.

---

## 3. Dataset

The project uses the `vulnerabilities_50k.csv` dataset.

The main columns used are:

| Column   | Description                        |
| -------- | ---------------------------------- |
| `func`   | Source-code function               |
| `target` | Vulnerability classification label |

The dataset contains source-code functions labelled according to whether they contain a vulnerability.

### Dataset Split

The dataset was divided using an **80/20 train-test split**:

* **80%** → Training data
* **20%** → Testing data

The split uses stratification to maintain the class distribution between the training and testing datasets.

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True
)
```

---

## 4. NLP Pipeline

The implemented NLP pipeline consists of the following stages:

```text
Dataset
   ↓
Data Cleaning
   ↓
Train-Test Split
   ↓
Tokenization
   ↓
Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
```

### 4.1 Data Cleaning

The source-code data is cleaned and preprocessed before being passed to the machine learning models.

The purpose of this stage is to remove unnecessary information and prepare the source code for tokenization and feature extraction.

### 4.2 Train-Test Split

The cleaned dataset is divided into:

* 80% training data
* 20% testing data

The test data is kept separate so that the trained models can be evaluated on previously unseen source code.

### 4.3 Tokenization

Source-code functions are converted into sequences of programming tokens.

Examples of tokens include:

```text
int
main
(
)
{
return
0
;
}
```

Tokenization allows the source code to be represented as structured sequences that can be processed by machine learning and deep learning models.

### 4.4 Feature Extraction

Different feature extraction approaches are used for the two models.

#### Random Forest

Random Forest uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert tokenized source code into numerical feature vectors.

```text
Tokenized Code
      ↓
TF-IDF
      ↓
Numerical Feature Vectors
      ↓
Random Forest
```

The TF-IDF vectorizer is fitted using the training data and then used to transform both training and testing data.

#### CNN

The CNN uses a different representation because CNNs can learn patterns from sequential input.

```text
Tokenized Code
      ↓
Tokenizer
      ↓
Integer Sequences
      ↓
Padding
      ↓
CNN
```

The tokenizer converts tokens into integer IDs, and padding ensures that all source-code sequences have the same length.

---

# 5. Individual Model Implementations

## 5.1 Random Forest

### Model Explanation

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to perform classification.

For this project, Random Forest was selected because it can work effectively with high-dimensional numerical features such as TF-IDF vectors and provides a relatively efficient classification approach.

### Architecture / Algorithm

```text
Tokenized Source Code
        ↓
TF-IDF Vectorization
        ↓
Multiple Decision Trees
        ↓
Voting / Ensemble
        ↓
Vulnerability Classification
```

### Training Process

The Random Forest model is trained using the TF-IDF features generated from the training dataset.

The saved TF-IDF vectorizer is then used to transform the test dataset before generating predictions.

The trained model is saved as:

```text
models/random_forest.pkl
```

### Hyperparameters

| Hyperparameter        |  Value |
| --------------------- | -----: |
| Number of Trees       |     50 |
| Maximum Depth         |     30 |
| Minimum Samples Split |      5 |
| Minimum Samples Leaf  |      2 |
| Maximum Features      | `sqrt` |
| Random State          |     42 |
| CPU Jobs              |   `-1` |

---

## 5.2 Convolutional Neural Network (CNN)

### Model Explanation

A Convolutional Neural Network is a deep learning model capable of learning local patterns from sequential data.

CNN was selected because source code contains meaningful local patterns formed by combinations of keywords, identifiers, operators, and function calls.

### Architecture

The CNN consists of:

1. Embedding layer
2. Conv1D layer
3. Global Max Pooling layer
4. Dense layer
5. Sigmoid output layer

```text
Tokenized Source Code
        ↓
Integer Sequences
        ↓
Padding
        ↓
Embedding
        ↓
Conv1D
        ↓
Global Max Pooling
        ↓
Dense Layer
        ↓
Sigmoid Output
        ↓
Vulnerability Classification
```

### Training Process

The tokenized source code is converted into integer sequences using the tokenizer.

The sequences are padded to a fixed length of **300 tokens**.

The CNN is then trained using the training dataset.

The model uses:

* Adam optimizer
* Binary cross-entropy loss
* 10 training epochs
* Batch size of 32
* 10% validation split

The trained CNN model is saved as:

```text
models/cnn_model.keras
```

### Hyperparameters

| Hyperparameter          |                Value |
| ----------------------- | -------------------: |
| Maximum Sequence Length |                  300 |
| Embedding Dimension     |                  128 |
| Conv1D Filters          |                  128 |
| Kernel Size             |                    5 |
| Activation              |                 ReLU |
| Dense Units             |                   64 |
| Output Activation       |              Sigmoid |
| Optimizer               |                 Adam |
| Loss                    | Binary Cross-Entropy |
| Epochs                  |                   10 |
| Batch Size              |                   32 |
| Validation Split        |                  10% |

---

# 6. Model Evaluation

Both models are evaluated using the same **20% test dataset**.

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

## Evaluation Results

| Model         |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------- | ---------: | ---------: | ---------: | ---------: |
| Random Forest |     74.51% |     67.41% | **67.66%** |     67.53% |
| CNN           | **77.48%** | **74.89%** |     63.97% | **69.00%** |

---

# 7. Model Comparison

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

### Final Model Selection

Based on the evaluation results, **CNN was selected as the final model**.

CNN achieved:

* The highest accuracy: **77.48%**
* The highest precision: **74.89%**
* The highest F1-score: **69.00%**

Although Random Forest achieved a higher recall, CNN provided better overall performance across the majority of the evaluation metrics.

---

# 8. Project Structure

```text
NLP_Group_02-features-CIT-24-01-0249-Random-Forest+CNN/
│
├── data/
│   ├── vulnerabilities_50k.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_tokenized.csv
│   └── test_tokenized.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   ├── random_forest.pkl
│   ├── cnn_model.keras
│   └── model_comparison.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── data_split.py
│   ├── tokenization.py
│   ├── feature_extraction_rf.py
│   ├── feature_extraction_cnn.py
│   ├── random_forest.py
│   ├── cnn.py
│   └── evaluation.py
│
└── README.md
```

---

# 9. Installation

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

# 10. Running the Project

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

# 11. Performance Visualization

The model comparison results can be visualized using a Jupyter Notebook.

The notebook generates graphs for:

* Accuracy
* Precision
* Recall
* F1-Score
* Overall Random Forest vs CNN comparison

The graphs are based on:

```text
models/model_comparison.csv
```

The visualization makes it easier to compare the strengths and weaknesses of each model.

---

# 12. Generated Model Files

| File                   | Purpose                                |
| ---------------------- | -------------------------------------- |
| `tfidf_vectorizer.pkl` | Stores the fitted TF-IDF vectorizer    |
| `tokenizer.pkl`        | Stores the fitted CNN tokenizer        |
| `random_forest.pkl`    | Stores the trained Random Forest model |
| `cnn_model.keras`      | Stores the trained CNN model           |
| `model_comparison.csv` | Stores evaluation results              |

---

# 13. Key Findings

The experimental results demonstrate that both machine learning approaches can be used for source-code vulnerability detection.

Random Forest achieved a **74.51% accuracy**, while CNN achieved a higher **77.48% accuracy**.

CNN also achieved better precision and F1-score, suggesting that it produced more accurate overall vulnerability classifications. Random Forest, however, achieved better recall, indicating stronger performance in identifying actual vulnerable samples.

Overall, **CNN was selected as the final model** because it provided the best balance of accuracy, precision, and F1-score among the evaluated models.

---

# 14. Conclusion

This project implemented a complete NLP-based pipeline for source-code vulnerability detection.

The pipeline included data cleaning, train-test splitting, tokenization, model-specific feature extraction, model training, and evaluation.

Two different approaches were implemented:

* **Random Forest with TF-IDF features**
* **CNN with tokenized and padded source-code sequences**

Based on the experimental results, the CNN model achieved the best overall performance with **77.48% accuracy** and **69.00% F1-score**.

The results demonstrate that deep learning approaches using sequential source-code representations can provide promising results for automated vulnerability detection.

