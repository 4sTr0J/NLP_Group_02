from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diversevul_stratified_10000.csv"
)


# ---------------------------------------------------------
# Load and validate dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

df = (
    df.dropna(subset=["func", "target"])
    .copy()
)

df["func"] = df["func"].astype(str)
df["target"] = df["target"].astype(int)

assert len(df) == 10_000
assert set(df["target"].unique()) == {0, 1}

X = df["func"]
y = df["target"]


# ---------------------------------------------------------
# Reproduce baseline 70 / 15 / 15 split
# ---------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp,
)


# ---------------------------------------------------------
# Verify split
# ---------------------------------------------------------

assert len(X_train) == 7000
assert len(X_validation) == 1500
assert len(X_test) == 1500

print("Character n-gram SVM experiment")
print("=" * 45)

print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_validation)}")
print(f"Testing records:    {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_validation.value_counts().sort_index())

print("\nTest set reserved and untouched.")
print("Dataset preparation completed successfully.")

# ---------------------------------------------------------
# Character n-gram TF-IDF
# ---------------------------------------------------------

print("\nBuilding character n-gram TF-IDF features...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    dtype=np.float32,
)

# Fit ONLY on training data.
X_train_char = char_vectorizer.fit_transform(X_train)

# Validation data is transformed using the training vocabulary.
X_validation_char = char_vectorizer.transform(X_validation)

print("Character TF-IDF completed successfully.")

print("\nFeature matrix shapes:")
print("Training:", X_train_char.shape)
print("Validation:", X_validation_char.shape)

print(f"\nVocabulary size: {len(char_vectorizer.vocabulary_):,}")

assert X_train_char.shape[0] == 7000
assert X_validation_char.shape[0] == 1500
assert X_train_char.shape[1] <= 50_000
assert X_validation_char.shape[1] == X_train_char.shape[1]

print("\nFeature extraction validation completed successfully.")

# ---------------------------------------------------------
# Character n-gram SVM hyperparameter tuning
# ---------------------------------------------------------

print("\nTraining character n-gram SVM candidates...")

C_VALUES = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
]

tuning_results = []
candidate_models = {}

for c_value in C_VALUES:

    print(f"Training C = {c_value}...")

    candidate_model = LinearSVC(
        C=c_value,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=10_000,
    )

    candidate_model.fit(
        X_train_char,
        y_train,
    )

    validation_predictions = candidate_model.predict(
        X_validation_char
    )

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    tuning_results.append(
        {
            "C": c_value,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }
    )

    candidate_models[c_value] = candidate_model


# ---------------------------------------------------------
# Select best C using validation F1
# ---------------------------------------------------------

tuning_results_df = (
    pd.DataFrame(tuning_results)
    .sort_values(
        by="F1-score",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nCharacter SVM validation results:")
print(
    tuning_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

best_c = tuning_results_df.loc[0, "C"]
best_validation_f1 = tuning_results_df.loc[0, "F1-score"]

print(f"\nBest C value: {best_c}")
print(
    f"Best validation F1-score: "
    f"{best_validation_f1:.4f}"
)

print("\nTest set has NOT been evaluated.")

# ---------------------------------------------------------
# Validation-only decision threshold tuning
# ---------------------------------------------------------

print("\nTuning decision threshold using validation data...")

best_char_model = candidate_models[best_c]

validation_scores = best_char_model.decision_function(
    X_validation_char
)

precision_values, recall_values, threshold_values = (
    precision_recall_curve(
        y_validation,
        validation_scores,
    )
)

# precision_recall_curve returns one extra precision/recall
# value that has no corresponding threshold.
threshold_precision = precision_values[:-1]
threshold_recall = recall_values[:-1]

threshold_f1 = np.divide(
    2 * threshold_precision * threshold_recall,
    threshold_precision + threshold_recall,
    out=np.zeros_like(threshold_precision),
    where=(threshold_precision + threshold_recall) != 0,
)

best_threshold_index = np.argmax(threshold_f1)

best_threshold = threshold_values[
    best_threshold_index
]

best_threshold_precision = threshold_precision[
    best_threshold_index
]

best_threshold_recall = threshold_recall[
    best_threshold_index
]

best_threshold_f1 = threshold_f1[
    best_threshold_index
]


# ---------------------------------------------------------
# Apply selected threshold to validation set
# ---------------------------------------------------------

optimized_validation_predictions = (
    validation_scores >= best_threshold
).astype(int)

validation_average_precision = (
    average_precision_score(
        y_validation,
        validation_scores,
    )
)

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_scores,
)

validation_confusion_matrix = confusion_matrix(
    y_validation,
    optimized_validation_predictions,
)


# ---------------------------------------------------------
# Report validation results
# ---------------------------------------------------------

print("\nOptimized character SVM validation results")
print("=" * 50)

print(f"Selected C: {best_c}")
print(
    f"Selected decision threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Precision: "
    f"{best_threshold_precision:.4f}"
)

print(
    f"Recall:    "
    f"{best_threshold_recall:.4f}"
)

print(
    f"F1-score:  "
    f"{best_threshold_f1:.4f}"
)

print(
    f"Average precision: "
    f"{validation_average_precision:.4f}"
)

print(
    f"ROC-AUC: "
    f"{validation_roc_auc:.4f}"
)

print("\nConfusion matrix:")
print(validation_confusion_matrix)

print("\nOriginal token-SVM validation F1: 0.2359")
print(
    "Character-SVM optimized validation F1: "
    f"{best_threshold_f1:.4f}"
)

print("\nTest set remains untouched.")