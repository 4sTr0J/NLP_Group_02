from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from cit_24_01_0251_svm_pipeline import code_tokenizer


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
# Load dataset
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

assert len(X_train) == 7000
assert len(X_validation) == 1500
assert len(X_test) == 1500

print("Hybrid token + character SVM experiment")
print("=" * 50)

print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_validation)}")
print(f"Testing records:    {len(X_test)}")


# ---------------------------------------------------------
# Token-level TF-IDF
# Same representation as original SVM baseline
# ---------------------------------------------------------

print("\nBuilding token-level TF-IDF features...")

token_vectorizer = TfidfVectorizer(
    tokenizer=code_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=50_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_token = token_vectorizer.fit_transform(
    X_train
)

X_validation_token = token_vectorizer.transform(
    X_validation
)

print("Token TF-IDF completed.")


# ---------------------------------------------------------
# Character-level TF-IDF
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

X_train_char = char_vectorizer.fit_transform(
    X_train
)

X_validation_char = char_vectorizer.transform(
    X_validation
)

print("Character TF-IDF completed.")


# ---------------------------------------------------------
# Combine token + character representations
# ---------------------------------------------------------

print("\nCombining token and character features...")

X_train_hybrid = hstack(
    [
        X_train_token,
        X_train_char,
    ],
    format="csr",
)

X_validation_hybrid = hstack(
    [
        X_validation_token,
        X_validation_char,
    ],
    format="csr",
)


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\nFeature matrix shapes:")

print(
    "Token training:     ",
    X_train_token.shape,
)

print(
    "Character training: ",
    X_train_char.shape,
)

print(
    "Hybrid training:    ",
    X_train_hybrid.shape,
)

print(
    "Hybrid validation:  ",
    X_validation_hybrid.shape,
)

print(
    f"\nToken vocabulary size: "
    f"{len(token_vectorizer.vocabulary_):,}"
)

print(
    f"Character vocabulary size: "
    f"{len(char_vectorizer.vocabulary_):,}"
)

assert X_train_hybrid.shape[0] == 7000
assert X_validation_hybrid.shape[0] == 1500

assert (
    X_train_hybrid.shape[1]
    ==
    X_train_token.shape[1]
    + X_train_char.shape[1]
)

print(
    "\nHybrid feature extraction "
    "completed successfully."
)

print("Test set remains untouched.")

# ---------------------------------------------------------
# Hybrid SVM hyperparameter tuning
# ---------------------------------------------------------

print("\nTraining hybrid SVM candidates...")

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
        X_train_hybrid,
        y_train,
    )

    validation_predictions = candidate_model.predict(
        X_validation_hybrid
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

print("\nTest set has NOT been evaluated.")

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

print("\nReference results:")
print("Original token SVM optimized F1: 0.2359")
print("Character SVM optimized F1:      0.2577")

print("\nTest set has NOT been evaluated.")

# ---------------------------------------------------------
# Validation-only decision threshold tuning
# ---------------------------------------------------------

print("\nTuning hybrid decision threshold using validation data...")

best_hybrid_model = candidate_models[best_c]

validation_scores = best_hybrid_model.decision_function(
    X_validation_hybrid
)

precision_values, recall_values, threshold_values = (
    precision_recall_curve(
        y_validation,
        validation_scores,
    )
)

# precision_recall_curve returns one additional
# precision/recall point without a corresponding threshold.
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
# Apply selected threshold to validation data
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
# Report optimized validation performance
# ---------------------------------------------------------

print("\nOptimized hybrid SVM validation results")
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

print("\nReference validation F1 scores:")
print("Original token SVM: 0.2359")
print("Character SVM:      0.2577")

print(
    "Hybrid SVM:         "
    f"{best_threshold_f1:.4f}"
)

print("\nTest set remains untouched.")

# ---------------------------------------------------------
# Controlled weighted hybrid experiment
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("WEIGHTED TOKEN + CHARACTER HYBRID EXPERIMENT")
print("=" * 60)

# Keep C fixed at the best value found by the
# standard hybrid validation experiment.
WEIGHTED_EXPERIMENT_C = 0.05

# Small controlled set of relative feature weights.
#
# token_weight > char_weight  -> favour token features
# char_weight > token_weight  -> favour character features
#
# Equal weighting is included as a reference check.
WEIGHT_CONFIGURATIONS = [
    {
        "name": "Token-favoured",
        "token_weight": 1.00,
        "char_weight": 0.75,
    },
    {
        "name": "Equal",
        "token_weight": 1.00,
        "char_weight": 1.00,
    },
    {
        "name": "Slight char-favoured",
        "token_weight": 1.00,
        "char_weight": 1.25,
    },
    {
        "name": "Char-favoured",
        "token_weight": 0.75,
        "char_weight": 1.00,
    },
]

weighted_results = []


for config in WEIGHT_CONFIGURATIONS:

    name = config["name"]
    token_weight = config["token_weight"]
    char_weight = config["char_weight"]

    print(
        f"\nTesting: {name} "
        f"(token={token_weight}, char={char_weight})"
    )

    # -----------------------------------------------------
    # Apply feature-block weights
    # -----------------------------------------------------

    X_train_weighted = hstack(
        [
            X_train_token * token_weight,
            X_train_char * char_weight,
        ],
        format="csr",
    )

    X_validation_weighted = hstack(
        [
            X_validation_token * token_weight,
            X_validation_char * char_weight,
        ],
        format="csr",
    )

    # -----------------------------------------------------
    # Train SVM
    # -----------------------------------------------------

    weighted_model = LinearSVC(
        C=WEIGHTED_EXPERIMENT_C,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=10_000,
    )

    weighted_model.fit(
        X_train_weighted,
        y_train,
    )

    # -----------------------------------------------------
    # Validation decision scores
    # -----------------------------------------------------

    validation_scores = weighted_model.decision_function(
        X_validation_weighted
    )

    # -----------------------------------------------------
    # Tune threshold using VALIDATION ONLY
    # -----------------------------------------------------

    precision_values, recall_values, threshold_values = (
        precision_recall_curve(
            y_validation,
            validation_scores,
        )
    )

    threshold_precision = precision_values[:-1]
    threshold_recall = recall_values[:-1]

    threshold_f1 = np.divide(
        2 * threshold_precision * threshold_recall,
        threshold_precision + threshold_recall,
        out=np.zeros_like(threshold_precision),
        where=(
            threshold_precision + threshold_recall
        ) != 0,
    )

    best_index = np.argmax(threshold_f1)

    selected_threshold = threshold_values[
        best_index
    ]

    selected_precision = threshold_precision[
        best_index
    ]

    selected_recall = threshold_recall[
        best_index
    ]

    selected_f1 = threshold_f1[
        best_index
    ]

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

    weighted_results.append(
        {
            "Configuration": name,
            "Token weight": token_weight,
            "Char weight": char_weight,
            "Threshold": selected_threshold,
            "Precision": selected_precision,
            "Recall": selected_recall,
            "F1-score": selected_f1,
            "Average precision": validation_average_precision,
            "ROC-AUC": validation_roc_auc,
        }
    )


# ---------------------------------------------------------
# Compare weighted configurations
# ---------------------------------------------------------

weighted_results_df = (
    pd.DataFrame(weighted_results)
    .sort_values(
        by="F1-score",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nWeighted hybrid validation results:")
print(
    weighted_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

best_weighted = weighted_results_df.iloc[0]

print("\nBest weighted configuration")
print("=" * 50)

print(
    f"Configuration: "
    f"{best_weighted['Configuration']}"
)

print(
    f"Token weight: "
    f"{best_weighted['Token weight']:.2f}"
)

print(
    f"Character weight: "
    f"{best_weighted['Char weight']:.2f}"
)

print(
    f"Decision threshold: "
    f"{best_weighted['Threshold']:.4f}"
)

print(
    f"Precision: "
    f"{best_weighted['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_weighted['Recall']:.4f}"
)

print(
    f"F1-score: "
    f"{best_weighted['F1-score']:.4f}"
)

print(
    f"Average precision: "
    f"{best_weighted['Average precision']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_weighted['ROC-AUC']:.4f}"
)

print("\nCurrent validation references:")
print("Token SVM:      F1 = 0.2359")
print("Character SVM:  F1 = 0.2577")
print("Equal hybrid:   F1 = 0.2544")

print("\nTEST SET REMAINS UNTOUCHED.")