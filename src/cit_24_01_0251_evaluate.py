"""Evaluate Ravishka's SVM, LSTM, and combined vulnerability detector.

Student: Ravishka Rathnayake
Student ID: CIT-24-01-0251
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from keras.utils import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .cit_24_01_0251_svm_pipeline import load_svm_bundle
    from .cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        prepare_code_for_lstm,
    )
except ImportError:
    from cit_24_01_0251_svm_pipeline import load_svm_bundle
    from cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        prepare_code_for_lstm,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diversevul_test_1500.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "CIT-24-01-0251_combined_evaluation.json"
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def main() -> None:
    """Evaluate the trained models on the held-out test set."""

    print("\n" + "=" * 70)
    print("CIT-24-01-0251 - HELD-OUT MODEL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load held-out test dataset
    # ---------------------------------------------------------

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    test_df = pd.read_csv(
        TEST_DATA_PATH,
        usecols=["func", "target"],
    )

    test_df = test_df.dropna(
        subset=["func", "target"]
    ).copy()

    test_df["func"] = test_df["func"].astype(str)
    test_df["target"] = test_df["target"].astype(int)

    if len(test_df) != 1500:
        raise ValueError(
            f"Expected 1500 test records, found {len(test_df)}."
        )

    print("\nTest dataset loaded successfully.")
    print("Records:", len(test_df))

    print("\nClass distribution:")
    print(
        test_df["target"]
        .value_counts()
        .sort_index()
    )

    source_code = test_df["func"].tolist()

    y_true = test_df["target"].to_numpy(
        dtype=np.int32
    )

    # ---------------------------------------------------------
    # Load trained models
    # ---------------------------------------------------------

    print("\nLoading trained SVM model...")
    svm_bundle = load_svm_bundle()

    print("Loading trained LSTM model...")
    lstm_model, lstm_preprocessing = (
        load_lstm_artifacts()
    )

    print("Both models loaded successfully.")

    # ---------------------------------------------------------
    # Batch SVM evaluation
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("Evaluating SVM")
    print("-" * 70)

    svm_vectorizer = svm_bundle["vectorizer"]
    svm_model = svm_bundle["model"]

    svm_threshold = float(
        svm_bundle["decision_threshold"]
    )

    svm_features = svm_vectorizer.transform(
        source_code
    )

    svm_scores = svm_model.decision_function(
        svm_features
    )

    svm_predictions = (
        svm_scores >= svm_threshold
    ).astype(np.int32)

    svm_metrics = calculate_metrics(
        y_true,
        svm_predictions,
    )

    svm_metrics["roc_auc"] = float(
        roc_auc_score(
            y_true,
            svm_scores,
        )
    )

    svm_metrics["average_precision"] = float(
        average_precision_score(
            y_true,
            svm_scores,
        )
    )

    print("\nSVM results:")

    for metric, value in svm_metrics.items():
        print(
            f"{metric:20s}: {value:.4f}"
        )

    # ---------------------------------------------------------
    # Batch LSTM evaluation
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("Evaluating LSTM")
    print("-" * 70)

    tokenizer = lstm_preprocessing["tokenizer"]

    maximum_sequence_length = int(
        lstm_preprocessing[
            "maximum_sequence_length"
        ]
    )

    lstm_threshold = float(
        lstm_preprocessing[
            "decision_threshold"
        ]
    )

    print("Preparing source-code sequences...")

    prepared_code = [
        prepare_code_for_lstm(code)
        for code in source_code
    ]

    sequences = tokenizer.texts_to_sequences(
        prepared_code
    )

    padded_sequences = pad_sequences(
        sequences,
        maxlen=maximum_sequence_length,
        padding="post",
        truncating="post",
        dtype="int32",
    )

    print(
        "Running LSTM batch prediction..."
    )

    lstm_probabilities = (
        lstm_model.predict(
            padded_sequences,
            batch_size=64,
            verbose=1,
        )
        .ravel()
    )

    lstm_predictions = (
        lstm_probabilities >= lstm_threshold
    ).astype(np.int32)

    lstm_metrics = calculate_metrics(
        y_true,
        lstm_predictions,
    )

    lstm_metrics["roc_auc"] = float(
        roc_auc_score(
            y_true,
            lstm_probabilities,
        )
    )

    lstm_metrics["average_precision"] = float(
        average_precision_score(
            y_true,
            lstm_probabilities,
        )
    )

    print("\nLSTM results:")

    for metric, value in lstm_metrics.items():
        print(
            f"{metric:20s}: {value:.4f}"
        )

    # ---------------------------------------------------------
    # Combined system evaluation
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("Evaluating Combined SVM + LSTM System")
    print("-" * 70)

    models_agree = (
        svm_predictions == lstm_predictions
    )

    automatic_mask = models_agree

    review_mask = ~models_agree

    automatic_count = int(
        automatic_mask.sum()
    )

    review_count = int(
        review_mask.sum()
    )

    agreement_rate = float(
        models_agree.mean()
    )

    coverage = float(
        automatic_count / len(y_true)
    )

    automatic_y_true = y_true[
        automatic_mask
    ]

    automatic_predictions = (
        svm_predictions[automatic_mask]
    )

    if automatic_count > 0:
        automatic_metrics = calculate_metrics(
            automatic_y_true,
            automatic_predictions,
        )
    else:
        automatic_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    correct_automatic = int(
        (
            automatic_predictions
            == automatic_y_true
        ).sum()
    )

    incorrect_automatic = (
        automatic_count
        - correct_automatic
    )

    combined_results = {
        "total_test_records": int(
            len(y_true)
        ),
        "models_agree": automatic_count,
        "agreement_rate": agreement_rate,
        "automatic_decisions": automatic_count,
        "review_required": review_count,
        "coverage": coverage,
        "correct_automatic_decisions": (
            correct_automatic
        ),
        "incorrect_automatic_decisions": (
            incorrect_automatic
        ),
        "automatic_decision_metrics": (
            automatic_metrics
        ),
    }

    print(
        "\nTotal test records:",
        len(y_true),
    )

    print(
        "Models agreed:",
        automatic_count,
    )

    print(
        "Models disagreed / review required:",
        review_count,
    )

    print(
        "Agreement rate:",
        f"{agreement_rate:.4f}",
    )

    print(
        "Automatic-decision coverage:",
        f"{coverage:.4f}",
    )

    print(
        "Correct automatic decisions:",
        correct_automatic,
    )

    print(
        "Incorrect automatic decisions:",
        incorrect_automatic,
    )

    print(
        "\nAutomatic-decision metrics:"
    )

    for (
        metric,
        value,
    ) in automatic_metrics.items():
        print(
            f"{metric:20s}: {value:.4f}"
        )

    # ---------------------------------------------------------
    # Save evaluation report
    # ---------------------------------------------------------

    complete_report = {
        "student_id": "CIT-24-01-0251",
        "student_name": "Ravishka Rathnayake",
        "dataset": {
            "file": str(
                TEST_DATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "records": int(
                len(y_true)
            ),
            "non_vulnerable_records": int(
                (y_true == 0).sum()
            ),
            "vulnerable_records": int(
                (y_true == 1).sum()
            ),
        },
        "svm": svm_metrics,
        "lstm": lstm_metrics,
        "combined_system": combined_results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            complete_report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_PATH.relative_to(
            PROJECT_ROOT
        )
    )


if __name__ == "__main__":
    main()