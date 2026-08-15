from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cit_24_01_0251_svm_pipeline import load_svm_bundle
from cit_24_01_0251_lstm_pipeline import (
    load_lstm_artifacts,
    prepare_code_for_lstm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "common"
    / "common_test.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "CIT-24-01-0251_common_test_results.json"
)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict:
    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
        "confusion_matrix": (
            confusion_matrix(
                labels,
                predictions,
            ).tolist()
        ),
    }


def main() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Common test file not found: {TEST_FILE}"
        )

    test_data = pd.read_csv(TEST_FILE)

    required_columns = {"func", "target"}

    if not required_columns.issubset(test_data.columns):
        raise ValueError(
            "The common test file must contain "
            "'func' and 'target' columns."
        )

    test_data = test_data.dropna(
        subset=["func", "target"]
    ).copy()

    source_code = (
        test_data["func"]
        .astype(str)
        .tolist()
    )

    labels = test_data["target"].to_numpy(
        dtype=np.int32
    )

    print("Common test file loaded.")
    print("Records:", len(test_data))
    print("\nClass distribution:")
    print(
        test_data["target"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # SVM evaluation
    # -----------------------------------------------------

    print("\nLoading final SVM...")
    svm_bundle = load_svm_bundle()

    token_features = svm_bundle[
        "token_vectorizer"
    ].transform(source_code)

    char_features = svm_bundle[
        "char_vectorizer"
    ].transform(source_code)

    svm_features = hstack(
        [
            token_features
            * float(svm_bundle["token_weight"]),
            char_features
            * float(svm_bundle["char_weight"]),
        ],
        format="csr",
    )

    svm_scores = svm_bundle[
        "model"
    ].decision_function(svm_features)

    svm_threshold = float(
        svm_bundle["decision_threshold"]
    )

    svm_predictions = (
        svm_scores >= svm_threshold
    ).astype(np.int32)

    svm_metrics = calculate_metrics(
        labels,
        svm_predictions,
        svm_scores,
    )

    # -----------------------------------------------------
    # LSTM evaluation
    # -----------------------------------------------------

    print("\nLoading final LSTM...")
    lstm_model, lstm_bundle = (
        load_lstm_artifacts()
    )

    tokenizer = lstm_bundle["tokenizer"]

    maximum_sequence_length = int(
        lstm_bundle.get(
            "max_sequence_length",
            lstm_bundle.get(
                "maximum_sequence_length"
            ),
        )
    )

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

    print("Running LSTM predictions...")

    lstm_probabilities = (
        lstm_model.predict(
            padded_sequences,
            batch_size=64,
            verbose=1,
        )
        .ravel()
    )

    lstm_threshold = float(
        lstm_bundle["decision_threshold"]
    )

    lstm_predictions = (
        lstm_probabilities >= lstm_threshold
    ).astype(np.int32)

    lstm_metrics = calculate_metrics(
        labels,
        lstm_predictions,
        lstm_probabilities,
    )

    results = {
        "student_id": "CIT-24-01-0251",
        "common_test_records": int(
            len(test_data)
        ),
        "svm": svm_metrics,
        "lstm": lstm_metrics,
    }

    print("\n" + "=" * 60)
    print("COMMON TEST RESULTS")
    print("=" * 60)

    for model_name, metrics in [
        ("SVM", svm_metrics),
        ("LSTM", lstm_metrics),
    ]:
        print(f"\n{model_name}")
        print("-" * 30)

        print(
            f"Accuracy:          "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"Precision:         "
            f"{metrics['precision']:.4f}"
        )
        print(
            f"Recall:            "
            f"{metrics['recall']:.4f}"
        )
        print(
            f"F1-score:          "
            f"{metrics['f1_score']:.4f}"
        )
        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.4f}"
        )
        print(
            f"ROC-AUC:           "
            f"{metrics['roc_auc']:.4f}"
        )
        print(
            "Confusion Matrix:",
            metrics["confusion_matrix"],
        )

    REPORT_FILE.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\nReport saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()