"""Combined SVM and LSTM vulnerability-prediction pipeline.

Student: Ravishka Rathnayake
Student ID: CIT-24-01-0251
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Support both direct execution and package-based imports.
try:
    from .cit_24_01_0251_svm_pipeline import (
        load_svm_bundle,
        predict_vulnerability as predict_with_svm,
        read_source_code,
    )
    from .cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        predict_vulnerability as predict_with_lstm,
    )
except ImportError:
    from cit_24_01_0251_svm_pipeline import (
        load_svm_bundle,
        predict_vulnerability as predict_with_svm,
        read_source_code,
    )
    from cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        predict_vulnerability as predict_with_lstm,
    )


def compare_predictions(
    svm_result: dict[str, Any],
    lstm_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare the predictions returned by both models."""

    svm_class = int(svm_result["prediction"])
    lstm_class = int(lstm_result["prediction"])

    models_agree = svm_class == lstm_class

    if models_agree:
        comparison_message = (
            "Both models produced the same prediction."
        )
        agreed_label = svm_result["label"]
    else:
        comparison_message = (
            "The models produced different predictions. "
            "This result should be reviewed instead of being "
            "treated as a final ensemble decision."
        )
        agreed_label = None

    return {
        "models_agree": models_agree,
        "agreed_label": agreed_label,
        "vulnerable_votes": svm_class + lstm_class,
        "message": comparison_message,
    }


def run_combined_pipeline(
    source_code: str,
) -> dict[str, Any]:
    """Run the SVM and LSTM pipelines on the same source code."""

    print("Loading SVM artifacts...")

    svm_bundle = load_svm_bundle()

    print("Loading LSTM artifacts...")

    lstm_model, lstm_preprocessing_bundle = (
        load_lstm_artifacts()
    )

    print("Generating SVM prediction...")

    svm_result = predict_with_svm(
        source_code=source_code,
        bundle=svm_bundle,
    )

    print("Generating LSTM prediction...")

    lstm_result = predict_with_lstm(
        source_code=source_code,
        model=lstm_model,
        preprocessing_bundle=lstm_preprocessing_bundle,
    )

    comparison = compare_predictions(
        svm_result=svm_result,
        lstm_result=lstm_result,
    )

    return {
        "student_id": "CIT-24-01-0251",
        "pipeline": "Combined SVM and LSTM comparison",
        "svm_result": svm_result,
        "lstm_result": lstm_result,
        "comparison": comparison,
    }


def main() -> None:
    """Run the combined command-line pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the trained SVM and LSTM vulnerability "
            "models on the same source-code function."
        )
    )

    parser.add_argument(
        "--code",
        type=str,
        help="Source code supplied directly as text.",
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a file containing source code.",
    )

    arguments = parser.parse_args()

    try:
        source_code = read_source_code(
            code_argument=arguments.code,
            file_argument=arguments.file,
        )

        result = run_combined_pipeline(
            source_code
        )

        print(
            "\nCombined vulnerability prediction "
            "completed successfully.\n"
        )

        print(
            json.dumps(
                result,
                indent=4,
            )
        )

    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        print(f"\nPipeline error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()