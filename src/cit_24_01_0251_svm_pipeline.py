"""Reusable SVM vulnerability-prediction pipeline.

Student: Ravishka Rathnayake
Student ID: CIT-24-01-0251
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib


# This tokenizer must match the tokenizer used when training the SVM.
CODE_TOKEN_PATTERN = re.compile(
    r"""
    0x[0-9A-Fa-f]+
    |
    [A-Za-z_]\w*
    |
    \d+(?:\.\d+)?
    |
    ==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>
    |
    [{}()\[\];,.+\-*/%&|^~!<>=?:]
    """,
    re.VERBOSE,
)


def code_tokenizer(source_code: str) -> list[str]:
    """Convert source code into programming-related tokens."""
    return CODE_TOKEN_PATTERN.findall(str(source_code))


# Compatibility support for loading the saved TF-IDF vectorizer.
# The vectorizer was originally created in a Jupyter notebook.
setattr(
    sys.modules["__main__"],
    "code_tokenizer",
    code_tokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_svm_model_bundle.joblib"
)


def load_svm_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load the SVM model, TF-IDF vectorizer and metadata."""

    if not model_path.exists():
        raise FileNotFoundError(
            "The SVM model bundle was not found:\n"
            f"{model_path}\n\n"
            "Make sure the trained model exists inside the models folder."
        )

    bundle = joblib.load(model_path)

    required_items = {
        "model",
        "vectorizer",
        "decision_threshold",
        "class_labels",
    }

    missing_items = required_items.difference(bundle)

    if missing_items:
        raise KeyError(
            "The model bundle is missing these items: "
            + ", ".join(sorted(missing_items))
        )

    return bundle


def predict_vulnerability(
    source_code: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict whether one source-code function is vulnerable."""

    cleaned_code = source_code.strip()

    if not cleaned_code:
        raise ValueError("The source-code input cannot be empty.")

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    threshold = float(bundle["decision_threshold"])
    class_labels = bundle["class_labels"]

    features = vectorizer.transform([cleaned_code])

    decision_score = float(
        model.decision_function(features)[0]
    )

    predicted_class = int(
        decision_score >= threshold
    )

    predicted_label = class_labels.get(
        predicted_class,
        str(predicted_class),
    )

    return {
        "student_id": "CIT-24-01-0251",
        "model": "Linear Support Vector Machine",
        "prediction": predicted_class,
        "label": predicted_label,
        "decision_score": decision_score,
        "decision_threshold": threshold,
        "token_count": len(code_tokenizer(cleaned_code)),
    }


def read_source_code(
    code_argument: str | None,
    file_argument: Path | None,
) -> str:
    """Read source code from a command argument or text file."""

    if code_argument and file_argument:
        raise ValueError(
            "Use either --code or --file, not both."
        )

    if code_argument:
        return code_argument

    if file_argument:
        if not file_argument.exists():
            raise FileNotFoundError(
                f"Input file was not found: {file_argument}"
            )

        return file_argument.read_text(
            encoding="utf-8",
        )

    raise ValueError(
        "No source code was provided. "
        "Use --code or --file."
    )


def main() -> None:
    """Run the command-line prediction pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict whether a source-code function is "
            "vulnerable using the trained SVM model."
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

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Optional path to the saved SVM model bundle.",
    )

    arguments = parser.parse_args()

    try:
        source_code = read_source_code(
            code_argument=arguments.code,
            file_argument=arguments.file,
        )

        svm_bundle = load_svm_bundle(
            arguments.model
        )

        result = predict_vulnerability(
            source_code,
            svm_bundle,
        )

        print("\nSVM vulnerability prediction completed.\n")
        print(
            json.dumps(
                result,
                indent=4,
            )
        )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        print(f"\nPipeline error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()