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
from scipy.sparse import hstack


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

# This tokenizer must match the tokenizer used when
# training the token-based TF-IDF vectorizer.

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


# ---------------------------------------------------------
# Joblib compatibility
# ---------------------------------------------------------

# The TF-IDF vectorizer was originally trained using a
# tokenizer defined while Python was running as __main__.
# Registering it here allows joblib to load that vectorizer.

setattr(
    sys.modules["__main__"],
    "code_tokenizer",
    code_tokenizer,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_svm_weighted_hybrid_final.joblib"
)

LEGACY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_svm_model_bundle.joblib"
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_svm_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load and validate an SVM model bundle."""

    if not model_path.exists():
        raise FileNotFoundError(
            "The SVM model bundle was not found:\n"
            f"{model_path}\n\n"
            "Make sure the trained model exists inside "
            "the models folder."
        )

    bundle = joblib.load(model_path)

    basic_required_items = {
        "model",
        "decision_threshold",
    }

    missing_basic_items = basic_required_items.difference(
        bundle
    )

    if missing_basic_items:
        raise KeyError(
            "The model bundle is missing these items: "
            + ", ".join(
                sorted(missing_basic_items)
            )
        )

    # -----------------------------------------------------
    # New weighted-hybrid bundle
    # -----------------------------------------------------

    if (
        "token_vectorizer" in bundle
        and "char_vectorizer" in bundle
    ):
        hybrid_required_items = {
            "token_weight",
            "char_weight",
        }

        missing_hybrid_items = (
            hybrid_required_items.difference(bundle)
        )

        if missing_hybrid_items:
            raise KeyError(
                "The weighted-hybrid bundle is missing: "
                + ", ".join(
                    sorted(missing_hybrid_items)
                )
            )

        return bundle

    # -----------------------------------------------------
    # Legacy single-vectorizer bundle
    # -----------------------------------------------------

    if "vectorizer" in bundle:
        return bundle

    raise KeyError(
        "Unsupported SVM bundle format. "
        "Expected either a weighted-hybrid bundle "
        "or a legacy single-vectorizer bundle."
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_vulnerability(
    source_code: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict whether one source-code function is vulnerable."""

    cleaned_code = source_code.strip()

    if not cleaned_code:
        raise ValueError(
            "The source-code input cannot be empty."
        )

    model = bundle["model"]

    threshold = float(
        bundle["decision_threshold"]
    )

    class_labels = bundle.get(
        "class_labels",
        {
            0: "Non-vulnerable",
            1: "Vulnerable",
        },
    )

    # -----------------------------------------------------
    # New weighted token + character TF-IDF model
    # -----------------------------------------------------

    if (
        "token_vectorizer" in bundle
        and "char_vectorizer" in bundle
    ):
        token_vectorizer = bundle[
            "token_vectorizer"
        ]

        char_vectorizer = bundle[
            "char_vectorizer"
        ]

        token_weight = float(
            bundle["token_weight"]
        )

        char_weight = float(
            bundle["char_weight"]
        )

        token_features = (
            token_vectorizer.transform(
                [cleaned_code]
            )
        )

        char_features = (
            char_vectorizer.transform(
                [cleaned_code]
            )
        )

        features = hstack(
            [
                token_features * token_weight,
                char_features * char_weight,
            ],
            format="csr",
        )

        representation = (
            "Weighted token + character TF-IDF"
        )

    # -----------------------------------------------------
    # Legacy single-vectorizer model
    # -----------------------------------------------------

    elif "vectorizer" in bundle:
        vectorizer = bundle["vectorizer"]

        features = vectorizer.transform(
            [cleaned_code]
        )

        representation = "Token TF-IDF"

    else:
        raise KeyError(
            "No compatible vectorizer was found "
            "inside the SVM bundle."
        )

    # -----------------------------------------------------
    # SVM decision
    # -----------------------------------------------------

    decision_score = float(
        model.decision_function(
            features
        )[0]
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
        "representation": representation,
        "prediction": predicted_class,
        "label": predicted_label,
        "decision_score": decision_score,
        "decision_threshold": threshold,
        "token_count": len(
            code_tokenizer(cleaned_code)
        ),
    }


# ---------------------------------------------------------
# Input handling
# ---------------------------------------------------------

def read_source_code(
    code_argument: str | None,
    file_argument: Path | None,
) -> str:
    """Read source code from an argument or file."""

    if code_argument and file_argument:
        raise ValueError(
            "Use either --code or --file, not both."
        )

    if code_argument:
        return code_argument

    if file_argument:
        if not file_argument.exists():
            raise FileNotFoundError(
                "Input file was not found: "
                f"{file_argument}"
            )

        return file_argument.read_text(
            encoding="utf-8",
        )

    raise ValueError(
        "No source code was provided. "
        "Use --code or --file."
    )


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

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
        help=(
            "Path to a file containing source code."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Optional path to an SVM model bundle."
        ),
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

        print(
            "\nSVM vulnerability prediction "
            "completed.\n"
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
        ValueError,
    ) as error:
        print(
            f"\nPipeline error: {error}"
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()