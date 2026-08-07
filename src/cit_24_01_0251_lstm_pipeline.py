"""Reusable LSTM vulnerability-prediction pipeline.

Student: Ravishka Rathnayake
Student ID: CIT-24-01-0251
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

# Reduce unnecessary TensorFlow information messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

# This tokenizer must match the tokenizer used during
# LSTM training.

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


def prepare_code_for_lstm(source_code: str) -> str:
    """Convert source-code tokens into a space-separated string."""
    return " ".join(
        code_tokenizer(source_code)
    )


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_embedding32_final.keras"
)

DEFAULT_PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_embedding32_preprocessing.joblib"
)

LEGACY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_model.keras"
)

LEGACY_PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_preprocessing.joblib"
)


# ---------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------

def load_lstm_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessing_path: Path = DEFAULT_PREPROCESSING_PATH,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Load and validate the LSTM model and preprocessing bundle."""

    if not model_path.exists():
        raise FileNotFoundError(
            "The LSTM model was not found:\n"
            f"{model_path}"
        )

    if not preprocessing_path.exists():
        raise FileNotFoundError(
            "The LSTM preprocessing bundle was not found:\n"
            f"{preprocessing_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    preprocessing_bundle = joblib.load(
        preprocessing_path
    )

    required_items = {
        "tokenizer",
        "decision_threshold",
    }

    missing_items = required_items.difference(
        preprocessing_bundle
    )

    if missing_items:
        raise KeyError(
            "The preprocessing bundle is missing: "
            + ", ".join(
                sorted(missing_items)
            )
        )

    # Support both the final bundle:
    #     max_sequence_length
    #
    # and the original legacy bundle:
    #     maximum_sequence_length

    if (
        "max_sequence_length"
        not in preprocessing_bundle
        and "maximum_sequence_length"
        not in preprocessing_bundle
    ):
        raise KeyError(
            "The preprocessing bundle does not contain "
            "a sequence-length setting."
        )

    return model, preprocessing_bundle


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_vulnerability(
    source_code: str,
    model: tf.keras.Model,
    preprocessing_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict whether one source-code function is vulnerable."""

    cleaned_code = source_code.strip()

    if not cleaned_code:
        raise ValueError(
            "The source-code input cannot be empty."
        )

    tokenizer = preprocessing_bundle[
        "tokenizer"
    ]

    # -----------------------------------------------------
    # New final bundle / legacy compatibility
    # -----------------------------------------------------

    if "max_sequence_length" in preprocessing_bundle:
        maximum_sequence_length = int(
            preprocessing_bundle[
                "max_sequence_length"
            ]
        )
    else:
        maximum_sequence_length = int(
            preprocessing_bundle[
                "maximum_sequence_length"
            ]
        )

    threshold = float(
        preprocessing_bundle[
            "decision_threshold"
        ]
    )

    class_labels = preprocessing_bundle.get(
        "class_labels",
        {
            0: "Non-vulnerable",
            1: "Vulnerable",
        },
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    prepared_code = prepare_code_for_lstm(
        cleaned_code
    )

    sequence = tokenizer.texts_to_sequences(
        [prepared_code]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=maximum_sequence_length,
        padding="post",
        truncating="post",
        dtype="int32",
    )

    # -----------------------------------------------------
    # LSTM probability
    # -----------------------------------------------------

    probability = float(
        model.predict(
            padded_sequence,
            verbose=0,
        ).ravel()[0]
    )

    predicted_class = int(
        probability >= threshold
    )

    predicted_label = class_labels.get(
        predicted_class,
        str(predicted_class),
    )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    result = {
        "student_id": "CIT-24-01-0251",
        "model": "Long Short-Term Memory Network",
        "prediction": predicted_class,
        "label": predicted_label,
        "vulnerability_probability": probability,
        "decision_threshold": threshold,
        "token_count": len(
            code_tokenizer(cleaned_code)
        ),
        "maximum_sequence_length": (
            maximum_sequence_length
        ),
    }

    # Include final-model architecture information when
    # available in the new preprocessing bundle.

    if "embedding_dimension" in preprocessing_bundle:
        result["embedding_dimension"] = int(
            preprocessing_bundle[
                "embedding_dimension"
            ]
        )

    if "lstm_units" in preprocessing_bundle:
        result["lstm_units"] = int(
            preprocessing_bundle[
                "lstm_units"
            ]
        )

    return result


# ---------------------------------------------------------
# Input handling
# ---------------------------------------------------------

def read_source_code(
    code_argument: str | None,
    file_argument: Path | None,
) -> str:
    """Read source code from an argument or a file."""

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
            encoding="utf-8"
        )

    raise ValueError(
        "No source code was provided. "
        "Use --code or --file."
    )


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main() -> None:
    """Run the command-line LSTM prediction pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict whether source code is vulnerable "
            "using the trained LSTM model."
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
            "Optional path to the saved LSTM model."
        ),
    )

    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=DEFAULT_PREPROCESSING_PATH,
        help=(
            "Optional path to the preprocessing bundle."
        ),
    )

    arguments = parser.parse_args()

    try:
        source_code = read_source_code(
            code_argument=arguments.code,
            file_argument=arguments.file,
        )

        model, preprocessing_bundle = (
            load_lstm_artifacts(
                model_path=arguments.model,
                preprocessing_path=arguments.preprocessing,
            )
        )

        result = predict_vulnerability(
            source_code=source_code,
            model=model,
            preprocessing_bundle=preprocessing_bundle,
        )

        print(
            "\nLSTM vulnerability prediction "
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
        OSError,
        ValueError,
    ) as error:
        print(
            f"\nPipeline error: {error}"
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()