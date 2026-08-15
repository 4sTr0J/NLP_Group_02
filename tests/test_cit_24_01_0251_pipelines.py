"""Automated tests for Ravishka's SVM and LSTM pipelines."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from cit_24_01_0251_svm_pipeline import (
    code_tokenizer as svm_code_tokenizer,
    load_svm_bundle,
    predict_vulnerability as predict_with_svm,
    read_source_code,
)

from cit_24_01_0251_lstm_pipeline import (
    code_tokenizer as lstm_code_tokenizer,
    load_lstm_artifacts,
    predict_vulnerability as predict_with_lstm,
)

from cit_24_01_0251_combined_pipeline import compare_predictions


SAFE_SAMPLE_PATH = (
    PROJECT_ROOT
    / "examples"
    / "safe_sample.c"
)

SVM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_svm_model_bundle.joblib"
)

LSTM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_model.keras"
)

LSTM_PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_preprocessing.joblib"
)

MODEL_ARTIFACTS_AVAILABLE = all(
    path.exists()
    for path in (
        SVM_MODEL_PATH,
        LSTM_MODEL_PATH,
        LSTM_PREPROCESSING_PATH,
    )
)


class TestCodeTokenization(unittest.TestCase):
    """Test source-code tokenization."""

    def test_svm_tokenizer_preserves_code_operators(self) -> None:
        source_code = (
            "if (value == 10 && enabled != 0) "
            "{ return value + 1; }"
        )

        tokens = svm_code_tokenizer(source_code)

        self.assertIn("if", tokens)
        self.assertIn("==", tokens)
        self.assertIn("&&", tokens)
        self.assertIn("!=", tokens)
        self.assertIn("+", tokens)

    def test_both_models_use_matching_tokens(self) -> None:
        source_code = (
            "int add(int a, int b) "
            "{ return a + b; }"
        )

        svm_tokens = svm_code_tokenizer(source_code)
        lstm_tokens = lstm_code_tokenizer(source_code)

        self.assertEqual(svm_tokens, lstm_tokens)


class TestInputHandling(unittest.TestCase):
    """Test command-line input handling."""

    def test_read_source_code_from_file(self) -> None:
        source_code = read_source_code(
            code_argument=None,
            file_argument=SAFE_SAMPLE_PATH,
        )

        self.assertIn("add_numbers", source_code)
        self.assertIn("return", source_code)

    def test_rejects_missing_input(self) -> None:
        with self.assertRaises(ValueError):
            read_source_code(
                code_argument=None,
                file_argument=None,
            )

    def test_rejects_two_input_methods(self) -> None:
        with self.assertRaises(ValueError):
            read_source_code(
                code_argument="int main() { return 0; }",
                file_argument=SAFE_SAMPLE_PATH,
            )


@unittest.skipUnless(
    MODEL_ARTIFACTS_AVAILABLE,
    "Local trained model files are unavailable.",
)
class TestModelPredictions(unittest.TestCase):
    """Test predictions using locally saved trained models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source_code = read_source_code(
            code_argument=None,
            file_argument=SAFE_SAMPLE_PATH,
        )

        cls.svm_bundle = load_svm_bundle()

        (
            cls.lstm_model,
            cls.lstm_preprocessing,
        ) = load_lstm_artifacts()

    def test_svm_prediction_structure(self) -> None:
        result = predict_with_svm(
            source_code=self.source_code,
            bundle=self.svm_bundle,
        )

        self.assertIn(result["prediction"], {0, 1})
        self.assertIn(
            result["label"],
            {"Vulnerable", "Non-vulnerable"},
        )
        self.assertIn("decision_score", result)
        self.assertIn("decision_threshold", result)

    def test_lstm_prediction_structure(self) -> None:
        result = predict_with_lstm(
            source_code=self.source_code,
            model=self.lstm_model,
            preprocessing_bundle=self.lstm_preprocessing,
        )

        self.assertIn(result["prediction"], {0, 1})
        self.assertIn(
            result["label"],
            {"Vulnerable", "Non-vulnerable"},
        )
        self.assertGreaterEqual(
            result["vulnerability_probability"],
            0.0,
        )
        self.assertLessEqual(
            result["vulnerability_probability"],
            1.0,
        )

    def test_model_comparison(self) -> None:
        svm_result = predict_with_svm(
            source_code=self.source_code,
            bundle=self.svm_bundle,
        )

        lstm_result = predict_with_lstm(
            source_code=self.source_code,
            model=self.lstm_model,
            preprocessing_bundle=self.lstm_preprocessing,
        )

        comparison = compare_predictions(
            svm_result=svm_result,
            lstm_result=lstm_result,
        )

        expected_votes = (
            svm_result["prediction"]
            + lstm_result["prediction"]
        )

        self.assertEqual(
            comparison["vulnerable_votes"],
            expected_votes,
        )

        self.assertIsInstance(
            comparison["models_agree"],
            bool,
        )


if __name__ == "__main__":
    unittest.main()