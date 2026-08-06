"""Automated tests for Ravishka's group-integration adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from cit_24_01_0251_adapter import (
    RavishkaVulnerabilityPredictor,
)


SAFE_SAMPLE_PATH = (
    PROJECT_ROOT
    / "examples"
    / "safe_sample.c"
)

MODEL_PATHS = (
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_svm_model_bundle.joblib",
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_model.keras",
    PROJECT_ROOT
    / "models"
    / "CIT-24-01-0251_lstm_preprocessing.joblib",
)

MODEL_ARTIFACTS_AVAILABLE = all(
    path.exists()
    for path in MODEL_PATHS
)


class TestAdapterValidation(unittest.TestCase):
    """Test adapter input validation without loading models."""

    def setUp(self) -> None:
        self.predictor = RavishkaVulnerabilityPredictor(
            svm_bundle={},
            lstm_model=None,
            lstm_preprocessing_bundle={},
        )

    def test_rejects_non_string_input(self) -> None:
        with self.assertRaises(TypeError):
            self.predictor.predict(None)  # type: ignore[arg-type]

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            self.predictor.predict("   ")


@unittest.skipUnless(
    MODEL_ARTIFACTS_AVAILABLE,
    "Local trained model files are unavailable.",
)
class TestAdapterIntegration(unittest.TestCase):
    """Test the adapter using the trained model artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = (
            RavishkaVulnerabilityPredictor.load()
        )

        cls.source_code = SAFE_SAMPLE_PATH.read_text(
            encoding="utf-8"
        )

    def test_svm_prediction_structure(self) -> None:
        result = self.predictor.predict_svm(
            self.source_code
        )

        self.assertEqual(
            result["student_id"],
            "CIT-24-01-0251",
        )
        self.assertIn(result["prediction"], {0, 1})
        self.assertIn(
            result["label"],
            {"Vulnerable", "Non-vulnerable"},
        )
        self.assertIn("decision_score", result)

    def test_lstm_prediction_structure(self) -> None:
        result = self.predictor.predict_lstm(
            self.source_code
        )

        self.assertEqual(
            result["student_id"],
            "CIT-24-01-0251",
        )
        self.assertIn(result["prediction"], {0, 1})
        self.assertIn(
            result["label"],
            {"Vulnerable", "Non-vulnerable"},
        )

        probability = result[
            "vulnerability_probability"
        ]

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_combined_prediction_structure(self) -> None:
        result = self.predictor.predict(
            self.source_code
        )

        self.assertEqual(
            result["student_id"],
            "CIT-24-01-0251",
        )
        self.assertEqual(
            result["student_name"],
            "Ravishka Rathnayake",
        )

        self.assertIn("svm", result["models"])
        self.assertIn("lstm", result["models"])
        self.assertIn("comparison", result)

        comparison = result["comparison"]

        self.assertIsInstance(
            comparison["models_agree"],
            bool,
        )
        self.assertIn(
            comparison["vulnerable_votes"],
            {0, 1, 2},
        )


if __name__ == "__main__":
    unittest.main()