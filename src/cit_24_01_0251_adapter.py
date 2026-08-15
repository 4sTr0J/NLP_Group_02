"""Reusable group-integration adapter for Ravishka's models.

Student: Ravishka Rathnayake
Student ID: CIT-24-01-0251
"""

from __future__ import annotations

from typing import Any, Self

try:
    from .cit_24_01_0251_svm_pipeline import (
        load_svm_bundle,
        predict_vulnerability as predict_with_svm,
    )
    from .cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        predict_vulnerability as predict_with_lstm,
    )
    from .cit_24_01_0251_combined_pipeline import (
        compare_predictions,
    )
except ImportError:
    from cit_24_01_0251_svm_pipeline import (
        load_svm_bundle,
        predict_vulnerability as predict_with_svm,
    )
    from cit_24_01_0251_lstm_pipeline import (
        load_lstm_artifacts,
        predict_vulnerability as predict_with_lstm,
    )
    from cit_24_01_0251_combined_pipeline import (
        compare_predictions,
    )


class RavishkaVulnerabilityPredictor:
    """Load and run Ravishka's SVM and LSTM models."""

    def __init__(
        self,
        svm_bundle: dict[str, Any],
        lstm_model: Any,
        lstm_preprocessing_bundle: dict[str, Any],
    ) -> None:
        self.svm_bundle = svm_bundle
        self.lstm_model = lstm_model
        self.lstm_preprocessing_bundle = (
            lstm_preprocessing_bundle
        )

    @classmethod
    def load(cls) -> Self:
        """Load all required model artifacts once."""

        print("Loading Ravishka's SVM model...")
        svm_bundle = load_svm_bundle()

        print("Loading Ravishka's LSTM model...")
        (
            lstm_model,
            lstm_preprocessing_bundle,
        ) = load_lstm_artifacts()

        print("Both models loaded successfully.")

        return cls(
            svm_bundle=svm_bundle,
            lstm_model=lstm_model,
            lstm_preprocessing_bundle=(
                lstm_preprocessing_bundle
            ),
        )

    def predict_svm(
        self,
        source_code: str,
    ) -> dict[str, Any]:
        """Generate a prediction using the SVM model."""

        self._validate_source_code(source_code)

        return predict_with_svm(
            source_code=source_code,
            bundle=self.svm_bundle,
        )

    def predict_lstm(
        self,
        source_code: str,
    ) -> dict[str, Any]:
        """Generate a prediction using the LSTM model."""

        self._validate_source_code(source_code)

        return predict_with_lstm(
            source_code=source_code,
            model=self.lstm_model,
            preprocessing_bundle=(
                self.lstm_preprocessing_bundle
            ),
        )

    def predict(
        self,
        source_code: str,
    ) -> dict[str, Any]:
        """Run both models and compare their predictions."""

        self._validate_source_code(source_code)

        svm_result = self.predict_svm(source_code)
        lstm_result = self.predict_lstm(source_code)

        comparison = compare_predictions(
            svm_result=svm_result,
            lstm_result=lstm_result,
        )

        return {
            "student_id": "CIT-24-01-0251",
            "student_name": "Ravishka Rathnayake",
            "models": {
                "svm": svm_result,
                "lstm": lstm_result,
            },
            "comparison": comparison,
        }

    def predict_for_group(
    self,
    source_code: str,
    ) -> dict[str, Any]:
        """Return a simplified result for group integration."""

        result = self.predict(source_code)

        svm_result = result["models"]["svm"]
        comparison = result["comparison"]

        models_agree = bool(
            comparison["models_agree"]
        )

        if models_agree:
            final_prediction = int(
                svm_result["prediction"]
            )
            final_label = str(
                comparison["agreed_label"]
            )
            review_required = False
        else:
            final_prediction = None
            final_label = "Review required"
            review_required = True

        return {
            "student_id": "CIT-24-01-0251",
            "student_name": "Ravishka Rathnayake",
            "final_prediction": final_prediction,
            "final_label": final_label,
            "review_required": review_required,
            "models_agree": models_agree,
            "vulnerable_votes": int(
                comparison["vulnerable_votes"]
            ),
            "model_results": result["models"],
        }
    @staticmethod
    def _validate_source_code(
        source_code: str,
    ) -> None:
        """Validate source-code input."""

        if not isinstance(source_code, str):
            raise TypeError(
                "Source code must be provided as text."
            )

        if not source_code.strip():
            raise ValueError(
                "Source code cannot be empty."
            )