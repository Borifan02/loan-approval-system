"""Prediction engine for the Loan Approval system.

This module provides the core high-level inference functions, returning rich,
structured results containing decisions, probabilities, confidence disclaimers,
and intermediate feature representations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import CLASS_LABELS, FINAL_MODEL_FEATURES
from .model_loader import load_artifacts
from .preprocessing import preprocess_applicant_data
from .utils import logger
from .validation import ValidationError, validate_applicant_data


@dataclass
class PredictionResult:
    """Structured response model representing an individual loan approval prediction."""

    success: bool
    prediction: str
    prediction_class: int
    probabilities: Dict[str, float]
    confidence: float
    confidence_note: str
    features_used: Dict[str, float]
    engineered_metrics: Dict[str, float]
    raw_input: Dict[str, Any]
    explanation_available: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the prediction result to a JSON-serializable dictionary."""
        return asdict(self)


CONFIDENCE_DISCLAIMER: str = (
    "Model probability is a statistical estimate based on historical training data "
    "and does not constitute a guaranteed lending decision, legal approval, or financial advice."
)


def predict_loan_approval(
    applicant_data: Dict[str, Any],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> PredictionResult:
    """Executes loan approval prediction for an individual applicant.

    Args:
        applicant_data: Dictionary containing applicant financial and contextual details.
        model: Optional pre-loaded classifier model.
        scaler: Optional pre-loaded feature scaler.

    Returns:
        PredictionResult containing approval decision, probabilities, and feature details.
    """
    try:
        # 1. Validate applicant input
        val_result = validate_applicant_data(applicant_data)
        if not val_result.is_valid:
            error_msg = "; ".join(val_result.errors)
            return PredictionResult(
                success=False,
                prediction="Error",
                prediction_class=-1,
                probabilities={"approved": 0.0, "rejected": 0.0},
                confidence=0.0,
                confidence_note=CONFIDENCE_DISCLAIMER,
                features_used={},
                engineered_metrics={},
                raw_input=applicant_data,
                explanation_available=False,
                error=f"Validation Error: {error_msg}",
            )

        # 2. Load model and scaler if needed
        artifacts = load_artifacts()
        clf = model if model is not None else artifacts.model
        scl = scaler if scaler is not None else artifacts.scaler

        # 3. Preprocess applicant data
        prep_output = preprocess_applicant_data(
            applicant_data=val_result.cleaned_data,
            scaler=scl,
            validate=False,
        )

        features_df = prep_output.features_df

        # 4. Execute prediction
        raw_pred = clf.predict(features_df)
        pred_class = int(raw_pred[0])
        pred_label = CLASS_LABELS.get(pred_class, "Unknown")

        # 5. Extract probabilities
        prob_dict = {"rejected": 0.5, "approved": 0.5}
        confidence = 0.5

        if hasattr(clf, "predict_proba"):
            raw_proba = clf.predict_proba(features_df)[0]
            # Assumes classes_ are [0, 1] as verified during model audit
            classes = list(getattr(clf, "classes_", [0, 1]))
            if len(raw_proba) >= 2 and 0 in classes and 1 in classes:
                idx_0 = classes.index(0)
                idx_1 = classes.index(1)
                p_rejected = float(raw_proba[idx_0])
                p_approved = float(raw_proba[idx_1])
                prob_dict = {
                    "rejected": round(p_rejected, 4),
                    "approved": round(p_approved, 4),
                }
                confidence = round(max(p_rejected, p_approved), 4)

        # 6. Extract features used for reference
        features_used = {
            col: round(float(features_df[col].iloc[0]), 4)
            for col in FINAL_MODEL_FEATURES
        }

        # 7. Extract engineered metrics for explainability
        eng_metrics = {
            "Total_Income": round(prep_output.engineered_features["Total_Income"], 2),
            "Monthly_EMI": round(prep_output.engineered_features["Monthly_Installment"], 2),
            "Balance_Income": round(prep_output.engineered_features["Balance_Income"], 2),
        }

        return PredictionResult(
            success=True,
            prediction=pred_label,
            prediction_class=pred_class,
            probabilities=prob_dict,
            confidence=confidence,
            confidence_note=CONFIDENCE_DISCLAIMER,
            features_used=features_used,
            engineered_metrics=eng_metrics,
            raw_input=val_result.cleaned_data,
            explanation_available=True,
            error=None,
        )

    except ValidationError as ve:
        return PredictionResult(
            success=False,
            prediction="Error",
            prediction_class=-1,
            probabilities={"approved": 0.0, "rejected": 0.0},
            confidence=0.0,
            confidence_note=CONFIDENCE_DISCLAIMER,
            features_used={},
            engineered_metrics={},
            raw_input=applicant_data,
            explanation_available=False,
            error=str(ve),
        )
    except Exception as exc:
        logger.exception("Unexpected error during loan approval prediction: %s", exc)
        return PredictionResult(
            success=False,
            prediction="Error",
            prediction_class=-1,
            probabilities={"approved": 0.0, "rejected": 0.0},
            confidence=0.0,
            confidence_note=CONFIDENCE_DISCLAIMER,
            features_used={},
            engineered_metrics={},
            raw_input=applicant_data,
            explanation_available=False,
            error=f"Prediction System Error: {str(exc)}",
        )


def predict_batch(
    applicants: List[Dict[str, Any]],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> List[PredictionResult]:
    """Processes a batch list of applicant records, returning predictions for each."""
    return [predict_loan_approval(app, model=model, scaler=scaler) for app in applicants]
