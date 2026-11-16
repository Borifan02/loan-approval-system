"""Explainable AI (XAI) engine using SHAP for the Loan Approval system.

This module provides:
  - Local feature attribution per applicant (positive/negative contributors)
  - Global feature importance from Random Forest ensemble trees
  - Headless Matplotlib waterfall plot generation
  - Plain-English human-readable decision narratives
  - Defensive fallbacks for diverse SHAP tensor shapes and environments
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server and CLI safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import (
    FEATURE_DISPLAY_NAMES,
    FINAL_MODEL_FEATURES,
    SHAP_WATERFALL_MAX_DISPLAY,
)
from .model_loader import load_artifacts
from .preprocessing import preprocess_applicant_data
from .utils import logger
from .validation import validate_applicant_data


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution to the prediction."""

    feature: str
    display_name: str
    value: float
    shap_value: float
    direction: str  # 'positive' (favors approval) or 'negative' (favors rejection)
    impact_magnitude: float
    summary: str


@dataclass
class ExplanationResult:
    """Structured response containing all local and global explanation details."""

    success: bool
    prediction: str
    base_value: float
    model_output_value: float
    contributions: List[FeatureContribution]
    positive_factors: List[Dict[str, Any]]
    negative_factors: List[Dict[str, Any]]
    global_importance: Dict[str, float]
    human_readable_summary: str
    disclaimer: str
    waterfall_plot_base64: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts explanation to a JSON-serializable dictionary."""
        return asdict(self)


EXPLANATION_DISCLAIMER: str = (
    "SHAP explanations illustrate feature weights within the statistical model. "
    "They do not constitute financial advice, credit counseling, or legal justification for lending decisions."
)

# Cached TreeExplainer instance
_cached_explainer: Optional[shap.TreeExplainer] = None


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Returns or lazily creates a cached SHAP TreeExplainer for the given model."""
    global _cached_explainer
    if _cached_explainer is None:
        logger.info("Initializing SHAP TreeExplainer for Random Forest model")
        _cached_explainer = shap.TreeExplainer(model)
    return _cached_explainer


def _extract_class1_shap(
    raw_shap_values: Any,
    feature_count: int,
) -> Tuple[np.ndarray, float]:
    """Defensively extracts Class 1 (Approval) SHAP values and base value across versions."""
    # SHAP Explanation object
    values = getattr(raw_shap_values, "values", raw_shap_values)
    base_values = getattr(raw_shap_values, "base_values", 0.5)

    vals = np.array(values)
    base = np.array(base_values)

    # If 3D array: (n_samples, n_features, n_classes)
    if vals.ndim == 3:
        # Index 1 corresponds to class 1 (Approved)
        class_idx = 1 if vals.shape[2] > 1 else 0
        extracted_vals = vals[0, :, class_idx]
        if base.ndim >= 2:
            extracted_base = float(base[0, class_idx])
        elif base.ndim == 1 and len(base) > 1:
            extracted_base = float(base[class_idx])
        else:
            extracted_base = float(base.ravel()[0])

    elif vals.ndim == 2:
        # (n_samples, n_features)
        extracted_vals = vals[0, :]
        extracted_base = float(base.ravel()[0]) if hasattr(base, "ravel") else float(base)

    elif vals.ndim == 1:
        extracted_vals = vals
        extracted_base = float(base)

    else:
        raise ValueError(f"Unexpected SHAP values dimensionality: {vals.shape}")

    return extracted_vals, extracted_base


def _generate_narrative(
    prediction: str,
    positive_factors: List[FeatureContribution],
    negative_factors: List[FeatureContribution],
) -> str:
    """Generates a plain-English explanation narrative based on top drivers."""
    paragraphs = []
    if prediction == "Approved":
        paragraphs.append("The applicant meets the requirements for loan approval.")
        if positive_factors:
            top_pos = [f"{f.display_name} ({f.summary})" for f in positive_factors[:2]]
            paragraphs.append(f"Key positive drivers supporting approval include: {'; '.join(top_pos)}.")
        if negative_factors:
            top_neg = [f"{f.display_name}" for f in negative_factors[:1]]
            paragraphs.append(f"Minor risk factors noted: {'; '.join(top_neg)}.")
    else:
        paragraphs.append("The applicant does not meet the approval threshold for this loan application.")
        if negative_factors:
            top_neg = [f"{f.display_name} ({f.summary})" for f in negative_factors[:2]]
            paragraphs.append(f"Primary factors contributing to rejection include: {'; '.join(top_neg)}.")
        if positive_factors:
            top_pos = [f"{f.display_name}" for f in positive_factors[:1]]
            paragraphs.append(f"Favorable factors that partially offset risk: {'; '.join(top_pos)}.")

    return " ".join(paragraphs)


def _feature_summary(feature: str, val: float, shap_val: float) -> str:
    """Returns contextual text explaining how a feature value affected the score."""
    if feature == "Credit_History":
        if val >= 0.5:
            return "Positive repayment record meeting lender guidelines"
        return "Lack of verified satisfactory repayment history"
    if feature == "Total_Income_log":
        if shap_val > 0:
            return "Strong combined household earning capacity"
        return "Household income below typical threshold for requested amount"
    if feature == "EMI_log":
        if shap_val > 0:
            return "Manageable monthly installment relative to term"
        return "Higher monthly installment relative to loan duration"
    if feature == "Balance_Income_log":
        if shap_val > 0:
            return "Adequate disposable buffer remaining after debt servicing"
        return "Constrained net cash flow after debt obligations"
    return f"Value: {val:.2f}"


def explain_prediction(
    applicant_data: Dict[str, Any],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    generate_plot: bool = False,
) -> ExplanationResult:
    """Computes comprehensive SHAP explainability metrics for an applicant.

    Args:
        applicant_data: Input dictionary of applicant parameters.
        model: Optional pre-loaded classifier.
        scaler: Optional pre-loaded scaler.
        generate_plot: Whether to render a Matplotlib waterfall plot.

    Returns:
        ExplanationResult with per-feature contributions, narrative, and importance.
    """
    try:
        val_result = validate_applicant_data(applicant_data)
        val_result.raise_if_invalid()

        artifacts = load_artifacts()
        clf = model if model is not None else artifacts.model
        scl = scaler if scaler is not None else artifacts.scaler

        prep_output = preprocess_applicant_data(val_result.cleaned_data, scaler=scl, validate=False)
        features_df = prep_output.features_df

        # Prediction outcome
        pred_raw = clf.predict(features_df)[0]
        prediction = "Approved" if pred_raw == 1 else "Rejected"

        # Global feature importance
        importances = getattr(clf, "feature_importances_", None)
        global_importance: Dict[str, float] = {}
        if importances is not None and len(importances) == len(FINAL_MODEL_FEATURES):
            for feat, imp in zip(FINAL_MODEL_FEATURES, importances):
                global_importance[feat] = round(float(imp), 4)

        # Compute SHAP
        explainer = get_tree_explainer(clf)
        raw_shap = explainer(features_df)

        extracted_shap, base_val = _extract_class1_shap(raw_shap, len(FINAL_MODEL_FEATURES))

        contributions: List[FeatureContribution] = []
        pos_factors: List[FeatureContribution] = []
        neg_factors: List[FeatureContribution] = []

        for idx, feat in enumerate(FINAL_MODEL_FEATURES):
            disp_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            feat_val = float(features_df[feat].iloc[0])
            shap_val = float(extracted_shap[idx])
            direction = "positive" if shap_val >= 0 else "negative"
            summary_text = _feature_summary(feat, feat_val, shap_val)

            fc = FeatureContribution(
                feature=feat,
                display_name=disp_name,
                value=round(feat_val, 4),
                shap_value=round(shap_val, 4),
                direction=direction,
                impact_magnitude=round(abs(shap_val), 4),
                summary=summary_text,
            )
            contributions.append(fc)
            if direction == "positive":
                pos_factors.append(fc)
            else:
                neg_factors.append(fc)

        # Sort factors by impact magnitude
        pos_factors.sort(key=lambda x: x.impact_magnitude, reverse=True)
        neg_factors.sort(key=lambda x: x.impact_magnitude, reverse=True)

        narrative = _generate_narrative(prediction, pos_factors, neg_factors)

        final_output = base_val + float(np.sum(extracted_shap))

        return ExplanationResult(
            success=True,
            prediction=prediction,
            base_value=round(base_val, 4),
            model_output_value=round(final_output, 4),
            contributions=contributions,
            positive_factors=[asdict(f) for f in pos_factors],
            negative_factors=[asdict(f) for f in neg_factors],
            global_importance=global_importance,
            human_readable_summary=narrative,
            disclaimer=EXPLANATION_DISCLAIMER,
            error=None,
        )

    except Exception as exc:
        logger.exception("Failed to generate SHAP explanation: %s", exc)
        return ExplanationResult(
            success=False,
            prediction="Error",
            base_value=0.5,
            model_output_value=0.5,
            contributions=[],
            positive_factors=[],
            negative_factors=[],
            global_importance={},
            human_readable_summary="Explanation could not be generated due to an error.",
            disclaimer=EXPLANATION_DISCLAIMER,
            error=str(exc),
        )


def render_waterfall_plot(
    explanation_res: ExplanationResult,
    title: str = "SHAP Waterfall Plot: Feature Contributions",
) -> plt.Figure:
    """Renders a standalone Matplotlib figure displaying feature attributions.

    Returns:
        matplotlib.figure.Figure instance safe for Streamlit or image saving.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    contributions = sorted(explanation_res.contributions, key=lambda c: c.impact_magnitude)

    feature_names = [c.display_name for c in contributions]
    shap_vals = [c.shap_value for c in contributions]
    colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in shap_vals]

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, shap_vals, color=colors, align="center", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.axvline(0, color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax.set_xlabel("SHAP Value (Impact on Approval Probability)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Annotate value labels
    for i, v in enumerate(shap_vals):
        offset = 0.01 if v >= 0 else -0.01
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9, fontweight="medium")

    plt.tight_layout()
    return fig
