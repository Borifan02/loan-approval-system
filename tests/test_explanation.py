"""Unit tests for the SHAP explainability engine."""

import matplotlib
import pytest
from loan_approval import explain_prediction, render_waterfall_plot


def test_explain_prediction_success(valid_applicant, artifacts):
    res = explain_prediction(valid_applicant, model=artifacts.model, scaler=artifacts.scaler)
    assert res.success is True
    assert res.prediction == "Approved"
    assert isinstance(res.base_value, float)
    assert isinstance(res.model_output_value, float)
    assert len(res.contributions) == 4

    # Verify factors
    total_factors = len(res.positive_factors) + len(res.negative_factors)
    assert total_factors == 4

    # Check narrative
    assert len(res.human_readable_summary) > 20
    assert "Credit History" in res.human_readable_summary or "income" in res.human_readable_summary.lower()

    # Check global importance
    assert len(res.global_importance) == 4
    assert "Credit_History" in res.global_importance


def test_explain_prediction_rejected(rejected_applicant, artifacts):
    res = explain_prediction(rejected_applicant, model=artifacts.model, scaler=artifacts.scaler)
    assert res.success is True
    assert res.prediction == "Rejected"
    # Rejection should have strong negative factor for Credit History
    assert any(f["feature"] == "Credit_History" for f in res.negative_factors)


def test_render_waterfall_plot(valid_applicant):
    res = explain_prediction(valid_applicant)
    fig = render_waterfall_plot(res)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0


def test_explain_prediction_invalid_data():
    bad_data = {"ApplicantIncome": -100}
    res = explain_prediction(bad_data)
    assert res.success is False
    assert res.error is not None
    assert len(res.contributions) == 0
