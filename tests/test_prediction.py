"""Unit tests for the prediction engine."""

import pytest
from loan_approval import predict_batch, predict_loan_approval


def test_predict_loan_approval_success(valid_applicant, artifacts):
    res = predict_loan_approval(valid_applicant, model=artifacts.model, scaler=artifacts.scaler)
    assert res.success is True
    assert res.prediction == "Approved"
    assert res.prediction_class == 1
    assert "approved" in res.probabilities
    assert "rejected" in res.probabilities
    assert res.probabilities["approved"] > 0.5
    assert pytest.approx(res.probabilities["approved"] + res.probabilities["rejected"], 0.01) == 1.0
    assert res.confidence == res.probabilities["approved"]
    assert res.explanation_available is True
    assert res.error is None
    assert len(res.features_used) == 4


def test_predict_loan_approval_rejection(rejected_applicant, artifacts):
    res = predict_loan_approval(rejected_applicant, model=artifacts.model, scaler=artifacts.scaler)
    assert res.success is True
    assert res.prediction == "Rejected"
    assert res.prediction_class == 0
    assert res.probabilities["rejected"] > 0.5
    assert res.confidence == res.probabilities["rejected"]


def test_predict_loan_approval_validation_failure():
    bad_applicant = {"ApplicantIncome": -500}
    res = predict_loan_approval(bad_applicant)
    assert res.success is False
    assert res.prediction == "Error"
    assert res.prediction_class == -1
    assert "Validation Error" in res.error


def test_predict_batch(valid_applicant, rejected_applicant):
    batch = [valid_applicant, rejected_applicant]
    results = predict_batch(batch)
    assert len(results) == 2
    assert results[0].prediction == "Approved"
    assert results[1].prediction == "Rejected"


def test_prediction_result_to_dict(valid_applicant):
    res = predict_loan_approval(valid_applicant)
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["success"] is True
    assert d["prediction"] == "Approved"
    assert "probabilities" in d
