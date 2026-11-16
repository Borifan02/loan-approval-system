"""Unit tests for feature engineering and preprocessing pipeline."""

import math
import numpy as np
import pytest
from loan_approval import compute_engineered_features, preprocess_applicant_data
from loan_approval.config import FINAL_MODEL_FEATURES, SCALED_NUMERICAL_FEATURES


def test_compute_engineered_features_math():
    eng = compute_engineered_features(
        applicant_income=5000.0,
        coapplicant_income=2000.0,
        loan_amount=150.0,
        loan_term=300.0,
    )
    assert eng["Total_Income"] == 7000.0
    assert eng["EMI"] == 150.0 / 300.0  # 0.5
    assert eng["Monthly_Installment"] == 0.5 * 1000.0  # 500.0
    assert eng["Balance_Income"] == 7000.0 - 500.0  # 6500.0

    # Verify log outputs are strictly finite
    assert not math.isnan(eng["Total_Income_log"])
    assert not math.isinf(eng["Total_Income_log"])
    assert not math.isnan(eng["EMI_log"])
    assert not math.isnan(eng["Balance_Income_log"])


def test_balance_income_negative_safe_log():
    # Extreme debt where balance income is negative
    eng = compute_engineered_features(
        applicant_income=500.0,
        coapplicant_income=0.0,
        loan_amount=500.0,
        loan_term=12.0,  # EMI = 41.67k -> monthly installment = $41,666.67
    )
    assert eng["Balance_Income"] < -1769.0  # severely negative
    # Safe log must not produce NaN or raise runtime error
    assert not math.isnan(eng["Balance_Income_log"])
    assert not math.isinf(eng["Balance_Income_log"])


def test_preprocess_applicant_data_columns_and_order(valid_applicant, artifacts):
    output = preprocess_applicant_data(valid_applicant, scaler=artifacts.scaler)
    df = output.features_df

    assert list(df.columns) == FINAL_MODEL_FEATURES
    assert len(df) == 1
    # Check that scaled numerical features are present in output dictionary
    for col in SCALED_NUMERICAL_FEATURES:
        assert col in output.scaled_features
        assert not math.isnan(output.scaled_features[col])

    # Check that Credit_History is unscaled binary float
    assert df["Credit_History"].iloc[0] == 1.0


def test_preprocess_invalid_data_raises_validation_error():
    bad_data = {"ApplicantIncome": -100}
    with pytest.raises(Exception):
        preprocess_applicant_data(bad_data)
