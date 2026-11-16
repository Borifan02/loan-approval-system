"""Centralized feature engineering and preprocessing pipeline.

This module replicates the exact mathematical transformations used during model
training in the development notebook (Loan_prediction_main.ipynb):
  - Total Income: ApplicantIncome + CoapplicantIncome
  - EMI: LoanAmount / Loan_Amount_Term
  - Balance Income: Total_Income - (EMI * 1000)
  - Log transformations: log(Total_Income), log(EMI), log(Balance_Income + 1769)
  - Scaling: StandardScaler normalization on the 3 numerical log features
  - Final feature vector: ['Credit_History', 'Total_Income_log', 'EMI_log', 'Balance_Income_log']

Mathematical boundaries and safeguards prevent NaN, -inf, or ZeroDivisionError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BALANCE_INCOME_OFFSET,
    EMI_CURRENCY_MULTIPLIER,
    FINAL_MODEL_FEATURES,
    MIN_BALANCE_INCOME_ARG,
    SAFE_LOG_FLOOR,
    SCALED_NUMERICAL_FEATURES,
)
from .model_loader import load_artifacts
from .utils import logger
from .validation import validate_applicant_data


@dataclass
class PreprocessingOutput:
    """Encapsulates the preprocessed features ready for inference and display."""

    features_df: pd.DataFrame
    raw_input: Dict[str, Any]
    engineered_features: Dict[str, float] = field(default_factory=dict)
    scaled_features: Dict[str, float] = field(default_factory=dict)


def compute_engineered_features(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_term: float,
) -> Dict[str, float]:
    """Computes Total_Income, EMI, and Balance_Income with safe mathematical bounds.

    Args:
        applicant_income: Monthly applicant income in standard currency units.
        coapplicant_income: Monthly coapplicant income in standard currency units.
        loan_amount: Requested loan amount in thousands (e.g. 100 = $100k).
        loan_term: Repayment term in months (e.g. 360).

    Returns:
        Dictionary of raw and log-transformed engineered features.
    """
    total_income = float(applicant_income + coapplicant_income)
    safe_term = max(float(loan_term), 1.0)
    emi = float(loan_amount / safe_term)
    monthly_installment = float(emi * EMI_CURRENCY_MULTIPLIER)
    balance_income = float(total_income - monthly_installment)

    # Safe log transformations
    total_income_log = float(np.log(max(total_income, SAFE_LOG_FLOOR)))
    emi_log = float(np.log(max(emi, SAFE_LOG_FLOOR)))

    # Balance income offset c = 1769 is derived from Cell 41 of the training notebook:
    # c = abs(min(Balance_Income)) + 1 = 1769.0.
    # We defensively clamp the argument to >= MIN_BALANCE_INCOME_ARG (1.0) so that
    # np.log() is always finite and non-NaN even for extreme debt-to-income inputs.
    shifted_balance = balance_income + BALANCE_INCOME_OFFSET
    safe_balance_arg = max(shifted_balance, MIN_BALANCE_INCOME_ARG)
    balance_income_log = float(np.log(safe_balance_arg))

    return {
        "Total_Income": total_income,
        "EMI": emi,
        "Monthly_Installment": monthly_installment,
        "Balance_Income": balance_income,
        "Total_Income_log": total_income_log,
        "EMI_log": emi_log,
        "Balance_Income_log": balance_income_log,
    }


def preprocess_applicant_data(
    applicant_data: Dict[str, Any],
    scaler: Optional[Any] = None,
    validate: bool = True,
) -> PreprocessingOutput:
    """Preprocesses a single applicant's data dictionary into a model-ready DataFrame.

    Args:
        applicant_data: Raw input dictionary containing applicant fields.
        scaler: Optional pre-loaded StandardScaler; loaded automatically if omitted.
        validate: Whether to run the validation layer prior to preprocessing.

    Returns:
        PreprocessingOutput containing the model-ready 1-row DataFrame and metadata.
    """
    cleaned = applicant_data
    if validate:
        val_result = validate_applicant_data(applicant_data)
        val_result.raise_if_invalid()
        cleaned = val_result.cleaned_data

    # Obtain scaler from cached artifacts if not provided
    if scaler is None:
        artifacts = load_artifacts()
        scaler = artifacts.scaler

    eng = compute_engineered_features(
        applicant_income=float(cleaned["ApplicantIncome"]),
        coapplicant_income=float(cleaned["CoapplicantIncome"]),
        loan_amount=float(cleaned["LoanAmount"]),
        loan_term=float(cleaned["Loan_Amount_Term"]),
    )

    credit_history = float(cleaned["Credit_History"])

    # Build unscaled DataFrame for the numerical features
    unscaled_num_df = pd.DataFrame(
        [[eng["Total_Income_log"], eng["EMI_log"], eng["Balance_Income_log"]]],
        columns=SCALED_NUMERICAL_FEATURES,
    )

    # Apply fitted scaler
    scaled_array = scaler.transform(unscaled_num_df)
    scaled_num_df = pd.DataFrame(scaled_array, columns=SCALED_NUMERICAL_FEATURES)

    scaled_dict = {
        col: float(scaled_num_df[col].iloc[0]) for col in SCALED_NUMERICAL_FEATURES
    }

    # Construct final feature DataFrame in exact model order
    features_df = pd.DataFrame(
        [
            [
                credit_history,
                scaled_dict["Total_Income_log"],
                scaled_dict["EMI_log"],
                scaled_dict["Balance_Income_log"],
            ]
        ],
        columns=FINAL_MODEL_FEATURES,
    )

    return PreprocessingOutput(
        features_df=features_df,
        raw_input=cleaned,
        engineered_features=eng,
        scaled_features=scaled_dict,
    )
