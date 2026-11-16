"""Configuration and environment settings for the Loan Approval system.

This module centralizes filesystem paths, domain constants, feature schemas,
and default hyperparameters used across the validation, preprocessing,
prediction, and explanation pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Ensure non-privileged writable cache directory for Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# -----------------------------------------------------------------------------
# Base Directories and Path Resolution
# -----------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Models directory and artifact paths
MODELS_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
DEFAULT_MODEL_FILENAME = "rf_model.pkl"
DEFAULT_SCALER_FILENAME = "scaler.pkl"
DEFAULT_METADATA_FILENAME = "metadata.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / DEFAULT_MODEL_FILENAME)))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(MODELS_DIR / DEFAULT_SCALER_FILENAME)))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(MODELS_DIR / DEFAULT_METADATA_FILENAME)))

# Fallback paths
FALLBACK_MODEL_PATH = PROJECT_ROOT / DEFAULT_MODEL_FILENAME
FALLBACK_SCALER_PATH = PROJECT_ROOT / DEFAULT_SCALER_FILENAME

# Data directory (checks lowercase 'data' then 'Data')
DATA_DIR = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").is_dir() else PROJECT_ROOT / "Data"

# -----------------------------------------------------------------------------
# Preprocessing and Mathematical Constants
# -----------------------------------------------------------------------------
# Offset c = 1769.0 from Cell 41 of Loan_prediction_main.ipynb:
# min_value = min(train['Balance_Income'].min(), test['Balance_Income'].min()) = -1768.0
# c = abs(min_value) + 1 = 1769.0
BALANCE_INCOME_OFFSET: float = 1769.0

# Mathematical safeguard epsilons
EPSILON: float = 1e-6
SAFE_LOG_FLOOR: float = 1e-4
MIN_BALANCE_INCOME_ARG: float = 1.0

# LoanAmount in thousands vs Income in single currency units
EMI_CURRENCY_MULTIPLIER: float = 1000.0

# -----------------------------------------------------------------------------
# Feature Schemas and Column Definitions
# -----------------------------------------------------------------------------
REQUIRED_FINANCIAL_FIELDS: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

OPTIONAL_CONTEXTUAL_FIELDS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

ALL_INPUT_FIELDS: List[str] = REQUIRED_FINANCIAL_FIELDS + OPTIONAL_CONTEXTUAL_FIELDS

SCALED_NUMERICAL_FEATURES: List[str] = [
    "Total_Income_log",
    "EMI_log",
    "Balance_Income_log",
]

FINAL_MODEL_FEATURES: List[str] = [
    "Credit_History",
    "Total_Income_log",
    "EMI_log",
    "Balance_Income_log",
]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "Credit_History": "Credit History",
    "Total_Income_log": "Total Income (Log)",
    "EMI_log": "Monthly EMI (Log)",
    "Balance_Income_log": "Balance Income (Log)",
}

# -----------------------------------------------------------------------------
# Classification Labels and Thresholds
# -----------------------------------------------------------------------------
CLASS_LABELS: Dict[int, str] = {
    0: "Rejected",
    1: "Approved",
}

DECISION_THRESHOLD: float = 0.50

# -----------------------------------------------------------------------------
# Explainability Settings
# -----------------------------------------------------------------------------
SHAP_DEFAULT_MAX_BACKGROUND: int = 100
SHAP_WATERFALL_MAX_DISPLAY: int = 4
