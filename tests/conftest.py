"""Pytest fixtures and test configuration for Loan Approval AI."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure non-privileged writable cache directory for Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loan_approval import load_artifacts


@pytest.fixture(scope="session")
def artifacts():
    """Provides cached model artifacts across test session."""
    return load_artifacts()


@pytest.fixture
def valid_applicant() -> Dict[str, Any]:
    """Provides a standard valid applicant profile expected to be approved."""
    return {
        "ApplicantIncome": 5849.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 128.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }


@pytest.fixture
def rejected_applicant() -> Dict[str, Any]:
    """Provides an applicant profile expected to be rejected (no credit history)."""
    return {
        "ApplicantIncome": 2500.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 200.0,
        "Loan_Amount_Term": 180.0,
        "Credit_History": 0.0,
        "Gender": "Female",
        "Married": "No",
        "Dependents": "3+",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "Property_Area": "Rural",
    }
