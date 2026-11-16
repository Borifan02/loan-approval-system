"""Input validation layer for the Loan Approval system.

This module validates and cleans applicant data payloads prior to preprocessing
and inference, rejecting missing or invalid data with informative error messages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .config import REQUIRED_FINANCIAL_FIELDS


class ValidationError(ValueError):
    """Raised when input applicant data fails business rule or schema validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors: List[str] = errors or [message]


@dataclass
class ValidationResult:
    """Encapsulates the outcome of input validation."""

    is_valid: bool
    cleaned_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Convenience helper to raise ValidationError if is_valid is False."""
        if not self.is_valid:
            summary = "; ".join(self.errors)
            raise ValidationError(f"Validation failed: {summary}", errors=self.errors)


# Approved categorical vocabularies
VALID_GENDERS: Set[str] = {"male", "female"}
VALID_MARITAL_STATUS: Set[str] = {"yes", "no", "y", "n"}
VALID_EDUCATION: Set[str] = {"graduate", "not graduate"}
VALID_SELF_EMPLOYED: Set[str] = {"yes", "no", "y", "n"}
VALID_PROPERTY_AREAS: Set[str] = {"urban", "semiurban", "rural"}
VALID_DEPENDENTS: Set[str] = {"0", "1", "2", "3+", "3"}


def _parse_float(value: Any, field_name: str, errors: List[str]) -> Optional[float]:
    """Attempts to cast value to float, logging error if non-numeric or non-finite."""
    if value is None:
        errors.append(f"Field '{field_name}' cannot be None")
        return None
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            errors.append(f"Field '{field_name}' must be a finite number, got '{val}'")
            return None
        return val
    except (ValueError, TypeError):
        errors.append(f"Field '{field_name}' must be numeric, got '{value}'")
        return None


def _parse_dependents(value: Any, errors: List[str]) -> Optional[int]:
    """Parses dependents supporting strings like '3+' and standard integers."""
    if value is None:
        return 0  # Default to 0 if omitted in optional context

    val_str = str(value).strip().lower()
    if val_str in ("3+", "3", "3.0"):
        return 3
    if val_str in ("0", "0.0"):
        return 0
    if val_str in ("1", "1.0"):
        return 1
    if val_str in ("2", "2.0"):
        return 2

    try:
        val_int = int(float(val_str))
        if val_int < 0:
            errors.append(f"Dependents cannot be negative, got '{value}'")
            return None
        return min(val_int, 3)
    except (ValueError, TypeError):
        errors.append(
            f"Invalid value for 'Dependents': '{value}'. Expected 0, 1, 2, or '3+'"
        )
        return None


def validate_applicant_data(
    data: Dict[str, Any],
    require_demographics: bool = False,
) -> ValidationResult:
    """Validates raw applicant dictionary against required financial and domain rules.

    Args:
        data: Key-value dictionary containing applicant attributes.
        require_demographics: Whether demographic fields are strictly required.

    Returns:
        ValidationResult containing validation status, cleaned dict, and error list.
    """
    if not isinstance(data, dict):
        return ValidationResult(
            is_valid=False,
            cleaned_data={},
            errors=["Applicant payload must be a JSON object / dictionary"],
        )

    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    # 1. Check presence of core financial fields
    for req_field in REQUIRED_FINANCIAL_FIELDS:
        if req_field not in data or data[req_field] is None:
            errors.append(f"Missing required financial field: '{req_field}'")

    if errors:
        return ValidationResult(is_valid=False, cleaned_data={}, errors=errors)

    # 2. Validate Applicant Income
    app_income = _parse_float(data.get("ApplicantIncome"), "ApplicantIncome", errors)
    if app_income is not None:
        if app_income < 0.0:
            errors.append(f"ApplicantIncome cannot be negative, got {app_income}")
        else:
            cleaned["ApplicantIncome"] = app_income

    # 3. Validate Coapplicant Income
    coapp_income = _parse_float(data.get("CoapplicantIncome"), "CoapplicantIncome", errors)
    if coapp_income is not None:
        if coapp_income < 0.0:
            errors.append(f"CoapplicantIncome cannot be negative, got {coapp_income}")
        else:
            cleaned["CoapplicantIncome"] = coapp_income

    # 4. Check total household income
    if "ApplicantIncome" in cleaned and "CoapplicantIncome" in cleaned:
        total_income = cleaned["ApplicantIncome"] + cleaned["CoapplicantIncome"]
        if total_income <= 0.0:
            errors.append(
                f"Combined household income (ApplicantIncome + CoapplicantIncome) "
                f"must be greater than 0, got {total_income}"
            )

    # 5. Validate Loan Amount
    loan_amount = _parse_float(data.get("LoanAmount"), "LoanAmount", errors)
    if loan_amount is not None:
        if loan_amount <= 0.0:
            errors.append(f"LoanAmount must be strictly positive, got {loan_amount}")
        elif loan_amount > 10000.0:
            # LoanAmount in dataset is in thousands (e.g. 100 = $100k). > 10,000 = $10M
            errors.append(
                f"LoanAmount exceeds maximum allowable threshold (10,000 in thousands), "
                f"got {loan_amount}"
            )
        else:
            cleaned["LoanAmount"] = loan_amount

    # 6. Validate Loan Amount Term
    loan_term = _parse_float(data.get("Loan_Amount_Term"), "Loan_Amount_Term", errors)
    if loan_term is not None:
        if loan_term < 12.0 or loan_term > 480.0:
            errors.append(
                f"Loan_Amount_Term must be between 12 and 480 months, got {loan_term}"
            )
        else:
            cleaned["Loan_Amount_Term"] = loan_term

    # 7. Validate Credit History
    credit_hist = _parse_float(data.get("Credit_History"), "Credit_History", errors)
    if credit_hist is not None:
        if credit_hist in (1.0, 1):
            cleaned["Credit_History"] = 1.0
        elif credit_hist in (0.0, 0):
            cleaned["Credit_History"] = 0.0
        else:
            errors.append(
                f"Credit_History must be binary (1 for meets guidelines, 0 for does not), "
                f"got {credit_hist}"
            )

    # 8. Demographic and Contextual Fields (Validated when provided or when required)
    # Gender
    gender = data.get("Gender")
    if gender is not None:
        g_clean = str(gender).strip().lower()
        if g_clean in ("male", "m", "1"):
            cleaned["Gender"] = "Male"
        elif g_clean in ("female", "f", "0"):
            cleaned["Gender"] = "Female"
        else:
            errors.append(f"Invalid Gender '{gender}'. Expected 'Male' or 'Female'")
    elif require_demographics:
        errors.append("Missing required field: 'Gender'")

    # Married
    married = data.get("Married")
    if married is not None:
        m_clean = str(married).strip().lower()
        if m_clean in ("yes", "y", "1", "true"):
            cleaned["Married"] = "Yes"
        elif m_clean in ("no", "n", "0", "false"):
            cleaned["Married"] = "No"
        else:
            errors.append(f"Invalid Married status '{married}'. Expected 'Yes' or 'No'")
    elif require_demographics:
        errors.append("Missing required field: 'Married'")

    # Dependents
    if "Dependents" in data:
        dep = _parse_dependents(data.get("Dependents"), errors)
        if dep is not None:
            cleaned["Dependents"] = dep
    elif require_demographics:
        errors.append("Missing required field: 'Dependents'")

    # Education
    education = data.get("Education")
    if education is not None:
        e_clean = str(education).strip().lower()
        if e_clean in ("graduate", "0"):
            cleaned["Education"] = "Graduate"
        elif e_clean in ("not graduate", "under graduate", "undergraduate", "1"):
            cleaned["Education"] = "Not Graduate"
        else:
            errors.append(
                f"Invalid Education '{education}'. Expected 'Graduate' or 'Not Graduate'"
            )
    elif require_demographics:
        errors.append("Missing required field: 'Education'")

    # Self Employed
    self_emp = data.get("Self_Employed")
    if self_emp is not None:
        s_clean = str(self_emp).strip().lower()
        if s_clean in ("yes", "y", "1", "true"):
            cleaned["Self_Employed"] = "Yes"
        elif s_clean in ("no", "n", "0", "false"):
            cleaned["Self_Employed"] = "No"
        else:
            errors.append(
                f"Invalid Self_Employed '{self_emp}'. Expected 'Yes' or 'No'"
            )
    elif require_demographics:
        errors.append("Missing required field: 'Self_Employed'")

    # Property Area
    prop_area = data.get("Property_Area")
    if prop_area is not None:
        p_clean = str(prop_area).strip().lower()
        if p_clean in ("urban", "0"):
            cleaned["Property_Area"] = "Urban"
        elif p_clean in ("semiurban", "semi-urban", "1"):
            cleaned["Property_Area"] = "Semiurban"
        elif p_clean in ("rural", "2"):
            cleaned["Property_Area"] = "Rural"
        else:
            errors.append(
                f"Invalid Property_Area '{prop_area}'. Expected 'Urban', 'Semiurban', or 'Rural'"
            )
    elif require_demographics:
        errors.append("Missing required field: 'Property_Area'")

    return ValidationResult(
        is_valid=len(errors) == 0,
        cleaned_data=cleaned if len(errors) == 0 else {},
        errors=errors,
    )
