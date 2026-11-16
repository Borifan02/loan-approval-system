"""Unit tests for the applicant validation layer."""

import pytest
from loan_approval import ValidationError, validate_applicant_data


def test_valid_applicant_passes(valid_applicant):
    result = validate_applicant_data(valid_applicant)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.cleaned_data["ApplicantIncome"] == 5849.0
    assert result.cleaned_data["LoanAmount"] == 128.0
    assert result.cleaned_data["Credit_History"] == 1.0


def test_missing_required_fields():
    data = {"ApplicantIncome": 5000}
    result = validate_applicant_data(data)
    assert result.is_valid is False
    assert any("CoapplicantIncome" in err for err in result.errors)
    assert any("LoanAmount" in err for err in result.errors)
    assert any("Loan_Amount_Term" in err for err in result.errors)
    assert any("Credit_History" in err for err in result.errors)


def test_negative_income_rejected(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["ApplicantIncome"] = -500.0
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("ApplicantIncome cannot be negative" in err for err in result.errors)


def test_zero_total_household_income_rejected(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["ApplicantIncome"] = 0.0
    bad_data["CoapplicantIncome"] = 0.0
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("Combined household income" in err for err in result.errors)


def test_invalid_loan_amount_rejected(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["LoanAmount"] = 0.0
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("LoanAmount must be strictly positive" in err for err in result.errors)

    bad_data["LoanAmount"] = 999999.0
    result2 = validate_applicant_data(bad_data)
    assert result2.is_valid is False
    assert any("LoanAmount exceeds maximum allowable threshold" in err for err in result2.errors)


def test_invalid_loan_term_rejected(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["Loan_Amount_Term"] = 5.0
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("Loan_Amount_Term must be between 12 and 480" in err for err in result.errors)

    bad_data["Loan_Amount_Term"] = 600.0
    result2 = validate_applicant_data(bad_data)
    assert result2.is_valid is False
    assert any("Loan_Amount_Term must be between 12 and 480" in err for err in result2.errors)


def test_invalid_credit_history_rejected(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["Credit_History"] = 5.0
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("Credit_History must be binary" in err for err in result.errors)


@pytest.mark.parametrize("dep_input,expected_int", [
    ("0", 0),
    (0, 0),
    ("1", 1),
    ("2", 2),
    ("3+", 3),
    ("3", 3),
    (3, 3),
    (4, 3),
])
def test_dependents_parsing(valid_applicant, dep_input, expected_int):
    data = valid_applicant.copy()
    data["Dependents"] = dep_input
    result = validate_applicant_data(data)
    assert result.is_valid is True
    assert result.cleaned_data["Dependents"] == expected_int


def test_invalid_dependents_string(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["Dependents"] = "many"
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("Invalid value for 'Dependents'" in err for err in result.errors)


def test_invalid_categorical_values(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["Gender"] = "Alien"
    bad_data["Education"] = "Doctorate"
    bad_data["Property_Area"] = "Mars"
    result = validate_applicant_data(bad_data)
    assert result.is_valid is False
    assert any("Invalid Gender" in err for err in result.errors)
    assert any("Invalid Education" in err for err in result.errors)
    assert any("Invalid Property_Area" in err for err in result.errors)


def test_raise_if_invalid_throws_exception(valid_applicant):
    bad_data = valid_applicant.copy()
    bad_data["ApplicantIncome"] = -100
    res = validate_applicant_data(bad_data)
    with pytest.raises(ValidationError) as exc_info:
        res.raise_if_invalid()
    assert "ApplicantIncome cannot be negative" in str(exc_info.value)


def test_non_dict_payload():
    result = validate_applicant_data(["not", "a", "dict"])
    assert result.is_valid is False
    assert "must be a JSON object" in result.errors[0]
