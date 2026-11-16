# Applicant Input Schema Reference

## Overview
This document specifies the exact JSON input schema required by the Loan Approval AI engine.
All requests must provide valid JSON objects containing the required financial parameters.

## Field Specifications

### Required Financial Fields
| Field Name | Type | Valid Range / Format | Description | Required? |
|---|---|---|---|---|
| `ApplicantIncome` | Float / Int | `>= 0.0` (USD) | Primary applicant's monthly gross income. | **Yes** |
| `CoapplicantIncome` | Float / Int | `>= 0.0` (USD) | Secondary / co-borrower monthly gross income. | **Yes** |
| `LoanAmount` | Float / Int | `> 0.0` and `<= 10000.0` | Requested loan principal in **thousands** of USD (e.g., `150` = $150,000). | **Yes** |
| `Loan_Amount_Term` | Float / Int | `12.0` to `480.0` (months) | Loan repayment period in months (e.g. `360` = 30 years, `180` = 15 years). | **Yes** |
| `Credit_History` | Float / Int | `1` or `0` | Binary indicator whether applicant meets credit guidelines (1 = yes, 0 = no). | **Yes** |

*Note: Combined household income (`ApplicantIncome + CoapplicantIncome`) must be strictly greater than $0.00.*

---

### Optional Contextual & Demographic Fields
These fields are preserved for application completeness and record-keeping, but are **excluded from the active machine learning model** in accordance with fair lending principles (ECOA).

| Field Name | Type | Permitted Values | Normalization Mapping |
|---|---|---|---|
| `Gender` | String | `"Male"`, `"Female"` | Case-insensitive ("m", "f", "1", "0" supported) |
| `Married` | String | `"Yes"`, `"No"` | Case-insensitive ("y", "n", "true", "false" supported) |
| `Dependents` | String / Int | `"0"`, `"1"`, `"2"`, `"3+"` | Normalized to integer `0`, `1`, `2`, `3` |
| `Education` | String | `"Graduate"`, `"Not Graduate"` | Case-insensitive ("undergraduate" mapped to "Not Graduate") |
| `Self_Employed` | String | `"Yes"`, `"No"` | Case-insensitive |
| `Property_Area` | String | `"Urban"`, `"Semiurban"`, `"Rural"` | Case-insensitive |

---

## Example Valid JSON Payload

```json
{
  "ApplicantIncome": 5849,
  "CoapplicantIncome": 0,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Gender": "Male",
  "Married": "No",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "Property_Area": "Urban"
}
```

## Validation Error Handling
If any required field is missing or contains invalid data types (e.g., negative income, non-numeric strings), the system returns an informative validation error with exit code `1`:

```json
{
  "success": false,
  "error_type": "ValidationError",
  "message": "Validation failed: ApplicantIncome cannot be negative, got -500.0"
}
```
