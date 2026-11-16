# Applicant Input Schema Reference

## Overview
This document specifies the exact JSON input schema required by the Loan Approval AI engine.
All requests must provide valid JSON objects containing the required financial parameters.

## Field Specifications

### Required Financial Fields
| Field Name | Type | Valid Range / Format | Description | Required? |
|---|---|---|---|---|
| `ApplicantIncome` | Float / Int | `>= 0.0` (currency units) | Primary applicant's monthly gross income. | **Yes** |
| `CoapplicantIncome` | Float / Int | `>= 0.0` (currency units) | Secondary / co-borrower monthly gross income. | **Yes** |
| `LoanAmount` | Float / Int | `> 0.0` and `<= 10000.0` | Requested principal in **thousands** of currency units (e.g., `120` = 120,000 units). | **Yes** |
| `Loan_Amount_Term` | Float / Int | `12.0` to `480.0` (months) | Loan repayment period in months (e.g. `360` = 30 years, `180` = 15 years). | **Yes** |
| `Credit_History` | Float / Int | `1` or `0` | Binary indicator whether applicant meets credit guidelines (1 = meets, 0 = does not). | **Yes** |

*Note: Combined household income (`ApplicantIncome + CoapplicantIncome`) must be strictly greater than 0.*

---

### Optional Contextual & Demographic Fields
These fields are preserved for applicant profile completeness, but are **excluded from the active machine learning model** to mitigate direct demographic bias.

| Field Name | Type | Permitted Values | Normalization Mapping |
|---|---|---|---|
| `Gender` | String | `"Male"`, `"Female"` | Case-insensitive ("m", "f", "1", "0" supported) |
| `Married` | String | `"Yes"`, `"No"` | Case-insensitive ("y", "n", "true", "false" supported) |
| `Dependents` | String / Int | `"0"`, `"1"`, `"2"`, `"3+"` | Normalized to integer `0`, `1`, `2`, `3` |
| `Education` | String | `"Graduate"`, `"Not Graduate"` | Case-insensitive ("undergraduate" mapped to "Not Graduate") |
| `Self_Employed` | String | `"Yes"`, `"No"` | Case-insensitive |
| `Property_Area` | String | `"Urban"`, `"Semiurban"`, `"Rural"` | Case-insensitive |

> **Fair Lending & Regulatory Notice:** Omitting demographic attributes reduces direct bias in model scoring. However, demographic feature omission **does not by itself guarantee legal compliance with the Equal Credit Opportunity Act (ECOA) or Regulation B**, because proxy discrimination or disparate impact may still arise through correlated variables. This software is an informational decision-support tool.

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
If any required field is missing or contains invalid values (e.g., negative income, non-numeric strings), the system returns an informative validation error with exit code `1`:

```json
{
  "success": false,
  "error_type": "ValidationError",
  "message": "Validation failed: ApplicantIncome cannot be negative, got -500.0"
}
```
