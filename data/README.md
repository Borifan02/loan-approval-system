# Loan Approval Dataset

## Source & Provenance
The datasets in this directory originate from the popular **Kaggle Loan Prediction Dataset**:
- **Dataset Title:** Loan Prediction Problem Dataset
- **Format:** CSV (Comma-Separated Values)
- **Primary Domain:** Retail banking, consumer credit assessment

## Files
- `Loan_dataset_train.csv`: 614 applicant records with 12 features and 1 target variable (`Loan_Status`).
- `Loan_dataset_test.csv`: 367 applicant records without target labels for out-of-sample evaluation.

## Feature Description
| Variable | Description | Type | Raw Values / Range |
|---|---|---|---|
| `Loan_ID` | Unique applicant identifier | Identifier | String (e.g., `LP001002`) |
| `Gender` | Applicant gender | Categorical | `Male`, `Female` |
| `Married` | Applicant marital status | Categorical | `Yes`, `No` |
| `Dependents` | Number of financial dependents | Categorical / Count | `0`, `1`, `2`, `3+` |
| `Education` | Highest educational attainment | Categorical | `Graduate`, `Not Graduate` |
| `Self_Employed` | Self-employment status | Categorical | `Yes`, `No` |
| `ApplicantIncome` | Primary applicant monthly income | Numeric | USD ($150 - $81,000) |
| `CoapplicantIncome` | Coapplicant monthly income | Numeric | USD ($0 - $41,667) |
| `LoanAmount` | Requested loan amount | Numeric | USD thousands ($9k - $700k) |
| `Loan_Amount_Term` | Loan repayment term | Numeric | Months (12 - 480) |
| `Credit_History` | Credit guidelines repayment history | Binary Indicator | `1.0` (meets guidelines), `0.0` (does not) |
| `Property_Area` | Property location category | Categorical | `Urban`, `Semiurban`, `Rural` |
| `Loan_Status` | Approval decision (target) | Binary Indicator | `Y` (Approved), `N` (Rejected) |

## Responsible AI & Fair Lending Notice
Demographic attributes (`Gender`, `Married`, `Education`, `Property_Area`, `Self_Employed`, `Dependents`) were present in the source dataset. In accordance with ethical lending guidelines (such as the US Equal Credit Opportunity Act - ECOA), the final production model is trained strictly on financial capability and credit history features:
- `Credit_History`
- `Total_Income_log`
- `EMI_log`
- `Balance_Income_log`

Demographic variables are not used in credit approval decisions.
