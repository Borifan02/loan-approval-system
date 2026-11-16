# User Guide: Loan Approval AI System

## 1. Introduction
The Loan Approval Prediction System provides automated, explainable credit decision support. It predicts loan outcomes (`Approved` or `Rejected`) and explains the decision through SHAP feature attributions.

---

## 2. Using the Streamlit Web Application

To launch the web interface:
```bash
streamlit run app.py
```

### Form Organization
The form is organized into 4 logical sections:
1. **Applicant Profile:** Demographic context (Gender, Marital Status, Dependents, Education Level). *Note: These are recorded for record completeness but excluded from scoring to protect against bias.*
2. **Employment & Income:** Self-employed status, Monthly Primary Applicant Income (USD), Monthly Coapplicant Income (USD), and Property Area.
3. **Loan Request Details:** Requested loan amount (in thousands of USD, e.g. $150 = 150) and Loan Term (months, e.g. 360 months = 30 years).
4. **Credit History Guidelines:** Indicator of whether applicant's credit history satisfies standard repayment guidelines (1 = Yes, 0 = No).

### Understanding the Results
- **Outcome Badge:** Displays `✅ Approved` or `❌ Rejected`.
- **Confidence Metric:** Displays statistical model probability (e.g. 90.9%).
- **Financial Breakdown:**
  - *Total Household Income:* Combined gross monthly income.
  - *Estimated Monthly EMI:* Monthly payment calculated as `(LoanAmount / Term) * 1000`.
  - *Net Balance Income:* Disposable income remaining after paying the monthly loan installment.
- **SHAP Waterfall Plot:**
  - **Green Bars (+):** Factors that increase approval likelihood.
  - **Red Bars (-):** Factors that decrease approval likelihood (risk factors).
- **Decision Narrative:** Plain-English explanation summarizing the key drivers.

---

## 3. Using the Command-Line Interface (CLI)

The CLI allows developers and automated systems to score applicants without opening a browser.

### Predict Loan Outcome
```bash
python -m cli.main predict --input examples/example_input.json
```

Output:
```json
{
  "success": true,
  "prediction": "Approved",
  "prediction_class": 1,
  "probabilities": {
    "rejected": 0.0913,
    "approved": 0.9087
  },
  "confidence": 0.9087,
  "confidence_note": "Model probability is a statistical estimate...",
  "engineered_metrics": {
    "Total_Income": 5849.0,
    "Monthly_EMI": 355.56,
    "Balance_Income": 5493.44
  }
}
```

### Generate SHAP Explanations
```bash
python -m cli.main explain --input examples/example_input.json --plot docs/waterfall_sample.png
```

### Inspect Model Specifications
```bash
python -m cli.main model-info
```

### Batch Scoring
```bash
python -m cli.main batch --input data/Loan_dataset_test.csv --output /tmp/batch_predictions.json
```

---

## 4. Scenario Walkthroughs

### Scenario A: Standard Approved Application
- **Applicant Income:** $5,849 / month
- **Coapplicant Income:** $0 / month
- **Loan Amount:** $128,000 (entered as `128`) over 360 months
- **Credit History:** Meets guidelines (`1`)
- **Outcome:** **Approved** (90.9% confidence). Credit history and healthy balance income are the main positive drivers.

### Scenario B: High-Risk Application (Rejection)
- **Applicant Income:** $2,500 / month
- **Coapplicant Income:** $0 / month
- **Loan Amount:** $200,000 (entered as `200`) over 180 months
- **Credit History:** Does not meet guidelines (`0`)
- **Outcome:** **Rejected** (87.5% rejection probability). Delinquent credit history and insufficient balance income trigger the adverse decision.
