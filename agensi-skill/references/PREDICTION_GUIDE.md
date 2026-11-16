# Prediction Guide for AI Agents

## 1. Overview
This guide instructs AI coding agents on how to execute predictions, interpret outputs, and handle responses using the Loan Approval prediction engine from within the packaged skill.

---

## 2. Invocation Methods

### Method A: Standalone Script (Primary Method for Agents)
Execute from the extracted package root:

```bash
# Using input JSON file
python scripts/predict.py --input examples/example_input.json

# Using inline JSON string
python scripts/predict.py --input '{"ApplicantIncome": 5000, "CoapplicantIncome": 2000, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1}'
```

### Method B: In-Process Python API
```python
from loan_approval import predict_loan_approval

applicant = {
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}

result = predict_loan_approval(applicant)
print(f"Decision: {result.prediction}, Confidence: {result.confidence:.2%}")
```

---

## 3. Interpreting Output Structure

A successful prediction response returns machine-readable JSON on `stdout`:

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
  "confidence_note": "Model probability is a statistical estimate from a 150-tree Random Forest...",
  "features_used": {
    "Credit_History": 1.0,
    "Total_Income_log": 0.0085,
    "EMI_log": -0.1604,
    "Balance_Income_log": 0.0146
  },
  "engineered_metrics": {
    "Total_Income": 6500.0,
    "Monthly_EMI": 333.33,
    "Balance_Income": 6166.67
  },
  "raw_input": { ... },
  "explanation_available": true,
  "error": null
}
```

### Key Fields to Report to Users
1. **Decision (`prediction`):** `"Approved"` or `"Rejected"`.
2. **Statistical Model Probability (`confidence` / `probabilities`):** Empirical tree vote distribution across classes.
3. **Monthly Financial Obligations (`engineered_metrics`):**
   - `Total_Income`: Combined monthly earning capacity.
   - `Monthly_EMI`: Linear monthly installment approximation `(LoanAmount * 1000) / Loan_Amount_Term`.
   - `Balance_Income`: Disposable income buffer remaining after paying the monthly EMI (`Total_Income - Monthly_EMI`).
4. **Mandatory Disclaimer:** Always inform the user that the outcome is a statistical model estimate for decision support and does not guarantee a formal credit approval or binding underwriting decision.
