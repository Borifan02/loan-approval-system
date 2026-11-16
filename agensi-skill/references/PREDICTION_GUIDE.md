# Prediction Guide for AI Agents

## 1. Overview
This guide instructs AI coding agents on how to execute predictions, interpret outputs, and handle responses using the Loan Approval prediction engine.

## 2. Invocation Methods

### Method A: Command-Line Interface (Preferred for Agents)
Execute from project root or via script:

```bash
# Using CLI module
python -m cli.main predict --input examples/example_input.json

# Using inline JSON
python -m cli.main predict --input '{"ApplicantIncome": 5000, "CoapplicantIncome": 2000, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1}'

# Using Agensi Skill Script
python agensi-skill/scripts/predict.py --input agensi-skill/examples/example_input.json
```

### Method B: Python API (In-Process Execution)
```python
import sys
sys.path.insert(0, "src")
from loan_approval import predict_loan_approval

applicant = {
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}

result = predict_loan_approval(applicant)
print(f"Status: {result.prediction}, Confidence: {result.confidence:.2%}")
```

---

## 3. Interpreting Output Structure

A successful prediction response returns:

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
  "features_used": {
    "Credit_History": 1.0,
    "Total_Income_log": 0.0085,
    "EMI_log": -0.1604,
    "Balance_Income_log": 0.0146
  },
  "engineered_metrics": {
    "Total_Income": 5849.0,
    "Monthly_EMI": 355.56,
    "Balance_Income": 5493.44
  },
  "raw_input": { ... },
  "explanation_available": true,
  "error": null
}
```

### Key Fields to Report to Users
1. **Decision (`prediction`):** `"Approved"` or `"Rejected"`.
2. **Model Probability (`probabilities`):** Probability distribution across classes.
3. **Monthly Financial Obligations (`engineered_metrics`):**
   - `Total_Income`: Combined earning capacity.
   - `Monthly_EMI`: Calculated monthly loan installment.
   - `Balance_Income`: Disposable income remaining after paying EMI.
4. **Disclaimers:** Always communicate that the prediction represents a statistical model score and does not guarantee an official banking credit decision.
