# Interpreting SHAP Model Explanations Responsibly

## 1. Overview
The Loan Approval System integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, quantitative explanations for every prediction.

## 2. Generating Explanations

### CLI Command
```bash
python -m cli.main explain --input examples/example_input.json --plot docs/waterfall.png
```

### Agensi Script
```bash
python agensi-skill/scripts/explain.py --input agensi-skill/examples/example_input.json
```

---

## 3. Explanation Response Structure

```json
{
  "success": true,
  "prediction": "Approved",
  "base_value": 0.6877,
  "model_output_value": 0.9087,
  "contributions": [
    {
      "feature": "Credit_History",
      "display_name": "Credit History",
      "value": 1.0,
      "shap_value": 0.1134,
      "direction": "positive",
      "impact_magnitude": 0.1134,
      "summary": "Positive repayment record meeting lender guidelines"
    }
  ],
  "positive_factors": [ ... ],
  "negative_factors": [ ... ],
  "global_importance": {
    "Credit_History": 0.5585,
    "Total_Income_log": 0.1725,
    "Balance_Income_log": 0.1495,
    "EMI_log": 0.1196
  },
  "human_readable_summary": "The applicant meets the requirements for loan approval...",
  "disclaimer": "SHAP explanations illustrate feature weights..."
}
```

---

## 4. How to Interpret Feature Contributions

1. **Base Value (`base_value`):**
   - The expected probability of loan approval across the entire training dataset (~68.8%).
2. **SHAP Value (`shap_value`):**
   - The degree to which a specific applicant feature shifted the probability away from the base value.
   - **Positive SHAP Value:** Feature increased the probability of approval.
   - **Negative SHAP Value:** Feature decreased the probability of approval (risk factor).
3. **Global Feature Importance:**
   - `Credit_History` (~55.8%): Dominant factor across all applicants. Delinquency history is the most common reason for rejection.
   - `Total_Income_log` (~17.3%): Overall earning power.
   - `Balance_Income_log` (~15.0%): Net buffer after paying the monthly loan installment.
   - `EMI_log` (~12.0%): Loan installment burden relative to term.

---

## 5. Responsible AI & Advisory Boundaries

When reporting SHAP explanations to end users:
- **Never present model explanations as financial advice:** SHAP values represent statistical attribution within an empirical Random Forest model.
- **Do not promise approval based on adjustments:** E.g., do not say "If you increase your income by $500, your loan is guaranteed to be approved." State instead: "In this statistical model, higher income positively correlates with approval probability."
- **Highlight adverse factors respectfully:** When explaining rejection, clearly point out the primary risk driver (e.g. `Credit_History = 0` or insufficient `Balance_Income`).
