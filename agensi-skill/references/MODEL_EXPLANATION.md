# Interpreting SHAP Model Explanations Responsibly

## 1. Overview
The Loan Approval System integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, quantitative feature attributions for each applicant prediction.

---

## 2. Generating Explanations

Execute from the extracted package root:

```bash
# Generate JSON feature attribution report
python scripts/explain.py --input examples/example_input.json

# Optionally render and save a visual waterfall plot
python scripts/explain.py --input examples/example_input.json --plot waterfall.png
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
  "disclaimer": "SHAP explanations illustrate feature weights in an empirical model..."
}
```

---

## 4. How to Interpret Feature Contributions

1. **Base Value (`base_value`):**
   - The average predicted approval probability across the historical training baseline (~68.8%).
2. **SHAP Value (`shap_value`):**
   - The marginal contribution of each applicant feature shifting the probability toward approval (`+`) or rejection (`-`).
   - **Positive SHAP Value:** Feature increased the likelihood of approval.
   - **Negative SHAP Value:** Feature decreased the likelihood of approval (identified risk factor).
3. **Global Feature Importance:**
   - `Credit_History` (~55.8%): The single most decisive factor. A value of `0` almost universally drives rejection.
   - `Total_Income_log` (~17.3%): Overall earning power.
   - `Balance_Income_log` (~15.0%): Net disposable buffer after paying the monthly loan installment.
   - `EMI_log` (~12.0%): Monthly debt installment obligation relative to requested term.

---

## 5. Responsible AI & Advisory Boundaries

When communicating SHAP explanations to users or loan applicants:
- **Do Not Promise Approvals:** Never state that changing an attribute guarantees approval (e.g., do not say "If you increase your income by $500, your loan will be approved"). State instead: "Higher income positively correlates with approval probability in this model."
- **Fair Lending Context:** The model omits demographic features (`Gender`, `Married`, `Education`, `Property_Area`) to avoid direct disparate treatment. However, agents must understand that demographic exclusion alone does not eliminate indirect proxy correlation.
- **Decision-Support Status:** Emphasize that the explanation reflects the internal reasoning of an empirical Random Forest model and is intended for preliminary evaluation only.
