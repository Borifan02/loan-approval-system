# Loan Approval AI — Agensi AI Agent Skill

**GitHub Repository:** https://github.com/Borifan02/loan-approval-system

This package provides a production-quality, explainable loan approval prediction engine packaged as a standardized, self-contained **Agensi.io AI Agent Skill**. It enables AI coding agents (such as Claude Code, OpenAI Codex, and Cursor) to evaluate consumer loan eligibility, compute financial debt metrics, and generate SHAP-based feature attributions.

---

## 1. Directory Structure

When extracted, the package provides the following self-contained structure:

```
loan-approval-ai-skill/
├── SKILL.md                  # Main skill specification with valid YAML frontmatter
├── README.md                 # Package documentation and usage instructions
├── requirements-skill.txt    # Lightweight runtime dependencies (no UI/heavy packages)
├── scripts/                  # Standalone execution entry points for agents
│   ├── predict.py            # Headless prediction script
│   └── explain.py            # Headless SHAP explanation and waterfall plot script
├── src/
│   └── loan_approval/        # Modular core inference engine & mathematical safeguards
├── models/                   # Serialized ML artifacts & metadata
│   ├── rf_model.pkl          # Trained Random Forest classifier (150 trees)
│   ├── scaler.pkl            # Trained StandardScaler (3 continuous log features)
│   └── metadata.json         # Feature schema and training metadata
├── references/               # Detailed documentation for coding agents
│   ├── INPUT_SCHEMA.md       # JSON payload parameters, ranges, and types
│   ├── PREDICTION_GUIDE.md   # CLI execution guide and response schemas
│   ├── MODEL_EXPLANATION.md  # How to interpret SHAP feature attributions
│   └── TROUBLESHOOTING.md    # Exit codes, validation errors, and diagnostics
└── examples/
    └── example_input.json    # Standard benchmark applicant payload
```

---

## 2. Installation & Prerequisites

The skill requires **Python 3.9+** and standard scientific Python packages. It does not require Streamlit, PyArrow, or external database drivers.

From the extracted package root, install runtime dependencies:

```bash
pip install -r requirements-skill.txt
```

Verify dependencies in Python:
```bash
python -c "import sklearn, pandas, numpy, joblib, shap, matplotlib; print('Environment OK')"
```

---

## 3. Quick Start (Execution from Package Root)

All commands are executed from the root of the extracted skill directory:

### 3.1. Predict Loan Approval
```bash
# Predict using example input file
python scripts/predict.py --input examples/example_input.json

# Predict using inline JSON
python scripts/predict.py --input '{"ApplicantIncome": 5000, "CoapplicantIncome": 1500, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": 1}'
```

*Expected output (`stdout`):*
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
  "engineered_metrics": {
    "Total_Income": 6500.0,
    "Monthly_EMI": 333.33,
    "Balance_Income": 6166.67
  }
}
```

### 3.2. Generate SHAP Explanations
```bash
# Generate JSON explanation of positive and negative drivers
python scripts/explain.py --input examples/example_input.json

# Optionally render and save a visual waterfall plot (PNG)
python scripts/explain.py --input examples/example_input.json --plot waterfall.png
```

---

## 4. Financial Units & Responsible AI Principles

### Feature Units & Definitions
- **`ApplicantIncome` & `CoapplicantIncome`**: Monthly gross income in nominal currency units.
- **`LoanAmount`**: Requested principal in **thousands** of currency units (e.g., `120` represents 120,000 currency units).
- **`Loan_Amount_Term`**: Repayment schedule expressed in **months** (`12` to `480`).
- **`EMI`**: Calculated linear monthly installment: `(LoanAmount * 1000) / Loan_Amount_Term`.
- **`Balance_Income`**: Estimated disposable buffer after installment: `Total_Income - EMI`.

### Decision-Support Tool (Not an Automated Underwriter)
- **Statistical Estimates, Not Guarantees:** Model output probabilities reflect tree vote fractions across a 150-tree Random Forest trained on 614 benchmark samples. They do not represent actuarial probabilities of default or binding legal commitments.
- **Advisory Function:** This tool is designed strictly for preliminary credit risk assessment and applicant pre-screening. Final credit decisions must involve qualified human underwriters.

### Fair Lending & Demographic Exclusions
- Demographic variables (`Gender`, `Married`, `Education`, `Property_Area`) are intentionally excluded from the active classifier to mitigate direct bias.
- **Fair Lending Limitation Notice:** Demographic feature omission alone **does not eliminate indirect proxy discrimination** (e.g., through correlated financial variables) and **does not constitute legal compliance with the Equal Credit Opportunity Act (ECOA) or Regulation B**. Any production deployment must undergo independent fair lending audits.

---

## 5. Security & Privacy

- **Zero Network Egress:** The engine runs entirely locally. Applicant data is processed in-memory and is never logged to disk or transmitted over the internet.
- **Artifact Security:** Models are stored in Python `pickle` / `joblib` format. Loading unverified pickle files from untrusted sources can introduce security vulnerabilities. Only load the verified model artifacts shipped in `models/`.

---

## 6. License, Attribution & Repository

- **GitHub Repository:** https://github.com/Borifan02/loan-approval-system
- **Original Baseline Prototype & Training Notebook:** Copyright (c) 2024 Gianluca Sposito, PhD ([MIT License](LICENSE)).
- **Production Architecture, CLI, & Agensi AI Skill Packaging:** Copyright (c) 2026 Loan Approval System Contributors ([MIT License](LICENSE)).
