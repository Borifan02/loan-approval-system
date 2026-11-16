# Loan Approval AI — Production System & Agensi AI Agent Skill

**GitHub Repository:** [https://github.com/Borifan02/loan-approval-system](https://github.com/Borifan02/loan-approval-system)

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/SHAP-Explainable%20AI-green)](https://shap.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40%2B-red)](https://streamlit.io/)

A production-grade machine learning application and **Agensi-compatible AI Agent Skill** for predicting consumer loan approvals and generating quantitative SHAP explainability insights.

---

## 1. Project Overview

The **Loan Approval AI System** enhances retail lending decisions by combining an ensemble **Random Forest Classifier** with **SHAP (SHapley Additive exPlanations)**. It transforms empirical applicant financial data into actionable, interpretable credit decisions with statistical confidence ratings and plain-English factor narratives.

### Key Capabilities
- **Dual Deployment Modes:**
  1. **Interactive Web App:** Modern, responsive Streamlit dashboard with real-time SHAP waterfall charts.
  2. **Headless AI Agent Skill:** Agensi-compatible skill with CLI scripts for autonomous coding agents (Claude Code, OpenAI Codex, Cursor).
- **Decoupled Architecture:** Core prediction and explainability logic is completely separated from the presentation layer.
- **Fail-Fast Input Validation:** Rejects missing fields, negative earnings, and invalid inputs before inference.
- **Ethical & Fair Lending Design:** In compliance with ECOA principles, demographic attributes (`Gender`, `Married`, `Education`, `Property_Area`) are excluded from credit decisions.
- **100% Offline & Private:** Operates entirely locally with zero external API calls or telemetry.

---

## 2. System Architecture

```
loan-approval-system/
├── app.py                      # Production Streamlit UI
├── requirements.txt            # Production dependencies
├── pyproject.toml              # Modern package metadata
├── README.md                   # This documentation
├── LICENSE                     # MIT License
├── LICENSE_REVIEW.md           # Legal provenance & IP audit
├── AUDIT_REPORT.md             # Initial project audit
│
├── models/                     # Model artifact repository
│   ├── rf_model.pkl            # Trained Random Forest classifier
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── metadata.json           # Model specification & schema
│
├── data/                       # Dataset directory
│   ├── README.md               # Dataset provenance
│   ├── Loan_dataset_train.csv  # Training dataset
│   └── Loan_dataset_test.csv   # Evaluation dataset
│
├── src/                        # Core Python package
│   └── loan_approval/
│       ├── __init__.py         # Public exports
│       ├── config.py           # Configuration & constants
│       ├── model_loader.py     # Thread-safe model artifact caching
│       ├── validation.py       # Input validation & business constraints
│       ├── preprocessing.py    # Log transforms, EMI calculation, scaling
│       ├── predictor.py        # High-level inference engine
│       ├── explanation.py      # SHAP feature attribution & waterfall plotting
│       └── utils.py            # Helpers & logging
│
├── cli/                        # Command-Line Interface
│   ├── __init__.py
│   └── main.py                 # predict, explain, model-info, batch
│
├── agensi-skill/               # Agensi-Compatible AI Agent Skill
│   ├── SKILL.md                # Agent instructions with YAML frontmatter
│   ├── README.md               # Skill overview
│   ├── references/             # Schema, guides, and troubleshooting
│   ├── scripts/                # predict.py, explain.py
│   └── examples/               # example_input.json
│
├── tests/                      # Automated test suite (pytest)
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   ├── test_explanation.py
│   └── test_cli.py
│
├── examples/                   # Usage scripts and sample inputs
│   ├── example_input.json
│   ├── example_rejected.json
│   └── example_usage.py
│
├── docs/                       # Technical documentation suite
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   ├── MODEL_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   ├── TROUBLESHOOTING.md
│   ├── LIMITATIONS.md
│   └── SECURITY.md
│
└── scripts/                    # Automation utilities
    ├── verify_installation.py
    ├── run_tests.sh
    └── package_skill.sh
```

---

## 3. Technology Stack

- **Machine Learning:** `scikit-learn`, `joblib`, `numpy`, `pandas`
- **Explainability:** `shap` (TreeExplainer, Waterfall attributions)
- **Visualization:** `matplotlib` (headless backend), `seaborn`
- **User Interface:** `streamlit`
- **Testing & Quality Assurance:** `pytest`, `pytest-cov`
- **Agent Skill Distribution:** `Agensi.io` skill package specification

---

## 4. Quick Start & Installation

### Linux / Ubuntu Setup
```bash
# 1. Clone the repository
git clone https://github.com/Borifan02/loan-approval-system.git
cd loan-approval-system

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt pytest pytest-cov

# 4. Verify system installation and model artifacts
python scripts/verify_installation.py
```

---

## 5. Running the Application

### Option 1: Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Option 2: Command-Line Interface (CLI)
Score an applicant JSON file:
```bash
python -m cli.main predict --input examples/example_input.json
```

Generate a SHAP explanation and waterfall plot:
```bash
python -m cli.main explain --input examples/example_input.json --plot docs/waterfall.png
```

Inspect model parameters and metadata:
```bash
python -m cli.main model-info
```

Run batch predictions on a dataset:
```bash
python -m cli.main batch --input data/Loan_dataset_test.csv --output /tmp/batch_results.json
```

---

## 6. Example Prediction & Output

### Input Payload (`examples/example_input.json`)
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

### Prediction Output
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
  "confidence_note": "Model probability is a statistical estimate based on historical training data and does not constitute a guaranteed lending decision, legal approval, or financial advice.",
  "engineered_metrics": {
    "Total_Income": 5849.0,
    "Monthly_EMI": 355.56,
    "Balance_Income": 5493.44
  }
}
```

---

## 7. AI Agent Skill Integration (Agensi.io, Claude Code, Cursor)

The repository includes a ready-to-use skill under `agensi-skill/` with full YAML frontmatter in `SKILL.md`.

### Standalone Skill Execution
```bash
# Prediction
python agensi-skill/scripts/predict.py --input agensi-skill/examples/example_input.json

# Explanation
python agensi-skill/scripts/explain.py --input agensi-skill/examples/example_input.json
```

### Packaging for Agensi.io
To package the skill into a distributable ZIP:
```bash
bash scripts/package_skill.sh
```
This generates `loan-approval-ai-skill.zip` following Agensi distribution standards.

---

## 8. Automated Testing

Run the automated test suite with coverage:
```bash
pytest --cov=src/loan_approval tests/
```
Or execute the automated runner script:
```bash
bash scripts/run_tests.sh
```

---

## 9. Model Limitations & Ethical Considerations

- **Statistical Model, Not a Guarantee:** Predictions represent statistical probabilities based on a historical sample of 614 records and do not constitute formal lending approval or financial advice.
- **Credit History Sensitivity:** `Credit_History` accounts for ~55.8% of feature importance. Applicants lacking verified credit history will be predicted as Rejected regardless of income.
- **Fair Lending:** Demographic attributes are excluded to eliminate bias based on gender, marital status, education, or geographic area.

---

## 10. License & Attribution

This project is licensed under the **MIT License**.
- Original Prototype & Exploration Notebook: Copyright (c) 2024 Gianluca Sposito, PhD
- Production Architecture, CLI & Agensi AI Skill: Copyright (c) 2026 Loan Approval System Contributors
- See [LICENSE](LICENSE) and [LICENSE_REVIEW.md](LICENSE_REVIEW.md) for full intellectual property details.
