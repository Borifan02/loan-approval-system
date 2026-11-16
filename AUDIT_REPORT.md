# Loan Approval AI System — Comprehensive Project Audit Report

**Date:** 2026-11-16  
**Auditor:** Senior ML & Systems Engineering Review  
**Repository:** `loan-approval-system` (originating from `Borifan02/loan-approval-system`)  
**Status:** Audit Complete — Phase 1 Deliverable

---

## 1. Executive Summary

The existing repository implements a loan approval classification prototype trained on the Kaggle Loan Prediction dataset. It uses a `scikit-learn` Random Forest Classifier (`rf_model.pkl`), a `StandardScaler` (`scaler.pkl`), and a Streamlit front-end (`app.py`) with SHAP interpretability.

While the core machine learning concept is sound, the initial implementation suffers from critical design, architectural, and runtime vulnerabilities that prevent it from being production-grade or safely usable as an AI agent skill. Specifically:
- The prediction logic is tightly coupled inside the Streamlit UI file (`app.py`), preventing programmatic or CLI execution.
- Missing and fragile input validation leads to crashes on non-numeric strings, negative numbers, zero divisions, or inputs like `"3+"` for dependents.
- Logarithmic transformations produce `NaN` or `-inf` on zero or negative incomes/balance incomes, crashing model inference with `ValueError: Input contains NaN`.
- SHAP value extraction makes brittle assumptions about tensor dimensions (`shap_values.values[0, :, 0]`) and re-initializes explainers on every invocation.
- Model and scaler loaders use naive relative paths, failing when invoked outside the repository root.
- The codebase lacks unit/integration tests, packaging metadata, CLI interfaces, and AI agent integration files.
- Upstream copyright attribution exists (`Gianluca Sposito, PhD`), requiring formal documentation and legal compliance regarding provenance and redistribution.

This audit details all findings across 10 assessment criteria and outlines the architectural blueprint required to elevate this system to a production-ready application and Agensi-compatible AI Agent Skill.

---

## 2. Inventory of Existing Project Artifacts

| File / Directory | Type / Size | Purpose | Key Observations & Health |
|---|---|---|---|
| `app.py` | Python (8.0 KB, 204 lines) | Streamlit web application | Monolithic script mixing UI, feature engineering, loading, inference, and SHAP plotting. |
| `rf_model.pkl` | Joblib Pickle (~687 KB) | Trained Random Forest Classifier | `RandomForestClassifier` trained on 4 engineered features: `['Credit_History', 'Total_Income_log', 'EMI_log', 'Balance_Income_log']`. |
| `scaler.pkl` | Joblib Pickle (~975 B) | Fitted `StandardScaler` | Scales 3 log-transformed features: `['Total_Income_log', 'EMI_log', 'Balance_Income_log']`. |
| `Loan_prediction_main.ipynb` | Jupyter Notebook (~782 KB) | ML training & experimentation | Compares KNN, SVM, RF, XGBoost; documents removal of demographic features; justifies `c = 1769` offset for negative balance income. |
| `Data/Loan_dataset_train.csv` | CSV Dataset (38 KB, 614 rows) | Training dataset | Standard Kaggle Loan Prediction dataset (contains `Loan_ID`, demographics, income, loan amount/term, credit history, target). |
| `Data/Loan_dataset_test.csv` | CSV Dataset (22 KB, 367 rows) | Test dataset | Unlabeled applicant dataset (367 rows). |
| `Loan_Prediction_Image.jpg` | JPEG Image (9.3 KB) | Banner image for Streamlit | Static asset used in Streamlit UI header. |
| `README.md` | Markdown (3.1 KB, 69 lines) | Project documentation | Minimal instructions referencing `Borifan02/loan-approval-system`; lacks architecture, API, CLI, or agent documentation. |
| `requirements.txt` | Dependency list (204 B) | Python packages | Unpinned/pinned packages: `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.5.1`, `shap==0.46.0`, `streamlit==1.40.1`, `xgboost==2.1.1`. Includes heavy packages not required at runtime. |
| `LICENSE` | MIT License (1.1 KB) | Legal licensing | Copyright (c) 2024 Gianluca Sposito, PhD. Attribution must be preserved according to MIT terms. |

---

## 3. Deep-Dive Audit Findings (10 Assessment Dimensions)

### 3.1. Bugs and Potential Runtime Errors
1. **Unsafe Dependents Conversion:**
   In `app.py` (line 115): `int(Dependents)`. In credit datasets (including Kaggle's), the Dependents column commonly contains `"3+"`. Calling `int("3+")` raises `ValueError: invalid literal for int() with base 10: '3+'`.
2. **Division by Zero in EMI:**
   `EMI = LoanAmount / Loan_Amount_Term`. If `Loan_Amount_Term` is 0, Python raises `ZeroDivisionError: float division by zero`.
3. **Log of Zero and Negative Numbers:**
   `np.log(Total_Income)` and `np.log(EMI)` will produce `-inf` when `Total_Income == 0` or `LoanAmount == 0`.
   Furthermore, `Balance_Income = Total_Income - (EMI * 1000)`. If `Balance_Income + 1769 <= 0` (e.g. Total Income = $500, Loan Amount = $500k over 120 months -> EMI = 4.167k -> Balance Income = $500 - $4167 = -$3667; `-$3667 + 1769 = -1898`), `np.log(-1898)` returns `NaN` with a `RuntimeWarning: invalid value encountered in log`.
4. **NaN Ingestion in Scikit-Learn:**
   When `NaN` or `-inf` is passed into `scaler.transform()` or `model.predict()`, scikit-learn raises:
   `ValueError: Input contains NaN, infinity or a value too large for dtype('float64')`.
5. **Relative File Path Vulnerabilities:**
   `joblib.load('rf_model.pkl')` and `joblib.load('scaler.pkl')` in `app.py` use relative paths to the current working directory. Executing `streamlit run path/to/app.py` or importing from another directory immediately crashes with `FileNotFoundError`.

### 3.2. Incorrect or Inconsistent Preprocessing
1. **Unnecessary Demographic Field Requirements:**
   `app.py` prompts users for `Gender`, `Married`, `Education`, `Property_Area`, and `Self_Employed`, maps them to integers, creates a DataFrame, and then immediately drops them! While demographic features were present in the raw data, the final trained model (`rf_model.pkl`) utilizes strictly 4 features:
   `['Credit_History', 'Total_Income_log', 'EMI_log', 'Balance_Income_log']`.
   Demographic inputs should be optional metadata (or separated from model-critical inputs) to avoid giving false impressions that gender or marital status dictate creditworthiness.
2. **Hardcoded Mathematical Constants without Validation:**
   The constant `c = 1769` in `Balance_Income_log = np.log(Balance_Income + c)` originates from Cell 41 of the training notebook (`min_value = min(train_data['Balance_Income'].min(), test_data['Balance_Income'].min()) = -1768`, so `abs(-1768) + 1 = 1769`). In production inference, extreme edge-case applicant financials could yield `Balance_Income < -1769`, causing negative logarithm exceptions. Preprocessing must defensively clamp or safely validate lower bounds.

### 3.3. Missing Input Validation
1. **Free-form Text Inputs in UI:**
   `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount` are collected via `st.sidebar.text_input` with simple `float(...)` casts. Non-numeric strings, negative amounts, or empty fields cause unhandled exceptions that crash the user interface.
2. **Absence of Domain Rules:**
   No checks exist for:
   - Negative income values (e.g. `ApplicantIncome < 0`).
   - Non-positive loan amounts (`LoanAmount <= 0`).
   - Unrealistic loan terms (`Loan_Amount_Term <= 0` or `> 600` months).
   - Missing required keys in JSON payloads.
   - Categorical field validation against approved vocabularies.

### 3.4. Model Compatibility & Lifecycle Management
1. **Repeated Disk Reads:**
   `load_model()` and `load_scaler()` are executed synchronously inside the `Submit` button handler without caching. This causes redundant file I/O and deserialization on every prediction request.
2. **Model Metadata Obscurity:**
   No metadata class or configuration file documents the expected input columns, data types, scaler parameters, or output class mappings (`0 -> Rejected`, `1 -> Approved`).
3. **Probability Information Suppressed:**
   `app.py` commented out `prediction_proba = model.predict_proba(processed_data)`, withholding calibrated confidence probabilities from applicants and downstream agent systems.

### 3.5. Fragile SHAP Implementation
1. **Dynamic Explainer Initialization:**
   `explainer = shap.Explainer(model)` is instantiated on every click. Initializing SHAP tree explainers on the fly is computationally expensive and causes latency spikes.
2. **Brittle Multi-Index Slicing:**
   `shap_values.values[0, :, 0]` and `shap_values.base_values[0, 0]` index class index 0 (Rejected) rather than the predicted class or positive class (Approved). Furthermore, SHAP output dimensions vary between versions and model types; indexing into 3D tensors directly without shape checking causes `IndexError` on differing SHAP versions.
3. **Global Matplotlib State Leak:**
   `fig_inc = plt.figure()` followed by `st.pyplot(fig_inc)` without explicitly closing figures or managing thread safety leads to memory accumulation under continuous usage.

### 3.6. Security, Privacy, and Dependency Footprint
1. **Arbitrary Code Execution via Pickle:**
   Loading `.pkl` files with `joblib.load` carries standard Python pickle risks. Paths must be locked to internal project directories, and checksum/integrity checks should be established.
2. **Heavyweight Unused Dependencies:**
   `requirements.txt` specifies `xgboost==2.1.1` and `seaborn==0.13.2`. On Linux systems, `xgboost` pulls massive CUDA/NCCL wheels (>350 MB) despite not being used by the runtime prediction engine. The production runtime should distinguish core inference dependencies from training/visualization dependencies.
3. **Information Disclosure in UI/CLI:**
   Raw stack traces are currently displayed directly to users on invalid input instead of graceful, sanitized error responses.

### 3.7. Missing Automated Test Suite
- Zero automated tests exist in the repository (no `tests/` directory, no `pytest` configurations, no test coverage reporting).
- Edge cases (zero income, extreme loan amounts, invalid text strings, missing fields) have never been systematically verified.

### 3.8. Missing Reusable Architecture & CLI
- All logic is confined to `app.py`. There is no importable Python package (`src/loan_approval/`).
- No Command Line Interface (CLI) exists for batch inference, headless environments, CI/CD, or automated tooling.

### 3.9. Missing AI Agent Integration
- No Agensi skill definition (`SKILL.md`) exists.
- Coding agents (Claude Code, OpenAI Codex, Cursor) cannot programmatically discover, invoke, or understand the system's input requirements, execution commands, or output formats.

### 3.10. Missing Distribution & Packaging Metadata
- No packaging script (`package_skill.sh`) or distributable ZIP (`loan-approval-ai-skill.zip`).
- No `.env.example`, `.gitignore` enhancements, or standard project layout.

---

## 4. Proposed Modular Architecture

To resolve all audit findings, the project will be reorganized into a clean, layered architecture:

```
loan-approval-system/
├── app.py                      # Production Streamlit UI (presentation layer only)
├── requirements.txt            # Production runtime dependencies
├── pyproject.toml              # Modern package definition
├── README.md                   # Comprehensive documentation
├── LICENSE                     # MIT License preserving upstream copyright
├── LICENSE_REVIEW.md           # Legal provenance and IP audit
├── AUDIT_REPORT.md             # This comprehensive audit
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment configuration template
│
├── models/                     # Dedicated model artifact repository
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl              # Fitted feature scaler
│   └── metadata.json           # Model schema, versioning, feature order
│
├── data/                       # Dataset directory
│   ├── README.md               # Dataset provenance and Kaggle reference
│   ├── Loan_dataset_train.csv  # Training data (preserved)
│   └── Loan_dataset_test.csv   # Test data (preserved)
│
├── src/                        # Core Python package
│   └── loan_approval/
│       ├── __init__.py         # Package exports
│       ├── config.py           # Absolute paths, feature names, defaults
│       ├── model_loader.py     # Thread-safe cached artifact loader
│       ├── validation.py       # Pydantic/dataclass schema & validation rules
│       ├── preprocessing.py    # Robust pipeline (safe logs, EMI, scaling)
│       ├── predictor.py        # Prediction engine with probabilities & labels
│       ├── explanation.py      # Resilient SHAP explanations (waterfall/importance)
│       └── utils.py            # Formatting, logging, and helpers
│
├── cli/                        # Command-Line Interface
│   ├── __init__.py
│   └── main.py                 # predict, explain, model-info, batch subcommands
│
├── tests/                      # Automated pytest suite
│   ├── __init__.py
│   ├── conftest.py             # Reusable fixtures (samples, engine, loader)
│   ├── test_validation.py      # Schema and edge-case boundary tests
│   ├── test_preprocessing.py   # Transformation, log, and scaling tests
│   ├── test_prediction.py      # Inference output structure and probability tests
│   ├── test_explanation.py     # SHAP contribution and visualization tests
│   └── test_cli.py             # CLI command, exit code, and JSON output tests
│
├── examples/                   # Sample payloads and usage scripts
│   ├── example_input.json      # Standard valid applicant JSON
│   ├── example_rejected.json   # Expected rejection payload
│   └── example_usage.py        # Python SDK quickstart script
│
├── docs/                       # Technical documentation suite
│   ├── ARCHITECTURE.md         # Component diagrams and data flows
│   ├── INSTALLATION.md         # Linux/macOS/Windows setup guide
│   ├── USER_GUIDE.md           # Streamlit and CLI user manual
│   ├── MODEL_DOCUMENTATION.md  # Training, hyperparameters, metrics, ethics
│   ├── API_REFERENCE.md        # Python functions and CLI arguments
│   ├── TROUBLESHOOTING.md      # Common errors and remedies
│   ├── LIMITATIONS.md          # Model scope, boundaries, and disclaimers
│   └── SECURITY.md             # Security, privacy, and responsible AI policies
│
├── agensi-skill/               # Agensi-compatible AI Agent Skill
│   ├── SKILL.md                # Agent instructions with YAML frontmatter
│   ├── README.md               # Skill overview
│   ├── references/
│   │   ├── INPUT_SCHEMA.md     # Input parameters and data types
│   │   ├── PREDICTION_GUIDE.md # Execution commands and response interpretation
│   │   ├── MODEL_EXPLANATION.md# Interpreting SHAP values responsibly
│   │   └── TROUBLESHOOTING.md  # Agent error handling procedures
│   ├── scripts/
│   │   ├── predict.py          # Standalone prediction CLI script
│   │   └── explain.py          # Standalone explanation CLI script
│   └── examples/
│       └── example_input.json  # Reference agent payload
│
└── scripts/                    # Automation and verification utilities
    ├── verify_installation.py  # Self-test script checking environment & artifacts
    ├── run_tests.sh            # One-step test runner with coverage
    └── package_skill.sh        # Deterministic packaging script for agensi ZIP
```

---

## 5. Implementation Roadmap

1. **Stage 1:** Deliver `AUDIT_REPORT.md` and `docs/ARCHITECTURE.md`; establish target directory structure.
2. **Stage 2:** Construct `src/loan_approval/` package: `config.py`, `model_loader.py`, `preprocessing.py`.
3. **Stage 3:** Build comprehensive input validation (`validation.py`) and prediction engine (`predictor.py`).
4. **Stage 4:** Implement robust SHAP explainability engine (`explanation.py`) with fallback handling.
5. **Stage 5:** Modernize Streamlit application (`app.py`) to consume the clean engine.
6. **Stage 6:** Build production CLI (`cli/main.py`) with JSON stdout and POSIX exit codes.
7. **Stage 7:** Author Agensi AI Agent Skill (`agensi-skill/SKILL.md`, scripts, references).
8. **Stage 8:** Author complete technical documentation in `docs/` and rewrite `README.md`.
9. **Stage 9:** Develop and execute comprehensive automated test suite (`pytest`).
10. **Stage 10:** Conduct security, privacy, ethical AI, and intellectual property review (`LICENSE_REVIEW.md`).
11. **Stage 11:** Package and verify distributable `loan-approval-ai-skill.zip`.
12. **Stage 12:** Final end-to-end verification and comprehensive completion report.
