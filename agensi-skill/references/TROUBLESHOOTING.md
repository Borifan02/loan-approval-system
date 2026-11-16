# Troubleshooting Guide for AI Agents

## 1. Common Operational Errors & Resolutions

### Error: `ModelLoadError: Unable to locate Random Forest model`
- **Root Cause:** Model artifacts (`models/rf_model.pkl`, `models/scaler.pkl`) are missing or unreadable.
- **Resolution:**
  1. Verify artifacts exist in `models/`: `ls -la models/`
  2. Ensure the working directory is the package root or set `MODEL_DIR`:
     `export MODEL_DIR=models`
  3. Ensure file permissions permit reading: `chmod 644 models/*.pkl`

---

### Error: `ValidationError: Missing required financial field`
- **Root Cause:** Input JSON payload lacks one or more required financial attributes:
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- **Resolution:**
  Verify that the input payload adheres to `references/INPUT_SCHEMA.md`.

---

### Error: `ValidationError: Combined household income must be greater than 0`
- **Root Cause:** Both `ApplicantIncome` and `CoapplicantIncome` are `0.0`.
- **Resolution:**
  Ensure the applicant has non-zero total household earnings before requesting evaluation.

---

### Error: `ValidationError: Loan_Amount_Term must be between 12 and 480 months`
- **Root Cause:** The term entered is either negative, zero, or exceeds 40 years (480 months).
- **Resolution:**
  Enter a realistic loan term in months (e.g. 120, 180, 240, or 360).

---

### Error: `DependencyError: No module named 'loan_approval'`
- **Root Cause:** The virtual environment is either not activated or the package directory is not in Python path.
- **Resolution:**
  1. Activate virtual environment: `source .venv/bin/activate` or use `.venv/bin/python`
  2. Install required dependencies: `pip install -r requirements-skill.txt`
  3. Run from the package root: `python scripts/predict.py --input examples/example_input.json`

---

### Error: `InvalidInputError: Input file not found`
- **Root Cause:** The file path passed to `--input` does not exist or has a typo.
- **Resolution:**
  Check path spelling or provide inline JSON.

---

## 2. Process Exit Codes Summary

| Exit Code | Meaning | Agent Action |
|---|---|---|
| `0` | Success | Parse stdout JSON for `prediction` or `contributions`. |
| `1` | Input / Validation Error | Inspect stdout JSON `message` and re-prompt user for valid inputs. |
| `2` | System / Environment Error | Inspect dependency installation or file paths. |

---

## 3. Security Notice on Pickle / Joblib Artifacts

Model artifacts are serialized using Python's `pickle`/`joblib` protocol.
> ⚠️ **Security Warning:** Unpickling files from untrusted third parties can lead to arbitrary code execution. Never replace `models/rf_model.pkl` or `models/scaler.pkl` with unverified external files. Only use the authentic artifacts shipped with this package.
