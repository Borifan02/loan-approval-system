# Troubleshooting Guide for AI Agents

## 1. Common Operational Errors & Resolutions

### Error: `ModelLoadError: Unable to locate Random Forest model`
- **Root Cause:** Model artifacts (`rf_model.pkl`, `scaler.pkl`) are missing from `models/` or root directory.
- **Resolution:**
  1. Check if model files exist: `ls -la models/`
  2. If missing, restore from backup or project root: `cp rf_model.pkl scaler.pkl models/`
  3. Verify environment variables: Ensure `MODEL_PATH` points to a readable `.pkl` file.

---

### Error: `ValidationError: Missing required financial field`
- **Root Cause:** Input JSON payload lacks one or more required financial attributes:
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- **Resolution:**
  Verify that the input payload adheres to `agensi-skill/references/INPUT_SCHEMA.md`.

---

### Error: `ValidationError: Combined household income must be greater than 0`
- **Root Cause:** Both `ApplicantIncome` and `CoapplicantIncome` are set to `0.0`.
- **Resolution:**
  Ensure the applicant has non-zero total household earnings before requesting evaluation.

---

### Error: `ValidationError: Loan_Amount_Term must be between 12 and 480 months`
- **Root Cause:** The term entered is either negative, zero, or exceeds 40 years (480 months).
- **Resolution:**
  Enter a realistic loan term (e.g. 120, 180, 240, or 360 months).

---

### Error: `DependencyError: No module named 'loan_approval'`
- **Root Cause:** The virtual environment is either not activated or the package is not in the Python path.
- **Resolution:**
  1. Verify virtualenv: `source .venv/bin/activate` or use `.venv/bin/python`
  2. Ensure dependencies are installed: `pip install -r requirements.txt`
  3. Execute via module syntax: `python -m cli.main ...`

---

### Process Exit Codes Summary
| Exit Code | Meaning | Agent Action |
|---|---|---|
| `0` | Success | Parse stdout JSON for `prediction` or `contributions`. |
| `1` | Input / Validation Error | Inspect stdout/stderr JSON `message` and re-prompt user for valid inputs. |
| `2` | System / Environment Error | Inspect dependency installation or file paths. |
