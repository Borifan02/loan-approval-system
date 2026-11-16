# Troubleshooting & Operational Remedies

## 1. Quick Diagnostic Checklist
Before troubleshooting specific errors, run the automated verification script:
```bash
python scripts/verify_installation.py
```

---

## 2. Common Issues and Resolutions

### 2.1. Model Artifact Loading Failures
**Symptom:**
`ModelLoadError: Unable to locate Random Forest model. Checked: models/rf_model.pkl, rf_model.pkl`
**Cause:**
Artifacts were moved or the script is executing from an unrecognized working directory.
**Resolution:**
1. Confirm artifact existence:
   ```bash
   ls -la models/rf_model.pkl models/scaler.pkl
   ```
2. If absent, copy from the repository root:
   ```bash
   cp rf_model.pkl scaler.pkl models/
   ```
3. Set explicit environment variables if storing in custom directories:
   ```bash
   export MODEL_DIR=/absolute/path/to/models
   ```

---

### 2.2. Validation Failures
**Symptom:**
`ValidationError: Missing required financial field: 'ApplicantIncome'`
**Cause:**
Input JSON payload did not include one or more required keys (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`).
**Resolution:**
Check spelling and casing. Field names are case-sensitive. See `agensi-skill/references/INPUT_SCHEMA.md`.

**Symptom:**
`ValidationError: Combined household income must be greater than 0`
**Cause:**
Both `ApplicantIncome` and `CoapplicantIncome` are zero.
**Resolution:**
A loan application cannot be evaluated without household earnings. Provide valid income figures.

---

### 2.3. Scikit-Learn Inconsistent Version Warning
**Symptom:**
`InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.6.1 when using version 1.5.1`
**Cause:**
`rf_model.pkl` was saved using scikit-learn 1.6.1.
**Resolution:**
The package internally filters this warning safely. To eliminate it completely in external scripts:
```bash
pip install "scikit-learn>=1.6.0"
```

---

### 2.4. Matplotlib Permission Warning on Headless Server
**Symptom:**
`UserWarning: /home/user/.config/matplotlib is not a writable directory`
**Cause:**
Matplotlib defaults to `~/.config/matplotlib`, which may be read-only in sandboxes or container environments.
**Resolution:**
The system sets `MPLCONFIGDIR=/tmp/matplotlib` automatically in `src/loan_approval/config.py`.

---

### 2.5. Streamlit Port Conflicts
**Symptom:**
`Port 8501 is already in use`
**Cause:**
Another instance of Streamlit or another service is bound to port 8501.
**Resolution:**
Launch on an alternate port:
```bash
streamlit run app.py --server.port 8502
```
