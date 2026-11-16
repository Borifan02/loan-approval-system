# Security, Privacy, and Responsible AI Policy

## 1. Data Privacy & Confidentiality
- **Zero External Telemetry:** The application does not collect, track, or transmit applicant data, telemetry, or analytics to external servers or cloud services.
- **Zero Sensitive Data Logging:** Applicant financial records (incomes, loan requests, credit scores) are never written to disk, application logs, or temporary files.
- **Stateless Execution:** In-memory applicant representations are garbage-collected immediately following prediction/explanation response generation.

---

## 2. Secure Machine Learning Operations

### Model Pickle Safety
- Machine learning artifacts (`rf_model.pkl`, `scaler.pkl`) are serialized via `joblib`/`pickle`.
- **Policy:** The application only loads artifacts from known, trusted local paths (`models/` or project root).
- **Prohibition:** Never configure `MODEL_PATH` or `SCALER_PATH` to point to untrusted remote URLs or unverified third-party files. Pickle deserialization can execute arbitrary code if artifacts are compromised.

### Input Sanitization & Bounds Checking
- The validation layer (`src/loan_approval/validation.py`) enforces strict type casting, NaN/infinity rejection, and domain range validation before inputs enter numpy/pandas operations.
- Non-finite numbers, extreme numbers exceeding realistic bounds, and injection payloads are rejected with explicit validation errors.

---

## 3. Fair Lending & Responsible AI

### Compliance with Non-Discrimination Guidelines
In accordance with ethical lending standards such as the **Equal Credit Opportunity Act (ECOA)**:
1. **Protected Attributes Excluded:**
   - `Gender`: Excluded to prevent gender-based credit disparity.
   - `Married`: Excluded to eliminate bias based on marital or household status.
   - `Education`: Excluded to avoid socioeconomic educational bias.
   - `Property_Area`: Excluded to eliminate geographic redlining risks.
2. **Transparent Explanations:**
   - Every decision is accompanied by SHAP feature attributions, explaining the exact quantitative factors influencing the score.
3. **No Fabricated Compliance Claims:**
   - While demographic variables are removed from the active model, the system does not claim that removal of demographic features guarantees complete absence of proxy bias in historical data. Regular algorithmic audits are recommended for any commercial deployment.
