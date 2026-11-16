# API & Technical Reference

## 1. Python Package API (`loan_approval`)

### 1.1. `predict_loan_approval`
```python
def predict_loan_approval(
    applicant_data: Dict[str, Any],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> PredictionResult
```
Executes loan approval prediction for an individual applicant.
- **Parameters:**
  - `applicant_data`: Dictionary containing applicant financial fields (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`) and optional demographic fields.
  - `model`: (Optional) pre-loaded model object.
  - `scaler`: (Optional) pre-loaded scaler object.
- **Returns:** `PredictionResult` dataclass with attributes:
  - `success` (bool): Whether prediction succeeded.
  - `prediction` (str): `"Approved"` or `"Rejected"`.
  - `prediction_class` (int): `1` or `0`.
  - `probabilities` (dict): `{"approved": float, "rejected": float}`.
  - `confidence` (float): Highest class probability.
  - `engineered_metrics` (dict): Calculated `Total_Income`, `Monthly_EMI`, `Balance_Income`.
  - `error` (Optional[str]): Error message if `success` is False.

---

### 1.2. `explain_prediction`
```python
def explain_prediction(
    applicant_data: Dict[str, Any],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    generate_plot: bool = False,
) -> ExplanationResult
```
Computes SHAP feature contributions for an applicant.
- **Returns:** `ExplanationResult` dataclass with attributes:
  - `success` (bool): Status of computation.
  - `prediction` (str): `"Approved"` or `"Rejected"`.
  - `base_value` (float): Baseline model expected value (~0.688).
  - `model_output_value` (float): Final model prediction score.
  - `contributions` (List[FeatureContribution]): Full list of per-feature contributions.
  - `positive_factors` (List[dict]): Factors favoring approval.
  - `negative_factors` (List[dict]): Risk factors favoring rejection.
  - `global_importance` (dict): Overall feature importances.
  - `human_readable_summary` (str): Plain-English narrative.

---

### 1.3. `validate_applicant_data`
```python
def validate_applicant_data(
    data: Dict[str, Any],
    require_demographics: bool = False,
) -> ValidationResult
```
Validates raw applicant dictionary against required data types and business constraints.
- **Returns:** `ValidationResult` with `is_valid` (bool), `cleaned_data` (dict), and `errors` (List[str]).

---

### 1.4. `load_artifacts`
```python
def load_artifacts(force_reload: bool = False) -> ModelArtifacts
```
Loads and caches `rf_model.pkl`, `scaler.pkl`, and `metadata.json` in a thread-safe singleton.
- **Returns:** `ModelArtifacts` instance.

---

## 2. Command-Line Interface (CLI)

### Command Syntax
```bash
python -m cli.main <subcommand> [options]
```

### Subcommands
1. **`predict`**
   - `--input`, `-i`: Path to JSON file or inline JSON string (Required).
2. **`explain`**
   - `--input`, `-i`: Path to JSON file or inline JSON string (Required).
   - `--plot`, `-p`: Path to save waterfall plot image as PNG (Optional).
3. **`model-info`**
   - Prints model metadata, expected features, scaler parameters, and paths.
4. **`batch`**
   - `--input`, `-i`: Path to CSV or JSON array file (Required).
   - `--output`, `-o`: Output JSON file path (Optional).

### Exit Codes
- `0`: Normal successful execution.
- `1`: Input validation or argument parsing error.
- `2`: Internal model or environment exception.
