#!/usr/bin/env python3
"""Automated verification script for the Loan Approval system.

Performs health checks on:
  1. Python runtime environment
  2. Required third-party dependencies
  3. Model and scaler artifact files
  4. Metadata integrity
  5. Validation engine
  6. Inference engine (end-to-end prediction)
  7. SHAP explainability engine
  8. Agensi skill scripts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure non-privileged writable cache directory for Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def report(step: str, status: bool, detail: str = "") -> None:
    badge = "[\033[92mOK\033[0m]" if status else "[\033[91mFAIL\033[0m]"
    msg = f"{badge} {step}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def main() -> int:
    print("=" * 70)
    print(" Loan Approval AI System: Environment & Artifact Verification")
    print("=" * 70)

    all_passed = True

    # 1. Python version check
    py_ver = sys.version_info
    py_ok = (py_ver.major == 3) and (py_ver.minor >= 9)
    report("Python Version (>= 3.9)", py_ok, f"Current: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if not py_ok:
        all_passed = False

    # 2. Dependency checks
    packages = ["sklearn", "pandas", "numpy", "joblib", "shap", "matplotlib", "streamlit"]
    for pkg in packages:
        try:
            __import__(pkg)
            report(f"Dependency '{pkg}'", True)
        except ImportError as err:
            report(f"Dependency '{pkg}'", False, str(err))
            all_passed = False

    # 3. Model and Scaler Artifact Existence
    model_path = PROJECT_ROOT / "models" / "rf_model.pkl"
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    meta_path = PROJECT_ROOT / "models" / "metadata.json"

    report("Model File (models/rf_model.pkl)", model_path.is_file(), str(model_path))
    report("Scaler File (models/scaler.pkl)", scaler_path.is_file(), str(scaler_path))
    report("Metadata File (models/metadata.json)", meta_path.is_file(), str(meta_path))

    if not (model_path.is_file() and scaler_path.is_file()):
        all_passed = False

    # 4. Artifact Loading & Schema Verification
    try:
        from loan_approval import load_artifacts
        artifacts = load_artifacts()
        report("Model Artifact Loading", True, f"Type: {type(artifacts.model).__name__}")
        report("Scaler Artifact Loading", True, f"Type: {type(artifacts.scaler).__name__}")
    except Exception as exc:
        report("Artifact Loading", False, str(exc))
        all_passed = False

    # 5. Validation Check
    try:
        from loan_approval import validate_applicant_data
        valid_sample = {
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 1000,
            "LoanAmount": 100,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
        }
        val_res = validate_applicant_data(valid_sample)
        report("Validation Engine (Valid Payload)", val_res.is_valid)

        invalid_sample = {"ApplicantIncome": -100}
        inv_res = validate_applicant_data(invalid_sample)
        report("Validation Engine (Rejection of Invalid)", not inv_res.is_valid)
    except Exception as exc:
        report("Validation Engine", False, str(exc))
        all_passed = False

    # 6. End-to-End Prediction Check
    try:
        from loan_approval import predict_loan_approval
        pred = predict_loan_approval(valid_sample)
        report(
            "Prediction Engine",
            pred.success and pred.prediction in ("Approved", "Rejected"),
            f"Result: {pred.prediction} (Confidence: {pred.confidence * 100:.1f}%)",
        )
    except Exception as exc:
        report("Prediction Engine", False, str(exc))
        all_passed = False

    # 7. SHAP Explanation Check
    try:
        from loan_approval import explain_prediction
        expl = explain_prediction(valid_sample)
        report(
            "SHAP Explainability Engine",
            expl.success and len(expl.contributions) == 4,
            f"Factors: {len(expl.positive_factors)} positive, {len(expl.negative_factors)} negative",
        )
    except Exception as exc:
        report("SHAP Explainability Engine", False, str(exc))
        all_passed = False

    # 8. Agensi Skill Files
    skill_md = PROJECT_ROOT / "agensi-skill" / "SKILL.md"
    skill_predict = PROJECT_ROOT / "agensi-skill" / "scripts" / "predict.py"
    skill_explain = PROJECT_ROOT / "agensi-skill" / "scripts" / "explain.py"

    report("Agensi SKILL.md", skill_md.is_file())
    report("Agensi scripts/predict.py", skill_predict.is_file())
    report("Agensi scripts/explain.py", skill_explain.is_file())

    print("=" * 70)
    if all_passed:
        print("\033[92mALL SYSTEM VERIFICATION CHECKS PASSED!\033[0m")
        print("The system is ready for production use and Agensi skill distribution.")
        return 0
    else:
        print("\033[91mSOME VERIFICATION CHECKS FAILED.\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
