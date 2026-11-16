#!/usr/bin/env python3
"""Example programmatic usage of the loan_approval Python package."""

import json
import sys
from pathlib import Path

# Add src to python path for development execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from loan_approval import (
    explain_prediction,
    predict_loan_approval,
    validate_applicant_data,
)


def main():
    example_path = Path(__file__).parent / "example_input.json"
    with open(example_path, "r", encoding="utf-8") as f:
        applicant = json.load(f)

    print("=" * 60)
    print("1. Validating Applicant Input")
    print("=" * 60)
    val = validate_applicant_data(applicant)
    if not val.is_valid:
        print(f"Validation failed: {val.errors}")
        sys.exit(1)
    print("Applicant data is valid!")

    print("\n" + "=" * 60)
    print("2. Predicting Loan Approval")
    print("=" * 60)
    prediction = predict_loan_approval(applicant)
    print(f"Decision:     {prediction.prediction}")
    print(f"Class:        {prediction.prediction_class}")
    print(f"Confidence:   {prediction.confidence * 100:.1f}%")
    print(f"Probabilities: Approved={prediction.probabilities['approved']}, Rejected={prediction.probabilities['rejected']}")
    print(f"Metrics:      Total Income=${prediction.engineered_metrics['Total_Income']:,.2f}, Monthly EMI=${prediction.engineered_metrics['Monthly_EMI']:,.2f}")

    print("\n" + "=" * 60)
    print("3. Generating SHAP Explanation")
    print("=" * 60)
    explanation = explain_prediction(applicant)
    print(f"Summary:      {explanation.human_readable_summary}")
    print("\nPositive Factors (Favoring Approval):")
    for factor in explanation.positive_factors:
        print(f"  + {factor['display_name']}: {factor['summary']} (SHAP: +{factor['impact_magnitude']:.4f})")

    print("\nNegative Factors (Risk Drivers):")
    for factor in explanation.negative_factors:
        print(f"  - {factor['display_name']}: {factor['summary']} (SHAP: -{factor['impact_magnitude']:.4f})")

    print("\n" + "=" * 60)
    print("Execution complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
