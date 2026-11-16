"""Command-Line Interface (CLI) for Loan Approval AI.

Provides subcommands for prediction, SHAP explanation, model inspection,
and batch scoring with structured JSON outputs and standard POSIX exit codes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src/ is on python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loan_approval import (
    explain_prediction,
    load_artifacts,
    predict_batch,
    predict_loan_approval,
    render_waterfall_plot,
    validate_applicant_data,
)
from loan_approval.utils import safe_json_dumps


def _parse_input_data(input_arg: str) -> Dict[str, Any]:
    """Parses applicant input from either a file path or an inline JSON string.

    Args:
        input_arg: Path to JSON file or string of JSON content.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If JSON is invalid or file cannot be read.
    """
    input_path = Path(input_arg)
    if input_path.is_file():
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to read JSON input from file '{input_arg}': {exc}")

    # Attempt parsing as raw inline JSON
    try:
        data = json.loads(input_arg)
        if isinstance(data, dict):
            return data
        raise ValueError("JSON input must represent a key-value object / dictionary")
    except json.JSONDecodeError:
        raise ValueError(
            f"Input argument is neither an existing file path nor valid inline JSON: '{input_arg}'"
        )


def cmd_predict(args: argparse.Namespace) -> int:
    """Executes prediction subcommand."""
    try:
        data = _parse_input_data(args.input)
    except ValueError as err:
        error_output = {
            "success": False,
            "error_type": "InvalidInputError",
            "message": str(err),
        }
        print(safe_json_dumps(error_output))
        return 1

    result = predict_loan_approval(data)
    print(safe_json_dumps(result.to_dict()))
    return 0 if result.success else 1


def cmd_explain(args: argparse.Namespace) -> int:
    """Executes explain subcommand."""
    try:
        data = _parse_input_data(args.input)
    except ValueError as err:
        error_output = {
            "success": False,
            "error_type": "InvalidInputError",
            "message": str(err),
        }
        print(safe_json_dumps(error_output))
        return 1

    result = explain_prediction(data)

    # Optional plot saving
    if args.plot and result.success:
        try:
            fig = render_waterfall_plot(result)
            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(plot_path), bbox_inches="tight", dpi=150)
            print(f"[CLI] Waterfall plot saved to: {plot_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[CLI WARNING] Failed to save plot: {exc}", file=sys.stderr)

    print(safe_json_dumps(result.to_dict()))
    return 0 if result.success else 1


def cmd_model_info(args: argparse.Namespace) -> int:
    """Executes model-info subcommand."""
    try:
        artifacts = load_artifacts()
        info = {
            "success": True,
            "model_path": str(artifacts.model_path),
            "scaler_path": str(artifacts.scaler_path),
            "model_type": type(artifacts.model).__name__,
            "scaler_type": type(artifacts.scaler).__name__,
            "expected_features": artifacts.expected_features,
            "scaled_features": artifacts.scaled_features,
            "classes": [0, 1],
            "class_labels": {0: "Rejected", 1: "Approved"},
            "metadata": artifacts.metadata,
        }
        print(safe_json_dumps(info))
        return 0
    except Exception as exc:
        error_output = {
            "success": False,
            "error_type": "ModelLoadError",
            "message": str(exc),
        }
        print(safe_json_dumps(error_output))
        return 2


def cmd_batch(args: argparse.Namespace) -> int:
    """Executes batch predictions from a CSV or JSON array file."""
    input_path = Path(args.input)
    if not input_path.is_file():
        error_output = {
            "success": False,
            "error_type": "FileNotFoundError",
            "message": f"Input batch file not found: '{args.input}'",
        }
        print(safe_json_dumps(error_output))
        return 1

    records: List[Dict[str, Any]] = []
    try:
        if input_path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(input_path)
            records = df.to_dict(orient="records")
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    raise ValueError("JSON file must contain an array of applicant objects")
    except Exception as exc:
        error_output = {
            "success": False,
            "error_type": "BatchLoadError",
            "message": f"Failed to parse batch input file: {exc}",
        }
        print(safe_json_dumps(error_output))
        return 1

    results = predict_batch(records)
    results_dict = [r.to_dict() for r in results]

    summary = {
        "success": True,
        "total_records": len(results),
        "approved_count": sum(1 for r in results if r.prediction == "Approved"),
        "rejected_count": sum(1 for r in results if r.prediction == "Rejected"),
        "error_count": sum(1 for r in results if not r.success),
        "results": results_dict,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(safe_json_dumps(summary))
        print(f"[CLI] Batch results written to: {out_path}", file=sys.stderr)

    print(safe_json_dumps(summary))
    return 0


def main(argv: List[str] | None = None) -> int:
    """CLI entry point dispatcher."""
    parser = argparse.ArgumentParser(
        prog="loan-approval",
        description="Loan Approval AI — Prediction and Explainability CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict loan approval for an applicant")
    predict_parser.add_argument(
        "--input", "-i", required=True, help="Path to JSON file or inline JSON string"
    )

    # explain
    explain_parser = subparsers.add_parser("explain", help="Generate SHAP explanations for an applicant")
    explain_parser.add_argument(
        "--input", "-i", required=True, help="Path to JSON file or inline JSON string"
    )
    explain_parser.add_argument(
        "--plot", "-p", required=False, help="Optional output image path to save waterfall chart (PNG)"
    )

    # model-info
    subparsers.add_parser("model-info", help="Inspect loaded model metadata and feature schemas")

    # batch
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple applicant records")
    batch_parser.add_argument(
        "--input", "-i", required=True, help="Path to CSV or JSON array file"
    )
    batch_parser.add_argument(
        "--output", "-o", required=False, help="Optional output JSON file path"
    )

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help(sys.stderr)
        return 1

    if args.subcommand == "predict":
        return cmd_predict(args)
    elif args.subcommand == "explain":
        return cmd_explain(args)
    elif args.subcommand == "model-info":
        return cmd_model_info(args)
    elif args.subcommand == "batch":
        return cmd_batch(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
