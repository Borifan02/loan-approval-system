#!/usr/bin/env python3
"""Standalone SHAP explanation script for Agensi AI coding agents.

This script executes explainable AI analysis for an applicant using the
shared Python explanation engine from src/loan_approval. It accepts a JSON file
path or inline JSON string and outputs machine-readable JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SKILL_ROOT.parent

# Check for src/ either in skill root (standalone zip) or project root (repo)
for candidate in [SKILL_ROOT / "src", PROJECT_ROOT / "src"]:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Check for models/ either in skill root or project root
for candidate_model_dir in [SKILL_ROOT / "models", PROJECT_ROOT / "models"]:
    if candidate_model_dir.is_dir() and "MODEL_DIR" not in os.environ:
        os.environ["MODEL_DIR"] = str(candidate_model_dir)
        break

try:
    from loan_approval import explain_prediction, render_waterfall_plot
    from loan_approval.utils import safe_json_dumps
except ImportError as err:
    print(
        json.dumps(
            {
                "success": False,
                "error_type": "DependencyError",
                "message": (
                    f"Failed to import 'loan_approval' package: {err}. "
                    "Ensure virtualenv is activated and dependencies are installed "
                    "(pip install -r requirements-skill.txt)."
                ),
            },
            indent=2,
        ),
        file=sys.stdout,
    )
    sys.exit(2)


def parse_input(raw_input: str) -> dict:
    """Parses JSON input from file path or inline JSON string."""
    candidate_path = Path(raw_input)
    if candidate_path.is_file():
        try:
            with open(candidate_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError(
                f"JSON in '{raw_input}' must be a key-value dictionary object, got {type(data).__name__}"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"File '{raw_input}' contains invalid JSON: {exc}")

    # Check if the user clearly intended a file path that does not exist
    if raw_input.endswith(".json") or "/" in raw_input or "\\" in raw_input:
        raise ValueError(f"Input file not found: '{raw_input}'")

    # Otherwise try parsing as inline JSON string
    try:
        data = json.loads(raw_input)
        if isinstance(data, dict):
            return data
        raise ValueError(
            f"Inline JSON must be a key-value dictionary object, got {type(data).__name__}"
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Input is neither an existing file path nor valid inline JSON: '{raw_input}'"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agensi Skill: Explain Loan Approval Decision Using SHAP"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to JSON file or inline JSON string containing applicant data",
    )
    parser.add_argument(
        "--plot",
        "-p",
        required=False,
        help="Optional path to save waterfall plot image (PNG)",
    )
    args = parser.parse_args()

    try:
        applicant_data = parse_input(args.input)
    except Exception as exc:
        print(
            safe_json_dumps(
                {
                    "success": False,
                    "error_type": "InvalidInputError",
                    "message": str(exc),
                }
            )
        )
        return 1

    result = explain_prediction(applicant_data)

    if args.plot and result.success:
        try:
            fig = render_waterfall_plot(result)
            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(plot_path), bbox_inches="tight", dpi=150)
            print(f"[Skill] Waterfall plot saved to: {plot_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[Skill WARNING] Failed to render plot: {exc}", file=sys.stderr)

    print(safe_json_dumps(result.to_dict()))
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
