"""Tests for Agensi AI Agent Skill packaging, structure, and execution."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_final_zip_structure():
    """Verifies loan-approval-ai-skill-final.zip contents and internal structure."""
    zip_path = PROJECT_ROOT / "loan-approval-ai-skill-final.zip"
    assert zip_path.is_file(), f"Expected package {zip_path} does not exist"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        required_entries = [
            "loan-approval-ai-skill/SKILL.md",
            "loan-approval-ai-skill/README.md",
            "loan-approval-ai-skill/requirements-skill.txt",
            "loan-approval-ai-skill/LICENSE",
            "loan-approval-ai-skill/scripts/predict.py",
            "loan-approval-ai-skill/scripts/explain.py",
            "loan-approval-ai-skill/models/rf_model.pkl",
            "loan-approval-ai-skill/models/scaler.pkl",
            "loan-approval-ai-skill/models/metadata.json",
            "loan-approval-ai-skill/examples/example_input.json",
            "loan-approval-ai-skill/references/INPUT_SCHEMA.md",
            "loan-approval-ai-skill/references/PREDICTION_GUIDE.md",
            "loan-approval-ai-skill/references/MODEL_EXPLANATION.md",
            "loan-approval-ai-skill/references/TROUBLESHOOTING.md",
        ]
        for entry in required_entries:
            assert entry in namelist, f"Missing required entry in zip: {entry}"


def test_skill_md_frontmatter():
    """Ensures SKILL.md has valid YAML frontmatter conforming to Agensi standard."""
    skill_md = PROJECT_ROOT / "agensi-skill" / "SKILL.md"
    assert skill_md.is_file()
    content = skill_md.read_text(encoding="utf-8")
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "Frontmatter must be delimited by ---"
    frontmatter_text = parts[1]
    assert "name: loan-approval-ai" in frontmatter_text
    assert "description:" in frontmatter_text
    assert "version:" in frontmatter_text


def test_skill_predict_execution(tmp_path):
    """Extracts zip to tmp_path and executes predict.py successfully."""
    zip_path = PROJECT_ROOT / "loan-approval-ai-skill-final.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_path)

    pkg_root = tmp_path / "loan-approval-ai-skill"
    cmd = [
        sys.executable,
        str(pkg_root / "scripts" / "predict.py"),
        "--input",
        str(pkg_root / "examples" / "example_input.json"),
    ]
    res = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    assert res.returncode == 0, f"predict.py failed: {res.stderr}"
    data = json.loads(res.stdout)
    assert data["success"] is True
    assert data["prediction"] in ("Approved", "Rejected")
    assert "probabilities" in data


def test_skill_explain_execution(tmp_path):
    """Extracts zip to tmp_path and executes explain.py successfully."""
    zip_path = PROJECT_ROOT / "loan-approval-ai-skill-final.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_path)

    pkg_root = tmp_path / "loan-approval-ai-skill"
    cmd = [
        sys.executable,
        str(pkg_root / "scripts" / "explain.py"),
        "--input",
        str(pkg_root / "examples" / "example_input.json"),
    ]
    res = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    assert res.returncode == 0, f"explain.py failed: {res.stderr}"
    data = json.loads(res.stdout)
    assert data["success"] is True
    assert "contributions" in data
    assert len(data["contributions"]) == 4


def test_skill_predict_error_handling(tmp_path):
    """Verifies that predict.py returns JSON error and exit code 1 on nonexistent file."""
    zip_path = PROJECT_ROOT / "loan-approval-ai-skill-final.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_path)

    pkg_root = tmp_path / "loan-approval-ai-skill"
    cmd = [
        sys.executable,
        str(pkg_root / "scripts" / "predict.py"),
        "--input",
        "nonexistent_file.json",
    ]
    res = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["success"] is False
    assert data["error_type"] == "InvalidInputError"
