"""Unit tests for the Command-Line Interface (CLI)."""

import json
from pathlib import Path

import pytest
from cli.main import main


def test_cli_model_info(capsys):
    ret = main(["model-info"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert data["model_type"] == "RandomForestClassifier"
    assert "expected_features" in data


def test_cli_predict_valid_file(capsys):
    example_path = str(Path("examples/example_input.json").resolve())
    ret = main(["predict", "--input", example_path])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert data["prediction"] in ("Approved", "Rejected")
    assert "probabilities" in data


def test_cli_predict_inline_json(capsys):
    inline_json = (
        '{"ApplicantIncome": 5000, "CoapplicantIncome": 1500, '
        '"LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": 1}'
    )
    ret = main(["predict", "--input", inline_json])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert data["prediction"] == "Approved"


def test_cli_explain_valid_file(capsys, tmp_path):
    example_path = str(Path("examples/example_input.json").resolve())
    plot_path = str(tmp_path / "waterfall_test.png")
    ret = main(["explain", "--input", example_path, "--plot", plot_path])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is True
    assert len(data["contributions"]) == 4
    assert Path(plot_path).is_file()


def test_cli_predict_invalid_json(capsys):
    ret = main(["predict", "--input", '{"invalid": 123}'])
    assert ret == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is False
    assert "Validation Error" in data["error"]


def test_cli_predict_nonexistent_file(capsys):
    ret = main(["predict", "--input", "does_not_exist.json"])
    assert ret == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["success"] is False
    assert data["error_type"] == "InvalidInputError"


def test_cli_no_subcommand(capsys):
    ret = main([])
    assert ret == 1
