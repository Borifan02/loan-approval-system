# Agensi Skill: Loan Approval AI & Explainable Credit Decisioning

This directory packages the Loan Approval Prediction System as an **Agensi-compatible AI Agent Skill** suitable for distribution on Agensi.io and direct integration into AI coding agents such as Claude Code, OpenAI Codex, and Cursor.

## Directory Layout
```
agensi-skill/
├── SKILL.md                  # Main skill specification with valid YAML frontmatter
├── README.md                 # This overview document
├── references/               # Detailed technical references for agents
│   ├── INPUT_SCHEMA.md       # JSON payload parameters and validation rules
│   ├── PREDICTION_GUIDE.md   # Step-by-step CLI and API invocation guide
│   ├── MODEL_EXPLANATION.md  # How to interpret SHAP feature attributions
│   └── TROUBLESHOOTING.md    # Common errors, exit codes, and diagnostics
├── scripts/                  # Standalone execution scripts for agents
│   ├── predict.py            # Headless prediction script
│   └── explain.py            # Headless explanation & waterfall plot script
└── examples/
    └── example_input.json    # Standard benchmark applicant payload
```

## Quick Start for Coding Agents
1. Verify prerequisites:
   ```bash
   python -m cli.main model-info
   ```
2. Execute a prediction:
   ```bash
   python agensi-skill/scripts/predict.py --input agensi-skill/examples/example_input.json
   ```
3. Generate a SHAP explanation:
   ```bash
   python agensi-skill/scripts/explain.py --input agensi-skill/examples/example_input.json
   ```
