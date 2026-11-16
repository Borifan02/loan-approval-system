# Installation & Setup Guide

## 1. System Requirements & Environment
The Loan Approval Prediction System is optimized for lightweight execution on standard desktop or server hardware, including resource-constrained environments (e.g. 2 GB RAM, 1 CPU core).

### Minimum Hardware
- **CPU:** 1 x86_64 or ARM64 core (no GPU required)
- **RAM:** 2 GB minimum (4 GB recommended)
- **Disk Space:** ~500 MB free space
- **Operating System:** Linux (Ubuntu 20.04+, Debian 11+), macOS 12+, or Windows 10/11 (WSL2 recommended)

### Software Requirements
- **Python:** Version 3.9, 3.10, 3.11, or 3.12
- **pip:** Modern pip (v23.0+)
- **virtualenv:** Python built-in `venv` module

---

## 2. Step-by-Step Installation (Ubuntu / Linux)

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/Borifan02/loan-approval-system.git
cd loan-approval-system
```

### Step 2: Create a Dedicated Python Virtual Environment
Using a virtual environment prevents conflicts with system packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip and Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*For developers or test runners, install test dependencies:*
```bash
pip install pytest pytest-cov
```

*Optional: Editable installation of the core Python package:*
```bash
pip install -e .
```

### Step 4: Verify Installation & Artifact Integrity
Run the built-in self-test verification script:
```bash
python scripts/verify_installation.py
```
*Expected output:* All checks report `[OK]`.

---

## 3. Launching the System

### Option A: Interactive Streamlit Web Interface
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Option B: Command-Line Interface (Headless / Agent Mode)
```bash
python -m cli.main predict --input examples/example_input.json
```

### Option C: Agensi AI Agent Skill Mode
```bash
python agensi-skill/scripts/predict.py --input agensi-skill/examples/example_input.json
```

---

## 4. Running Without Internet Access (Offline Mode)
Once wheels are downloaded to your local cache, the entire application, CLI, and AI agent skill operate 100% offline with zero network connectivity or external API calls.
