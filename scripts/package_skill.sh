#!/usr/bin/env bash
# Reproducible packaging script for Agensi AI Agent Skill

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DIST_ZIP="$PROJECT_ROOT/loan-approval-ai-skill.zip"
STAGE_DIR="/tmp/agensi-pack-$$"
TARGET_DIR="$STAGE_DIR/loan-approval-ai-skill"

echo "======================================================================"
echo " Packaging Agensi AI Agent Skill: loan-approval-ai-skill.zip"
echo "======================================================================"

# Clean previous build artifacts
rm -rf "$STAGE_DIR" "$DIST_ZIP"
mkdir -p "$TARGET_DIR"

# 1. Copy core Agensi skill metadata and documents
cp "$PROJECT_ROOT/agensi-skill/SKILL.md" "$TARGET_DIR/SKILL.md"
cp "$PROJECT_ROOT/agensi-skill/README.md" "$TARGET_DIR/README.md"
cp "$PROJECT_ROOT/LICENSE" "$TARGET_DIR/LICENSE"
cp "$PROJECT_ROOT/requirements.txt" "$TARGET_DIR/requirements.txt"

# 2. Copy directories (dereferencing symlinks for standalone distribution)
cp -rL "$PROJECT_ROOT/agensi-skill/references" "$TARGET_DIR/references"
cp -rL "$PROJECT_ROOT/agensi-skill/scripts" "$TARGET_DIR/scripts"
cp -rL "$PROJECT_ROOT/agensi-skill/examples" "$TARGET_DIR/examples"

# 3. Include lightweight runtime inference engine and models
mkdir -p "$TARGET_DIR/models"
cp "$PROJECT_ROOT/models/rf_model.pkl" "$TARGET_DIR/models/rf_model.pkl"
cp "$PROJECT_ROOT/models/scaler.pkl" "$TARGET_DIR/models/scaler.pkl"
cp "$PROJECT_ROOT/models/metadata.json" "$TARGET_DIR/models/metadata.json"

mkdir -p "$TARGET_DIR/src"
cp -r "$PROJECT_ROOT/src/loan_approval" "$TARGET_DIR/src/loan_approval"

# 4. Clean any python cache or temporary files inside stage directory
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TARGET_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$TARGET_DIR" -type f -name ".DS_Store" -delete 2>/dev/null || true

# 5. Create the ZIP archive
cd "$STAGE_DIR"
zip -r -q "$DIST_ZIP" "loan-approval-ai-skill"

# 6. Clean stage directory
rm -rf "$STAGE_DIR"

# 7. Verify ZIP contents
echo "Archive created successfully at: $DIST_ZIP"
echo "File size: $(du -h "$DIST_ZIP" | cut -f1)"
echo "----------------------------------------------------------------------"
echo "Verifying archive structure:"
unzip -l "$DIST_ZIP" | head -n 35
echo "----------------------------------------------------------------------"

# 8. Check that SKILL.md is located at the root of the folder
if unzip -l "$DIST_ZIP" | grep -q "loan-approval-ai-skill/SKILL.md"; then
    echo "[OK] Verified: SKILL.md is located directly at root of skill directory."
else
    echo "[ERROR] SKILL.md is missing or improperly nested in ZIP!"
    exit 1
fi

echo "======================================================================"
echo " Agensi packaging completed and verified!"
echo "======================================================================"
