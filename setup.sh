#!/bin/bash
# G42 Financial Intelligence Agent — Setup Script
# Creates a virtual environment and installs all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

VENV_DIR="finAgent"
PYTHON_CMD="python3"

echo "═══════════════════════════════════════════════════"
echo "  G42 Financial Intelligence Agent — Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# Check Python version
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using: $PYTHON_VERSION"

# Check minimum version (3.10+)
MIN_VERSION="3.10"
CURRENT=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$(printf '%s\n' "$MIN_VERSION" "$CURRENT" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "ERROR: Python $MIN_VERSION+ required (found $CURRENT)"
    exit 1
fi

# Check minimum version (3.10+)
MIN_VERSION="3.10"
CURRENT=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$(printf '%s\n' "$MIN_VERSION" "$CURRENT" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "ERROR: Python $MIN_VERSION+ required (found $CURRENT)"
    exit 1
fi

# Create virtual environment
echo ""
echo "→ Creating virtual environment in ./$VENV_DIR ..."
$PYTHON_CMD -m venv $VENV_DIR

# Activate
echo "→ Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "→ Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
    echo "→ Creating .env from .env.example..."
    cp .env.example .env
fi

# Generate sample data
echo "→ Generating sample financial data..."
python data/generate_sample.py

# Remind about Groq key
echo ""
if grep -q "^GROQ_API_KEY=gsk_" .env 2>/dev/null; then
    echo "⚠  IMPORTANT: Edit .env and replace the GROQ_API_KEY placeholder:"
    echo "   Get a free key at: https://console.groq.com"
    echo ""
fi

# Run tests to verify installation
echo ""
echo "→ Running test suite..."
python -m pytest tests/ -v --tb=short

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  1. Add your Groq API key to .env:"
echo "     GROQ_API_KEY=gsk_..."
echo "     (free key at https://console.groq.com)"
echo ""
echo "  To activate the environment:"
echo "    source finAgent/bin/activate"
echo ""
echo "  To run locally:"
echo "    streamlit run ui/app.py"
echo ""
echo "  To deploy to Streamlit Cloud:"
echo "    git push origin main"
echo "    Then add GROQ_API_KEY in your Streamlit app secrets"
echo ""
echo "  To deactivate when done:"
echo "    deactivate"
echo ""
