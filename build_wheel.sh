#!/bin/bash
# Build Python wheel for Databricks Asset Bundle deployment
# This script creates a wheel file that can be deployed to Databricks

set -e

echo "=================================="
echo "Building Python Wheel"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel build

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info src/*.egg-info

# Build wheel
echo "Building wheel..."
python -m build --wheel

# Verify wheel was created
if [ -f dist/*.whl ]; then
    echo ""
    echo "=================================="
    echo "✓ Wheel built successfully!"
    echo "=================================="
    ls -lh dist/*.whl
    echo ""
    echo "Wheel location: $(ls dist/*.whl)"
    echo ""

    # Test entry point
    echo "Testing entry point installation..."
    pip install dist/*.whl --force-reinstall

    echo ""
    echo "Testing train_ditto command..."
    train_ditto --help

    echo ""
    echo "=================================="
    echo "✓ Entry point verified!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. Deploy bundle: databricks bundle deploy -t dev"
    echo "  2. The wheel will be uploaded to workspace"
    echo "  3. Phase 2 jobs can now use python_wheel_task"
    echo ""
else
    echo ""
    echo "=================================="
    echo "✗ Wheel build failed!"
    echo "=================================="
    exit 1
fi
