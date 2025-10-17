#!/bin/bash
# ===================================
# LLMOps Monitoring Build and Publish Script (Linux/Mac)
# ===================================

set -e  # Exit on error

echo ""
echo "==================================="
echo "LLMOps Monitoring Build and Publish"
echo "==================================="
echo ""

# [1/5] Clean old builds
echo "[1/5] Cleaning old builds..."
rm -rf build dist *.egg-info llamonitor_async.egg-info
echo "    Done."
echo ""

# [2/5] Verify version in pyproject.toml
echo "[2/5] Checking version..."
grep "version" pyproject.toml
echo ""
read -p "Is this the correct version to publish? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo ""
    echo "Publishing cancelled."
    exit 1
fi
echo ""

# [3/5] Build package
echo "[3/5] Building package..."
python -m build
echo "    Done."
echo ""

# [4/5] Check package
echo "[4/5] Checking package with twine..."
python -m twine check dist/*
echo "    Done."
echo ""

# [5/5] Upload to PyPI
echo "[5/5] Publishing to PyPI..."
echo ""
echo "IMPORTANT: This will publish to PyPI!"
echo ""
read -p "Continue with upload? (y/n): " UPLOAD_CONFIRM
if [ "$UPLOAD_CONFIRM" != "y" ]; then
    echo ""
    echo "Publishing cancelled."
    exit 1
fi
echo ""

python -m twine upload dist/*

echo ""
echo "==================================="
echo "SUCCESS: Package published to PyPI!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Create git tag: git tag v{version}"
echo "  2. Push tag: git push origin v{version}"
echo "  3. Create GitHub release"
echo ""
