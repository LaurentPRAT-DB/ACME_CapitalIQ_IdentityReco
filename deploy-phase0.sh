#!/bin/bash
# Phase 0: Create Unity Catalog
# This script creates the Unity Catalog before deploying any bundles

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get target environment (default: dev)
TARGET="${1:-dev}"

echo "========================================================================"
echo "Phase 0: Unity Catalog Setup"
echo "Target: $TARGET"
echo "========================================================================"
echo ""

# Determine catalog name based on target
case "$TARGET" in
  dev)
    CATALOG_NAME="laurent_prat_entity_matching_dev"
    OWNER="laurent.prat@databricks.com"
    ;;
  staging)
    CATALOG_NAME="entity_matching_staging"
    OWNER="laurent.prat@databricks.com"  # Change to service principal if needed
    ;;
  prod)
    CATALOG_NAME="entity_matching"
    OWNER="laurent.prat@databricks.com"  # Change to service principal if needed
    ;;
  *)
    echo -e "${RED}✗ Unknown target: $TARGET${NC}"
    echo "Usage: $0 [dev|staging|prod]"
    exit 1
    ;;
esac

echo "Catalog Name: $CATALOG_NAME"
echo "Owner: $OWNER"
echo ""

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo -e "${RED}✗ Databricks CLI not found${NC}"
    echo "Please install: uv pip install databricks-cli databricks-sdk"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv not found${NC}"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Found .venv, activating...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}✓ Found venv, activating...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠ No virtual environment found${NC}"
    echo "Creating virtual environment with uv..."
    uv venv
    source .venv/bin/activate
fi

# Check if databricks-sdk is installed
if ! python3 -c "import databricks.sdk" 2>/dev/null; then
    echo -e "${YELLOW}⚠ databricks-sdk not installed${NC}"
    echo "Installing databricks-sdk with uv..."
    uv pip install databricks-sdk
fi

echo -e "${YELLOW}Running Phase 0 setup...${NC}"
echo ""

# Run the catalog creation script
python3 scripts/create_catalog.py \
    --catalog-name "$CATALOG_NAME" \
    --owner "$OWNER" \
    --comment "Entity matching to S&P Capital IQ identifiers"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================================================"
    echo "✓ Phase 0 completed successfully!"
    echo "========================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Deploy Phase 1: databricks bundle deploy -t $TARGET"
    echo "  2. Run setup_unity_catalog job to create schemas"
    echo "  3. Run train_ditto_model job to train the model"
else
    echo ""
    echo -e "${RED}========================================================================"
    echo "✗ Phase 0 failed"
    echo "========================================================================${NC}"
    exit $EXIT_CODE
fi
