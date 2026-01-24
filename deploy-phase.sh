#!/bin/bash
# Phased Deployment Helper Script
# Switches databricks.yml to the appropriate phase configuration

set -e

PHASE=$1
TARGET=${2:-dev}

if [ -z "$PHASE" ]; then
    echo "Usage: ./deploy-phase.sh <phase> [target]"
    echo ""
    echo "Phases:"
    echo "  1 - Setup and Training only"
    echo "  2 - Setup, Training, and Model Serving"
    echo "  3 - Complete deployment (all resources)"
    echo ""
    echo "Examples:"
    echo "  ./deploy-phase.sh 1        # Deploy Phase 1 to dev"
    echo "  ./deploy-phase.sh 2 dev    # Deploy Phase 2 to dev"
    echo "  ./deploy-phase.sh 3 prod   # Deploy Phase 3 to prod"
    exit 1
fi

# Validate phase
if [[ ! "$PHASE" =~ ^[1-3]$ ]]; then
    echo "Error: Phase must be 1, 2, or 3"
    exit 1
fi

CONFIG_FILE="databricks-phase${PHASE}.yml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

echo "=========================================="
echo "Databricks Asset Bundle - Phase $PHASE Deployment"
echo "=========================================="
echo ""
echo "Target: $TARGET"
echo "Config: $CONFIG_FILE"
echo ""

# Backup current databricks.yml if it exists and is not a symlink
if [ -f "databricks.yml" ] && [ ! -L "databricks.yml" ]; then
    echo "Backing up current databricks.yml to databricks.yml.backup"
    cp databricks.yml databricks.yml.backup
fi

# Copy phase config to databricks.yml
echo "Copying $CONFIG_FILE to databricks.yml"
cp "$CONFIG_FILE" databricks.yml

echo ""
echo "Validating bundle..."
databricks bundle validate -t "$TARGET"

if [ $? -ne 0 ]; then
    echo ""
    echo "Validation failed. Please fix errors before deploying."
    exit 1
fi

echo ""
echo "Validation successful!"
echo ""
read -p "Deploy to $TARGET environment? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deploying Phase $PHASE to $TARGET..."
    databricks bundle deploy -t "$TARGET"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Phase $PHASE deployed successfully!"
        echo "=========================================="

        if [ "$PHASE" == "1" ]; then
            echo ""
            echo "Next steps:"
            echo "  1. Run setup job: databricks bundle run setup_unity_catalog -t $TARGET"
            echo "  2. Run training job: databricks bundle run train_ditto_model -t $TARGET"
            echo "  3. Deploy Phase 2: ./deploy-phase.sh 2 $TARGET"
        elif [ "$PHASE" == "2" ]; then
            echo ""
            echo "Next steps:"
            echo "  1. Wait for model serving endpoint to be ready (~5 minutes)"
            echo "  2. Check status: databricks serving-endpoints get ditto-em-$TARGET"
            echo "  3. Deploy Phase 3: ./deploy-phase.sh 3 $TARGET"
        else
            echo ""
            echo "Deployment complete! All resources deployed."
            echo ""
            echo "Verify in Databricks UI:"
            echo "  - Workflows → Jobs"
            echo "  - Serving → Endpoints"
            echo "  - Data → Unity Catalog"
        fi
    else
        echo ""
        echo "Deployment failed. Check errors above."
        exit 1
    fi
else
    echo ""
    echo "Deployment cancelled."
    exit 0
fi
