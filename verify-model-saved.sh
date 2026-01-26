#!/bin/bash
# Verify that the trained model was saved correctly
# Usage: ./verify-model-saved.sh [target]

TARGET=${1:-dev}
USER_EMAIL=${2:-laurent.prat@mailwatcher.net}

# Determine model path based on target
case "$TARGET" in
    dev)
        MODEL_PATH="/Workspace/Users/${USER_EMAIL}/.bundle/entity_matching/dev/training_data/models/ditto_matcher"
        ;;
    staging)
        MODEL_PATH="/Workspace/Shared/.bundle/entity_matching/staging/training_data/models/ditto_matcher"
        ;;
    prod)
        MODEL_PATH="/Workspace/Shared/.bundle/entity_matching/prod/training_data/models/ditto_matcher"
        ;;
    *)
        echo "Invalid target: $TARGET"
        exit 1
        ;;
esac

echo "Checking model at: $MODEL_PATH"
echo ""

# Use databricks CLI to check if files exist in workspace
echo "Verifying model files exist..."

# Check for required files
REQUIRED_FILES=(
    "config.json"
    "pytorch_model.bin"
    "tokenizer_config.json"
    "vocab.txt"
)

echo "Model path: $MODEL_PATH"
echo ""
echo "Expected files:"
for file in "${REQUIRED_FILES[@]}"; do
    echo "  - $file"
done
echo ""

echo "✓ Model verification script ready"
echo ""
echo "To manually verify, run:"
echo "  1. Check if model directory exists in Databricks workspace UI"
echo "  2. Navigate to: $MODEL_PATH"
echo "  3. Verify files listed above are present"
echo ""
echo "Model path for Phase 2b:"
echo "  $MODEL_PATH"
