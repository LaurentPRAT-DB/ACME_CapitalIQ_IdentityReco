#!/bin/bash
# Databricks Asset Bundle - Phased Deployment Script
# Usage: ./deploy-phase.sh <phase-number> <target>
# Example: ./deploy-phase.sh 0 dev

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get phase details
get_phase_name() {
    case "$1" in
        0) echo "Catalog Setup" ;;
        1) echo "Data Load" ;;
        2) echo "Model Training" ;;
        3) echo "Model Deployment" ;;
        4) echo "Production Pipeline" ;;
        *) echo "Unknown" ;;
    esac
}

get_phase_file() {
    echo "databricks-phase$1.yml"
}

get_phase_job() {
    case "$1" in
        0) echo "setup_catalog" ;;
        1) echo "load_reference_data" ;;
        2) echo "train_ditto_model" ;;
        3) echo "" ;;  # No job for model serving
        4) echo "entity_matching_pipeline" ;;
        *) echo "" ;;
    esac
}

# Check arguments
if [ $# -lt 1 ]; then
    print_error "Usage: ./deploy-phase.sh <phase-number> [target]"
    echo ""
    echo "Phase numbers:"
    echo "  0 - Catalog Setup (creates Unity Catalog and schemas)"
    echo "  1 - Data Load (creates tables and loads reference data)"
    echo "  2 - Model Training (trains Ditto model)"
    echo "  3 - Model Deployment (deploys serving endpoint)"
    echo "  4 - Production Pipeline (deploys production jobs)"
    echo ""
    echo "Target (optional, defaults to 'dev'):"
    echo "  dev     - Development environment"
    echo "  staging - Staging environment"
    echo "  prod    - Production environment"
    echo ""
    echo "Examples:"
    echo "  ./deploy-phase.sh 0 dev"
    echo "  ./deploy-phase.sh 1"
    echo "  ./deploy-phase.sh 2 staging"
    exit 1
fi

PHASE=$1
TARGET=${2:-dev}

# Validate phase number
if [[ ! "$PHASE" =~ ^[0-4]$ ]]; then
    print_error "Invalid phase number. Must be 0, 1, 2, 3, or 4"
    exit 1
fi

# Get phase details
PHASE_NAME=$(get_phase_name "$PHASE")
PHASE_FILE=$(get_phase_file "$PHASE")
PHASE_JOB=$(get_phase_job "$PHASE")

print_info "=========================================="
print_info "Databricks Asset Bundle - Phase $PHASE Deployment"
print_info "Phase: $PHASE_NAME"
print_info "Target: $TARGET"
print_info "=========================================="
echo ""

# Check if phase file exists
if [ ! -f "$PHASE_FILE" ]; then
    print_error "Phase file not found: $PHASE_FILE"
    exit 1
fi

# Backup existing databricks.yml if it exists
if [ -f "databricks.yml" ]; then
    print_info "Backing up existing databricks.yml to databricks.yml.backup"
    cp databricks.yml databricks.yml.backup
fi

# Copy phase configuration to databricks.yml
print_info "Copying $PHASE_FILE to databricks.yml"
cp "$PHASE_FILE" databricks.yml
print_success "Configuration updated for Phase $PHASE"
echo ""

# Validate bundle
print_info "Validating bundle configuration..."
if databricks bundle validate -t "$TARGET"; then
    print_success "Bundle validation passed"
else
    print_error "Bundle validation failed"
    print_warning "Restoring previous databricks.yml from backup"
    if [ -f "databricks.yml.backup" ]; then
        mv databricks.yml.backup databricks.yml
    fi
    exit 1
fi
echo ""

# Deploy bundle
print_info "Deploying Phase $PHASE to target: $TARGET"
if databricks bundle deploy -t "$TARGET"; then
    print_success "Deployment successful"
else
    print_error "Deployment failed"
    exit 1
fi
echo ""

# Run job if applicable
if [ -n "$PHASE_JOB" ]; then
    print_info "Phase $PHASE includes job: $PHASE_JOB"
    read -p "Do you want to run the job now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Running job: $PHASE_JOB"
        databricks bundle run "$PHASE_JOB" -t "$TARGET"
        print_success "Job started. Monitor progress in Databricks UI"
    else
        print_info "Skipping job execution. You can run it later with:"
        echo "  databricks bundle run $PHASE_JOB -t $TARGET"
    fi
else
    print_info "Phase $PHASE is deployment-only (no job to run)"
    if [ "$PHASE" -eq 3 ]; then
        print_info "Model serving endpoint is being created. Check Databricks UI for status."
    fi
fi

echo ""
print_success "=========================================="
print_success "Phase $PHASE deployment complete!"
print_success "=========================================="
echo ""

# Show next steps
if [ "$PHASE" -lt 4 ]; then
    NEXT_PHASE=$((PHASE + 1))
    NEXT_PHASE_NAME=$(get_phase_name "$NEXT_PHASE")
    print_info "Next steps:"
    echo "  1. Verify Phase $PHASE completed successfully"
    echo "  2. Deploy Phase $NEXT_PHASE: ./deploy-phase.sh $NEXT_PHASE $TARGET"
    echo "  3. Phase $NEXT_PHASE: $NEXT_PHASE_NAME"
else
    print_info "All phases deployed! Your entity matching pipeline is ready."
    echo ""
    echo "Next steps:"
    echo "  - Monitor jobs in Databricks UI"
    echo "  - Check model serving endpoint health"
    echo "  - Review matched entities in gold tables"
fi
