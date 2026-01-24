# Phased Deployment Guide

This guide walks through the phased deployment process for the Entity Matching project using Databricks Asset Bundles.

## Overview

The deployment is split into four phases to ensure proper dependency management:

0. **Phase 0**: Create Unity Catalog (one-time setup)
1. **Phase 1**: Setup and Training (create schemas, train model)
2. **Phase 2**: Model Serving (deploy trained model endpoint)
3. **Phase 3**: Production Pipelines (deploy scheduled matching jobs)

## Deployment Configuration Files

Each phase has dedicated setup:

| Phase | Description | Command |
|-------|-------------|---------|
| Phase 0 | Create Unity Catalog | `./deploy-phase0.sh dev` |
| Phase 1 | Setup & Training | `databricks bundle deploy -t dev` |
| Phase 2 | Model Serving | (Enable in databricks.yml, then deploy) |
| Phase 3 | Production Pipelines | (Enable in databricks.yml, then deploy) |

**Benefits of separate config files:**
- No need to edit/uncomment sections
- Clear separation of deployment phases
- Easy to re-run specific phases
- Automated validation before deployment
- Reduces deployment errors

**How it works:**
The `deploy-phase.sh` script copies the appropriate phase configuration to `databricks.yml` and runs validation + deployment.

## Prerequisites

- Databricks CLI installed and configured
- Python 3.7+ with `databricks-sdk` installed
- Access to a Databricks workspace
- Proper permissions for the target environment (catalog creation rights)

## Phase 0: Create Unity Catalog (One-time Setup)

### What Gets Created
- Unity Catalog for the environment
- Catalog ownership and permissions

### Why Phase 0 is Separate
Phase 0 creates the Unity Catalog using the Databricks SDK/API, which is more reliable than SQL for catalog creation. This phase is idempotent and safe to run multiple times.

**Important:** Catalog names cannot contain special characters like `@` or `.`. Phase 0 ensures the catalog name is properly sanitized.

### Steps

1. **Install dependencies** (if not already installed):
```bash
pip install databricks-sdk
```

2. **Run Phase 0 setup**:
```bash
./deploy-phase0.sh dev
```

The script will:
- Validate prerequisites
- Create the Unity Catalog with name: `laurent_prat_entity_matching_dev`
- Set ownership to: `laurent.prat@databricks.com`
- Grant necessary permissions
- Verify catalog accessibility

3. **Verify catalog creation**:
```bash
# Via CLI
databricks catalogs get laurent_prat_entity_matching_dev

# Or check in the Databricks UI under Data > Catalogs
```

### Configuration

Catalog settings are defined in `catalog-config.yml`:
- Catalog names for each environment
- Owners and permissions
- Schema structure

### For Other Environments

For staging or prod:
```bash
./deploy-phase0.sh staging
./deploy-phase0.sh prod
```

**Note:** Update the owner in `deploy-phase0.sh` to use service principals for production environments.

## Phase 1: Setup and Training

### Prerequisites
- Phase 0 must be completed (Unity Catalog created)
- Catalog `laurent_prat_entity_matching_dev` exists

### What Gets Deployed
- Unity Catalog setup job (`setup_unity_catalog`) - Creates schemas
- Model training job (`train_ditto_model`) - Trains and registers model

### Steps

1. **Validate the bundle**:
```bash
databricks bundle validate -t dev
```

2. **Deploy Phase 1**:
```bash
databricks bundle deploy -t dev
```

This deploys two jobs:
- `[dev] Entity Matching - Setup Unity Catalog` - Creates bronze/silver/gold/models schemas
- `[dev] Entity Matching - Train Ditto Model` - Trains and registers the model

3. **Run the setup job** (creates schemas):
```bash
# Via UI: Go to Workflows and run "[dev] Entity Matching - Setup Unity Catalog"
# Or via CLI:
databricks jobs list | grep "Setup Unity Catalog"
# Then run by job ID
```

4. **Verify schemas created**:
Check in the Databricks UI that these schemas exist:
- `laurent_prat_entity_matching_dev.bronze`
- `laurent_prat_entity_matching_dev.silver`
- `laurent_prat_entity_matching_dev.gold`
- `laurent_prat_entity_matching_dev.models`

5. **Run the training job**:
```bash
# Via UI: Go to Workflows and run "[dev] Entity Matching - Train Ditto Model"
# Or via CLI:
databricks jobs run-now --job-name "[dev] Entity Matching - Train Ditto Model"
```

6. **Verify model registration**:
After the training job completes, verify the model exists:
```bash
# Check in Unity Catalog that the model is registered
# Model name: laurent.prat@databricks.com_entity_matching_dev.models.entity_matching_ditto
```

## Phase 2: Model Serving

### What Gets Deployed
- Ditto model serving endpoint (plus all Phase 1 resources)

### Steps

1. **Deploy Phase 2**:
```bash
./deploy-phase.sh 2 dev
```

4. **Wait for endpoint to be ready**:
The model serving endpoint will take several minutes to provision and become ready. Monitor in the UI under "Serving".

5. **Test the endpoint**:
```bash
# Test via UI or API call once endpoint is ready
```

## Phase 3: Production Pipelines

### What Gets Deployed
- Scheduled production matching pipeline (`entity_matching_pipeline`)
- Ad-hoc matching job (`adhoc_entity_matching`)
- (Plus all Phase 1 and Phase 2 resources)

### Steps

1. **Deploy Phase 3**:
```bash
./deploy-phase.sh 3 dev
```

4. **Verify deployment**:
Check that all jobs are visible in the Workflows UI:
- `[dev] Entity Matching - Setup Unity Catalog`
- `[dev] Entity Matching - Train Ditto Model`
- `[dev] Entity Matching - Production Pipeline`
- `[dev] Entity Matching - Ad-hoc Run`

## Permissions (Dev Environment)

The dev environment is configured with restricted permissions:

**Bundle-level permissions:**
- CAN_MANAGE: `laurent.prat@databricks.com` (current user)
- CAN_VIEW: Group `users`

**Model Serving permissions:**
- CAN_MANAGE: `laurent.prat@databricks.com`
- CAN_QUERY: `laurent.prat@databricks.com`

## Troubleshooting

### Model Not Found Error
If you see "RegisteredModel does not exist" during Phase 2:
- Ensure Phase 1 training job completed successfully
- Verify the model is registered in Unity Catalog
- Check the model name matches: `${workspace.current_user.userName}_entity_matching_dev.models.entity_matching_ditto`

### Endpoint Name Too Long
If you see "Endpoint name must be maximum 63 characters":
- This is fixed in `model_serving.yml` with the shorter name `ditto-em-${bundle.target}`

### Python Wheel Build Errors
If artifacts fail to build:
- The training job uses `python_wheel_task` which requires the wheel to be built
- Currently artifacts are commented out in databricks.yml
- Uncomment and fix the build command if needed

## Rolling Back

To roll back to a previous phase, simply redeploy using an earlier phase:

```bash
# Roll back from Phase 3 to Phase 2
./deploy-phase.sh 2 dev

# Roll back from Phase 2 to Phase 1
./deploy-phase.sh 1 dev
```

**Note:** This removes resources from the bundle definition but doesn't delete them. To fully clean up:

```bash
# Destroy all resources and start fresh
databricks bundle destroy -t dev
./deploy-phase.sh 1 dev
```

## Next Steps

After successful deployment:
- Monitor job runs in the Workflows UI
- Check model serving endpoint metrics
- Review Unity Catalog tables for data quality
- Adjust schedules and parameters as needed
