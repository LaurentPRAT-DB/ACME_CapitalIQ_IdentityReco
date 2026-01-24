# Phased Deployment Guide

This guide walks through the phased deployment process for the Entity Matching project using Databricks Asset Bundles.

## Overview

The deployment is split into three phases to ensure proper dependency management:

1. **Phase 1**: Setup and Training (create catalog, train model)
2. **Phase 2**: Model Serving (deploy trained model endpoint)
3. **Phase 3**: Production Pipelines (deploy scheduled matching jobs)

## Deployment Configuration Files

Each phase has a dedicated configuration file:

| Phase | Config File | Command |
|-------|-------------|---------|
| Phase 1 | `databricks-phase1.yml` | `./deploy-phase.sh 1 dev` |
| Phase 2 | `databricks-phase2.yml` | `./deploy-phase.sh 2 dev` |
| Phase 3 | `databricks-phase3.yml` | `./deploy-phase.sh 3 dev` |

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
- Access to a Databricks workspace
- Proper permissions for the target environment

## Phase 1: Setup and Training

### What Gets Deployed
- Unity Catalog setup job (`setup_unity_catalog`)
- Model training job (`train_ditto_model`)

### Steps

1. **Deploy Phase 1** (includes validation):
```bash
./deploy-phase.sh 1 dev
```

The script will:
- Copy `databricks-phase1.yml` to `databricks.yml`
- Validate the configuration
- Prompt for confirmation
- Deploy to the dev environment

4. **Run the setup job**:
```bash
# Via UI: Go to Workflows and run "[dev] Entity Matching - Setup Unity Catalog"
# Or via CLI:
databricks jobs run-now --job-name "[dev] Entity Matching - Setup Unity Catalog"
```

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
