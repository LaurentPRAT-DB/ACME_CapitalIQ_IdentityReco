# Phased Deployment Guide

This guide walks through the phased deployment process for the Entity Matching project using Databricks Asset Bundles.

## Overview

The deployment is split into three phases to ensure proper dependency management:

1. **Phase 1**: Setup and Training (create catalog, train model)
2. **Phase 2**: Model Serving (deploy trained model endpoint)
3. **Phase 3**: Production Pipelines (deploy scheduled matching jobs)

## Prerequisites

- Databricks CLI installed and configured
- Access to a Databricks workspace
- Proper permissions for the target environment

## Phase 1: Setup and Training

### What Gets Deployed
- Unity Catalog setup job (`setup_unity_catalog`)
- Model training job (`train_ditto_model`)

### Steps

1. **Edit databricks.yml** - Ensure only Phase 1 is uncommented:
```yaml
include:
  - resources/jobs_setup_training.yml  # Phase 1: Setup and training
  # - resources/model_serving.yml      # Phase 2: Model serving (enable after training)
  # - resources/jobs_pipeline.yml      # Phase 3: Production pipelines (enable after serving)
```

2. **Validate the bundle**:
```bash
databricks bundle validate -t dev
```

3. **Deploy Phase 1**:
```bash
databricks bundle deploy -t dev
```

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
- Ditto model serving endpoint

### Steps

1. **Edit databricks.yml** - Enable Phase 2:
```yaml
include:
  - resources/jobs_setup_training.yml  # Phase 1: Setup and training
  - resources/model_serving.yml        # Phase 2: Model serving (enable after training)
  # - resources/jobs_pipeline.yml      # Phase 3: Production pipelines (enable after serving)
```

2. **Validate the bundle**:
```bash
databricks bundle validate -t dev
```

3. **Deploy Phase 2**:
```bash
databricks bundle deploy -t dev
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

### Steps

1. **Edit databricks.yml** - Enable Phase 3:
```yaml
include:
  - resources/jobs_setup_training.yml  # Phase 1: Setup and training
  - resources/model_serving.yml        # Phase 2: Model serving
  - resources/jobs_pipeline.yml        # Phase 3: Production pipelines
```

2. **Validate the bundle**:
```bash
databricks bundle validate -t dev
```

3. **Deploy Phase 3**:
```bash
databricks bundle deploy -t dev
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
- CAN_VIEW: Service principal `b1edc2eb-4f83-4d2c-a4b6-5f626e3024dd`
- CAN_VIEW: Group `users`
- CAN_MANAGE: Group `databricks_workshop`

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

To roll back a phase:

1. Edit `databricks.yml` to comment out the phase
2. Run `databricks bundle deploy -t dev`
3. Manually delete resources if needed via `databricks bundle destroy -t dev`

## Next Steps

After successful deployment:
- Monitor job runs in the Workflows UI
- Check model serving endpoint metrics
- Review Unity Catalog tables for data quality
- Adjust schedules and parameters as needed
