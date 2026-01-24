# Quick Start Guide

## TL;DR - Complete Deployment Steps

For a complete dev environment deployment, run these commands in order:

```bash
# Phase 0: Create Unity Catalog (one-time setup)
./deploy-phase0.sh dev

# Phase 1: Deploy setup and training jobs
databricks bundle deploy -t dev

# Run the setup job (creates schemas)
# → Go to Workflows UI and run "[dev] Entity Matching - Setup Unity Catalog"

# Run the training job (trains and registers model)
# → Go to Workflows UI and run "[dev] Entity Matching - Train Ditto Model"

# Phase 2: Enable model serving
# Edit databricks.yml:
#   - Uncomment: - resources/model_serving.yml
#   - Uncomment the resources.model_serving_endpoints section in dev target
databricks bundle deploy -t dev

# Wait for model serving endpoint to be ready (~5-10 minutes)

# Phase 3: Enable production pipelines
# Edit databricks.yml:
#   - Uncomment: - resources/jobs_pipeline.yml
databricks bundle deploy -t dev
```

## File Locations

- **Phase 0 Script**: `./deploy-phase0.sh`
- **Catalog Config**: `catalog-config.yml`
- **Bundle Config**: `databricks.yml`
- **Phase 1 Resources**: `resources/jobs_setup_training.yml`
- **Phase 2 Resources**: `resources/model_serving.yml`
- **Phase 3 Resources**: `resources/jobs_pipeline.yml`

## Key Catalog Names

| Environment | Catalog Name |
|-------------|--------------|
| Dev | `laurent_prat_entity_matching_dev` |
| Staging | `entity_matching_staging` |
| Prod | `entity_matching` |

## Troubleshooting Quick Fixes

### "Catalog does not exist"
```bash
./deploy-phase0.sh dev
```

### "Endpoint name too long"
Already fixed in `model_serving.yml` - endpoint name is `ditto-em-${bundle.target}`

### "RegisteredModel does not exist"
Ensure Phase 1 training job completed successfully. Check:
```bash
# Verify model exists in Unity Catalog
databricks models list --catalog laurent_prat_entity_matching_dev
```

### Clean slate restart
```bash
databricks bundle destroy -t dev
./deploy-phase0.sh dev
databricks bundle deploy -t dev
```

## Permissions (Dev Environment)

- **CAN_MANAGE**: laurent.prat@databricks.com
- **CAN_VIEW**: Group "users"

## Next Steps

After successful deployment, see:
- **Full Guide**: `DEPLOYMENT_GUIDE.md`
- **Testing**: Run the ad-hoc matching job to test the pipeline
- **Monitoring**: Check job runs in Workflows UI
