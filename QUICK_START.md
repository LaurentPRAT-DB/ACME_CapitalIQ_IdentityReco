# Quick Start Guide

## TL;DR - Complete Deployment Steps

For a complete dev environment deployment, run these commands in order:

```bash
# Make script executable (first time only)
chmod +x deploy-phase.sh

# Phase 0: Create Unity Catalog and schemas
./deploy-phase.sh 0 dev

# Phase 1: Create tables and load reference data
./deploy-phase.sh 1 dev

# Phase 2: Train Ditto model
./deploy-phase.sh 2 dev

# Phase 3: Deploy model serving endpoint
./deploy-phase.sh 3 dev

# Phase 4: Deploy production pipeline
./deploy-phase.sh 4 dev
```

## File Locations

- **Deployment Script**: `./deploy-phase.sh`
- **Phase Config Files**: `databricks-phase0.yml` through `databricks-phase4.yml`
- **Phase 0 Resources**: `resources/jobs_phase0_catalog.yml`
- **Phase 1 Resources**: `resources/jobs_phase1_data.yml`
- **Phase 2 Resources**: `resources/jobs_phase2_training.yml`
- **Phase 3 Resources**: `resources/jobs_phase3_serving.yml`
- **Phase 4 Resources**: `resources/jobs_phase4_pipeline.yml`

## Key Catalog Names

| Environment | Catalog Name |
|-------------|--------------|
| Dev | `laurent_prat_entity_matching_dev` |
| Staging | `entity_matching_staging` |
| Prod | `entity_matching` |

## Troubleshooting Quick Fixes

### "Catalog does not exist"
```bash
./deploy-phase.sh 0 dev
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
./deploy-phase.sh 0 dev
./deploy-phase.sh 1 dev
```

## Permissions (Dev Environment)

- **CAN_MANAGE**: laurent.prat@databricks.com (current user)
- **CAN_VIEW**: Group "account users"

## Next Steps

After successful deployment, see:
- **Full Guide**: `DEPLOYMENT_GUIDE.md`
- **Testing**: Run the ad-hoc matching job to test the pipeline
- **Monitoring**: Check job runs in Workflows UI
