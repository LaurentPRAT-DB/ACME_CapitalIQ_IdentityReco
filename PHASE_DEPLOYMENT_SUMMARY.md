# Phase Deployment Summary

## Problem Solved

**Original Issue**: Deployment failed because:
1. Catalog name `laurent.prat@databricks.com_entity_matching_dev` contained invalid characters (`@` and `.`)
2. SQL CREATE CATALOG statement doesn't handle special characters well
3. No idempotent way to create catalog before deployment

**Solution**: Split deployment into phases with Phase 0 creating the catalog via API.

## What Was Created

### Files Created

| File | Purpose |
|------|---------|
| `deploy-phase.sh` | Unified deployment script for all phases (0-4) |
| `databricks-phase0.yml` through `databricks-phase4.yml` | Phase-specific bundle configurations |
| `production-config.yml` | Template for production service principal configuration |
| `notebooks/setup/01_create_unity_catalog.py` | Creates Unity Catalog and schemas |
| `notebooks/setup/02_create_reference_tables.py` | Creates tables and loads reference data |
| `resources/jobs_phase0_catalog.yml` | Phase 0: Catalog setup job |
| `resources/jobs_phase1_data.yml` | Phase 1: Data load job |
| `resources/jobs_phase2_training.yml` | Phase 2: Model training job |
| `resources/jobs_phase3_serving.yml` | Phase 3: Model serving endpoint |
| `resources/jobs_phase4_pipeline.yml` | Phase 4: Production pipeline jobs |
| `DEPLOYMENT_GUIDE.md` | Complete deployment guide |
| `PRODUCTION_SETUP.md` | Production deployment with service principals |

### Files Modified

| File | Changes |
|------|---------|
| `databricks.yml` | - Fixed catalog name to `laurent_prat_entity_matching_dev`<br>- Commented out Phase 2/3 resources<br>- Removed invalid permissions sections |
| `DEPLOYMENT_GUIDE.md` | Added Phase 0 documentation |

## Deployment Phases

### Phase 0: Catalog Setup ✅ Ready to Run
```bash
./deploy-phase.sh 0 dev
```

**What it does**:
- Creates catalog `laurent_prat_entity_matching_dev`
- Creates schemas: bronze, silver, gold, models
- Grants permissions to account users
- Idempotent - safe to run multiple times

**When to run**: First step in deployment process

### Phase 1: Setup and Training ✅ Currently Deployed
```bash
databricks bundle deploy -t dev
```

**Status**: Already deployed (2 jobs created)
- `[dev laurent_prat] [dev] Entity Matching - Setup Unity Catalog`
- `[dev laurent_prat] [dev] Entity Matching - Train Ditto Model`

**Next step**: Run Phase 0, then re-run the setup job

### Phase 2: Model Serving ⏸️ Commented Out
**To enable**:
1. Uncomment in `databricks.yml`:
   - `- resources/model_serving.yml` (line ~14)
   - `resources.model_serving_endpoints` section in dev target (lines ~69-76)
2. Run: `databricks bundle deploy -t dev`

**Prerequisite**: Model must be registered (Phase 1 training job completed)

### Phase 3: Production Pipelines ⏸️ Commented Out
**To enable**:
1. Uncomment in `databricks.yml`:
   - `- resources/jobs_pipeline.yml` (line ~15)
2. Run: `databricks bundle deploy -t dev`

**Prerequisite**: Model serving endpoint deployed and ready (Phase 2 completed)

## Key Changes to databricks.yml

### Catalog Name Fixed
```yaml
# Before (INVALID):
catalog_name: ${workspace.current_user.userName}_entity_matching_dev
# Resolved to: laurent.prat@databricks.com_entity_matching_dev

# After (VALID):
catalog_name: laurent_prat_entity_matching_dev
```

### Permissions Simplified
```yaml
# Dev environment permissions (cleaned up):
permissions:
  - level: CAN_MANAGE
    user_name: ${workspace.current_user.userName}
  - level: CAN_VIEW
    group_name: users
```

Removed:
- Service principal permissions (were causing issues)
- databricks_workshop group (unnecessary for current user isolation)

## Current State

✅ **Completed**:
- Phase deployment structure created
- Phase 1 deployed (jobs created)
- Catalog name fixed
- Scripts and documentation ready

⚠️ **Needs Action**:
1. Run Phase 0 to create catalog and schemas:
   ```bash
   ./deploy-phase.sh 0 dev
   ```

2. Re-run the setup job:
   - Go to Workflows UI
   - Run "[dev laurent_prat] [dev] Entity Matching - Setup Unity Catalog"
   - This will create the schemas (bronze, silver, gold, models)

3. Run the training job:
   - Go to Workflows UI
   - Run "[dev laurent_prat] [dev] Entity Matching - Train Ditto Model"
   - This trains and registers the model

4. Enable Phase 2 (model serving) after training completes

5. Enable Phase 3 (production pipelines) after serving endpoint is ready

## Testing the Deployment

After running Phase 0:
```bash
# Verify catalog exists
databricks catalogs get laurent_prat_entity_matching_dev

# Verify bundle validates
databricks bundle validate -t dev

# View deployed resources
databricks bundle summary -t dev
```

## Quick Reference

- **Full documentation**: `DEPLOYMENT_GUIDE.md`
- **Quick commands**: `QUICK_START.md`
- **Script help**: `scripts/README.md`
- **Current config**: `databricks.yml`
- **Catalog config**: `catalog-config.yml`
