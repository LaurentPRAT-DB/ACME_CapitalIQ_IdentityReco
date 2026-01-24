# Deployment Guide - Entity Matching for S&P Capital IQ

This guide outlines the step-by-step deployment process for the Entity Matching project using Databricks Asset Bundles.

## Overview

The deployment is split into 5 phases, each building on the previous one. Use the `deploy-phase.sh` script to deploy each phase sequentially.

### Deployment Phases

- **Phase 0**: Catalog Setup - Creates Unity Catalog and schemas
- **Phase 1**: Data Load - Creates tables and loads reference data
- **Phase 2**: Model Training - Trains the Ditto matching model
- **Phase 3**: Model Deployment - Deploys model to serving endpoint
- **Phase 4**: Production Pipeline - Deploys production matching jobs

## Prerequisites

- Databricks CLI installed and configured
- Access to a Databricks workspace
- Appropriate permissions to create catalogs, schemas, and resources
- Databricks Runtime 13.3 LTS or higher

## Quick Start

### Deploy All Phases Sequentially

```bash
# Make script executable (first time only)
chmod +x deploy-phase.sh

# Phase 0: Catalog Setup
./deploy-phase.sh 0 dev

# Phase 1: Data Load
./deploy-phase.sh 1 dev

# Phase 2: Model Training
./deploy-phase.sh 2 dev

# Phase 3: Model Deployment
./deploy-phase.sh 3 dev

# Phase 4: Production Pipeline
./deploy-phase.sh 4 dev
```

Each phase will:
1. Copy the phase configuration to `databricks.yml`
2. Validate the bundle
3. Deploy to Databricks
4. Prompt to run the associated job (if applicable)

---

## How the deploy-phase.sh Script Works

The `deploy-phase.sh` script automates the phased deployment process:

1. **Copies phase configuration**: Replaces `databricks.yml` with the phase-specific file (e.g., `databricks-phase0.yml`)
2. **Validates**: Runs `databricks bundle validate` to check configuration
3. **Deploys**: Executes `databricks bundle deploy` to create resources
4. **Runs jobs**: Prompts to run associated jobs (when applicable)
5. **Shows next steps**: Displays what to do next

### Script Usage

```bash
./deploy-phase.sh <phase-number> [target]
```

**Parameters:**
- `phase-number`: 0, 1, 2, 3, or 4
- `target`: dev (default), staging, or prod

**Examples:**
```bash
./deploy-phase.sh 0          # Deploy Phase 0 to dev
./deploy-phase.sh 1 dev      # Deploy Phase 1 to dev
./deploy-phase.sh 2 staging  # Deploy Phase 2 to staging
```

---

## Phase-by-Phase Details

### Phase 0: Catalog Setup

**What it creates:**
- Unity Catalog: `laurent_prat_entity_matching_dev` (dev) or `entity_matching` (prod)
- Schemas: `bronze`, `silver`, `gold`, `models`
- Permissions for account users

**Deploy:**
```bash
./deploy-phase.sh 0 dev
```

**What happens:**
1. Script copies `databricks-phase0.yml` to `databricks.yml`
2. Validates and deploys the bundle
3. Prompts to run the `setup_catalog` job
4. Job creates catalog and schemas

**Verify:**
```sql
SHOW SCHEMAS IN `laurent_prat_entity_matching_dev`;
```

**Manual deployment (without script):**
```bash
cp databricks-phase0.yml databricks.yml
databricks bundle deploy -t dev
databricks bundle run setup_catalog -t dev
```

---

### Phase 1: Data Load

**Prerequisites:** Phase 0 must be completed

**What it creates:**
- Bronze tables: `spglobal_reference`, `source_entities`
- Gold table: `matched_entities`
- Views: `review_queue`, `daily_stats`
- Sample S&P 500 reference data
- Table-level SELECT permissions

**Deploy:**
```bash
./deploy-phase.sh 1 dev
```

**What happens:**
1. Script copies `databricks-phase1.yml` to `databricks.yml`
2. Validates and deploys the bundle
3. Prompts to run the `load_reference_data` job
4. Job creates tables and loads sample data

**Verify:**
```sql
SELECT COUNT(*) FROM `laurent_prat_entity_matching_dev`.bronze.spglobal_reference;
DESCRIBE TABLE `laurent_prat_entity_matching_dev`.gold.matched_entities;
```

---

### Phase 2: Model Training

**Prerequisites:** Phases 0 and 1 must be completed

**What it creates:**
- Training job that generates 1000 positive and 1000 negative training pairs
- Trains Ditto model on entity matching task
- Registers model in Unity Catalog as `<catalog>.models.entity_matching_ditto`

**Deploy:**
```bash
./deploy-phase.sh 2 dev
```

**What happens:**
1. Script copies `databricks-phase2.yml` to `databricks.yml`
2. Validates and deploys the bundle
3. Prompts to run the `train_ditto_model` job
4. Job trains model (may take several hours)

**Note:** This phase may take several hours depending on cluster configuration.

**Verify:**
```sql
SELECT * FROM `laurent_prat_entity_matching_dev`.models.models
WHERE name = 'entity_matching_ditto';
```

---

### Phase 3: Model Deployment

**Prerequisites:** Phases 0, 1, and 2 must be completed

**What it creates:**
- Model serving endpoint: `ditto-em-dev`
- Small workload size with scale-to-zero enabled
- Traffic routing configuration

**Deploy:**
```bash
./deploy-phase.sh 3 dev
```

**What happens:**
1. Script copies `databricks-phase3.yml` to `databricks.yml`
2. Validates and deploys the bundle
3. Creates model serving endpoint (no job to run)
4. Endpoint takes ~5 minutes to become ready

**Note:** This is deployment-only (no job to run).

**Verify:**
- Check Databricks UI: Serving → Endpoints
- Or use CLI: `databricks serving-endpoints list`

---

### Phase 4: Production Pipeline

**Prerequisites:** All previous phases (0, 1, 2, 3) must be completed

**What it creates:**
- Scheduled daily matching pipeline job
- Ad-hoc matching job for on-demand runs
- Complete pipeline with 5 stages:
  1. Ingest source entities
  2. Exact match on identifiers
  3. Vector search + Ditto matching
  4. Write results to gold table
  5. Generate metrics

**Deploy:**
```bash
./deploy-phase.sh 4 dev
```

**What happens:**
1. Script copies `databricks-phase4.yml` to `databricks.yml`
2. Validates and deploys the bundle
3. Prompts to run the `entity_matching_pipeline` job
4. Job runs the complete matching pipeline

**Run ad-hoc job:**
```bash
databricks bundle run adhoc_entity_matching -t dev
```

**Verify:**
```sql
SELECT * FROM `laurent_prat_entity_matching_dev`.gold.matched_entities LIMIT 10;
SELECT * FROM `laurent_prat_entity_matching_dev`.gold.daily_stats;
```

---

## Environment Targets

### Development (`dev`)
- Catalog: `laurent_prat_entity_matching_dev`
- User-owned resources
- Default target
- Deploy: `./deploy-phase.sh <phase> dev`

### Staging (`staging`)
- Catalog: `entity_matching_staging`
- Service principal owned (requires `var.staging_service_principal`)
- Production mode
- Deploy: `./deploy-phase.sh <phase> staging`

### Production (`prod`)
- Catalog: `entity_matching`
- Service principal owned (requires `var.prod_service_principal`)
- Production mode with larger clusters (i3.2xlarge)
- Deploy: `./deploy-phase.sh <phase> prod`

---

## Troubleshooting

### Issue: Script permission denied
```bash
chmod +x deploy-phase.sh
```

### Issue: Phase file not found
Ensure all `databricks-phase*.yml` files exist in the project root.

### Issue: Validation failed
- Check error messages from `databricks bundle validate`
- Script automatically restores previous `databricks.yml` on validation failure

### Issue: GRANT syntax errors
The notebooks have been updated with correct GRANT syntax. Redeploy Phase 0 and Phase 1 if you encounter these errors.

### Issue: VIEW COMMENT syntax error
The notebooks have been updated with correct VIEW syntax (COMMENT before AS). Redeploy Phase 1 if you encounter this error.

### Issue: Missing model for Phase 3
- Ensure Phase 2 completed successfully
- Check that model is registered: `SELECT * FROM models.models WHERE name LIKE '%entity_matching_ditto%'`

### Issue: PRIMARY KEY constraint errors
- Ensure using Databricks Runtime 13.3 LTS or higher
- Check cluster spark version in `databricks-phase*.yml`

---

## Manual Deployment (Without Script)

If you prefer manual control:

```bash
# Phase 0
cp databricks-phase0.yml databricks.yml
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run setup_catalog -t dev

# Phase 1
cp databricks-phase1.yml databricks.yml
databricks bundle deploy -t dev
databricks bundle run load_reference_data -t dev

# Phase 2
cp databricks-phase2.yml databricks.yml
databricks bundle deploy -t dev
databricks bundle run train_ditto_model -t dev

# Phase 3
cp databricks-phase3.yml databricks.yml
databricks bundle deploy -t dev
# No job to run - just deployment

# Phase 4
cp databricks-phase4.yml databricks.yml
databricks bundle deploy -t dev
databricks bundle run entity_matching_pipeline -t dev
```

---

## Rollback

To rollback to a previous phase:

```bash
# Rollback to Phase 2 from Phase 3
./deploy-phase.sh 2 dev
```

The script will redeploy the earlier phase configuration.

To completely clean up:
```bash
databricks bundle destroy -t dev
```

**Note:** This removes deployed resources but does NOT drop the catalog. To drop the catalog:
```sql
DROP CATALOG IF EXISTS `laurent_prat_entity_matching_dev` CASCADE;
```

---

## Resource Files

Each phase has dedicated configuration files:

| Phase | Config File | Resource File | Job Name |
|-------|-------------|---------------|----------|
| 0 | `databricks-phase0.yml` | `resources/jobs_phase0_catalog.yml` | `setup_catalog` |
| 1 | `databricks-phase1.yml` | `resources/jobs_phase1_data.yml` | `load_reference_data` |
| 2 | `databricks-phase2.yml` | `resources/jobs_phase2_training.yml` | `train_ditto_model` |
| 3 | `databricks-phase3.yml` | `resources/jobs_phase3_serving.yml` | (deployment only) |
| 4 | `databricks-phase4.yml` | `resources/jobs_phase4_pipeline.yml` | `entity_matching_pipeline` |

---

## Next Steps

After successful deployment of all phases:

1. **Monitor Jobs**: Check Databricks UI → Workflows → Jobs
2. **Check Model Health**: Serving → Endpoints → ditto-em-dev
3. **Review Data**: Data → Unity Catalog → Browse tables
4. **Set Up Alerts**: Configure email notifications for job failures
5. **Performance Tuning**: Adjust cluster sizes and autoscaling settings
6. **Schedule Jobs**: Review cron schedule for production pipeline
7. **Documentation**: Update team documentation with deployment details

---

## Summary

✅ **Automated deployment** with `deploy-phase.sh` script
✅ **Step-by-step phases** from catalog to production
✅ **Validation** at each phase before deployment
✅ **Clear error handling** with automatic rollback on validation failure
✅ **Flexible targeting** for dev, staging, and production environments

For questions or issues, check the troubleshooting section or review the Databricks Asset Bundle logs.
