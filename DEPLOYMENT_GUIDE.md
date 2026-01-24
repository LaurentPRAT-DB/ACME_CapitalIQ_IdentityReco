# Deployment Guide - Entity Matching for S&P Capital IQ

This guide outlines the step-by-step deployment process for the Entity Matching project using Databricks Asset Bundles.

## Overview

The deployment is split into 5 phases, each building on the previous one:

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

## Phase-by-Phase Deployment

### Phase 0: Catalog Setup

**Purpose**: Creates the Unity Catalog and schemas (bronze, silver, gold, models)

**Deploy**:
```bash
databricks bundle deploy -t dev --config-file databricks-phase0.yml
```

**Run**:
```bash
databricks bundle run setup_catalog -t dev --config-file databricks-phase0.yml
```

**What it does**:
- Creates Unity Catalog: `laurent_prat_entity_matching_dev` (dev) or `entity_matching` (prod)
- Creates schemas: `bronze`, `silver`, `gold`, `models`
- Grants permissions to account users

**Verify**:
```sql
SHOW SCHEMAS IN `laurent_prat_entity_matching_dev`;
```

---

### Phase 1: Data Load

**Purpose**: Creates tables and loads sample reference data

**Prerequisites**: Phase 0 must be completed

**Deploy**:
```bash
databricks bundle deploy -t dev --config-file databricks-phase1.yml
```

**Run**:
```bash
databricks bundle run load_reference_data -t dev --config-file databricks-phase1.yml
```

**What it does**:
- Creates bronze tables: `spglobal_reference`, `source_entities`
- Creates gold table: `matched_entities`
- Creates views: `review_queue`, `daily_stats`
- Loads sample S&P 500 reference data
- Grants SELECT permissions

**Verify**:
```sql
SELECT COUNT(*) FROM `laurent_prat_entity_matching_dev`.bronze.spglobal_reference;
DESCRIBE TABLE `laurent_prat_entity_matching_dev`.gold.matched_entities;
```

---

### Phase 2: Model Training

**Purpose**: Generates training data and trains the Ditto model

**Prerequisites**: Phases 0 and 1 must be completed

**Deploy**:
```bash
databricks bundle deploy -t dev --config-file databricks-phase2.yml
```

**Run**:
```bash
databricks bundle run train_ditto_model -t dev --config-file databricks-phase2.yml
```

**What it does**:
- Generates positive and negative training pairs (1000 each)
- Trains Ditto model on entity matching task
- Registers model in Unity Catalog as `<catalog>.models.entity_matching_ditto`

**Note**: This phase may take several hours depending on cluster configuration

**Verify**:
```sql
SELECT * FROM `laurent_prat_entity_matching_dev`.models.models
WHERE name = 'entity_matching_ditto';
```

---

### Phase 3: Model Deployment

**Purpose**: Deploys the trained model to a serving endpoint

**Prerequisites**: Phases 0, 1, and 2 must be completed

**Deploy**:
```bash
databricks bundle deploy -t dev --config-file databricks-phase3.yml
```

**What it does**:
- Creates model serving endpoint: `ditto-em-dev`
- Configures endpoint with Small workload size
- Enables scale-to-zero
- Sets up traffic routing

**Note**: This is a deployment phase only - no job to run

**Verify**:
- Check serving endpoints in Databricks UI
- Or use CLI: `databricks serving-endpoints list`

---

### Phase 4: Production Pipeline

**Purpose**: Deploys production matching pipeline jobs

**Prerequisites**: Phases 0, 1, 2, and 3 must be completed

**Deploy**:
```bash
databricks bundle deploy -t dev --config-file databricks-phase4.yml
```

**Run** (scheduled job):
```bash
databricks bundle run entity_matching_pipeline -t dev --config-file databricks-phase4.yml
```

**Run** (ad-hoc job):
```bash
databricks bundle run adhoc_entity_matching -t dev --config-file databricks-phase4.yml
```

**What it does**:
- Deploys scheduled daily matching pipeline
- Deploys ad-hoc matching job for on-demand runs
- Pipeline includes:
  1. Ingest source entities
  2. Exact match on identifiers
  3. Vector search + Ditto matching
  4. Write results to gold table
  5. Generate metrics

**Verify**:
```sql
SELECT * FROM `laurent_prat_entity_matching_dev`.gold.matched_entities LIMIT 10;
SELECT * FROM `laurent_prat_entity_matching_dev`.gold.daily_stats;
```

---

## Complete Deployment (All Phases)

To deploy all phases at once (for environments where prerequisites are met):

```bash
# Deploy Phase 0
databricks bundle deploy -t dev --config-file databricks-phase0.yml
databricks bundle run setup_catalog -t dev --config-file databricks-phase0.yml

# Deploy Phase 1
databricks bundle deploy -t dev --config-file databricks-phase1.yml
databricks bundle run load_reference_data -t dev --config-file databricks-phase1.yml

# Deploy Phase 2
databricks bundle deploy -t dev --config-file databricks-phase2.yml
databricks bundle run train_ditto_model -t dev --config-file databricks-phase2.yml

# Deploy Phase 3 (no run command - just deployment)
databricks bundle deploy -t dev --config-file databricks-phase3.yml

# Deploy Phase 4
databricks bundle deploy -t dev --config-file databricks-phase4.yml
```

---

## Environment Targets

### Development (`dev`)
- Catalog: `laurent_prat_entity_matching_dev`
- User-owned resources
- Default target

### Staging (`staging`)
- Catalog: `entity_matching_staging`
- Service principal owned (requires `var.staging_service_principal`)
- Production mode

### Production (`prod`)
- Catalog: `entity_matching`
- Service principal owned (requires `var.prod_service_principal`)
- Production mode with larger clusters

To deploy to a specific target:
```bash
databricks bundle deploy -t staging --config-file databricks-phase0.yml
```

---

## Troubleshooting

### Issue: GRANT syntax errors
- **Solution**: Ensure you're using the updated notebooks with correct GRANT syntax (schema-level vs table-level privileges)

### Issue: Libraries field warning
- **Solution**: Ensure `libraries` is inside `new_cluster` block, not at `job_cluster` level

### Issue: VIEW COMMENT syntax error
- **Solution**: Ensure COMMENT comes before AS clause in CREATE VIEW statements

### Issue: Missing model for Phase 3
- **Error**: Model not found for serving endpoint
- **Solution**: Ensure Phase 2 completed successfully and model is registered in Unity Catalog

### Issue: PRIMARY KEY constraint errors
- **Solution**: Ensure using Databricks Runtime 13.3 LTS or higher

---

## Resource Files

Each phase includes specific resource files:

- `resources/jobs_phase0_catalog.yml` - Catalog setup job
- `resources/jobs_phase1_data.yml` - Data load job
- `resources/jobs_phase2_training.yml` - Model training job
- `resources/jobs_phase3_serving.yml` - Model serving endpoint
- `resources/jobs_phase4_pipeline.yml` - Production pipeline jobs

---

## Cleanup

To remove deployed resources:

```bash
# Destroy Phase 4 (pipeline jobs)
databricks bundle destroy -t dev --config-file databricks-phase4.yml

# Destroy Phase 3 (model serving)
databricks bundle destroy -t dev --config-file databricks-phase3.yml

# Destroy Phase 2 (training jobs)
databricks bundle destroy -t dev --config-file databricks-phase2.yml

# Destroy Phase 1 (data jobs)
databricks bundle destroy -t dev --config-file databricks-phase1.yml

# Destroy Phase 0 (catalog setup)
databricks bundle destroy -t dev --config-file databricks-phase0.yml
```

**Note**: Destroying Phase 0 will NOT drop the catalog - you must manually drop it:
```sql
DROP CATALOG IF EXISTS `laurent_prat_entity_matching_dev` CASCADE;
```

---

## Next Steps

After successful deployment:

1. Monitor job runs in Databricks UI
2. Check model serving endpoint health
3. Review matched entities in gold table
4. Set up alerting on job failures
5. Configure additional monitoring and logging
