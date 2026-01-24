# Databricks Asset Bundle Deployment Guide

**Deploy Entity Matching Solution to Databricks using Asset Bundles**

This guide explains how to deploy the complete entity matching solution to your Databricks workspace using Databricks Asset Bundles (DAB).

---

## 📋 Prerequisites

### Required

- ✅ Databricks CLI installed and configured
- ✅ Databricks workspace with Unity Catalog enabled
- ✅ Python 3.9+ installed locally
- ✅ DEFAULT profile configured in `~/.databrickscfg`

### Verify Prerequisites

```bash
# Check Databricks CLI
databricks --version
# Should show: Databricks CLI v0.209.0 or higher

# Verify authentication
databricks auth env --profile DEFAULT

# Check workspace access
databricks workspace ls /
```

---

## 🚀 Quick Deployment (5 Minutes)

### Step 1: Install Databricks CLI (if needed)

```bash
# Install or upgrade Databricks CLI
pip install --upgrade databricks-cli
```

### Step 2: Validate Bundle

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Expected output:
# ✓ Validation successful
```

### Step 3: Deploy to Development

```bash
# Deploy to dev environment
databricks bundle deploy -t dev

# This will:
# - Upload notebooks to Databricks workspace
# - Create job definitions
# - Configure model serving endpoints
# - Set up permissions
```

### Step 4: Run Setup Job

```bash
# Run Unity Catalog setup
databricks bundle run setup_unity_catalog -t dev

# Expected output:
# ✓ Catalog created: <user>_entity_matching_dev
# ✓ Schemas created: bronze, silver, gold, models
# ✓ Tables created
```

### Step 5: Verify Deployment

```bash
# List deployed resources
databricks bundle resources list -t dev

# Expected output:
# Jobs:
#   - setup_unity_catalog
#   - train_ditto_model
#   - entity_matching_pipeline
#   - adhoc_entity_matching
# Model Serving Endpoints:
#   - entity-matching-ditto-dev
```

---

## 📦 Bundle Structure

### Files and Directories

```
MET_CapitalIQ_identityReco/
├── databricks.yml              # Main bundle configuration
├── resources/                  # Resource definitions
│   ├── jobs.yml               # Job definitions
│   ├── model_serving.yml      # Model serving endpoints
│   └── pipelines.yml          # DLT pipelines (optional)
├── notebooks/                  # Notebooks to deploy
│   ├── setup/                 # Setup notebooks
│   │   ├── 01_create_unity_catalog.py
│   │   ├── 02_create_reference_tables.py
│   │   └── 03_register_model.py
│   ├── pipeline/              # Pipeline notebooks
│   │   ├── 01_ingest_source_entities.py
│   │   ├── 02_exact_match.py
│   │   ├── 03_vector_search_ditto.py
│   │   ├── 04_write_results.py
│   │   └── 05_generate_metrics.py
│   ├── 02_train_ditto_model.py
│   └── 03_full_pipeline_example.py
└── src/                       # Python package (bundled as wheel)
```

---

## 🎯 Deployment Targets

### Development (dev)

**Use for**: Local testing, development, experimentation

```bash
# Deploy to dev
databricks bundle deploy -t dev

# Resources created with prefix: <user>_entity_matching_dev
# Workspace path: /Users/<user>/.bundle/entity_matching/dev
```

**Configuration**:
- Mode: `development`
- Catalog: `<user>_entity_matching_dev`
- Run as: Your user
- Auto-cleanup: Yes (when redeploying)

### Staging (staging)

**Use for**: Pre-production testing, UAT

```bash
# Deploy to staging
databricks bundle deploy -t staging

# Resources created with prefix: entity_matching_staging
# Workspace path: /Workspace/Shared/.bundle/entity_matching/staging
```

**Configuration**:
- Mode: `production`
- Catalog: `entity_matching_staging`
- Run as: Service principal (configure first)
- Auto-cleanup: No

### Production (prod)

**Use for**: Production workloads

```bash
# Deploy to production
databricks bundle deploy -t prod

# Resources created with prefix: entity_matching
# Workspace path: /Workspace/Shared/.bundle/entity_matching/prod
```

**Configuration**:
- Mode: `production`
- Catalog: `entity_matching`
- Run as: Service principal (configure first)
- Auto-cleanup: No
- Cluster size: Larger (i3.2xlarge)

---

## 📝 Step-by-Step Deployment

### Phase 1: Initial Setup (10 minutes)

#### 1. Configure Service Principals (Staging/Prod only)

```bash
# Set service principal for staging
export TF_VAR_staging_service_principal="sp-entity-matching-staging@company.com"

# Set service principal for production
export TF_VAR_prod_service_principal="sp-entity-matching-prod@company.com"
```

#### 2. Validate Bundle

```bash
# Validate dev target
databricks bundle validate -t dev

# Check for errors
# Fix any validation issues before proceeding
```

#### 3. Deploy Bundle

```bash
# Deploy to development
databricks bundle deploy -t dev

# Monitor output for:
# ✓ Uploading notebooks...
# ✓ Creating jobs...
# ✓ Creating model serving endpoints...
# ✓ Setting permissions...
```

### Phase 2: Setup Unity Catalog (5 minutes)

```bash
# Run setup job
databricks bundle run setup_unity_catalog -t dev

# Monitor in Databricks UI:
# - Workflows → Jobs → [dev] Entity Matching - Setup Unity Catalog
```

**What this creates**:
- Catalog: `<user>_entity_matching_dev`
- Schemas: `bronze`, `silver`, `gold`, `models`
- Tables: `bronze.spglobal_reference`, `bronze.source_entities`, `gold.matched_entities`
- Views: `gold.review_queue`, `gold.daily_stats`
- Sample data (5 S&P 500 companies)

### Phase 3: Train Ditto Model (Optional - 2-4 hours)

```bash
# Run training job
databricks bundle run train_ditto_model -t dev

# This job will:
# 1. Generate training data (1000 positive + 1000 negative pairs)
# 2. Train Ditto model (20 epochs, ~2-3 hours on GPU)
# 3. Register model to Unity Catalog
```

**Skip if**: Using pre-trained model or testing without Ditto

### Phase 4: Deploy Model Serving (5 minutes)

The model serving endpoint is created during `deploy`, but needs a trained model:

```bash
# Check endpoint status
databricks serving-endpoints get entity-matching-ditto-dev

# Expected: READY state
```

**If not ready**: Endpoint will be ready once Ditto model is registered

### Phase 5: Run Pipeline (10 minutes)

#### Test with Ad-hoc Job

```bash
# Run ad-hoc matching
databricks bundle run adhoc_entity_matching -t dev \
  --params '{"source_table":"<user>_entity_matching_dev.bronze.source_entities","output_table":"<user>_entity_matching_dev.gold.matched_entities_adhoc"}'

# Monitor in Databricks UI
```

#### Schedule Production Pipeline

```bash
# Production pipeline is scheduled automatically
# Default: Daily at 2 AM PST

# To pause schedule:
databricks jobs reset --job-id <job-id> --schedule-pause-status PAUSED

# To resume:
databricks jobs reset --job-id <job-id> --schedule-pause-status UNPAUSED
```

---

## 🔧 Configuration

### Environment Variables

Set these before deployment (optional):

```bash
# Service principals (staging/prod)
export TF_VAR_staging_service_principal="sp@company.com"
export TF_VAR_prod_service_principal="sp@company.com"

# Override cluster node type
export BUNDLE_VAR_cluster_node_type="i3.2xlarge"

# Override schedule
export BUNDLE_VAR_matching_job_schedule="0 0 3 * * ?" # 3 AM instead of 2 AM
```

### Customize databricks.yml

Edit `databricks.yml` to customize:

```yaml
variables:
  catalog_name:
    default: my_custom_catalog  # Change catalog name

  cluster_node_type:
    default: i3.2xlarge  # Larger clusters

  matching_job_schedule:
    default: "0 0 */6 * * ?"  # Every 6 hours
```

---

## 📊 Deployed Resources

### Jobs

| Job Name | Purpose | Schedule |
|----------|---------|----------|
| **setup_unity_catalog** | Create catalog, schemas, tables | On-demand |
| **train_ditto_model** | Train/update Ditto matcher | On-demand |
| **entity_matching_pipeline** | Production matching pipeline | Daily 2 AM |
| **adhoc_entity_matching** | Ad-hoc matching runs | On-demand |

### Model Serving Endpoints

| Endpoint | Model | Workload Size |
|----------|-------|---------------|
| **entity-matching-ditto-{target}** | Ditto matcher | Small (scale-to-zero) |

### Unity Catalog Resources

```
<catalog_name>/
├── bronze/
│   ├── spglobal_reference (table)
│   └── source_entities (table)
├── silver/
│   └── (future normalized tables)
├── gold/
│   ├── matched_entities (table)
│   ├── review_queue (view)
│   └── daily_stats (view)
└── models/
    └── entity_matching_ditto (model)
```

---

## 🔍 Verification

### Check Deployed Jobs

```bash
# List all jobs
databricks bundle resources list -t dev

# Get specific job details
databricks jobs get --job-id <job-id>
```

### Verify Unity Catalog

```sql
-- In Databricks SQL or notebook
USE CATALOG <user>_entity_matching_dev;

-- List schemas
SHOW SCHEMAS;

-- Check tables
SHOW TABLES IN bronze;
SHOW TABLES IN gold;

-- Query reference data
SELECT * FROM bronze.spglobal_reference LIMIT 10;
```

### Test Model Serving

```bash
# Get endpoint URL
databricks serving-endpoints get entity-matching-ditto-dev

# Test prediction (requires trained model)
curl -X POST \
  "https://<workspace-url>/serving-endpoints/entity-matching-ditto-dev/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "entity1_name": "Apple Inc.",
        "entity1_ticker": "AAPL",
        "entity2_name": "Apple Computer Inc.",
        "entity2_ticker": "AAPL"
      }
    ]
  }'
```

---

## 🐛 Troubleshooting

### Issue: Bundle validation fails

```bash
# Check YAML syntax
databricks bundle validate -t dev

# Common issues:
# - Indentation errors in YAML
# - Missing required fields
# - Invalid resource references
```

**Fix**: Review error message and fix YAML syntax

### Issue: Deploy fails with permission error

```bash
# Error: User does not have permission to create jobs
```

**Fix**: Request Databricks workspace admin permissions or use service principal

### Issue: Job fails with "catalog not found"

```bash
# Error: Catalog <catalog_name> not found
```

**Fix**: Run `setup_unity_catalog` job first:
```bash
databricks bundle run setup_unity_catalog -t dev
```

### Issue: Model serving endpoint not ready

```bash
# Check endpoint status
databricks serving-endpoints get entity-matching-ditto-dev

# Status: NOT_READY or FAILED
```

**Fix**: Ensure Ditto model is registered. Run `train_ditto_model` job:
```bash
databricks bundle run train_ditto_model -t dev
```

### Issue: Job runs but no data

```bash
# Check source table
USE CATALOG <user>_entity_matching_dev;
SELECT COUNT(*) FROM bronze.source_entities;
-- Returns: 0
```

**Fix**: Load source data first. Modify `notebooks/pipeline/01_ingest_source_entities.py` with your data source

---

## 🔄 Update and Redeploy

### Update Notebooks or Code

```bash
# Make changes to notebooks or src/
# Redeploy
databricks bundle deploy -t dev

# Only changed files are re-uploaded
```

### Update Job Configuration

```bash
# Edit resources/jobs.yml
# Redeploy
databricks bundle deploy -t dev

# Jobs are updated in place
```

### Destroy and Recreate

```bash
# Destroy all resources (CAUTION: Deletes everything)
databricks bundle destroy -t dev

# Confirm when prompted

# Redeploy
databricks bundle deploy -t dev
```

---

## 📚 Common Commands

### Deployment

```bash
# Validate
databricks bundle validate -t dev

# Deploy
databricks bundle deploy -t dev

# List resources
databricks bundle resources list -t dev

# Destroy (delete all)
databricks bundle destroy -t dev
```

### Running Jobs

```bash
# Run a specific job
databricks bundle run setup_unity_catalog -t dev

# Run with parameters
databricks bundle run adhoc_entity_matching -t dev \
  --params '{"source_table":"catalog.schema.table"}'

# Check job run status
databricks runs get --run-id <run-id>
```

### Monitoring

```bash
# Watch job run logs
databricks runs get-output --run-id <run-id>

# List recent runs
databricks runs list --job-id <job-id> --limit 10
```

---

## 🎯 Production Deployment Checklist

Before deploying to production:

- [ ] Validate bundle: `databricks bundle validate -t prod`
- [ ] Configure service principal in `databricks.yml`
- [ ] Test in dev environment first
- [ ] Review job schedules
- [ ] Verify model serving endpoint configuration
- [ ] Set up monitoring and alerts
- [ ] Document runbook for operations team
- [ ] Train Ditto model on production data
- [ ] Test with sample data in prod
- [ ] Set up backup and recovery
- [ ] Configure cost alerts

---

## 📖 Additional Resources

- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [Bundle YAML Reference](https://docs.databricks.com/dev-tools/bundles/settings.html)
- [Project README](../README.md)
- [Production Deployment Guide](PRODUCTION_DEPLOYMENT.md)
- [Testing Guide](TESTING_GUIDE.md)

---

## 🆘 Getting Help

### Common Questions

**Q: Can I deploy to multiple workspaces?**
A: Yes, configure different profiles in `~/.databrickscfg` and specify with `--profile` flag

**Q: How do I rollback a deployment?**
A: Bundles don't support rollback. Use git to revert changes and redeploy, or manually fix resources in UI

**Q: Can I use existing resources?**
A: Yes, set `mode: production` to preserve existing resources during deployment

**Q: How do I debug job failures?**
A: Check job run logs in Databricks UI → Workflows → Jobs → Select job → View run logs

---

**Ready to deploy?** → `databricks bundle deploy -t dev`

**Target: Deploy in 5 minutes | Run in 10 minutes | Production-ready**
