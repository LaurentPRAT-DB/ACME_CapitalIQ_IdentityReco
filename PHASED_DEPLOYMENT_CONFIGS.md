# Phased Deployment Configuration Files

This project uses separate Databricks Asset Bundle configuration files for each deployment phase.

## Available Configuration Files

### Phase 1: Setup & Training (`databricks-phase1.yml`)
**Resources Deployed:**
- Unity Catalog setup job
- Ditto model training job

**Deploy Command:**
```bash
./deploy-phase.sh 1 dev
```

**When to Use:**
- Initial setup
- First-time deployment
- When you only need catalog and model training infrastructure

---

### Phase 2: Model Serving (`databricks-phase2.yml`)
**Resources Deployed:**
- All Phase 1 resources
- Model serving endpoint for Ditto matcher
- Model serving endpoint permissions

**Deploy Command:**
```bash
./deploy-phase.sh 2 dev
```

**When to Use:**
- After Phase 1 is complete and model is trained
- When you need to deploy or update the model serving endpoint
- Testing model serving before enabling production pipelines

**Prerequisites:**
- Phase 1 must be deployed
- Ditto model must be trained and registered in Unity Catalog

---

### Phase 3: Production Pipelines (`databricks-phase3.yml`)
**Resources Deployed:**
- All Phase 1 and Phase 2 resources
- Production matching pipeline (scheduled)
- Ad-hoc matching job

**Deploy Command:**
```bash
./deploy-phase.sh 3 dev
```

**When to Use:**
- Complete production deployment
- When model serving endpoint is ready and tested
- Enabling scheduled production workloads

**Prerequisites:**
- Phase 1 deployed (catalog and training)
- Phase 2 deployed (model serving endpoint ready)

---

### Deployment Script (`deploy-phase.sh`)

The `deploy-phase.sh` helper script automates phased deployment:
- Copies the appropriate phase config to `databricks.yml`
- Validates the bundle configuration
- Prompts for confirmation before deploying
- Provides next steps after successful deployment

**Usage:**
```bash
./deploy-phase.sh <phase> [target]

# Examples
./deploy-phase.sh 1        # Deploy Phase 1 to dev
./deploy-phase.sh 2 dev    # Deploy Phase 2 to dev
./deploy-phase.sh 3 prod   # Deploy Phase 3 to prod
```

---

## Deployment Workflow

### Recommended Phased Deployment

```bash
# Step 1: Deploy infrastructure and training
./deploy-phase.sh 1 dev

# Run setup jobs
databricks bundle run setup_unity_catalog -t dev
databricks bundle run train_ditto_model -t dev

# Step 2: Deploy model serving (after model training completes)
./deploy-phase.sh 2 dev
# Wait for endpoint to be ready (~5 minutes)

# Step 3: Deploy production pipelines
./deploy-phase.sh 3 dev
```

### Manual Deployment (Without Script)

If you prefer to deploy manually:

```bash
# Copy phase config to databricks.yml
cp databricks-phase1.yml databricks.yml

# Validate and deploy
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

---

## Validation

The deployment script automatically validates configurations. To validate manually:

```bash
# Copy phase config to databricks.yml
cp databricks-phase1.yml databricks.yml

# Validate
databricks bundle validate -t dev
```

---

## Rolling Back

To roll back to a previous phase:

```bash
# From Phase 3 back to Phase 2
./deploy-phase.sh 2 dev

# From Phase 2 back to Phase 1
./deploy-phase.sh 1 dev
```

**Note:** Resources are not automatically deleted when rolling back. Use `databricks bundle destroy -t dev` to clean up completely.

---

## Target Environments

All phase configuration files support these targets:

- `dev` - Development environment (runs as your user)
- `staging` - Staging environment (requires service principal)
- `prod` - Production environment (requires service principal)

**Example for staging:**
```bash
./deploy-phase.sh 3 staging
```

**Example for production:**
```bash
# Ensure service principal is configured
export TF_VAR_prod_service_principal="<service-principal-app-id>"

# Deploy to production
./deploy-phase.sh 3 prod
```

---

## Troubleshooting

### Issue: Model serving endpoint fails in Phase 2
**Cause:** Ditto model not yet registered
**Solution:** Ensure Phase 1 training job completed successfully before deploying Phase 2

### Issue: Production pipeline fails in Phase 3
**Cause:** Model serving endpoint not ready
**Solution:** Verify endpoint status: `databricks serving-endpoints get ditto-em-dev`

### Issue: Validation fails
**Cause:** YAML syntax error or missing resources
**Solution:** Check error message and fix YAML indentation/syntax

---

## See Also

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed phased deployment guide
- [README.md](README.md) - Project overview and quick start
- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
