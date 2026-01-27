# Custom Endpoint Name Configuration Guide

## Overview

This guide explains how to use custom serving endpoint names when you've created endpoints manually or want to override the default naming convention.

## Default Naming Convention

By default, the project uses these endpoint names:
- **Ditto Model Endpoint:** `ditto-em-{target}` (e.g., `ditto-em-dev`)
- **Vector Search Endpoint:** `entity-matching-vs-{target}` (e.g., `entity-matching-vs-dev`)

## Your Current Setup

You have created a custom endpoint:
- **Endpoint Name:** `ditto-em-dev-2`
- **Created:** Via Databricks Workspace UI
- **Configured for:** Phase 4 (Production Pipeline)

## How to Override Endpoint Names

### Option 1: Edit databricks-phase4.yml (Recommended)

**File:** `databricks-phase4.yml`

```yaml
targets:
  dev:
    variables:
      catalog_name: laurent_prat_entity_matching_dev
      ditto_endpoint_name: ditto-em-dev-2  # Your custom endpoint
      vector_search_endpoint_name: entity-matching-vs-dev  # Optional override
```

**Steps:**
1. Open `databricks-phase4.yml`
2. Find the `dev` target section
3. Add or update the `ditto_endpoint_name` variable
4. Save and redeploy Phase 4

### Option 2: Command Line Override

You can override variables at deployment time:

```bash
databricks bundle deploy -t dev \
  --var="ditto_endpoint_name=ditto-em-dev-2" \
  -c databricks-phase4.yml
```

### Option 3: Environment-Specific Override

Create a separate configuration file for custom endpoints:

**File:** `databricks-phase4-custom.yml`

```yaml
# Include base configuration
include:
  - databricks-phase4.yml

targets:
  dev:
    variables:
      ditto_endpoint_name: ditto-em-dev-2
```

Then deploy with:
```bash
databricks bundle deploy -t dev -c databricks-phase4-custom.yml
```

## Verification

After setting the custom endpoint name, verify it's configured correctly:

### 1. Check Bundle Configuration

```bash
databricks bundle validate -t dev -c databricks-phase4.yml
```

### 2. Check Deployed Job Parameters

```bash
databricks jobs list --profile LPT_FREE_EDITION | grep "Phase 4"
```

### 3. Test Endpoint Connection

Run the Phase 4 pipeline and check logs for:
```
✓ Connected to Ditto endpoint: ditto-em-dev-2
  Endpoint state: ready
```

## Phase 4 Pipeline Configuration

The Phase 4 pipeline automatically uses the configured endpoint name:

**File:** `resources/jobs_phase4_pipeline.yml`

```yaml
tasks:
  - task_key: vector_search_and_ditto
    notebook_task:
      notebook_path: ../notebooks/pipeline/03_vector_search_ditto.py
      base_parameters:
        ditto_endpoint: ${var.ditto_endpoint_name}  # Uses variable
```

**Notebook:** `notebooks/pipeline/03_vector_search_ditto.py`

```python
# Widget reads the parameter passed from job
dbutils.widgets.text("ditto_endpoint", "ditto-em-dev", "Ditto Endpoint")
ditto_endpoint = dbutils.widgets.get("ditto_endpoint")

# Uses the endpoint name throughout
print(f"Connecting to Ditto endpoint: {ditto_endpoint}")
```

## Common Scenarios

### Scenario 1: Manual Endpoint Creation

**Problem:** Created endpoint via UI with custom name `ditto-em-dev-2`

**Solution:** Update `databricks-phase4.yml`:
```yaml
dev:
  variables:
    ditto_endpoint_name: ditto-em-dev-2
```

### Scenario 2: Multiple Endpoints for Testing

**Problem:** Testing different model versions with different endpoints

**Solution:** Create target-specific overrides:
```yaml
dev:
  variables:
    ditto_endpoint_name: ditto-em-dev-v1

dev-test:
  variables:
    ditto_endpoint_name: ditto-em-dev-v2-test
```

### Scenario 3: Shared Endpoints Across Environments

**Problem:** Using same endpoint for dev and staging

**Solution:** Set the same endpoint name in both targets:
```yaml
dev:
  variables:
    ditto_endpoint_name: ditto-em-shared

staging:
  variables:
    ditto_endpoint_name: ditto-em-shared
```

## Deployment Workflow

### With Custom Endpoint Name

```bash
# 1. Update configuration
vim databricks-phase4.yml  # Add ditto_endpoint_name override

# 2. Validate configuration
databricks bundle validate -t dev -c databricks-phase4.yml

# 3. Deploy Phase 4
./deploy-phase.sh 4 dev

# 4. Verify endpoint in job logs
# Look for: "Connecting to Ditto endpoint: ditto-em-dev-2"
```

## Troubleshooting

### Issue: Pipeline Can't Find Endpoint

**Error:**
```
⚠ Could not connect to Ditto endpoint: ditto-em-dev
```

**Causes:**
1. Endpoint name mismatch
2. Endpoint not in "ready" state
3. Permissions issue

**Solutions:**

1. **Check Endpoint Name:**
   ```bash
   databricks serving-endpoints list --profile LPT_FREE_EDITION | grep ditto
   ```

2. **Verify Endpoint Status:**
   ```bash
   databricks serving-endpoints get ditto-em-dev-2 --profile LPT_FREE_EDITION
   ```

3. **Update Configuration:**
   - Ensure `ditto_endpoint_name` matches actual endpoint name
   - Redeploy Phase 4

### Issue: Endpoint Name Not Being Used

**Symptom:** Pipeline still tries to use `ditto-em-dev` instead of custom name

**Solution:**
1. Check that variable is set in target section (not global)
2. Ensure you're deploying with correct target: `-t dev`
3. Clear cached bundle state:
   ```bash
   rm -rf .databricks/bundle
   databricks bundle deploy -t dev -c databricks-phase4.yml
   ```

## Best Practices

### 1. Document Custom Endpoints

Keep a list of custom endpoints:
```yaml
# Custom Endpoints Log
# ditto-em-dev-2: Created 2026-01-26 for testing Phase 2b fixes
# ditto-em-dev-3: Reserved for future A/B testing
```

### 2. Use Descriptive Names

Good endpoint names:
- `ditto-em-dev-2` ✅ (version indicator)
- `ditto-em-dev-testing` ✅ (purpose indicator)
- `ditto-em-dev-champion` ✅ (model variant)

Avoid:
- `ditto-1` ❌ (not descriptive)
- `my-endpoint` ❌ (doesn't indicate purpose)

### 3. Keep Configuration in Version Control

Always commit endpoint name changes:
```bash
git add databricks-phase4.yml
git commit -m "Update dev endpoint to ditto-em-dev-2"
```

## Reference

### All Configurable Endpoint Variables

```yaml
variables:
  # Ditto model serving endpoint
  ditto_endpoint_name:
    description: Ditto model serving endpoint name
    default: ditto-em-${bundle.target}

  # Vector search endpoint
  vector_search_endpoint_name:
    description: Vector search endpoint name
    default: entity-matching-vs-${bundle.target}
```

### Where Endpoints Are Used

1. **Phase 3:** Creates the serving endpoint
   - File: `databricks-phase3.yml`
   - Resource: `serving_endpoints`

2. **Phase 4:** Queries the serving endpoint
   - File: `notebooks/pipeline/03_vector_search_ditto.py`
   - Task: `vector_search_and_ditto`

3. **Ad-hoc Jobs:** Uses endpoint for on-demand matching
   - File: `notebooks/03_full_pipeline_example.py`

## Summary

✅ **Your Configuration:**
- Endpoint Name: `ditto-em-dev-2`
- Configuration File: `databricks-phase4.yml`
- Override Location: `targets.dev.variables.ditto_endpoint_name`

✅ **To Deploy:**
```bash
./deploy-phase.sh 4 dev
```

✅ **To Verify:**
```bash
# Check endpoint exists
databricks serving-endpoints get ditto-em-dev-2 --profile LPT_FREE_EDITION

# Check job uses correct endpoint
databricks jobs list --profile LPT_FREE_EDITION | grep "Phase 4"
```

---

**Last Updated:** 2026-01-26
**Endpoint Configured:** ditto-em-dev-2
**Status:** Ready for Phase 4 Deployment
