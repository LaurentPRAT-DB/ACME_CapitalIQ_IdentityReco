# Test Data Integration Summary

## Overview

The large-scale test data generator has been **fully integrated** into the Databricks Asset Bundle (DAB) Phase 1 configuration. You can now generate and load test data at scale with a single command.

---

## What Was Added

### 1. **DAB Job Definition** (`resources/jobs_phase1_data.yml`)

Added new job: `load_large_test_data`

```yaml
load_large_test_data:
  name: "[${bundle.target}] Entity Matching - Phase 1: Load Large-Scale Test Data"

  parameters:
    - num_reference_entities (default: 1000)
    - num_source_entities (default: 3000)
    - match_ratio (default: 0.7)
    - mode (default: overwrite)

  tasks:
    - load_large_test_dataset (notebook: 03_load_large_test_dataset.py)
```

### 2. **Configuration Variables** (`databricks-phase1.yml`)

Added test data configuration:

```yaml
variables:
  test_data_reference_size: "1000"
  test_data_source_size: "3000"
  test_data_match_ratio: "0.7"
  test_data_write_mode: "overwrite"
```

### 3. **Updated Deployment Script** (`deploy-phase.sh`)

Added reminder to load test data after Phase 1 deployment.

### 4. **Documentation**

- **DAB_TEST_DATA_LOADING_GUIDE.md** - Complete DAB integration guide
- **QUICK_REFERENCE_TEST_DATA.md** - Quick command reference
- **TEST_DATA_INTEGRATION_SUMMARY.md** - This file

---

## How to Use

### Step 1: Deploy Phase 1

```bash
./deploy-phase.sh 1 dev
```

This deploys TWO jobs:
1. `load_reference_data` - Creates tables
2. `load_large_test_data` - Generates test data ✨ **NEW**

### Step 2: Run Test Data Loader

```bash
# Use defaults (1000 ref, 3000 source)
databricks bundle run load_large_test_data -t dev

# Or customize
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000
```

### Step 3: Verify

```sql
SELECT COUNT(*) FROM your_catalog.bronze.spglobal_reference;  -- 1000
SELECT COUNT(*) FROM your_catalog.bronze.source_entities;     -- 3000
```

---

## Benefits

### ✅ **Integrated Deployment**
- Single command to deploy: `./deploy-phase.sh 1 dev`
- Consistent with existing DAB workflow
- No manual notebook execution needed

### ✅ **Parameterized Job**
- Change dataset size without editing code
- Override at runtime: `--param num_reference_entities=5000`
- Configure per environment (dev/staging/prod)

### ✅ **Repeatable & Automated**
- Use in CI/CD pipelines
- Consistent test data generation
- Version controlled configuration

### ✅ **Multiple Deployment Options**
- Databricks CLI (command line)
- Databricks UI (interactive)
- Python API (programmatic)

---

## Configuration Examples

### Example 1: Small Dev Test

**Edit `databricks-phase1.yml`:**
```yaml
targets:
  dev:
    variables:
      test_data_reference_size: "100"
      test_data_source_size: "300"
```

**Run:**
```bash
./deploy-phase.sh 1 dev
databricks bundle run load_large_test_data -t dev
```

### Example 2: Large Staging Test

**Edit `databricks-phase1.yml`:**
```yaml
targets:
  staging:
    variables:
      test_data_reference_size: "5000"
      test_data_source_size: "15000"
      test_data_match_ratio: "0.8"
```

**Run:**
```bash
./deploy-phase.sh 1 staging
databricks bundle run load_large_test_data -t staging
```

### Example 3: Runtime Override

**Don't edit config, just override:**
```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=10000 \
  --param num_source_entities=30000 \
  --param match_ratio=0.9
```

---

## Architecture

### Before Integration

```
Manual Process:
1. Open Databricks notebook
2. Set widget values manually
3. Run notebook interactively
4. Repeat for each environment
```

### After Integration

```
Automated Process:
1. ./deploy-phase.sh 1 dev
2. databricks bundle run load_large_test_data -t dev
3. Done! ✅
```

### Deployment Flow

```
deploy-phase.sh 1 dev
    ↓
databricks-phase1.yml (config)
    ↓
resources/jobs_phase1_data.yml (job definition)
    ↓
notebooks/setup/03_load_large_test_dataset.py (execution)
    ↓
src/data/large_dataset_generator.py (data generation)
    ↓
bronze.spglobal_reference (1000 entities)
bronze.source_entities (3000 entities)
bronze.gold_standard (500 matches)
```

---

## Files Modified

### Configuration Files

1. **`resources/jobs_phase1_data.yml`**
   - Added `load_large_test_data` job definition
   - Configured parameters and notebook path

2. **`databricks-phase1.yml`**
   - Added test data configuration variables
   - Set defaults for reference/source sizes

3. **`deploy-phase.sh`**
   - Added reminder to load test data after Phase 1

### Documentation Files

4. **`DAB_TEST_DATA_LOADING_GUIDE.md`** (NEW)
   - Complete guide to using test data loader in DAB
   - Configuration options
   - Troubleshooting

5. **`QUICK_REFERENCE_TEST_DATA.md`** (NEW)
   - Quick command reference
   - Common scenarios

6. **`TEST_DATA_INTEGRATION_SUMMARY.md`** (NEW)
   - This file - integration summary

---

## Testing the Integration

### Test 1: Default Configuration

```bash
./deploy-phase.sh 1 dev
databricks bundle run load_large_test_data -t dev
```

**Expected:**
- 1000 reference entities loaded
- 3000 source entities loaded
- ~2-3 minutes execution time

### Test 2: Small Dataset

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=50 \
  --param num_source_entities=100
```

**Expected:**
- 50 reference entities loaded
- 100 source entities loaded
- ~30 seconds execution time

### Test 3: Large Dataset

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000
```

**Expected:**
- 5000 reference entities loaded
- 15000 source entities loaded
- ~10-15 minutes execution time

---

## Monitoring

### Via CLI

```bash
# List recent job runs
databricks runs list --limit 5

# Get run status
databricks runs get --run-id <run-id>

# View run output
databricks runs get-output --run-id <run-id>
```

### Via Databricks UI

1. Navigate to **Workflows**
2. Find: **[dev] Entity Matching - Phase 1: Load Large-Scale Test Data**
3. View run history and logs

---

## Next Steps

### For First-Time Users

1. **Deploy Phase 1:**
   ```bash
   ./deploy-phase.sh 1 dev
   ```

2. **Load test data:**
   ```bash
   databricks bundle run load_large_test_data -t dev
   ```

3. **Verify:**
   ```sql
   SELECT COUNT(*) FROM catalog.bronze.spglobal_reference;
   SELECT COUNT(*) FROM catalog.bronze.source_entities;
   ```

4. **Run pipeline:**
   ```bash
   ./deploy-phase.sh 4 dev
   databricks bundle run entity_matching_pipeline -t dev
   ```

### For Advanced Users

1. **Configure per environment:**
   - Edit `databricks-phase1.yml`
   - Set different sizes for dev/staging/prod

2. **Integrate with CI/CD:**
   - Add to GitHub Actions
   - Automate data refresh

3. **Create custom test scenarios:**
   - Different match ratios
   - Different data distributions
   - Edge case testing

---

## Troubleshooting

### Issue: Job not found

**Solution:**
```bash
./deploy-phase.sh 1 dev  # Redeploy Phase 1
```

### Issue: Parameters not working

**Solution:**
```bash
# Check current configuration
databricks bundle validate -t dev

# List jobs to verify deployment
databricks jobs list | grep "Load Large-Scale"
```

### Issue: Slow execution

**Solution:**
```bash
# Reduce dataset size
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=500 \
  --param num_source_entities=1500
```

---

## Summary

✅ **Fully Integrated** - Test data loader is now part of DAB Phase 1
✅ **Parameterized** - Configure dataset size via parameters or config
✅ **Automated** - Single command to generate and load data
✅ **Documented** - Comprehensive guides and references
✅ **Production Ready** - Error handling, logging, data quality checks

**Status:** Ready for use! Deploy Phase 1 and start testing at scale.

---

## Quick Reference

```bash
# Deploy Phase 1
./deploy-phase.sh 1 dev

# Load test data (defaults: 1000 ref, 3000 source)
databricks bundle run load_large_test_data -t dev

# Custom size
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000

# Verify
# SQL: SELECT COUNT(*) FROM catalog.bronze.spglobal_reference;
```

---

**Date:** 2026-01-27
**Phase:** 1 (Data Load)
**Status:** Complete
**Integration:** Databricks Asset Bundle
