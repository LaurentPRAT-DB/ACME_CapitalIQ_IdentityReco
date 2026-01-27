# DAB Test Data Loading Guide

## Overview

The large-scale test data loader is now integrated into the **Databricks Asset Bundle (DAB) Phase 1** configuration, making it easy to deploy and run as part of your data setup.

---

## Quick Start

### Deploy Phase 1 with Test Data Loader

```bash
# Deploy Phase 1 (includes both reference tables and test data loader jobs)
./deploy-phase.sh 1 dev
```

This deploys **two jobs**:
1. **load_reference_data** - Creates empty reference tables
2. **load_large_test_data** - Generates and loads test data (1000 ref, 3000 source)

---

## Running the Test Data Loader

### Method 1: Via Databricks CLI (Recommended)

```bash
# Deploy Phase 1 first
./deploy-phase.sh 1 dev

# Run the test data loader job with defaults (1000 ref, 3000 source)
databricks bundle run load_large_test_data -t dev

# Or with custom parameters
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000 \
  --param match_ratio=0.8 \
  --param mode=overwrite
```

### Method 2: Via Databricks Workspace UI

1. Navigate to **Workflows** in Databricks UI
2. Find job: **[dev] Entity Matching - Phase 1: Load Large-Scale Test Data**
3. Click **Run now**
4. Optionally override parameters:
   - `num_reference_entities`: 1000 (default)
   - `num_source_entities`: 3000 (default)
   - `match_ratio`: 0.7 (default)
   - `mode`: overwrite (default)

---

## Configuration

### Default Values (Phase 1)

These defaults are configured in `databricks-phase1.yml`:

```yaml
variables:
  test_data_reference_size: "1000"
  test_data_source_size: "3000"
  test_data_match_ratio: "0.7"
  test_data_write_mode: "overwrite"
```

### Override Defaults

**Option 1: Edit `databricks-phase1.yml`**

```yaml
targets:
  dev:
    variables:
      test_data_reference_size: "5000"
      test_data_source_size: "15000"
      test_data_match_ratio: "0.8"
```

**Option 2: Override at Runtime**

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000
```

**Option 3: Use Different Targets**

```yaml
# databricks-phase1.yml
targets:
  dev:
    variables:
      test_data_reference_size: "100"   # Small test
      test_data_source_size: "300"

  staging:
    variables:
      test_data_reference_size: "1000"  # Medium test
      test_data_source_size: "3000"

  prod:
    variables:
      test_data_reference_size: "10000" # Large test
      test_data_source_size: "30000"
```

---

## Complete Workflow

### Step 1: Deploy Phase 1

```bash
./deploy-phase.sh 1 dev
```

**What this does:**
- Deploys `load_reference_data` job
- Deploys `load_large_test_data` job
- Syncs notebooks to workspace
- Configures job parameters

### Step 2: Create Tables (First Time Only)

```bash
databricks bundle run load_reference_data -t dev
```

**What this does:**
- Creates `bronze.spglobal_reference` table
- Creates `bronze.source_entities` table
- Creates `gold.matched_entities` table
- Creates views (`review_queue`, `daily_stats`)

### Step 3: Load Test Data

```bash
# Small test (fast)
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=100 \
  --param num_source_entities=300

# Medium test (default)
databricks bundle run load_large_test_data -t dev

# Large test (stress)
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000
```

### Step 4: Verify Data

```sql
-- Check reference data
SELECT COUNT(*) FROM your_catalog.bronze.spglobal_reference;

-- Check source data
SELECT COUNT(*) FROM your_catalog.bronze.source_entities;

-- Sample data
SELECT * FROM your_catalog.bronze.spglobal_reference LIMIT 10;
SELECT * FROM your_catalog.bronze.source_entities LIMIT 10;
```

### Step 5: Run Pipeline

```bash
./deploy-phase.sh 4 dev
databricks bundle run entity_matching_pipeline -t dev
```

---

## Job Configuration Details

### Job Definition

**File:** `resources/jobs_phase1_data.yml`

```yaml
load_large_test_data:
  name: "[${bundle.target}] Entity Matching - Phase 1: Load Large-Scale Test Data"

  parameters:
    - name: num_reference_entities
      default: ${var.test_data_reference_size}
    - name: num_source_entities
      default: ${var.test_data_source_size}
    - name: match_ratio
      default: ${var.test_data_match_ratio}
    - name: mode
      default: ${var.test_data_write_mode}

  tasks:
    - task_key: load_large_test_dataset
      notebook_task:
        notebook_path: ../notebooks/setup/03_load_large_test_dataset.py
        base_parameters:
          catalog_name: ${var.catalog_name}
          num_reference_entities: "{{job.parameters.num_reference_entities}}"
          num_source_entities: "{{job.parameters.num_source_entities}}"
          match_ratio: "{{job.parameters.match_ratio}}"
          mode: "{{job.parameters.mode}}"
```

### Parameters Explained

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `num_reference_entities` | Number of reference entities to generate | 1000 | 100, 1000, 5000 |
| `num_source_entities` | Number of source entities to generate | 3000 | 300, 3000, 15000 |
| `match_ratio` | Ratio of entities that should match (0-1) | 0.7 | 0.5, 0.7, 0.9 |
| `mode` | Write mode for Delta tables | overwrite | append, overwrite |

---

## Common Scenarios

### Scenario 1: Quick Test (1 minute)

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=50 \
  --param num_source_entities=100
```

**Use case:** Smoke test, verify pipeline works

### Scenario 2: Standard Test (3 minutes)

```bash
databricks bundle run load_large_test_data -t dev
# Uses defaults: 1000 ref, 3000 source
```

**Use case:** Regular testing, demo purposes

### Scenario 3: Stress Test (15 minutes)

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=10000 \
  --param num_source_entities=30000
```

**Use case:** Performance testing, scalability validation

### Scenario 4: Precision Test (no matches)

```bash
databricks bundle run load_large_test_data -t dev \
  --param match_ratio=0.0
```

**Use case:** Test false positive rate

### Scenario 5: Recall Test (all match)

```bash
databricks bundle run load_large_test_data -t dev \
  --param match_ratio=1.0
```

**Use case:** Test recall/sensitivity

---

## Monitoring Job Execution

### Via CLI

```bash
# Get job run status
databricks runs list --limit 5

# Get specific job run details
databricks runs get --run-id <run-id>

# View run output
databricks runs get-output --run-id <run-id>
```

### Via Workspace UI

1. Navigate to **Workflows** → **Job runs**
2. Find: **[dev] Entity Matching - Phase 1: Load Large-Scale Test Data**
3. Click on latest run
4. View notebook output and logs

### Expected Output

```
Configuration:
  Catalog: laurent_prat_entity_matching_dev
  Reference entities: 1000
  Source entities: 3000
  Match ratio: 70%
  Write mode: overwrite

✅ Loaded 1000 reference entities
✅ Loaded 3000 source entities
✅ Created ground truth with 500 known matches

Reference Entity Statistics:
  Total entities: 1000
  Countries: 15
  Industries: 30
  With LEI: 1000 (100.0%)
  With CUSIP: 153 (15.3%)

Source Entity Statistics:
  Total entities: 3000
  Source systems: 5
  With ticker: 2340 (78.0%)
```

---

## Troubleshooting

### Issue: Job not found

**Error:** `Job 'load_large_test_data' not found`

**Solution:**
```bash
# Redeploy Phase 1
./deploy-phase.sh 1 dev
```

### Issue: Module not found

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Ensure DAB has synced the `src/` directory:
```bash
# Check sync configuration in databricks-phase1.yml
sync:
  include:
    - src/**/*.py
    - notebooks/**/*.py
```

### Issue: Table already exists

**Error:** `Table already exists`

**Solution:** Use `mode=overwrite`:
```bash
databricks bundle run load_large_test_data -t dev \
  --param mode=overwrite
```

### Issue: Out of memory

**Error:** Job fails with OOM

**Solution:** Reduce dataset size:
```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=500 \
  --param num_source_entities=1500
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy and Load Test Data

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy Phase 1
        run: ./deploy-phase.sh 1 dev

      - name: Load Test Data
        run: |
          databricks bundle run load_large_test_data -t dev \
            --param num_reference_entities=1000 \
            --param num_source_entities=3000
```

---

## Best Practices

### 1. Use Appropriate Dataset Sizes

- **Development:** 100 ref, 300 source (fast iteration)
- **Testing:** 1000 ref, 3000 source (realistic test)
- **Staging:** 5000 ref, 15000 source (stress test)
- **Production:** Use real data, not generated

### 2. Match Ratio Selection

- **Development:** 0.9 (easy matches, verify pipeline works)
- **Testing:** 0.7 (realistic, tests all stages)
- **Evaluation:** 0.5 (harder, tests edge cases)

### 3. Write Mode

- **Development:** `overwrite` (clean slate each run)
- **Testing:** `append` (accumulate data for testing)
- **CI/CD:** `overwrite` (reproducible)

### 4. Version Control

Track test data configurations:
```yaml
# config/test-scenarios.yml
small_test:
  reference: 100
  source: 300
  ratio: 0.7

standard_test:
  reference: 1000
  source: 3000
  ratio: 0.7

stress_test:
  reference: 10000
  source: 30000
  ratio: 0.7
```

---

## Summary

✅ **Integrated into DAB Phase 1**
- Deploy with `./deploy-phase.sh 1 dev`
- Run with `databricks bundle run load_large_test_data -t dev`

✅ **Configurable Parameters**
- Reference size: 100-10,000+
- Source size: 300-30,000+
- Match ratio: 0.0-1.0
- Write mode: append/overwrite

✅ **Multiple Deployment Methods**
- Databricks CLI (command line)
- Databricks UI (interactive)
- CI/CD pipelines (automated)

✅ **Production Ready**
- Error handling
- Progress logging
- Data quality checks
- Ground truth generation

**Ready to use:** Deploy Phase 1 and start testing at scale! 🚀

---

**Last Updated:** 2026-01-27
**Phase:** 1 (Data Load)
**Job Name:** `load_large_test_data`
**Status:** Production Ready
