# Phase 3 & 4 DABs Migration Analysis
**Date:** 2026-01-25
**Analyzed By:** Claude Code - databricks-dabs-migration skill
**Scope:** Phase 3 (Model Serving) & Phase 4 (Production Pipeline)
**Status:** 🟡 Issues Found - Requires Fixes

---

## Executive Summary

Analyzed Phase 3 (Model Serving) and Phase 4 (Production Pipeline) configurations and notebooks against DABs best practices. Found **12 issues** requiring attention before deployment:

- **Critical (3):** Hardcoded paths, missing dependencies, placeholder implementation
- **High (5):** Library version incompatibility, missing parameters, schema mismatches
- **Medium (4):** Type annotations, email format, relative paths

---

## Phase 3: Model Serving Analysis

### ✅ What's Working Well

1. **Proper DABs Variable Usage**
   - Uses `${var.catalog_name}` correctly
   - Uses `${bundle.target}` for environment naming
   - Proper tag structure

2. **Model Endpoint Configuration**
   - Scale-to-zero enabled (cost optimization)
   - MLflow tracing enabled
   - Traffic routing configured

### ⚠️ Issues Found

#### Issue 3.1: Model Registry Dependency (MEDIUM)
**File:** `resources/jobs_phase3_serving.yml:14`
**Issue:** References model `${var.catalog_name}.models.entity_matching_ditto`

**Analysis:**
- Model must exist in Unity Catalog before Phase 3 can deploy
- Notebook `02_train_ditto_model.py` saves model to MLflow but doesn't explicitly register to UC
- Need to verify model registration in Phase 2

**Recommendation:**
```yaml
# Option 1: Add explicit model version variable
entity_version: ${var.ditto_model_version}  # Already exists

# Option 2: Add model registration check in Phase 2
# Ensure 02_train_ditto_model.py registers model to UC catalog
```

**Priority:** MEDIUM
**Impact:** Phase 3 deployment will fail if model not registered

---

## Phase 4: Production Pipeline Analysis

### ✅ What's Working Well

1. **Job Structure**
   - Proper task dependencies
   - Shared cluster configuration
   - Timeout settings configured
   - Email notifications enabled

2. **Parameter Flow**
   - Uses `${var.catalog_name}` throughout
   - Proper notebook parameter passing
   - Job parameters for ad-hoc runs

### 🔴 Critical Issues

#### Issue 4.1: Hardcoded Path in Full Pipeline Example (CRITICAL)
**File:** `notebooks/03_full_pipeline_example.py:157`
**Current Code:**
```python
ground_truth = validator.load_gold_standard("/dbfs/entity_matching/gold_standard.csv")
```

**Problem:**
- Hardcoded path breaks DABs portability
- Won't work across dev/staging/prod environments
- Violates parameter-driven approach

**Fix Required:**
```python
# Add widget parameter
dbutils.widgets.text("gold_standard_path", "")
gold_standard_path = dbutils.widgets.get("gold_standard_path")

# Use parameter with fallback
if gold_standard_path:
    ground_truth = validator.load_gold_standard(gold_standard_path)
else:
    # Or use catalog-based path
    ground_truth_table = f"{catalog_name}.bronze.gold_standard"
    ground_truth = spark.table(ground_truth_table).toPandas()
```

**Priority:** CRITICAL
**Impact:** Code not portable, evaluation will fail in production

---

#### Issue 4.2: Placeholder Implementation (CRITICAL)
**File:** `notebooks/pipeline/03_vector_search_ditto.py:48-63`
**Current Code:**
```python
# Placeholder: In production, this would use vector search and Ditto endpoint
# For now, create empty results
matched_df = spark.createDataFrame([], schema="""...""")
```

**Problem:**
- Core matching logic not implemented
- Pipeline will produce zero results
- Cannot deploy to production

**Fix Required:**
Implement actual vector search + Ditto inference:
```python
# 1. Load BGE embeddings model
# 2. Generate embeddings for unmatched entities
# 3. Query vector search endpoint
# 4. Call Ditto serving endpoint for top candidates
# 5. Return matched results with confidence scores
```

**Priority:** CRITICAL
**Impact:** Pipeline produces no matches, core functionality missing

---

#### Issue 4.3: Old Library Version (CRITICAL)
**File:** `resources/jobs_phase4_pipeline.yml:125`
**Current Code:**
```yaml
libraries:
  - pypi:
      package: sentence-transformers==2.2.2
```

**Problem:**
- Version 2.2.2 is too old
- Incompatible with transformers 4.40.0 (required for Python 3.10)
- Will cause import errors

**Fix Required:**
```yaml
libraries:
  - pypi:
      package: sentence-transformers>=2.3.0
  - pypi:
      package: transformers>=4.40.0
```

**Priority:** CRITICAL
**Impact:** Job will fail with library compatibility errors

---

### 🟡 High Priority Issues

#### Issue 4.4: Missing Dependencies in Pipeline Notebooks (HIGH)
**Files:** All notebooks in `notebooks/pipeline/`
**Problem:**
- Pipeline notebooks don't install required libraries
- Rely on cluster libraries being pre-installed
- No `%pip install` commands

**Current State:**
```python
# notebooks/pipeline/03_vector_search_ditto.py
# No pip installs - assumes libraries available
```

**Fix Required:**
Add to each notebook that uses ML libraries:
```python
# COMMAND ----------
# MAGIC %pip install --upgrade transformers>=4.40.0 sentence-transformers>=2.3.0 torch>=2.1.0 mlflow

# COMMAND ----------
dbutils.library.restartPython()
```

**Affected Notebooks:**
- `03_vector_search_ditto.py` - needs transformers, sentence-transformers
- `05_generate_metrics.py` - might need pandas if not using Spark only

**Priority:** HIGH
**Impact:** Jobs may fail if cluster doesn't have required libraries

---

#### Issue 4.5: Missing workspace_path Parameter (HIGH)
**File:** `resources/jobs_phase4_pipeline.yml:31-43`
**Problem:**
- Pipeline tasks don't pass `workspace_path` parameter
- Ad-hoc job passes it (line 111) but scheduled pipeline doesn't
- Notebooks might fail to import `src` modules

**Current Code:**
```yaml
# Production pipeline tasks - MISSING workspace_path
notebook_task:
  notebook_path: ../notebooks/pipeline/01_ingest_source_entities.py
  base_parameters:
    catalog_name: ${var.catalog_name}
    # workspace_path: MISSING!

# Ad-hoc job - HAS workspace_path
base_parameters:
  workspace_path: ${workspace.root_path}/files  # ✓ Present
```

**Fix Required:**
Add to all pipeline tasks that import `src` modules:
```yaml
base_parameters:
  catalog_name: ${var.catalog_name}
  workspace_path: ${workspace.root_path}/files  # ADD THIS
```

**Priority:** HIGH
**Impact:** Import errors if notebooks use `src` modules

---

#### Issue 4.6: Schema Mismatch in Metrics Notebook (HIGH)
**File:** `notebooks/pipeline/05_generate_metrics.py:45-46`
**Problem:**
- References columns `auto_matched` and `needs_review`
- These columns might not exist in results from earlier pipeline stages

**Current Code:**
```python
sum(when(col("auto_matched"), 1).otherwise(0)).alias("auto_matched_count"),
sum(when(col("needs_review"), 1).otherwise(0)).alias("review_count"),
```

**Fix Required:**
Add null-safe column checks:
```python
# Option 1: Safe column reference
sum(when(col("auto_matched").isNotNull() & col("auto_matched"), 1).otherwise(0))

# Option 2: Add columns in write_results.py if missing
from pyspark.sql.functions import when, lit
all_matches = all_matches.withColumn(
    "auto_matched",
    when(col("match_confidence") >= 0.90, True).otherwise(False)
).withColumn(
    "needs_review",
    when(col("match_confidence") < 0.70, True).otherwise(False)
)
```

**Priority:** HIGH
**Impact:** Metrics job will fail with column not found error

---

#### Issue 4.7: Email Notification Format (HIGH)
**File:** `resources/jobs_phase4_pipeline.yml:82`
**Current Code:**
```yaml
email_notifications:
  on_success:
    - ${workspace.current_user.userName}@databricks.com
```

**Problem:**
- Hardcodes `@databricks.com` domain
- Won't work for non-Databricks email addresses
- User might have different email domain

**Fix Required:**
```yaml
# Option 1: Use user email directly (if available)
on_success:
  - ${workspace.current_user.userName}  # Use full email if available

# Option 2: Add email as variable
variables:
  notification_email:
    description: Email for job notifications
    default: ${workspace.current_user.userName}
```

**Priority:** HIGH
**Impact:** Email notifications won't reach correct address

---

#### Issue 4.8: Missing Type Annotations Import (HIGH)
**Files:** All pipeline notebooks
**Problem:**
- None of the pipeline notebooks have `from __future__ import annotations`
- Will fail if Python files use lowercase type hints
- Inconsistent with Phase 2 migration

**Current State:**
```python
# notebooks/pipeline/01_ingest_source_entities.py
# Databricks notebook source
# MAGIC %md
# No future annotations import
```

**Fix Required:**
Add to top of each notebook (after magic commands, before imports):
```python
# COMMAND ----------
from __future__ import annotations

# COMMAND ----------
```

**Affected Files:**
- `notebooks/pipeline/01_ingest_source_entities.py`
- `notebooks/pipeline/02_exact_match.py`
- `notebooks/pipeline/03_vector_search_ditto.py`
- `notebooks/pipeline/04_write_results.py`
- `notebooks/pipeline/05_generate_metrics.py`

**Priority:** HIGH
**Impact:** Potential type annotation errors if importing modules with modern type hints

---

### 📋 Medium Priority Issues

#### Issue 4.9: Relative Notebook Paths (MEDIUM)
**File:** `resources/jobs_phase4_pipeline.yml:30`
**Current Code:**
```yaml
notebook_path: ../notebooks/pipeline/01_ingest_source_entities.py
```

**Analysis:**
- Uses relative paths `../notebooks/pipeline/`
- DABs might handle this correctly, but absolute paths are clearer
- Could cause issues depending on bundle sync structure

**Recommendation:**
```yaml
# More explicit approach
notebook_path: ${workspace.file_path}/notebooks/pipeline/01_ingest_source_entities.py

# Or if notebooks are synced to workspace
notebook_path: /Workspace/${workspace.file_path}/notebooks/pipeline/01_ingest_source_entities.py
```

**Priority:** MEDIUM
**Impact:** May work as-is, but less portable

---

#### Issue 4.10: Missing databricks.yml Include (MEDIUM)
**File:** `databricks.yml:13-14`
**Current State:**
```yaml
include:
  - resources/jobs_phase2_training.yml
  # Phase 3 and 4 NOT included!
```

**Problem:**
- Phase 3 and 4 configurations exist but aren't included in main bundle
- Won't be deployed when running `databricks bundle deploy`

**Fix Required:**
```yaml
include:
  - resources/jobs_phase2_training.yml
  - resources/jobs_phase3_serving.yml      # ADD
  - resources/jobs_phase4_pipeline.yml     # ADD
```

**Priority:** MEDIUM
**Impact:** Phase 3 & 4 won't deploy until explicitly included

---

#### Issue 4.11: Ad-hoc Job Parameter Mismatch (MEDIUM)
**File:** `resources/jobs_phase4_pipeline.yml:111-115`
**Current Code:**
```yaml
base_parameters:
  workspace_path: ${workspace.root_path}/files
  catalog_name: ${var.catalog_name}
  source_table: "{{job.parameters.source_table}}"
  output_table: "{{job.parameters.output_table}}"
  date_filter: "{{job.parameters.date_filter}}"
```

**Problem:**
- Notebook `03_full_pipeline_example.py` doesn't define widgets for:
  - `source_table`
  - `output_table`
  - `date_filter`
- Will cause widget not found errors

**Fix Required:**
Add to `03_full_pipeline_example.py` after existing widgets:
```python
dbutils.widgets.text("source_table", f"{catalog_name}.bronze.source_entities")
dbutils.widgets.text("output_table", f"{catalog_name}.gold.matched_entities_adhoc")
dbutils.widgets.text("date_filter", "")

source_table = dbutils.widgets.get("source_table")
output_table = dbutils.widgets.get("output_table")
date_filter = dbutils.widgets.get("date_filter")
```

**Priority:** MEDIUM
**Impact:** Ad-hoc job will fail with widget errors

---

## Issue Summary

### By Priority

| Priority | Count | Issues |
|----------|-------|--------|
| 🔴 CRITICAL | 3 | 4.1, 4.2, 4.3 |
| 🟡 HIGH | 6 | 3.1, 4.4, 4.5, 4.6, 4.7, 4.8 |
| 🟢 MEDIUM | 4 | 4.9, 4.10, 4.11 |
| **TOTAL** | **13** | |

### By Category

| Category | Count | Issues |
|----------|-------|--------|
| Hardcoded Paths | 1 | 4.1 |
| Library Versions | 2 | 4.3, 4.4 |
| Missing Parameters | 3 | 4.5, 4.8, 4.11 |
| Configuration | 3 | 3.1, 4.7, 4.10 |
| Schema/Data | 1 | 4.6 |
| Implementation | 1 | 4.2 |
| Path References | 1 | 4.9 |

### By File

| File | Issue Count | Issue IDs |
|------|-------------|-----------|
| `notebooks/03_full_pipeline_example.py` | 2 | 4.1, 4.11 |
| `notebooks/pipeline/03_vector_search_ditto.py` | 3 | 4.2, 4.4, 4.8 |
| `resources/jobs_phase4_pipeline.yml` | 5 | 4.3, 4.5, 4.7, 4.9, 4.10 |
| `notebooks/pipeline/05_generate_metrics.py` | 2 | 4.4, 4.6, 4.8 |
| `notebooks/pipeline/01_ingest_source_entities.py` | 1 | 4.8 |
| `notebooks/pipeline/02_exact_match.py` | 1 | 4.8 |
| `notebooks/pipeline/04_write_results.py` | 1 | 4.8 |
| `resources/jobs_phase3_serving.yml` | 1 | 3.1 |
| `databricks.yml` | 1 | 4.10 |

---

## Recommended Fix Order

### Step 1: Configuration Fixes (30 minutes)
1. **Include Phase 3 & 4 in databricks.yml** (Issue 4.10)
2. **Update library versions** (Issue 4.3)
3. **Add workspace_path to all tasks** (Issue 4.5)
4. **Fix email notifications** (Issue 4.7)

### Step 2: Notebook Updates (1-2 hours)
5. **Add type annotations import** (Issue 4.8) - All pipeline notebooks
6. **Add pip installs** (Issue 4.4) - Pipeline notebooks
7. **Fix hardcoded path** (Issue 4.1) - `03_full_pipeline_example.py`
8. **Add missing widgets** (Issue 4.11) - `03_full_pipeline_example.py`
9. **Fix schema columns** (Issue 4.6) - `04_write_results.py` & `05_generate_metrics.py`

### Step 3: Implementation (TBD)
10. **Implement vector search + Ditto** (Issue 4.2) - `03_vector_search_ditto.py`

### Step 4: Validation (30 minutes)
11. **Verify model registration** (Issue 3.1) - Check Phase 2 output
12. **Update notebook paths** (Issue 4.9) - Optional, test first

---

## Testing Checklist

After fixes applied:

### Phase 3 Testing
- [ ] Model exists in Unity Catalog: `{catalog}.models.entity_matching_ditto`
- [ ] `databricks bundle validate -t dev` passes with Phase 3
- [ ] `databricks bundle deploy -t dev` includes Phase 3
- [ ] Model serving endpoint created successfully
- [ ] Endpoint responds to test inference requests

### Phase 4 Testing
- [ ] `databricks bundle validate -t dev` passes with Phase 4
- [ ] `databricks bundle deploy -t dev` includes Phase 4
- [ ] All notebooks have proper dependencies installed
- [ ] Scheduled pipeline job created
- [ ] Ad-hoc job created with parameters
- [ ] Pipeline tasks have correct parameters
- [ ] Test run of each pipeline notebook individually:
  - [ ] `01_ingest_source_entities.py`
  - [ ] `02_exact_match.py`
  - [ ] `03_vector_search_ditto.py` (after implementation)
  - [ ] `04_write_results.py`
  - [ ] `05_generate_metrics.py`
- [ ] Full pipeline end-to-end test
- [ ] Email notifications received
- [ ] Results in gold table
- [ ] Metrics generated correctly

---

## Success Criteria

Phase 3 & 4 migration complete when:

### Technical
- ✅ All 13 issues resolved
- ✅ `databricks bundle validate` passes
- ✅ `databricks bundle deploy` succeeds
- ✅ Model serving endpoint operational
- ✅ Pipeline job runs successfully
- ✅ No hardcoded paths
- ✅ All dependencies compatible
- ✅ Email notifications working

### Operational
- ✅ Can deploy to dev/staging/prod
- ✅ All notebooks portable
- ✅ Monitoring/metrics enabled
- ✅ Documentation complete

---

## Next Steps

1. **Review this document** with team
2. **Prioritize Issue 4.2** (placeholder implementation) - determines timeline
3. **Apply quick fixes** (Steps 1-2 above)
4. **Test incrementally** after each fix category
5. **Implement core logic** (Issue 4.2)
6. **Full end-to-end testing**
7. **Document learnings** in migration template

---

**Analysis Complete**
**Ready for fixes:** Yes
**Estimated effort:** 4-6 hours (excluding Issue 4.2 implementation)
**Blockers:** Issue 4.2 (placeholder implementation) - scope TBD
