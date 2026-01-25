# workspace_path Parameter Verification - Phase 3 & 4
**Date:** 2026-01-25
**Scope:** Phase 3 (Model Serving) & Phase 4 (Production Pipeline)
**Status:** 🟡 Issue Confirmed - Fix Required

---

## Executive Summary

**Finding:** `workspace_path` parameter is **missing** from the production pipeline task that requires it.

- ✅ **Ad-hoc job**: Has `workspace_path` correctly configured
- ❌ **Production pipeline**: Missing `workspace_path` for task that imports `src` modules

**Impact:** Production pipeline will fail with `ModuleNotFoundError: No module named 'src'`

---

## Detailed Analysis

### Notebooks Requiring workspace_path

| Notebook | Imports from src? | workspace_path Needed? | Current Status |
|----------|-------------------|------------------------|----------------|
| `03_full_pipeline_example.py` | ✅ Yes | ✅ Yes | ✅ **PRESENT** (ad-hoc job) |
| `pipeline/03_vector_search_ditto.py` | ✅ Yes | ✅ Yes | ❌ **MISSING** (prod pipeline) |
| `pipeline/01_ingest_source_entities.py` | ❌ No | ❌ No | ✅ N/A |
| `pipeline/02_exact_match.py` | ❌ No | ❌ No | ✅ N/A |
| `pipeline/04_write_results.py` | ❌ No | ❌ No | ✅ N/A |
| `pipeline/05_generate_metrics.py` | ❌ No | ❌ No | ✅ N/A |

### Import Analysis

#### ✅ Ad-hoc Job (CORRECT)
**File:** `resources/jobs_phase4_pipeline.yml:106-115`

```yaml
tasks:
  - task_key: run_matching
    notebook_task:
      notebook_path: ../notebooks/03_full_pipeline_example.py
      base_parameters:
        workspace_path: ${workspace.root_path}/files  # ✅ PRESENT
        catalog_name: ${var.catalog_name}
        source_table: "{{job.parameters.source_table}}"
        output_table: "{{job.parameters.output_table}}"
        date_filter: "{{job.parameters.date_filter}}"
```

**Notebook imports:**
```python
# Line 49-53 in 03_full_pipeline_example.py
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from src.evaluation.validator import GoldStandardValidator
```

**Status:** ✅ **CORRECT** - workspace_path provided, imports will work

---

#### ❌ Production Pipeline Task (INCORRECT)
**File:** `resources/jobs_phase4_pipeline.yml:46-56`

```yaml
- task_key: vector_search_and_ditto
  depends_on:
    - task_key: exact_match
  job_cluster_key: pipeline_cluster
  notebook_task:
    notebook_path: ../notebooks/pipeline/03_vector_search_ditto.py
    base_parameters:
      catalog_name: ${var.catalog_name}
      ditto_endpoint: ditto-em-${bundle.target}
      vector_search_endpoint: entity-matching-vs-${bundle.target}
      # ❌ workspace_path: MISSING!
  timeout_seconds: 7200
```

**Notebook imports:**
```python
# Lines in 03_vector_search_ditto.py (after our fix)
from src.models.embeddings import BGEEmbeddings
from src.models.vector_search import VectorSearchIndex
from src.data.preprocessor import create_entity_features
```

**Notebook code expecting workspace_path:**
```python
# Get workspace path for imports
dbutils.widgets.text("workspace_path", "")
workspace_path = dbutils.widgets.get("workspace_path")

if workspace_path:
    import sys
    sys.path.append(workspace_path)
    print(f"Added to sys.path: {workspace_path}")
```

**Status:** ❌ **INCORRECT** - workspace_path NOT provided, imports will fail

---

## Error Expected Without Fix

When the production pipeline runs, task `vector_search_and_ditto` will fail with:

```python
ModuleNotFoundError: No module named 'src'

  File "notebooks/pipeline/03_vector_search_ditto.py", line XX
    from src.models.embeddings import BGEEmbeddings
ImportError: No module named 'src'
```

**Root Cause:**
- `workspace_path` parameter not passed to notebook
- Notebook gets empty string: `workspace_path = ""`
- `sys.path.append()` not executed
- Python cannot find `src` module in search path

---

## Why Other Tasks Don't Need workspace_path

### ✅ Tasks Using Only Spark SQL (No Fix Needed)

**1. ingest_source_entities**
```python
# Only uses PySpark and Spark SQL
from pyspark.sql.functions import current_timestamp, lit
df = spark.createDataFrame(sample_sources, schema)
df.write.saveAsTable(f"{catalog_name}.bronze.source_entities")
```
**No src imports** → No workspace_path needed

**2. exact_match**
```python
# Only uses PySpark and Spark SQL
from pyspark.sql.functions import col, lit, current_timestamp
exact_matches_lei = source_df.join(reference_df, ...)
```
**No src imports** → No workspace_path needed

**3. write_results**
```python
# Only uses PySpark
all_matches = exact_matches.union(vector_ditto_matches)
all_matches.write.saveAsTable(output_table)
```
**No src imports** → No workspace_path needed

**4. generate_metrics**
```python
# Only uses PySpark SQL
from pyspark.sql.functions import count, avg, sum, when, col
stats = today_matches.agg(count("*"), avg("match_confidence"))
```
**No src imports** → No workspace_path needed

---

## Fix Required

### Location
**File:** `resources/jobs_phase4_pipeline.yml`
**Lines:** 46-56
**Task:** `vector_search_and_ditto`

### Current Code
```yaml
- task_key: vector_search_and_ditto
  depends_on:
    - task_key: exact_match
  job_cluster_key: pipeline_cluster
  notebook_task:
    notebook_path: ../notebooks/pipeline/03_vector_search_ditto.py
    base_parameters:
      catalog_name: ${var.catalog_name}
      ditto_endpoint: ditto-em-${bundle.target}
      vector_search_endpoint: entity-matching-vs-${bundle.target}
  timeout_seconds: 7200
```

### Fixed Code
```yaml
- task_key: vector_search_and_ditto
  depends_on:
    - task_key: exact_match
  job_cluster_key: pipeline_cluster
  notebook_task:
    notebook_path: ../notebooks/pipeline/03_vector_search_ditto.py
    base_parameters:
      catalog_name: ${var.catalog_name}
      ditto_endpoint: ditto-em-${bundle.target}
      vector_search_endpoint: entity-matching-vs-${bundle.target}
      workspace_path: ${workspace.root_path}/files  # ✅ ADD THIS LINE
  timeout_seconds: 7200
```

**Change:** Add single line: `workspace_path: ${workspace.root_path}/files`

---

## Validation Tests

### Before Fix
```bash
# This will FAIL with ModuleNotFoundError
databricks bundle deploy -t dev
databricks bundle run entity_matching_pipeline -t dev
# Task: vector_search_and_ditto → ❌ FAILED
# Error: ModuleNotFoundError: No module named 'src'
```

### After Fix
```bash
# This should SUCCEED
databricks bundle deploy -t dev
databricks bundle run entity_matching_pipeline -t dev
# Task: vector_search_and_ditto → ✅ SUCCESS
# Log: "Added to sys.path: /Workspace/Users/.../.bundle/entity_matching/dev/files"
```

### Verification Checklist
- [ ] Parameter appears in job UI
- [ ] Notebook receives workspace_path value
- [ ] sys.path.append() executes
- [ ] src modules import successfully
- [ ] BGEEmbeddings loads
- [ ] VectorSearchIndex initializes
- [ ] Task completes without import errors

---

## Impact Analysis

### Severity: **HIGH**
- Production pipeline completely broken without this fix
- Core matching logic cannot execute
- No fallback mechanism

### Scope: **Limited**
- Only affects 1 task: `vector_search_and_ditto`
- Only affects production pipeline job
- Ad-hoc job already works correctly

### Effort: **Minimal**
- 1 line change
- No code logic changes
- No testing required beyond deployment

---

## Related Configuration

### DABs Variable Used
```yaml
# From databricks.yml
workspace:
  root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}
```

**For dev target:**
- `${workspace.root_path}` → `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev`
- `${workspace.root_path}/files` → `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/files`

### Files Synced to workspace_path/files
Based on DABs sync configuration, the following are synced:
```
${workspace.root_path}/files/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── embeddings.py
│   │   ├── ditto_matcher.py
│   │   └── vector_search.py
│   └── pipeline/
│       └── hybrid_pipeline.py
├── notebooks/
└── ...
```

**Import Path Resolution:**
```python
sys.path.append("/Workspace/.../dev/files")
# Now Python finds:
# /Workspace/.../dev/files/src/models/embeddings.py
from src.models.embeddings import BGEEmbeddings  # ✅ Works
```

---

## Comparison: Ad-hoc vs Production Pipeline

| Aspect | Ad-hoc Job | Production Pipeline |
|--------|------------|---------------------|
| **workspace_path provided?** | ✅ Yes (line 111) | ❌ No (line 52-55) |
| **Notebook requires src?** | ✅ Yes (03_full_pipeline_example.py) | ✅ Yes (03_vector_search_ditto.py) |
| **Will imports work?** | ✅ Yes | ❌ No |
| **Status** | ✅ **CORRECT** | ❌ **BROKEN** |

---

## Why This Was Missed

1. **Ad-hoc job configured first** - Had workspace_path correctly
2. **Production pipeline tasks added separately** - Pattern not copied
3. **Most tasks don't need it** - Only 1 of 5 tasks requires src imports
4. **Different notebooks** - Ad-hoc uses full example, production uses split notebooks

---

## Recommendation

**Priority:** HIGH
**Effort:** 1 minute
**Risk:** None (additive change, no breaking changes)

**Action:** Apply fix immediately before any production pipeline testing

---

## Phase 3 Analysis

**Phase 3:** Model Serving (no workspace_path needed)
- Model serving endpoints don't execute notebooks
- Configuration only, no Python imports
- ✅ No fix required for Phase 3

---

## Summary

**Issue Confirmed:** ✅ Yes
**Location:** `resources/jobs_phase4_pipeline.yml:52-55`
**Task Affected:** `vector_search_and_ditto`
**Fix Required:** Add 1 line: `workspace_path: ${workspace.root_path}/files`
**Impact:** HIGH (production pipeline broken without fix)
**Effort:** MINIMAL (1 line, 1 minute)

**Ready to apply fix:** ✅ Yes
