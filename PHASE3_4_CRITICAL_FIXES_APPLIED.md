# Phase 3 & 4 Critical Fixes - Applied
**Date:** 2026-01-25
**Status:** ✅ Complete
**Scope:** 3 Critical Issues Fixed

---

## Summary

Fixed 3 critical issues blocking Phase 3 & 4 deployment:

1. ✅ **Hardcoded path removed** - `notebooks/03_full_pipeline_example.py`
2. ✅ **Placeholder implementation replaced** - `notebooks/pipeline/03_vector_search_ditto.py`
3. ✅ **Library versions updated** - `resources/jobs_phase4_pipeline.yml`

---

## Fix 1: Hardcoded Path → Parameterized (CRITICAL)

### File: `notebooks/03_full_pipeline_example.py`

### Changes Applied

**Added widget parameter:**
```python
# Line 28 (added)
dbutils.widgets.text("gold_standard_path", "")
gold_standard_path = dbutils.widgets.get("gold_standard_path")
```

**Replaced hardcoded path (line 157):**

**Before:**
```python
validator = GoldStandardValidator()
ground_truth = validator.load_gold_standard("/dbfs/entity_matching/gold_standard.csv")
```

**After:**
```python
validator = GoldStandardValidator()

# Use parameter if provided, otherwise load from catalog table
if gold_standard_path:
    print(f"Loading gold standard from: {gold_standard_path}")
    ground_truth = validator.load_gold_standard(gold_standard_path)
else:
    # Load from Unity Catalog table (preferred for DABs)
    print(f"Loading gold standard from catalog table: {catalog_name}.bronze.gold_standard")
    ground_truth_df = spark.table(f"{catalog_name}.bronze.gold_standard").toPandas()
    ground_truth = ground_truth_df
```

### Benefits
- ✅ No hardcoded paths
- ✅ Works across dev/staging/prod environments
- ✅ Supports both file and catalog-based gold standard
- ✅ Follows DABs best practices

---

## Fix 2: Placeholder → Full Implementation (CRITICAL)

### File: `notebooks/pipeline/03_vector_search_ditto.py`

### Changes Applied

Replaced 20-line placeholder with **120+ line full implementation**:

#### 1. Added Library Installation
```python
# MAGIC %pip install --upgrade transformers>=4.40.0 sentence-transformers>=2.3.0 torch>=2.1.0 faiss-cpu scikit-learn mlflow

dbutils.library.restartPython()
```

#### 2. Added workspace_path Support
```python
# Get workspace path for imports
dbutils.widgets.text("workspace_path", "")
workspace_path = dbutils.widgets.get("workspace_path")

if workspace_path:
    import sys
    sys.path.append(workspace_path)
    print(f"Added to sys.path: {workspace_path}")
```

#### 3. Implemented Vector Search
```python
from src.models.embeddings import BGEEmbeddings
from src.models.vector_search import VectorSearchIndex
from src.data.preprocessor import create_entity_features

# Initialize BGE embeddings model
embeddings_model = BGEEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Load reference data and build vector index
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference").toPandas()
reference_texts = [create_entity_features(entity) for entity in reference_df.to_dict('records')]
reference_embeddings = embeddings_model.encode(reference_texts, batch_size=32, show_progress_bar=True)

# Build vector search index
vector_index = VectorSearchIndex(embedding_dim=embeddings_model.embedding_dim)
vector_index.build_index(
    embeddings=reference_embeddings,
    ids=reference_df["ciq_id"].tolist(),
    metadata=reference_df.to_dict('records')
)
```

#### 4. Implemented Ditto Matching
```python
# Load Ditto model from Unity Catalog
ditto_matcher = DittoMatcher()
ditto_model_path = f"models:/{catalog_name}.models.entity_matching_ditto/1"
try:
    ditto_matcher.load_model(ditto_model_path)
except Exception as e:
    print(f"⚠ Could not load Ditto model: {e}")
    ditto_matcher = None  # Fall back to vector search only
```

#### 5. Implemented Matching Logic
```python
# For each unmatched entity:
# 1. Generate embedding
entity_embedding = embeddings_model.encode(entity_text)

# 2. Find top candidates with vector search
candidates = vector_index.search(entity_embedding, top_k=5)

# 3. Verify with Ditto matcher
for candidate in candidates[:3]:
    candidate_text = create_entity_features(candidate_entity)
    prediction, confidence = ditto_matcher.predict(entity_text, candidate_text)

    if prediction == 1 and confidence > best_confidence:
        best_match = candidate
        best_confidence = confidence
```

#### 6. Results Collection
```python
matched_results.append({
    "source_id": entity["source_id"],
    "source_system": entity["source_system"],
    "company_name": entity["company_name"],
    "matched_ciq_id": best_match["ciq_id"],
    "match_confidence": best_confidence,
    "match_method": match_method,  # "ditto_matcher" or "vector_search"
    "match_stage": "Stage 2 & 3: Vector Search + Ditto",
    "reasoning": best_reasoning,
    "matched_company_name": best_match["metadata"]["company_name"],
    "match_timestamp": pd.Timestamp.now(),
    "processing_time_ms": processing_time,
    "model_version": "v1.0"
})
```

### Implementation Features

- ✅ **Graceful degradation**: Falls back to vector search if Ditto unavailable
- ✅ **Batch processing**: Uses batch encoding for efficiency
- ✅ **Progress tracking**: Shows progress every 10 entities
- ✅ **Performance metrics**: Tracks processing time per entity
- ✅ **Confidence threshold**: Only accepts matches ≥ 0.70 confidence
- ✅ **Top-k verification**: Checks top 3 vector search candidates with Ditto
- ✅ **Proper imports**: Uses workspace_path for src module imports

### Performance Expectations

- **Vector search**: ~50ms per entity (embedding + FAISS search)
- **Ditto verification**: ~100ms per entity (3 candidates × 30ms)
- **Total**: ~150ms per entity
- **Batch of 100**: ~15 seconds

---

## Fix 3: Library Versions Updated (CRITICAL)

### File: `resources/jobs_phase4_pipeline.yml`

### Changes Applied

**Line 124-126**

**Before:**
```yaml
libraries:
  - pypi:
      package: sentence-transformers==2.2.2
```

**After:**
```yaml
libraries:
  - pypi:
      package: sentence-transformers>=2.3.0
  - pypi:
      package: transformers>=4.40.0
  - pypi:
      package: torch>=2.1.0
```

### Benefits
- ✅ Compatible with Python 3.10
- ✅ Matches Phase 2 migration standards
- ✅ Prevents type annotation errors
- ✅ Includes all required dependencies

---

## Validation Checklist

Before deploying, verify:

### Fix 1 Validation
- [ ] Widget `gold_standard_path` appears in job parameters
- [ ] Falls back to catalog table if path not provided
- [ ] No hardcoded `/dbfs/` paths remain

### Fix 2 Validation
- [ ] Ditto model registered in Unity Catalog: `{catalog}.models.entity_matching_ditto`
- [ ] Reference data available: `{catalog}.bronze.spglobal_reference`
- [ ] Workspace path parameter passed to notebook
- [ ] Vector search returns candidates
- [ ] Ditto matcher loads successfully
- [ ] Matches written to `{catalog}.silver.vector_ditto_matches_temp`
- [ ] Processing completes in reasonable time

### Fix 3 Validation
- [ ] Library versions updated in YAML
- [ ] Job cluster installs correct versions
- [ ] No version conflicts
- [ ] Imports work without errors

---

## Testing Commands

### Test Notebook Individually
```bash
# Deploy bundle first
databricks bundle deploy -t dev

# Test vector search + Ditto notebook
# (Requires Phase 1 & 2 data to be present)
databricks workspace export /Workspace/Users/your.email/.bundle/entity_matching/dev/files/notebooks/pipeline/03_vector_search_ditto.py
```

### Test Full Pipeline Job
```bash
# Validate bundle
databricks bundle validate -t dev

# Deploy bundle
databricks bundle deploy -t dev

# Run pipeline job
databricks bundle run entity_matching_pipeline -t dev
```

---

## Known Considerations

### Fix 2 Implementation Notes

1. **Model Dependency**
   - Requires Ditto model in Unity Catalog: `{catalog}.models.entity_matching_ditto/1`
   - If model not available, falls back to vector search only
   - Confidence threshold: 0.70 minimum for matches

2. **Performance**
   - Processes entities sequentially (not parallelized in this implementation)
   - For large batches, consider Spark UDF or distributed processing
   - Current: ~150ms per entity
   - For 10,000 entities: ~25 minutes

3. **Memory**
   - Loads full reference dataset into memory for vector index
   - BGE model: ~1.2GB
   - Ditto model: ~250MB
   - Vector index: ~size of reference × 4KB per entity
   - Total: ~2-3GB for 10K reference entities

4. **Fallback Behavior**
   - If Ditto model unavailable: Uses vector search only
   - If no candidates found: Entity not matched (not in results)
   - Only writes entities that meet confidence threshold

---

## Next Steps

### Immediate (Before Deploy)
1. **Verify Model Registration**
   - Check that Phase 2 training registered model to UC
   - Run: `mlflow search model_versions filter="name='{catalog}.models.entity_matching_ditto'"`
   - If missing, re-run Phase 2 training notebook

2. **Test Locally** (if possible)
   - Run `03_vector_search_ditto.py` with sample data
   - Verify embeddings generate correctly
   - Check Ditto predictions

3. **Update databricks.yml**
   - Include Phase 3 & 4 configurations (see Issue 4.10 in findings doc)
   - Add `workspace_path` to pipeline task parameters

### After Deploy
1. **Run End-to-End Test**
   - Full pipeline from ingest to metrics
   - Verify each stage produces expected outputs

2. **Monitor Performance**
   - Check processing time per entity
   - Review match rates by method
   - Validate confidence distributions

3. **Fix Remaining Issues**
   - Address 6 HIGH priority issues
   - Address 4 MEDIUM priority issues
   - See `PHASE3_4_MIGRATION_FINDINGS.md` for details

---

## Files Modified

| File | Lines Changed | Type |
|------|---------------|------|
| `notebooks/03_full_pipeline_example.py` | +14, -2 | Parameter addition |
| `notebooks/pipeline/03_vector_search_ditto.py` | +118, -20 | Full rewrite |
| `resources/jobs_phase4_pipeline.yml` | +3, -1 | Library update |
| **Total** | **+135, -23** | **3 files** |

---

## Success Indicators

### You'll know Fix 1 worked when:
- ✅ No hardcoded path errors in logs
- ✅ Gold standard loads from parameter or catalog
- ✅ Evaluation metrics calculate successfully

### You'll know Fix 2 worked when:
- ✅ Embeddings generate without errors
- ✅ Vector index builds successfully
- ✅ Ditto model loads from UC
- ✅ Matches written to silver table
- ✅ Match rate > 0%
- ✅ Processing completes within expected time

### You'll know Fix 3 worked when:
- ✅ Job cluster starts without library errors
- ✅ sentence-transformers imports successfully
- ✅ transformers version ≥ 4.40.0
- ✅ No type annotation errors

---

## Rollback Plan

If any fix causes issues:

```bash
# Revert specific file
git checkout main -- <file_path>

# Or revert all changes
git checkout main -- notebooks/03_full_pipeline_example.py
git checkout main -- notebooks/pipeline/03_vector_search_ditto.py
git checkout main -- resources/jobs_phase4_pipeline.yml
```

---

**Status:** ✅ All 3 Critical Fixes Applied
**Ready for:** Testing and deployment
**Remaining:** 10 HIGH/MEDIUM issues (see `PHASE3_4_MIGRATION_FINDINGS.md`)
