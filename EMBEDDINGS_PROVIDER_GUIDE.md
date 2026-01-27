# Embeddings Provider Configuration Guide

## Overview

The pipeline now supports **two embeddings providers**:
1. **Hugging Face** (default): Self-hosted models via sentence-transformers
2. **Databricks**: Native Databricks Foundation Model API

This guide explains how to choose and configure each provider.

---

## Quick Start

### Default Configuration (Hugging Face)
```yaml
# databricks-phase4.yml
variables:
  embeddings_provider: huggingface  # Default
  embeddings_model_name: ""  # Uses BAAI/bge-large-en-v1.5
```

### Databricks Configuration
```yaml
# databricks-phase4.yml
variables:
  embeddings_provider: databricks
  embeddings_model_name: "databricks-gte-large-en"
```

---

## Provider Comparison

| Feature | Hugging Face | Databricks |
|---------|-------------|------------|
| **Cost** | Cluster compute only ($0) | API calls ($0.0001-0.001 per call) |
| **Latency** | Low (local inference) | Medium (API call ~50-200ms) |
| **Setup** | Model download + GPU/CPU | API auth only |
| **Memory** | 1-2GB model in memory | No local memory |
| **Scaling** | Limited by cluster size | Serverless auto-scaling |
| **Best for** | Batch processing (100s-1000s) | Interactive/low-volume |
| **Dependencies** | torch, sentence-transformers | databricks-sdk |

### Cost Estimate (1000 entities)

**Hugging Face:**
- Model loading: ~30-60 seconds (one-time)
- Inference: ~100-200 entities/second
- Total time: ~5-10 seconds
- Cost: **$0** (uses existing cluster compute)

**Databricks:**
- API calls: 1000 entities × ~5 candidates = 6000 calls
- Latency: ~100-200ms per call
- Total time: ~10-20 minutes (with batching)
- Cost: **~$0.60-$6.00** (at $0.0001-0.001 per call)

---

## Configuration Methods

### Method 1: Global Configuration (databricks-phase4.yml)

**File:** `databricks-phase4.yml`

```yaml
variables:
  # Choose provider
  embeddings_provider:
    description: Embeddings provider (huggingface or databricks)
    default: huggingface  # or databricks

  # Optional: Override model name
  embeddings_model_name:
    description: Embeddings model name (provider-specific default if empty)
    default: ""  # Leave empty for defaults

targets:
  dev:
    variables:
      embeddings_provider: huggingface  # Dev uses free Hugging Face

  prod:
    variables:
      embeddings_provider: databricks  # Prod uses managed Databricks
      embeddings_model_name: "databricks-gte-large-en"
```

### Method 2: Runtime Override

Override at deployment/run time:

```bash
# Deploy with Databricks embeddings
databricks bundle deploy -t dev \
  --var="embeddings_provider=databricks" \
  --var="embeddings_model_name=databricks-gte-large-en" \
  -c databricks-phase4.yml

# Run job with Hugging Face embeddings
databricks bundle run entity_matching_pipeline -t dev \
  --var="embeddings_provider=huggingface"
```

### Method 3: Notebook Parameters

For ad-hoc runs, use notebook widgets:

```python
# In Databricks notebook
dbutils.widgets.dropdown("embeddings_provider", "huggingface", ["huggingface", "databricks"])
dbutils.widgets.text("embeddings_model_name", "")

provider = dbutils.widgets.get("embeddings_provider")
model_name = dbutils.widgets.get("embeddings_model_name")
```

---

## Available Models

### Hugging Face Models

**Default:** `BAAI/bge-large-en-v1.5`
- Dimensions: 1024
- Best for: General purpose entity matching
- Size: ~1.3GB

**Alternatives:**
- `BAAI/bge-base-en-v1.5`: 768 dims, smaller, faster
- `sentence-transformers/all-MiniLM-L6-v2`: 384 dims, very fast
- `intfloat/e5-large-v2`: 1024 dims, high quality

### Databricks Models

**Default:** `databricks-gte-large-en`
- Dimensions: 1024
- Optimized for general text embeddings
- Managed by Databricks

**Alternatives:**
- `databricks-bge-large-en`: BGE variant optimized by Databricks

---

## Usage Examples

### Example 1: Pipeline with Hugging Face (Default)

```python
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

pipeline = HybridMatchingPipeline(
    reference_df=reference_df,
    ditto_model_path=ditto_model_path,
    embeddings_provider="huggingface",  # Optional, this is default
    embeddings_model_name="BAAI/bge-large-en-v1.5",  # Optional, this is default
    databricks_client=None  # Not needed for Hugging Face
)
```

### Example 2: Pipeline with Databricks

```python
from databricks.sdk import WorkspaceClient
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

w = WorkspaceClient()

pipeline = HybridMatchingPipeline(
    reference_df=reference_df,
    ditto_model_path=ditto_model_path,
    embeddings_provider="databricks",
    embeddings_model_name="databricks-gte-large-en",
    databricks_client=w  # Required for Databricks provider
)
```

### Example 3: Direct Embeddings Model Creation

```python
from src.models.embeddings import create_embeddings_model
from databricks.sdk import WorkspaceClient

# Hugging Face
hf_model = create_embeddings_model(
    provider="huggingface",
    model_name="BAAI/bge-large-en-v1.5"
)

# Databricks
db_model = create_embeddings_model(
    provider="databricks",
    model_name="databricks-gte-large-en",
    databricks_client=WorkspaceClient()
)

# Use the same interface for both
embeddings = hf_model.encode(["Apple Inc.", "Microsoft Corporation"])
embeddings = db_model.encode(["Apple Inc.", "Microsoft Corporation"])
```

---

## Decision Guide

### Use Hugging Face When:

✅ Processing **large batches** (100s-1000s of entities)
✅ Cost optimization is priority
✅ Have GPU/CPU resources available
✅ Want faster throughput
✅ Offline/air-gapped environments

**Best for:** Production pipelines, scheduled jobs, high-volume processing

### Use Databricks When:

✅ Processing **small batches** (< 100 entities)
✅ Need **serverless** auto-scaling
✅ Limited cluster resources
✅ Interactive/ad-hoc queries
✅ Want managed infrastructure
✅ Latest model versions important

**Best for:** Interactive notebooks, low-volume queries, PoC/demos

---

## Hybrid Strategy (Recommended)

Use **different providers for different environments**:

```yaml
# databricks-phase4.yml
targets:
  dev:
    variables:
      embeddings_provider: huggingface
      # Dev: Use free Hugging Face for testing

  staging:
    variables:
      embeddings_provider: huggingface
      # Staging: Test at scale with Hugging Face

  prod:
    variables:
      embeddings_provider: databricks
      embeddings_model_name: "databricks-gte-large-en"
      # Prod: Use managed Databricks for reliability
```

Or use **different providers for different use cases**:

1. **Scheduled pipelines** → Hugging Face (cost-effective)
2. **Interactive notebooks** → Databricks (convenience)
3. **REST API endpoints** → Databricks (serverless scaling)

---

## Performance Optimization

### Hugging Face Optimization

```python
# Use GPU if available
embeddings_model = create_embeddings_model(
    provider="huggingface",
    model_name="BAAI/bge-large-en-v1.5",
    device="cuda"  # Use GPU
)

# Batch processing
embeddings = embeddings_model.encode(
    texts,
    batch_size=64,  # Larger batches for GPU
    show_progress_bar=True
)
```

### Databricks Optimization

```python
# Increase batch size for API efficiency
embeddings_model = create_embeddings_model(
    provider="databricks",
    model_name="databricks-gte-large-en",
    databricks_client=WorkspaceClient()
)

# Batch multiple texts per API call
embeddings = embeddings_model.encode(
    texts,
    batch_size=32,  # Databricks may have limits
    show_progress_bar=True
)
```

---

## Troubleshooting

### Issue: "Model not found" (Hugging Face)

**Error:**
```
OSError: Can't load model 'BAAI/bge-large-en-v1.5'
```

**Solution:**
1. Check internet connectivity (model downloads from Hugging Face)
2. Or pre-download model to workspace:
   ```bash
   huggingface-cli download BAAI/bge-large-en-v1.5
   ```

### Issue: "Endpoint not found" (Databricks)

**Error:**
```
databricks.sdk.errors.NotFound: Endpoint 'databricks-gte-large-en' not found
```

**Solution:**
1. Verify endpoint exists:
   ```bash
   databricks serving-endpoints list --profile YOUR_PROFILE
   ```
2. Check authentication:
   ```bash
   databricks auth login https://YOUR_WORKSPACE.cloud.databricks.com
   ```

### Issue: High API costs (Databricks)

**Symptom:** Unexpected costs from Databricks embedding API

**Solution:**
1. Switch to Hugging Face for batch processing
2. Reduce candidate retrieval (fewer embeddings needed)
3. Cache embeddings for reference data
4. Monitor usage:
   ```python
   # Track API calls
   print(f"Processed {len(entities)} entities")
   print(f"Estimated API calls: {len(entities) * 5}")  # Assuming 5 candidates
   ```

### Issue: Out of memory (Hugging Face)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size:
   ```python
   embeddings_model.encode(texts, batch_size=16)  # Smaller batches
   ```
2. Use CPU instead of GPU:
   ```python
   create_embeddings_model(provider="huggingface", device="cpu")
   ```
3. Use smaller model:
   ```python
   create_embeddings_model(
       provider="huggingface",
       model_name="BAAI/bge-base-en-v1.5"  # Smaller
   )
   ```

---

## Migration Guide

### Migrating from Hugging Face to Databricks

**Step 1:** Update configuration
```yaml
# databricks-phase4.yml
variables:
  embeddings_provider: databricks
  embeddings_model_name: "databricks-gte-large-en"
```

**Step 2:** Redeploy Phase 4
```bash
./deploy-phase.sh 4 dev
```

**Step 3:** Test with small batch
```python
# Test with 10 entities first
results = pipeline.batch_match(entities[:10])
pipeline.print_pipeline_stats(results)
```

**Step 4:** Monitor costs and performance

### Migrating from Databricks to Hugging Face

**Step 1:** Update configuration
```yaml
# databricks-phase4.yml
variables:
  embeddings_provider: huggingface
  embeddings_model_name: ""  # Use default
```

**Step 2:** Ensure cluster has sufficient resources
- Check memory: Need ~2GB for model
- Check disk: Need ~1.5GB for model download

**Step 3:** Redeploy and test
```bash
./deploy-phase.sh 4 dev
```

---

## FAQ

**Q: Can I use both providers in the same pipeline?**
A: Not in the same run, but you can switch between runs by changing the parameter.

**Q: Which provider gives better accuracy?**
A: Both use similar embeddings models (BGE/GTE family). Accuracy should be comparable. Choose based on cost/performance needs.

**Q: What happens if Databricks API is down?**
A: The pipeline will fail. For production, consider using Hugging Face for reliability or implement API fallback logic.

**Q: Can I use custom models?**
A: Yes! For Hugging Face, use any sentence-transformers compatible model. For Databricks, use any available Foundation Model endpoint.

**Q: How do I cache embeddings?**
A: Save reference embeddings to Delta table:
```python
# Generate once
reference_embeddings = embeddings_model.encode(reference_texts)
# Save to Delta
spark.createDataFrame(embeddings_df).write.mode("overwrite").saveAsTable("embeddings_cache")
```

---

## Summary

✅ **Dual provider support** implemented
✅ **Backward compatible** (defaults to Hugging Face)
✅ **Configurable** via YAML, CLI, or notebook widgets
✅ **Same interface** for both providers
✅ **Production-ready** with error handling and retries

**Recommendation:** Use **Hugging Face for production pipelines** and **Databricks for interactive work**.

---

**Last Updated:** 2026-01-27
**Status:** Production Ready
