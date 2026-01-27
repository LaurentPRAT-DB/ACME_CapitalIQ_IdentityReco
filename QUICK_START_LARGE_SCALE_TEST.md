# Quick Start: Large-Scale Testing

## 🚀 Quick Start (5 Minutes)

### Step 1: Load Test Data

**In Databricks Workspace:**
1. Open notebook: `notebooks/setup/03_load_large_test_dataset.py`
2. **Keep default parameters:**
   - Reference entities: 1000
   - Source entities: 3000
   - Match ratio: 0.7
3. Click **Run All**
4. ⏱️ Wait ~2-3 minutes for data generation

**Expected Output:**
```
✅ Loaded 1000 reference entities
✅ Loaded 3000 source entities
✅ Created ground truth with 500 known matches
```

### Step 2: Run Pipeline

```bash
# Deploy and run Phase 4
./deploy-phase.sh 4 dev
```

### Step 3: View Results

**In Databricks SQL Editor:**
```sql
-- Overall stats
SELECT * FROM your_catalog.gold.daily_stats;

-- Sample matches
SELECT * FROM your_catalog.gold.matched_entities LIMIT 100;

-- Review queue
SELECT * FROM your_catalog.gold.review_queue LIMIT 50;
```

---

## 📊 What You Get

### Reference Data (1000 entities)
- ✅ Realistic company names
- ✅ Valid tickers (AAPL, MSFT, etc.)
- ✅ Proper identifiers (LEI, CUSIP, ISIN)
- ✅ 15 countries, 30 industries

### Source Data (3000 entities)
- ✅ **2100 entities should match** (70%)
  - Name variations: "Microsoft Inc" vs "Microsoft Inc."
  - Abbreviations: "Microsoft Corporation" → "Microsoft"
  - Typos: "Mcirosoft"
  - Missing suffixes: "Microsoft Corp" → "Microsoft"
  - Ticker only: "MSFT"
  - With identifiers: LEI/CUSIP present
  - Partial info: Some fields missing

- ✅ **900 entities won't match** (30%)
  - Different companies not in reference
  - Tests precision (avoiding false positives)

---

## 🎯 Expected Results

### Pipeline Performance
- **Match Rate:** ~70-80%
- **Processing Time:** 5-10 minutes (with Hugging Face embeddings)
- **Matches Found:** ~2100-2400 out of 3000

### Stage Breakdown
- **Exact Match (Stage 1):** ~150-300 entities (5-10%)
- **Vector + Ditto (Stage 2-3):** ~1800-2100 entities (60-70%)
- **Foundation Model (Stage 4):** ~150-300 entities (5-10%)

### Quality Metrics
- **High Confidence (auto-match):** ~80% of matches
- **Needs Review:** ~20% of matches
- **Average Confidence:** ~85-90%

---

## 🔧 Configuration Options

### Smaller Test (Faster)

**In notebook widget:**
- Reference entities: `100`
- Source entities: `300`
- ⏱️ Time: ~1 minute to load, ~1 minute to process

### Larger Test (Stress Test)

**In notebook widget:**
- Reference entities: `5000`
- Source entities: `15000`
- ⏱️ Time: ~10 minutes to load, ~30-60 minutes to process

### Different Match Ratio

**In notebook widget:**
- Match ratio: `0.9` (90% should match - easier test)
- Match ratio: `0.5` (50% should match - harder test)
- Match ratio: `0.0` (0% should match - precision test)

---

## 📈 Evaluate Results

### Check Match Rate

```sql
SELECT
    COUNT(*) as total_source,
    COUNT(matched_ciq_id) as matched,
    COUNT(matched_ciq_id) * 100.0 / COUNT(*) as match_rate_pct
FROM your_catalog.gold.matched_entities;
```

**Expected:** ~70-80% match rate

### Check Accuracy (vs Ground Truth)

```sql
WITH pipeline_results AS (
    SELECT source_id, source_system, matched_ciq_id
    FROM your_catalog.gold.matched_entities
),
ground_truth AS (
    SELECT source_id, source_system, ciq_id as true_ciq_id
    FROM your_catalog.bronze.gold_standard
)
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN pr.matched_ciq_id = gt.true_ciq_id THEN 1 ELSE 0 END) as correct,
    SUM(CASE WHEN pr.matched_ciq_id = gt.true_ciq_id THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_pct
FROM ground_truth gt
LEFT JOIN pipeline_results pr
    ON gt.source_id = pr.source_id
    AND gt.source_system = pr.source_system;
```

**Expected:** ~90-95% accuracy

### Check Stage Performance

```sql
SELECT
    match_stage,
    COUNT(*) as count,
    AVG(match_confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_latency_ms
FROM your_catalog.gold.matched_entities
GROUP BY match_stage
ORDER BY count DESC;
```

---

## ⚡ Performance Tips

### Faster Processing (Use Hugging Face)

**In databricks-phase4.yml:**
```yaml
variables:
  embeddings_provider: huggingface  # Default, faster
```

### Serverless Scaling (Use Databricks)

**In databricks-phase4.yml:**
```yaml
variables:
  embeddings_provider: databricks
  embeddings_model_name: "databricks-gte-large-en"
```

**Note:** Databricks API is slower but serverless.

---

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution:** Update sys.path in notebook:
```python
import sys
sys.path.append("/Workspace/Users/YOUR_USER/.bundle/entity_matching/dev/files")
```

### Issue: "Table already exists"

**Solution:** Change "Write Mode" widget to "overwrite"

### Issue: Pipeline finds 0 matches

**Check:**
1. Reference table has data: `SELECT COUNT(*) FROM bronze.spglobal_reference`
2. Source table has data: `SELECT COUNT(*) FROM bronze.source_entities`
3. Pipeline ran successfully: Check job logs

---

## 📚 Full Documentation

For detailed information, see:
- **LARGE_SCALE_TEST_DATA_GUIDE.md** - Complete guide
- **EMBEDDINGS_PROVIDER_GUIDE.md** - Embeddings configuration
- **DEPLOYMENT_GUIDE.md** - Deployment instructions

---

## ✅ Success Checklist

- [ ] Loaded 1000+ reference entities
- [ ] Loaded 3000+ source entities
- [ ] Deployed Phase 4 successfully
- [ ] Pipeline completed without errors
- [ ] Match rate ~70-80%
- [ ] Matches written to gold.matched_entities
- [ ] Review queue populated for low-confidence matches

**Ready to scale to production!** 🎉

---

**Last Updated:** 2026-01-27
**Time to Complete:** ~5-10 minutes
**Difficulty:** Easy
