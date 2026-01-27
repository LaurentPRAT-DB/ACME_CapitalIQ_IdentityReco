## Large-Scale Test Data Generation Guide

## Overview

This guide explains how to generate and load large-scale test datasets for comprehensive entity matching testing.

### Dataset Sizes

**Default Configuration:**
- **Reference Entities (spglobal_reference):** 1,000 entities
- **Source Entities (source_entities):** 3,000 entities
- **Match Ratio:** 70% (2,100 should match, 900 won't match)

**Scalable:** Can easily scale to 10,000+ entities of each type.

---

## Quick Start

### Option 1: Run Setup Notebook (Recommended)

1. **Open the notebook:**
   ```
   notebooks/setup/03_load_large_test_dataset.py
   ```

2. **Configure parameters** (or use defaults):
   - Catalog Name: `entity_matching` (or your catalog)
   - Number of Reference Entities: `1000`
   - Number of Source Entities: `3000`
   - Match Ratio: `0.7` (70%)
   - Write Mode: `overwrite` or `append`

3. **Run all cells**

4. **Verify:**
   ```sql
   SELECT COUNT(*) FROM your_catalog.bronze.spglobal_reference;  -- Should be 1000
   SELECT COUNT(*) FROM your_catalog.bronze.source_entities;     -- Should be 3000
   ```

### Option 2: Python API

```python
from src.data.large_dataset_generator import LargeDatasetGenerator

# Initialize generator
generator = LargeDatasetGenerator(seed=42)

# Generate reference entities
reference_df = generator.generate_reference_entities(num_entities=1000)

# Generate source entities with 70% match ratio
source_df = generator.generate_source_entities(
    reference_df=reference_df,
    num_entities=3000,
    match_ratio=0.7
)

# Load to Spark and save
reference_spark = spark.createDataFrame(reference_df)
source_spark = spark.createDataFrame(source_df)

reference_spark.write.format("delta").mode("overwrite") \
    .saveAsTable("catalog.bronze.spglobal_reference")

source_spark.write.format("delta").mode("overwrite") \
    .saveAsTable("catalog.bronze.source_entities")
```

---

## Generated Data Characteristics

### Reference Entities (spglobal_reference)

**Schema:**
- `ciq_id`: Unique identifier (IQ100000-IQ101000)
- `company_name`: Realistic company names
- `ticker`: 3-4 character ticker symbols
- `lei`: 20-character Legal Entity Identifier
- `cusip`: 9-digit CUSIP (for US companies)
- `isin`: 12-character ISIN
- `country`: 15 different countries
- `industry`: 30 different industries
- `sector`: Mapped industry sector
- `market_cap`: Random market cap ($100M - $500B)
- `last_updated`: Recent timestamp

**Features:**
- ✅ Realistic company names with various suffixes (Inc, Corp, Ltd, etc.)
- ✅ Valid ticker symbols
- ✅ Proper identifiers (LEI, CUSIP, ISIN format)
- ✅ Diverse geography (US, UK, Germany, Japan, etc.)
- ✅ Comprehensive industry coverage (30 industries, 9 sectors)

### Source Entities (source_entities)

**Schema:**
- `source_id`: Unique per source system (SRC-00001)
- `source_system`: 10 different source systems (Salesforce, SAP, Oracle, etc.)
- `company_name`: Name variations of reference entities
- `ticker`: May or may not be present
- `lei/cusip/isin`: Occasionally present
- `country`: May vary slightly from reference
- `industry`: May vary slightly from reference
- `ingestion_timestamp`: Recent timestamp

**Variation Types (for entities that should match):**

1. **Name Variations (15%)**: "Microsoft Inc" vs "Microsoft Inc."
2. **Abbreviations (15%)**: "Microsoft Corporation" → "Microsoft"
3. **Typos (10%)**: "Mcirosoft Corporation"
4. **Missing Suffix (15%)**: "Microsoft Corporation" → "Microsoft"
5. **Ticker Only (10%)**: Company name is just "MSFT"
6. **With Identifiers (20%)**: Same name + LEI/CUSIP
7. **Partial Info (15%)**: Only 2-3 fields available

**Non-Matching Entities (30% by default):**
- Completely different company names
- Not in reference dataset
- Used to test precision/false positive rate

---

## Configuration Options

### Adjust Dataset Size

**Small Test (100 ref, 300 source):**
```python
generator.generate_reference_entities(num_entities=100)
generator.generate_source_entities(num_entities=300, match_ratio=0.7)
```

**Medium Test (1000 ref, 3000 source - Default):**
```python
generator.generate_reference_entities(num_entities=1000)
generator.generate_source_entities(num_entities=3000, match_ratio=0.7)
```

**Large Test (5000 ref, 15000 source):**
```python
generator.generate_reference_entities(num_entities=5000)
generator.generate_source_entities(num_entities=15000, match_ratio=0.7)
```

### Adjust Match Ratio

**High match rate (90% should match):**
```python
generator.generate_source_entities(match_ratio=0.9)
```

**Low match rate (50% should match):**
```python
generator.generate_source_entities(match_ratio=0.5)
```

**No matches (test precision):**
```python
generator.generate_source_entities(match_ratio=0.0)
```

### Change Seed for Different Data

```python
# Different random seed = different data
generator = LargeDatasetGenerator(seed=123)
```

---

## Data Quality Checks

### Verify Reference Data

```sql
SELECT
    COUNT(*) as total_entities,
    COUNT(DISTINCT ciq_id) as unique_ids,
    COUNT(DISTINCT ticker) as unique_tickers,
    SUM(CASE WHEN lei IS NOT NULL THEN 1 ELSE 0 END) as with_lei,
    SUM(CASE WHEN cusip IS NOT NULL THEN 1 ELSE 0 END) as with_cusip,
    COUNT(DISTINCT country) as countries,
    COUNT(DISTINCT industry) as industries
FROM catalog.bronze.spglobal_reference;
```

**Expected:**
- Total = num_entities specified
- Unique IDs = 100%
- LEI coverage = 100%
- CUSIP coverage = ~15% (only US companies)
- 15 countries
- 30 industries

### Verify Source Data

```sql
SELECT
    COUNT(*) as total_entities,
    COUNT(DISTINCT source_id, source_system) as unique_ids,
    COUNT(DISTINCT source_system) as num_systems,
    SUM(CASE WHEN ticker IS NOT NULL THEN 1 ELSE 0 END) as with_ticker
FROM catalog.bronze.source_entities;
```

**Expected:**
- Total = num_entities specified
- Unique IDs = 100%
- Source systems = 5-10
- Ticker coverage = 60-80%

### Sample Data

```sql
-- Sample reference entities
SELECT ciq_id, company_name, ticker, country, industry
FROM catalog.bronze.spglobal_reference
LIMIT 10;

-- Sample source entities
SELECT source_id, source_system, company_name, ticker
FROM catalog.bronze.source_entities
LIMIT 10;
```

---

## Testing Scenarios

### Scenario 1: Baseline Performance (Default)

**Configuration:**
- 1,000 reference
- 3,000 source
- 70% match ratio

**Expected Results:**
- ~2,100 matches found
- ~900 no matches
- Match rate: 70%+
- Processing time: 5-10 minutes

**Run:**
```bash
./deploy-phase.sh 4 dev
```

### Scenario 2: High-Volume Stress Test

**Configuration:**
- 5,000 reference
- 15,000 source
- 70% match ratio

**Expected Results:**
- ~10,500 matches found
- Processing time: 30-60 minutes
- Tests scalability and performance

**Load Data:**
```python
# In notebook
num_reference = 5000
num_source = 15000
```

### Scenario 3: Precision Test (No Matches)

**Configuration:**
- 1,000 reference
- 1,000 source (different companies)
- 0% match ratio

**Expected Results:**
- 0-50 matches (false positives)
- Precision test: Should minimize false matches

**Load Data:**
```python
generator.generate_source_entities(match_ratio=0.0)
```

### Scenario 4: Recall Test (All Match)

**Configuration:**
- 1,000 reference
- 1,000 source (all variations)
- 100% match ratio

**Expected Results:**
- ~950+ matches found
- Recall test: Should find most matches

**Load Data:**
```python
generator.generate_source_entities(match_ratio=1.0)
```

---

## Performance Benchmarks

### Expected Processing Times

| Config | Reference | Source | Embeddings | Time (HF) | Time (DB) |
|--------|-----------|--------|------------|-----------|-----------|
| Small | 100 | 300 | Hugging Face | 1-2 min | 3-5 min |
| Medium | 1,000 | 3,000 | Hugging Face | 5-10 min | 15-30 min |
| Large | 5,000 | 15,000 | Hugging Face | 30-60 min | 90-180 min |

**Notes:**
- Hugging Face (HF) = Local embeddings, faster
- Databricks (DB) = API embeddings, slower but serverless

### Expected Match Rates by Stage

**Exact Match (Stage 1):**
- ~5-10% (entities with matching LEI/CUSIP/ISIN)

**Vector Search + Ditto (Stage 2-3):**
- ~60-70% (main matching engine)

**Foundation Model (Stage 4):**
- ~5-10% (edge cases)

**Total Match Rate:**
- ~70-90% (depends on match_ratio configuration)

---

## Ground Truth Generation

The notebook creates a `gold_standard` table with known matches for evaluation:

```sql
-- View ground truth
SELECT * FROM catalog.bronze.gold_standard LIMIT 10;

-- Count known matches
SELECT COUNT(*) as known_matches
FROM catalog.bronze.gold_standard
WHERE is_match = 1;
```

**Note:** Ground truth is based on exact ticker matches. For comprehensive evaluation, you may want to manually review and expand this.

---

## Evaluation Against Ground Truth

### Calculate Accuracy

```sql
WITH pipeline_results AS (
    SELECT source_id, source_system, matched_ciq_id
    FROM catalog.gold.matched_entities
),
ground_truth AS (
    SELECT source_id, source_system, ciq_id as true_ciq_id
    FROM catalog.bronze.gold_standard
)
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN pr.matched_ciq_id = gt.true_ciq_id THEN 1 ELSE 0 END) as correct,
    SUM(CASE WHEN pr.matched_ciq_id = gt.true_ciq_id THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
FROM ground_truth gt
LEFT JOIN pipeline_results pr
    ON gt.source_id = pr.source_id
    AND gt.source_system = pr.source_system;
```

---

## Troubleshooting

### Issue: Module not found

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```python
import sys
sys.path.append("/Workspace/Users/your_user/.bundle/entity_matching/dev/files")
```

### Issue: Table already exists

**Error:** `Table already exists`

**Solution:** Change write mode to "overwrite":
```python
.write.format("delta").mode("overwrite").saveAsTable(...)
```

Or in notebook widget: Set "Write Mode" to "overwrite"

### Issue: Out of memory

**Error:** `OutOfMemoryError` when generating large datasets

**Solution:**
- Generate in smaller batches
- Use larger cluster
- Or reduce dataset size

---

## Next Steps

1. **Load Test Data:**
   ```bash
   # Run the setup notebook
   notebooks/setup/03_load_large_test_dataset.py
   ```

2. **Verify Data:**
   ```sql
   SELECT COUNT(*) FROM catalog.bronze.spglobal_reference;
   SELECT COUNT(*) FROM catalog.bronze.source_entities;
   ```

3. **Run Pipeline:**
   ```bash
   ./deploy-phase.sh 4 dev
   ```

4. **Analyze Results:**
   ```sql
   SELECT * FROM catalog.gold.daily_stats;
   ```

5. **Compare vs Ground Truth:**
   ```sql
   -- Use evaluation queries above
   ```

---

## Customization

### Add Custom Industries

Edit `large_dataset_generator.py`:
```python
self.industries = [
    "Technology Hardware",
    "Software",
    "Your Custom Industry",  # Add here
    ...
]
```

### Add Custom Countries

```python
self.countries = [
    "United States",
    "Your Custom Country",  # Add here
    ...
]
```

### Add Custom Variation Types

Add new variation in `_create_entity_variation()`:
```python
elif variation_type == "your_custom_variation":
    # Your custom logic here
    base_entity["company_name"] = custom_transform(ref_entity["company_name"])
```

---

## Summary

✅ **Generates 1000+ reference entities** with realistic data
✅ **Generates 3000+ source entities** with variations
✅ **Configurable match ratios** for different test scenarios
✅ **Multiple variation types** (typos, abbreviations, etc.)
✅ **Ground truth generation** for evaluation
✅ **Scalable to 10,000+ entities** for stress testing

**Ready to use:** Run the setup notebook and start testing!

---

**Created:** 2026-01-27
**Version:** 1.0
**Status:** Production Ready
