# Test Data Loading - Quick Reference

## 🚀 Quick Commands

### Deploy Phase 1 (One-Time Setup)
```bash
./deploy-phase.sh 1 dev
```

### Load Test Data

**Small test (100 ref, 300 source) - ~1 minute:**
```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=100 \
  --param num_source_entities=300
```

**Standard test (1000 ref, 3000 source) - ~3 minutes:**
```bash
databricks bundle run load_large_test_data -t dev
```

**Large test (5000 ref, 15000 source) - ~10 minutes:**
```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=5000 \
  --param num_source_entities=15000
```

---

## 📊 What Gets Loaded

| Dataset | Table | Default Size |
|---------|-------|--------------|
| Reference | `bronze.spglobal_reference` | 1,000 entities |
| Source | `bronze.source_entities` | 3,000 entities |
| Ground Truth | `bronze.gold_standard` | ~500 matches |

**Match Ratio:** 70% (2,100 should match, 900 won't match)

---

## 🎛️ Parameter Reference

```bash
databricks bundle run load_large_test_data -t dev \
  --param num_reference_entities=1000 \    # 100-10000+
  --param num_source_entities=3000 \       # 300-30000+
  --param match_ratio=0.7 \                # 0.0-1.0
  --param mode=overwrite                   # append|overwrite
```

---

## ✅ Verify Data Loaded

```sql
-- Check counts
SELECT COUNT(*) FROM catalog.bronze.spglobal_reference;  -- 1000
SELECT COUNT(*) FROM catalog.bronze.source_entities;     -- 3000

-- Sample data
SELECT * FROM catalog.bronze.spglobal_reference LIMIT 5;
SELECT * FROM catalog.bronze.source_entities LIMIT 5;
```

---

## 🔄 Complete Workflow

```bash
# 1. Deploy Phase 1
./deploy-phase.sh 1 dev

# 2. Create tables (first time only)
databricks bundle run load_reference_data -t dev

# 3. Load test data
databricks bundle run load_large_test_data -t dev

# 4. Run pipeline
./deploy-phase.sh 4 dev
databricks bundle run entity_matching_pipeline -t dev

# 5. Check results
# SQL: SELECT * FROM catalog.gold.matched_entities;
```

---

## 📚 Full Documentation

- **DAB_TEST_DATA_LOADING_GUIDE.md** - Complete guide
- **QUICK_START_LARGE_SCALE_TEST.md** - Getting started
- **LARGE_SCALE_TEST_DATA_GUIDE.md** - Detailed reference

---

**Quick Start:** `./deploy-phase.sh 1 dev` → `databricks bundle run load_large_test_data -t dev`
