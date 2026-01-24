# Databricks Bundle - Quick Start

**Deploy entity matching to Databricks in 5 minutes**

---

## ⚡ Quick Commands

```bash
# 1. Validate bundle
databricks bundle validate -t dev

# 2. Deploy to dev
databricks bundle deploy -t dev

# 3. Setup Unity Catalog
databricks bundle run setup_unity_catalog -t dev

# 4. Run test matching
databricks bundle run adhoc_entity_matching -t dev
```

---

## 📦 What Gets Deployed

### Jobs
- ✅ `setup_unity_catalog` - Create catalog, schemas, tables
- ✅ `train_ditto_model` - Train matching model
- ✅ `entity_matching_pipeline` - Production pipeline (scheduled daily)
- ✅ `adhoc_entity_matching` - On-demand matching

### Unity Catalog
- ✅ Catalog: `<user>_entity_matching_dev`
- ✅ Schemas: `bronze`, `silver`, `gold`, `models`
- ✅ Tables: Reference data, source entities, matched results

### Model Serving
- ✅ Endpoint: `entity-matching-ditto-dev` (when model is trained)

---

## 🎯 First Time Setup

```bash
# Install Databricks CLI (if needed)
pip install --upgrade databricks-cli

# Verify authentication
databricks auth env --profile DEFAULT

# Deploy
cd /path/to/MET_CapitalIQ_identityReco
databricks bundle deploy -t dev

# Setup catalog
databricks bundle run setup_unity_catalog -t dev
```

---

## 🔍 Verify Deployment

```bash
# List deployed resources
databricks bundle resources list -t dev

# Check in Databricks UI
# - Workflows → Jobs (4 jobs should be visible)
# - Data → Catalogs → <user>_entity_matching_dev
```

---

## 🚀 Run Your First Match

```bash
# Option 1: Use ad-hoc job
databricks bundle run adhoc_entity_matching -t dev

# Option 2: In Databricks notebook
# USE CATALOG <user>_entity_matching_dev;
# SELECT * FROM gold.matched_entities;
```

---

## 📚 Full Documentation

→ See [documentation/DATABRICKS_BUNDLE_DEPLOYMENT.md](documentation/DATABRICKS_BUNDLE_DEPLOYMENT.md)

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| Validation fails | Check YAML syntax errors |
| Permission denied | Request workspace admin access |
| Catalog not found | Run `setup_unity_catalog` job |
| No data in tables | Ingest source data first |

---

**Ready?** → `databricks bundle deploy -t dev`
