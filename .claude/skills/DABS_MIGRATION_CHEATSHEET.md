# Databricks Asset Bundles Migration - Quick Reference

## 🚀 Using the Skill

```bash
# Complete migration
/databricks-dabs-migration

# Focused analysis
/databricks-dabs-migration --focus paths
/databricks-dabs-migration --focus dependencies
/databricks-dabs-migration --focus mlflow
/databricks-dabs-migration --focus config
```

---

## 🔍 Common Issues & Fixes

### 1️⃣ Type Annotation Error
```
TypeError: 'type' object is not subscriptable
```

**Fix:** Add to **every** `.py` file:
```python
from __future__ import annotations
```

---

### 2️⃣ Hardcoded Paths
```python
# ❌ BAD
path = "/dbfs/my_project/data.csv"

# ✅ GOOD
path = f"{output_path}/data.csv"  # output_path from dbutils.widgets.get()
```

---

### 3️⃣ Directory Not Found
```
OSError: Cannot save file into a non-existent directory
```

**Fix:** Create directories first:
```python
from pathlib import Path
Path(filepath).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)
```

---

### 4️⃣ MLflow Experiment Error
```
RestException: Could not find experiment with ID None
```

**Fix:** Set experiment before run:
```python
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_path = f"/Users/{username}/{catalog_name}-experiment"
mlflow.set_experiment(experiment_path)

with mlflow.start_run(run_name="my-run"):
    mlflow.log_param("param", value)
```

---

### 5️⃣ Library Version Issues
```
# ❌ OLD (breaks in Databricks)
%pip install transformers>=4.36.0

# ✅ NEW (compatible)
%pip install --upgrade transformers>=4.40.0  # Note: --upgrade flag!
```

---

## 📋 Migration Checklist

### Before Deployment
- [ ] Add `from __future__ import annotations` to all `.py` files
- [ ] Replace hardcoded paths with DABs parameters
- [ ] Update library versions in `requirements.txt`
- [ ] Add `--upgrade` to notebook pip installs
- [ ] Add directory creation before file writes
- [ ] Set MLflow experiments before runs
- [ ] Validate bundle: `databricks bundle validate -t dev`

### After Deployment
- [ ] Deploy succeeds: `databricks bundle deploy -t dev`
- [ ] Jobs run without errors
- [ ] Files created in correct locations
- [ ] MLflow experiments appear in workspace
- [ ] Unity Catalog writes succeed
- [ ] Imports work correctly

---

## 🗂️ DABs Parameter Flow

```
databricks.yml
  variables:
    catalog_name: my_catalog
    ↓
resources/jobs.yml
  base_parameters:
    catalog_name: ${var.catalog_name}
    output_path: ${workspace.root_path}/output
    ↓
notebook.py
  catalog_name = dbutils.widgets.get("catalog_name")
  output_path = dbutils.widgets.get("output_path")
  final_path = f"{output_path}/data.csv"
```

---

## 📁 Directory Structure

```
/Workspace/Users/user@company.com/
├── .bundle/
│   └── project_name/
│       └── dev/
│           ├── files/              ← Source code (synced by DABs)
│           │   ├── src/
│           │   │   ├── data/
│           │   │   ├── models/
│           │   │   └── pipeline/
│           │   ├── notebooks/
│           │   └── resources/
│           └── output/             ← Job outputs (created by jobs)
│               ├── data/
│               ├── models/
│               └── training_data/
└── catalog_name-experiment/        ← MLflow experiments
    └── [runs]
```

---

## 🛠️ Essential Commands

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy bundle
databricks bundle deploy -t dev

# Run specific job
databricks bundle run job_name -t dev

# List deployed files
databricks workspace ls /Workspace/Users/$USER/.bundle/project/dev

# View job runs
databricks jobs list-runs --job-id <job-id>
```

---

## 🎯 Code Patterns

### Pattern: Path Parameterization
```python
# Setup (in notebook)
output_path = dbutils.widgets.get("output_path")

# Usage throughout code
training_data_path = f"{output_path}/training_data.csv"
model_path = f"{output_path}/models/my_model"
test_data_path = f"{output_path}/test_data.csv"
```

### Pattern: Safe File Writing
```python
def save_data(df, filepath):
    """Save DataFrame with automatic directory creation"""
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved to: {filepath}")
```

### Pattern: MLflow Experiment Setup
```python
# At the start of training notebook
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
catalog_name = dbutils.widgets.get("catalog_name")
experiment_name = f"{catalog_name}-training"
experiment_path = f"/Users/{username}/{experiment_name}"

mlflow.set_experiment(experiment_path)
print(f"Using experiment: {experiment_path}")
```

### Pattern: Widget Parameters
```python
# Define widgets with defaults (for interactive testing)
dbutils.widgets.text("workspace_path", "/Workspace/Users/user/.bundle/project/dev/files")
dbutils.widgets.text("catalog_name", "dev_catalog")
dbutils.widgets.text("output_path", "/Workspace/Users/user/.bundle/project/dev/output")

# Get values (overridden by job parameters)
workspace_path = dbutils.widgets.get("workspace_path")
catalog_name = dbutils.widgets.get("catalog_name")
output_path = dbutils.widgets.get("output_path")

# Add to Python path
import sys
sys.path.append(workspace_path)

# Now imports work
from src.data.loader import DataLoader
```

---

## 🔧 Library Versions

### Critical Updates
| Library | Old Version | New Version | Reason |
|---------|-------------|-------------|--------|
| transformers | 4.36.0 | **4.40.0+** | Python 3.10 type hints |
| sentence-transformers | 2.2.0 | **2.3.0+** | Compatibility |
| torch | 2.0.0 | **2.1.0+** | Databricks GPU |
| mlflow | 2.8.0 | **2.9.0+** | Unity Catalog |
| pandas | 1.5.0 | **2.0.0+** | Performance |

### Update Commands
```python
# In notebooks - ALWAYS use --upgrade
%pip install --upgrade transformers>=4.40.0 torch>=2.1.0 sentence-transformers>=2.3.0 scikit-learn mlflow
```

```bash
# In requirements.txt
transformers>=4.40.0
sentence-transformers>=2.3.0
torch>=2.1.0
mlflow>=2.9.0
pandas>=2.0.0
```

---

## ⚠️ Common Pitfalls

### ❌ DON'T
```python
# Hardcoded paths
path = "/dbfs/my_project/output.csv"

# No directory creation
df.to_csv(path)

# No experiment setup
with mlflow.start_run():
    pass

# Forget --upgrade
%pip install transformers>=4.40.0

# Variable shadowing
output_path = dbutils.widgets.get("output_path")
# ... later ...
output_path = "/dbfs/hardcoded/path"  # Overwrites parameter!
```

### ✅ DO
```python
# Parameterized paths
output_path = dbutils.widgets.get("output_path")
path = f"{output_path}/output.csv"

# Auto-create directories
Path(path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(path)

# Proper experiment setup
mlflow.set_experiment("/Users/user/experiment")
with mlflow.start_run(run_name="run-1"):
    pass

# Force upgrade
%pip install --upgrade transformers>=4.40.0

# Use different variable names
output_base = dbutils.widgets.get("output_path")
final_output_path = f"{output_base}/output.csv"
```

---

## 📊 Validation Steps

### 1. Pre-Deployment
```bash
# Check YAML syntax
databricks bundle validate -t dev

# Should output: "Validation OK!"
```

### 2. Post-Deployment
```bash
# Verify files synced
databricks workspace ls /Workspace/Users/$USER/.bundle/project/dev/files/src

# Check outputs can be created
databricks workspace mkdirs /Workspace/Users/$USER/.bundle/project/dev/output
```

### 3. Post-Execution
```bash
# Check job succeeded
databricks runs get --run-id <run-id>

# Verify outputs created
databricks workspace ls /Workspace/Users/$USER/.bundle/project/dev/output

# Check MLflow experiments
# Go to Databricks UI → Machine Learning → Experiments
```

---

## 🆘 Troubleshooting

### Issue: "RESOURCE_DOES_NOT_EXIST: Parent directory ... does not exist"
**Solution:** MLflow experiment path has nested directories. Use flat structure:
```python
# ❌ BAD
experiment_path = f"/Users/{username}/{catalog_name}/training"

# ✅ GOOD
experiment_path = f"/Users/{username}/{catalog_name}-training"
```

### Issue: "ImportError: No module named 'src'"
**Solution:** workspace_path not in sys.path:
```python
import sys
workspace_path = dbutils.widgets.get("workspace_path")
sys.path.append(workspace_path)
```

### Issue: Notebooks run but files not created
**Solution:** Check you're using output_path parameter, not hardcoded path:
```python
# Print to debug
output_path = dbutils.widgets.get("output_path")
print(f"Output path: {output_path}")
```

---

## 📚 Generated Documentation

After running `/databricks-dabs-migration`, review these files:

1. `MIGRATION_DISCOVERY.md` - Codebase overview
2. `PATH_MIGRATION_PLAN.md` - Path fixes needed
3. `DEPENDENCY_MIGRATION_PLAN.md` - Library updates
4. `CONFIG_REVIEW.md` - DABs YAML validation
5. `CODE_QUALITY_FIXES.md` - Code improvements
6. `MIGRATION_GUIDE.md` - Step-by-step instructions
7. `DATABRICKS_MIGRATION_COMPLETE.md` - Final reference

---

## 🎓 Quick Win: 5-Minute Check

```bash
# 1. Run the skill
/databricks-dabs-migration --mode analysis

# 2. Check for these issues:
grep -r "from __future__ import annotations" src/ | wc -l  # Should match number of .py files
grep -r "/dbfs/" --include="*.py" | grep -v "dbutils.widgets"  # Should be 0
grep -r "transformers>=" requirements.txt  # Should be >=4.40.0
grep -r "mlflow.set_experiment" --include="*.py" -B5 | grep "mlflow.start_run"  # Should find matches

# 3. Fix critical issues first:
# - Add future annotations
# - Update transformers version
# - Parameterize paths
# - Add MLflow experiment setup

# 4. Test
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

---

## ✨ Success Indicators

Your migration is successful when:
- ✅ `databricks bundle validate` passes
- ✅ `databricks bundle deploy -t dev` completes
- ✅ Jobs run without errors
- ✅ Files appear in expected locations
- ✅ MLflow experiments show up in workspace
- ✅ No hardcoded paths in code
- ✅ All imports work
- ✅ No type annotation errors

---

**Remember:** Test in `dev`, validate in `staging`, deploy to `prod`! 🚀
