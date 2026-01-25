# Claude Code Skills for Databricks Development

This directory contains custom Claude Code skills for Databricks development workflows.

## Available Skills

### 🚀 databricks-dabs-migration

**Purpose:** Systematically migrate local Python/Spark code to Databricks Asset Bundles (DABs)

**When to use:**
- Porting existing code to Databricks
- Fixing DABs deployment issues
- Reviewing path and dependency alignment
- Validating bundle configurations
- Troubleshooting deployment errors

**How to invoke:**
```bash
/databricks-dabs-migration
```

**What it does:**
1. ✅ Scans your codebase structure
2. ✅ Identifies hardcoded paths and suggests parameterization
3. ✅ Checks library versions for Databricks compatibility
4. ✅ Validates DABs YAML configuration alignment
5. ✅ Detects common issues (type hints, directory creation, MLflow setup)
6. ✅ Generates step-by-step migration guide
7. ✅ Creates comprehensive documentation

**Output files:**
- `MIGRATION_DISCOVERY.md` - Codebase structure analysis
- `PATH_MIGRATION_PLAN.md` - Path refactoring recommendations
- `DEPENDENCY_MIGRATION_PLAN.md` - Library compatibility report
- `CONFIG_REVIEW.md` - DABs configuration analysis
- `CODE_QUALITY_FIXES.md` - Code issues and fixes
- `MIGRATION_GUIDE.md` - Step-by-step instructions
- `DATABRICKS_MIGRATION_COMPLETE.md` - Final reference doc

---

## Skill Usage Examples

### Complete Migration
```bash
# Run full migration analysis and planning
/databricks-dabs-migration

# Follow the generated MIGRATION_GUIDE.md
# Apply suggested fixes
# Deploy and validate
```

### Focused Analysis
```bash
# Only analyze paths
/databricks-dabs-migration --focus paths

# Only check dependencies
/databricks-dabs-migration --focus dependencies

# Only validate MLflow configuration
/databricks-dabs-migration --focus mlflow

# Only review DABs YAML
/databricks-dabs-migration --focus config
```

### Different Modes
```bash
# Analysis only (no fixes)
/databricks-dabs-migration --mode analysis

# Generate plan without code changes
/databricks-dabs-migration --mode plan

# Validate existing migration
/databricks-dabs-migration --mode validate
```

---

## Common Issues Detected

### 1. Type Annotation Compatibility
**Issue:** `TypeError: 'type' object is not subscriptable`

**Detection:**
```python
# Missing future import with lowercase type hints
def process(items: list[str]) -> dict[str, int]:
    ...
```

**Fix:**
```python
from __future__ import annotations

def process(items: list[str]) -> dict[str, int]:
    ...
```

---

### 2. Hardcoded Paths
**Issue:** Paths break when deployed to Databricks

**Detection:**
```python
# Hardcoded absolute paths
output_path = "/dbfs/my_project/output.csv"
mlflow.set_experiment("/Shared/experiments/my_exp")
```

**Fix:**
```python
# Use DABs parameters
output_path = dbutils.widgets.get("output_path")
final_path = f"{output_path}/output.csv"

# Use user workspace for experiments
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_path = f"/Users/{username}/{catalog_name}-experiment"
mlflow.set_experiment(experiment_path)
```

---

### 3. Directory Creation
**Issue:** `OSError: Cannot save file into a non-existent directory`

**Detection:**
```python
# File write without directory creation
df.to_csv(filepath, index=False)
```

**Fix:**
```python
# Auto-create parent directories
from pathlib import Path
Path(filepath).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)
```

---

### 4. Library Version Compatibility
**Issue:** Old library versions incompatible with Databricks runtime

**Detection:**
```
transformers>=4.36.0  # Too old for Python 3.10
```

**Fix:**
```
transformers>=4.40.0  # Compatible with Python 3.10
```

And in notebooks:
```python
# Force upgrade (critical!)
%pip install --upgrade transformers>=4.40.0
```

---

### 5. MLflow Experiment Setup
**Issue:** `RestException: RESOURCE_DOES_NOT_EXIST: Could not find experiment with ID None`

**Detection:**
```python
# Missing experiment setup
with mlflow.start_run():
    mlflow.log_param("param", value)
```

**Fix:**
```python
# Set experiment before run
mlflow.set_experiment("/Users/user/my-experiment")
with mlflow.start_run(run_name="my-run"):
    mlflow.log_param("param", value)
```

---

## Quick Reference

### DABs Parameter Flow
```
databricks.yml (variables)
    ↓
resources/jobs.yml (${var.variable_name})
    ↓
notebook base_parameters
    ↓
dbutils.widgets.get("parameter_name")
```

### Path Structure
```
/Workspace/Users/user@company.com/
├── .bundle/project_name/dev/
│   ├── files/                    # Source code (synced)
│   │   ├── src/
│   │   ├── notebooks/
│   │   └── resources/
│   └── output/                   # Job outputs
│       ├── data/
│       └── models/
└── catalog_name-experiment/      # MLflow experiments
```

### Essential Commands
```bash
# Validate bundle
databricks bundle validate -t dev

# Deploy bundle
databricks bundle deploy -t dev

# Run job
databricks bundle run job_name -t dev

# Check files
databricks workspace ls /Workspace/Users/$USER/.bundle/project/dev
```

---

## Best Practices

### ✅ DO
- Use DABs parameters for all paths
- Create directories before writing files
- Set MLflow experiments before runs
- Use `--upgrade` flag for pip installs
- Add `from __future__ import annotations` to all Python files
- Test in `dev` before `staging`/`prod`

### ❌ DON'T
- Hardcode paths like `/dbfs/...`
- Skip directory creation
- Use nested experiment paths
- Forget to restart Python after pip installs
- Use outdated library versions
- Deploy directly to production

---

## Troubleshooting

### Bundle Validation Fails
```bash
# Check YAML syntax
databricks bundle validate -t dev

# Review error messages
# Common issues: missing parameters, invalid paths
```

### Deployment Fails
```bash
# Check workspace permissions
# Verify catalog exists
# Ensure cluster configuration is valid
```

### Job Execution Fails
```bash
# Check job logs in Databricks UI
# Verify parameters are passed correctly
# Test notebook interactively first
```

---

## Getting Help

1. Run the migration skill: `/databricks-dabs-migration`
2. Review generated documentation
3. Check the troubleshooting section
4. Test incrementally in dev environment
5. Consult Databricks Asset Bundles documentation

---

## Contributing

To improve this skill:
1. Identify new patterns or issues
2. Update the skill markdown
3. Add examples to this README
4. Share with the team

---

## Resources

- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [Databricks Runtime Compatibility](https://docs.databricks.com/release-notes/runtime/index.html)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
