# Databricks Asset Bundles Migration Assistant

**Skill Name:** `databricks-dabs-migration`
**Version:** 1.0.0
**Purpose:** Systematically migrate local Python/Spark code to Databricks Asset Bundles (DABs)

---

## Invocation

Use this skill when you need to:
- Port local code to Databricks Asset Bundles
- Fix DABs deployment issues
- Review path and dependency alignment
- Validate DABs configuration
- Troubleshoot bundle deployment errors

**How to invoke:**
```
/databricks-dabs-migration
```

Or with a specific focus:
```
/databricks-dabs-migration --focus paths
/databricks-dabs-migration --focus dependencies
/databricks-dabs-migration --focus mlflow
```

---

## Execution Phases

### Phase 1: Discovery & Analysis

**Objective:** Understand the current codebase structure and DABs setup

**Tasks:**
1. **Scan Repository Structure**
   ```bash
   find . -type f -name "*.py" | head -20
   find . -type f -name "*.yml" -o -name "*.yaml"
   find . -type f -name "*.ipynb"
   find . -name "requirements.txt" -o -name "setup.py" -o -name "pyproject.toml"
   ```

2. **Identify DABs Configuration**
   - Check for `databricks.yml`
   - Look for `resources/*.yml` files
   - Find job/pipeline configurations

3. **Map Current Organization**
   - Source code structure (Python modules)
   - Notebook organization
   - Configuration files
   - Test files
   - Documentation

**Output:** Create `MIGRATION_DISCOVERY.md` with:
- Directory tree
- File inventory
- Current vs. target structure
- DABs configuration status

---

### Phase 2: Path Analysis

**Objective:** Identify and fix hardcoded paths

**Detection Patterns:**
```python
# Hardcoded paths to find:
- /dbfs/...
- /tmp/...
- /Users/username/...
- C:\\ or D:\\ (Windows)
- ./relative/paths (without proper handling)
```

**Tasks:**
1. **Find All File I/O Operations**
   ```bash
   grep -r "open(" --include="*.py"
   grep -r "to_csv\|to_parquet\|to_json" --include="*.py"
   grep -r "read_csv\|read_parquet\|read_json" --include="*.py"
   grep -r "Path(" --include="*.py"
   grep -r "os.path.join" --include="*.py"
   ```

2. **Find MLflow Paths**
   ```bash
   grep -r "mlflow.set_experiment" --include="*.py"
   grep -r "mlflow.log_artifact" --include="*.py"
   grep -r "mlflow.log_model" --include="*.py"
   ```

3. **Find Unity Catalog References**
   ```bash
   grep -r "spark.table\|spark.sql" --include="*.py"
   grep -r "catalog\." --include="*.py"
   ```

**Analysis:**
For each hardcoded path:
- Current value
- Expected DABs parameter
- Suggested refactoring
- Impact assessment

**Output:** Create `PATH_MIGRATION_PLAN.md` with:
- Hardcoded path inventory
- Suggested parameter mappings
- Code change recommendations
- DABs YAML parameter additions

---

### Phase 3: Dependency Analysis

**Objective:** Ensure library compatibility with Databricks

**Tasks:**
1. **Scan Dependency Files**
   - Read `requirements.txt`
   - Read `setup.py` install_requires
   - Read `pyproject.toml` dependencies

2. **Check Known Compatibility Issues**

   **Critical Checks:**
   - `transformers` version (>=4.40.0 for Python 3.10 type hints)
   - `sentence-transformers` version (>=2.3.0)
   - `torch` version (>=2.1.0 for Databricks GPU)
   - `pyspark` version (must match DBR version)
   - `mlflow` version (>=2.9.0 for Unity Catalog)
   - `pandas` version (>=2.0.0 recommended)

3. **Detect Type Annotation Issues**
   ```bash
   # Find files with type hints
   grep -r "from typing import" --include="*.py"

   # Check for lowercase type hints (list[], dict[], etc.)
   grep -r ": list\[" --include="*.py"
   grep -r "-> list\[" --include="*.py"
   grep -r "-> dict\[" --include="*.py"

   # Check for future annotations import
   grep -r "from __future__ import annotations" --include="*.py"
   ```

4. **Python Version Compatibility**
   - Check Python version requirements
   - Verify Databricks Runtime compatibility
   - Identify deprecated features

**Analysis:**
For each dependency:
- Current version
- Required version for Databricks
- Compatibility issues
- Upgrade recommendations

**Output:** Create `DEPENDENCY_MIGRATION_PLAN.md` with:
- Library version matrix
- Upgrade commands
- Type annotation fixes needed
- Python compatibility notes

---

### Phase 4: Configuration Review

**Objective:** Verify DABs YAML structure and parameter flow

**Tasks:**
1. **Analyze databricks.yml**
   - Bundle name and targets
   - Workspace paths
   - Variables and their defaults
   - Sync configuration

2. **Review Job Configurations**
   - Job cluster definitions
   - Task definitions
   - Parameter passing
   - Dependency chains

3. **Trace Parameter Flow**

   **Check flow:** `databricks.yml` → `jobs/*.yml` → `notebook parameters`

   Example verification:
   ```yaml
   # In databricks.yml
   variables:
     catalog_name: dev_catalog

   # In jobs/training.yml
   base_parameters:
     catalog_name: ${var.catalog_name}

   # In notebook
   catalog_name = dbutils.widgets.get("catalog_name")
   ```

4. **Validate Naming Consistency**
   - Catalog names match
   - Schema names match
   - Table names consistent
   - Experiment names follow pattern

**Output:** Create `CONFIG_REVIEW.md` with:
- Parameter flow diagram
- Missing parameters
- Naming inconsistencies
- YAML structure improvements

---

### Phase 5: Code Quality Checks

**Objective:** Find common coding issues for DABs deployment

**Tasks:**
1. **Directory Creation Patterns**
   ```python
   # Find file writes without directory creation
   grep -B5 "to_csv\|to_parquet" --include="*.py" | grep -v "mkdir"
   ```

2. **MLflow Experiment Setup**
   ```python
   # Check if experiments are set before runs
   grep -B10 "mlflow.start_run" --include="*.py" | grep "mlflow.set_experiment"
   ```

3. **Import Path Configuration**
   ```python
   # Check for sys.path modifications
   grep -r "sys.path.append" --include="*.py"

   # Verify workspace_path usage
   grep -r "workspace_path" --include="*.py"
   ```

4. **Widget Parameter Usage**
   ```python
   # Find widget definitions
   grep -r "dbutils.widgets.text" --include="*.py"

   # Find widget gets
   grep -r "dbutils.widgets.get" --include="*.py"
   ```

**Common Issues to Fix:**
- ❌ File writes without `Path(...).parent.mkdir(parents=True, exist_ok=True)`
- ❌ MLflow runs without `mlflow.set_experiment()` first
- ❌ Hardcoded experiment paths like `/Shared/experiments`
- ❌ Widget parameters not matching job configuration
- ❌ Missing `%pip install --upgrade` for package updates

**Output:** Create `CODE_QUALITY_FIXES.md` with:
- Issue categories
- Code examples (before/after)
- File-by-file fix list
- Priority ranking

---

### Phase 6: Migration Plan Generation

**Objective:** Create actionable migration steps

**Generate:**

1. **Step-by-Step Migration Guide**
   ```markdown
   ## Migration Steps

   ### Step 1: Update Dependencies
   - [ ] Update requirements.txt
   - [ ] Add `from __future__ import annotations` to all .py files
   - [ ] Test imports locally

   ### Step 2: Refactor Paths
   - [ ] Replace hardcoded paths with parameters
   - [ ] Update file I/O methods to create directories
   - [ ] Test path resolution

   ### Step 3: Configure DABs
   - [ ] Update databricks.yml with parameters
   - [ ] Create/update job configurations
   - [ ] Set up catalog/schema variables

   ### Step 4: Fix MLflow Configuration
   - [ ] Add experiment setup before runs
   - [ ] Update artifact logging paths
   - [ ] Configure model registry

   ### Step 5: Test Deployment
   - [ ] Run databricks bundle validate
   - [ ] Deploy to dev: databricks bundle deploy -t dev
   - [ ] Run smoke tests

   ### Step 6: Verify Functionality
   - [ ] Check file outputs
   - [ ] Verify MLflow experiments
   - [ ] Validate Unity Catalog writes
   ```

2. **Updated DABs YAML Templates**
   Generate complete YAML files with proper parameters

3. **Code Refactoring Patterns**

   **Pattern 1: Path Parameterization**
   ```python
   # Before
   output_path = "/dbfs/my_project/output.csv"

   # After
   output_path = dbutils.widgets.get("output_path")
   final_path = f"{output_path}/output.csv"
   ```

   **Pattern 2: Directory Creation**
   ```python
   # Before
   df.to_csv(filepath, index=False)

   # After
   from pathlib import Path
   Path(filepath).parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(filepath, index=False)
   ```

   **Pattern 3: MLflow Experiment Setup**
   ```python
   # Before
   with mlflow.start_run():
       mlflow.log_param("param", value)

   # After
   username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
   experiment_name = f"{catalog_name}-my-experiment"
   experiment_path = f"/Users/{username}/{experiment_name}"
   mlflow.set_experiment(experiment_path)

   with mlflow.start_run(run_name="my-run"):
       mlflow.log_param("param", value)
   ```

**Output:** Create `MIGRATION_GUIDE.md` with complete instructions

---

### Phase 7: Documentation & Validation

**Objective:** Create comprehensive migration documentation

**Generate:**

1. **Path Mapping Documentation**
   ```markdown
   # Path Mapping Reference

   | Original Path | DABs Parameter | Final Path |
   |---------------|----------------|------------|
   | /dbfs/data/input.csv | ${workspace.root_path}/data | /Workspace/Users/user/.bundle/project/dev/data/input.csv |
   ```

2. **Testing Commands**
   ```bash
   # Validate bundle
   databricks bundle validate -t dev

   # Deploy bundle
   databricks bundle deploy -t dev

   # Run specific job
   databricks bundle run my_job -t dev

   # Check deployment
   databricks workspace ls /Workspace/Users/$USER/.bundle/project/dev
   ```

3. **Troubleshooting Guide**
   Common errors and solutions:
   - RestException: RESOURCE_DOES_NOT_EXIST
   - TypeError: 'type' object is not subscriptable
   - OSError: Cannot save file into a non-existent directory
   - ImportError: No module named 'src'

4. **Verification Checklist**
   ```markdown
   ## Pre-Deployment
   - [ ] All dependencies updated
   - [ ] No hardcoded paths in code
   - [ ] DABs YAML validates successfully
   - [ ] MLflow experiments configured
   - [ ] Directory creation in place

   ## Post-Deployment
   - [ ] Bundle deploys without errors
   - [ ] Files created in correct locations
   - [ ] MLflow experiments appear in workspace
   - [ ] Unity Catalog writes succeed
   - [ ] Imports work correctly
   ```

**Output:** Create `DATABRICKS_MIGRATION_COMPLETE.md` as final reference

---

## Skill Options

### Focus Areas
- `--focus paths` - Only analyze and fix path issues
- `--focus dependencies` - Only check library compatibility
- `--focus mlflow` - Only review MLflow configuration
- `--focus config` - Only validate DABs YAML
- `--focus all` - Complete migration (default)

### Modes
- `--mode analysis` - Only analyze, don't generate fixes
- `--mode plan` - Generate migration plan without code changes
- `--mode execute` - Analyze and apply fixes (default)
- `--mode validate` - Validate existing migration

### Output
- `--output docs` - Generate markdown documentation
- `--output code` - Generate code fixes
- `--output yaml` - Generate DABs configurations
- `--output all` - Generate everything (default)

---

## Key Patterns to Detect

### Anti-Patterns (Bad)
```python
# ❌ Hardcoded paths
path = "/dbfs/my_project/data.csv"

# ❌ No directory creation
df.to_csv(path)

# ❌ Missing MLflow experiment setup
with mlflow.start_run():
    ...

# ❌ Hardcoded experiment paths
mlflow.set_experiment("/Shared/experiments/my_exp")

# ❌ Type hints without future import
def process(items: list[str]) -> dict[str, int]:
    ...

# ❌ Missing --upgrade flag
%pip install transformers>=4.36.0
```

### Best Practices (Good)
```python
# ✅ Parameterized paths
path = f"{output_path}/data.csv"

# ✅ Auto-create directories
Path(path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(path)

# ✅ Proper MLflow setup
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_path = f"/Users/{username}/{catalog_name}-experiment"
mlflow.set_experiment(experiment_path)

# ✅ Future annotations for compatibility
from __future__ import annotations

def process(items: list[str]) -> dict[str, int]:
    ...

# ✅ Force package upgrade
%pip install --upgrade transformers>=4.40.0
```

---

## Success Criteria

Migration is complete when:
- ✅ `databricks bundle validate` passes
- ✅ `databricks bundle deploy -t dev` succeeds
- ✅ All jobs run without path errors
- ✅ MLflow experiments appear correctly
- ✅ Files are created in expected locations
- ✅ Imports work without sys.path hacks
- ✅ No hardcoded paths remain
- ✅ All dependencies compatible

---

## Example Usage

### Quick Migration
```bash
# Analyze everything and create migration plan
/databricks-dabs-migration

# Review the generated files:
# - MIGRATION_DISCOVERY.md
# - PATH_MIGRATION_PLAN.md
# - DEPENDENCY_MIGRATION_PLAN.md
# - CONFIG_REVIEW.md
# - CODE_QUALITY_FIXES.md
# - MIGRATION_GUIDE.md
# - DATABRICKS_MIGRATION_COMPLETE.md
```

### Focused Analysis
```bash
# Only check paths
/databricks-dabs-migration --focus paths

# Only validate existing migration
/databricks-dabs-migration --mode validate
```

---

## Integration with Other Skills

This skill works well with:
- `/gsd:map-codebase` - Before migration, map the codebase
- `/databricks-authentication` - Set up Databricks CLI
- `databricks-apps` - If migrating to Databricks Apps
- `/gsd:verify-work` - Validate migration results

---

## Notes

- Always backup code before applying fixes
- Test in `dev` target before `staging` or `prod`
- Review generated fixes before committing
- Keep migration documentation for team reference
- Update skill based on new patterns discovered

---

## Version History

**v1.0.0** (2026-01-25)
- Initial release
- Path analysis and refactoring
- Dependency compatibility checks
- MLflow configuration validation
- Type annotation fixes
- Complete migration workflow
