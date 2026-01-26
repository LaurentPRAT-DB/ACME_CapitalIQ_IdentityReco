# Databricks Asset Bundle Path Alignment Verification

## Overview
This document verifies that all paths in the MLflow training code are properly aligned with Databricks Asset Bundle (DABs) deployment paths.

## DABs Configuration

### Bundle Root Path (dev target)
```yaml
workspace.root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev
```

**Evaluated to:**
```
/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev
```

### Catalog Name (dev target)
```yaml
catalog_name: laurent_prat_entity_matching_dev
```

---

## Phase 2: Model Training Job Configuration

From `resources/jobs_phase2_training.yml`:

### Task: generate_training_data

**Notebook:** `notebooks/02_train_ditto_model.py`

**Parameters passed by DABs:**
| Parameter | Value | Evaluated Path |
|-----------|-------|----------------|
| `workspace_path` | `${workspace.root_path}/files` | `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/files` |
| `catalog_name` | `${var.catalog_name}` | `laurent_prat_entity_matching_dev` |
| `num_positive_pairs` | `"1000"` | `1000` |
| `num_negative_pairs` | `"1000"` | `1000` |
| `output_path` | `${workspace.root_path}/training_data` | `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/training_data` |

---

## Notebook Path Usage

### ✅ FIXED: Notebook `02_train_ditto_model.py`

#### Training Data Output
- **Line 137:** `training_data_path = f"{output_path}/ditto_training_data.csv"`
- **Resolves to:** `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/training_data/ditto_training_data.csv`
- **Status:** ✅ Uses DABs parameter

#### Model Output
- **Line 178:** `model_output_path = f"{output_path}/models/ditto_matcher"`
- **Resolves to:** `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/training_data/models/ditto_matcher`
- **Status:** ✅ Uses DABs parameter

#### Test Data Output
- **Line 214:** `test_data_path = f"{output_path}/ditto_test_data.csv"`
- **Resolves to:** `/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/training_data/ditto_test_data.csv`
- **Status:** ✅ Uses DABs parameter

---

## MLflow Experiment Path

### Experiment Configuration
- **Line 158:** `experiment_name = f"{catalog_name}-ditto-model-training"`
- **Line 159:** `experiment_path = f"/Users/{username}/{experiment_name}"`

**Resolves to:**
```
/Users/laurent.prat@databricks.com/laurent_prat_entity_matching_dev-ditto-model-training
```

**Status:** ✅ Properly configured
- Uses user's workspace directory (always exists)
- Includes catalog name for isolation
- Flat structure (no nested subdirectories)

---

## Directory Structure After Deployment

```
/Workspace/Users/laurent.prat@databricks.com/
├── .bundle/
│   └── entity_matching/
│       └── dev/
│           ├── files/                          # Source code (synced by DABs)
│           │   ├── src/
│           │   │   ├── data/
│           │   │   ├── models/
│           │   │   ├── pipeline/
│           │   │   ├── evaluation/
│           │   │   └── utils/
│           │   ├── notebooks/
│           │   └── resources/
│           └── training_data/                  # Training artifacts (created by job)
│               ├── ditto_training_data.csv
│               ├── ditto_test_data.csv
│               └── models/
│                   └── ditto_matcher/
│                       ├── config.json
│                       ├── pytorch_model.bin
│                       └── tokenizer files
└── laurent_prat_entity_matching_dev-ditto-model-training/  # MLflow experiment
    └── [MLflow runs]
```

---

## Import Path Configuration

### Python Path Setup
**Line 62:** `sys.path.append(workspace_path)`

**Result:**
```python
sys.path.append("/Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/files")
```

### Imports Work As Expected
```python
from src.data.loader import DataLoader                      # ✅ Works
from src.data.training_generator import TrainingDataGenerator  # ✅ Works
from src.models.ditto_matcher import DittoMatcher           # ✅ Works
```

---

## Unity Catalog Alignment

### Catalog Structure
```
laurent_prat_entity_matching_dev/
├── bronze/
│   ├── spglobal_reference      # Reference data
│   └── source_entities         # Source entities to match
├── silver/
│   └── entities_normalized     # Preprocessed entities
└── gold/
    └── matched_entities        # Final matching results
```

### Model Registration
- **Model Name:** `entity_matching_ditto` (registered in MLflow)
- **Catalog Path:** `laurent_prat_entity_matching_dev.models.entity_matching_ditto`

---

## Verification Checklist

- [x] **Workspace paths use DABs parameters** (not hardcoded)
- [x] **Output paths use DABs parameters** (not hardcoded)
- [x] **MLflow experiments use proper user workspace paths**
- [x] **Catalog names match between DABs config and notebooks**
- [x] **Source code imports work with workspace_path**
- [x] **Directory creation happens automatically** (Path.mkdir with parents=True)
- [x] **No hardcoded `/dbfs/entity_matching` paths remaining** (except defaults)

---

## Known Limitations

### Notebook 03 (full_pipeline_example.py)
- **Line 157:** Hardcoded path `/dbfs/entity_matching/gold_standard.csv` for testing
- **Impact:** Low - used only for manual testing/validation
- **Future Fix:** Add `test_data_path` parameter to DABs job config

---

## Testing Commands

### Deploy Bundle
```bash
databricks bundle deploy -t dev
```

### Run Training Job
```bash
databricks bundle run train_ditto_model -t dev
```

### Verify Paths
```bash
databricks workspace ls /Workspace/Users/laurent.prat@databricks.com/.bundle/entity_matching/dev/training_data
```

---

## Troubleshooting

### Issue: "Cannot save file into a non-existent directory"
**Resolution:** Fixed - All save methods now auto-create parent directories

### Issue: "Could not find experiment with ID None"
**Resolution:** Fixed - MLflow experiments now properly configured before runs

### Issue: Imports not working
**Resolution:** Verify `workspace_path` parameter is passed and `sys.path.append()` executes

---

## Summary

✅ **All critical paths are now aligned with DABs deployment**
✅ **MLflow experiments properly configured**
✅ **Directory creation automated**
✅ **Imports working correctly**

The training job should now run successfully end-to-end with proper path management.
