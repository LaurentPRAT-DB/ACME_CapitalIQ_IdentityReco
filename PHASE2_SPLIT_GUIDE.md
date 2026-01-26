# Phase 2 Split Guide: Training vs Registration

## Overview

Phase 2 has been split into two independent phases to allow you to work on model registration without needing to retrain the model each time.

```
┌─────────────────────────────────────────────────────┐
│                    PHASE 2                          │
│              Model Training Only                    │
│                                                     │
│  • Generates training data                         │
│  • Trains Ditto model                              │
│  • Saves model to disk                             │
│  • Duration: 2-4 hours                             │
│                                                     │
│  Output: Model saved at                            │
│    /Workspace/Users/{user}/.bundle/                │
│      entity_matching/dev/training_data/            │
│        models/ditto_matcher/                       │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  PHASE 2b                           │
│        Model Registration & Evaluation              │
│                                                     │
│  • Loads trained model from disk                   │
│  • Registers to Unity Catalog                      │
│  • Sets Champion alias                             │
│  • Evaluates performance                           │
│  • Duration: ~1 hour                               │
│                                                     │
│  Can run independently multiple times              │
└─────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files
- `databricks-phase2b.yml` - Configuration for Phase 2b
- `resources/jobs_phase2b_registration.yml` - Phase 2b job definition
- `notebooks/02a_train_ditto_model.py` - Training only
- `notebooks/02b_register_evaluate_model.py` - Registration & evaluation
- `verify-model-saved.sh` - Script to verify model was saved

### Modified Files
- `deploy-phase.sh` - Added Phase 2b support
- `databricks-phase2.yml` - Removed registration task
- `resources/jobs_phase2_training.yml` - Training only

## Deployment Workflow

### Full Workflow (First Time)

```bash
# Step 1: Deploy catalog and data
./deploy-phase.sh 0 dev    # Catalog Setup (~5 min)
./deploy-phase.sh 1 dev    # Data Load (~10 min)

# Step 2: Train model (long-running)
./deploy-phase.sh 2 dev    # Model Training (2-4 hours)
# Wait for training to complete...

# Step 3: Register model
./deploy-phase.sh 2b dev   # Model Registration (~1 hour)

# Step 4: Deploy and run
./deploy-phase.sh 3 dev    # Model Serving (~5 min)
./deploy-phase.sh 4 dev    # Production Pipeline (~15 min)
```

### Working on Registration (After Training)

If you've already trained the model and need to fix/retry registration:

```bash
# Just run Phase 2b as many times as needed
./deploy-phase.sh 2b dev

# No need to retrain!
```

## Model Storage Location

The trained model is saved to workspace storage at:

**Dev Environment:**
```
/Workspace/Users/{your-email}/.bundle/entity_matching/dev/training_data/models/ditto_matcher/
```

**Staging Environment:**
```
/Workspace/Shared/.bundle/entity_matching/staging/training_data/models/ditto_matcher/
```

**Prod Environment:**
```
/Workspace/Shared/.bundle/entity_matching/prod/training_data/models/ditto_matcher/
```

## Model Files

The saved model directory contains:

```
ditto_matcher/
├── config.json              # Model configuration
├── pytorch_model.bin        # Trained weights
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.txt               # Vocabulary
├── special_tokens_map.json # Special tokens
└── tokenizer.json          # Tokenizer state
```

## Verification

### Verify Training Completed

After Phase 2 completes, check the job output for:

```
✅ Training complete!
   Model saved to: /Workspace/Users/{user}/.bundle/entity_matching/dev/training_data/models/ditto_matcher
   MLflow Run ID: {run_id}
```

### Verify Model Files Exist

```bash
./verify-model-saved.sh dev
```

Or manually check in Databricks Workspace UI:
1. Navigate to Workspace
2. Browse to: `/Workspace/Users/{your-email}/.bundle/entity_matching/dev/training_data/models/ditto_matcher`
3. Verify files listed above are present

### Verify Registration Completed

After Phase 2b completes, check:

```bash
# List models in Unity Catalog
databricks models list --profile LPT_FREE_EDITION | grep entity_matching_ditto

# Check model details and Champion alias
databricks models get laurent_prat_entity_matching_dev.models.entity_matching_ditto \
  --profile LPT_FREE_EDITION
```

Expected output should show:
- Model exists
- Has version 1 (or higher)
- Champion alias is set

## Troubleshooting

### Problem: Phase 2b can't find model

**Error:**
```
⚠ Model not found at: /Workspace/Users/{user}/.bundle/entity_matching/dev/training_data/models/ditto_matcher
```

**Solutions:**

1. **Verify Phase 2 completed:**
   ```bash
   # Check Phase 2 job status in Databricks UI
   # Look for "✅ Training complete!" in job logs
   ```

2. **Use custom model path:**
   - In Databricks UI, go to Phase 2b job
   - Add parameter: `custom_model_path` = your model path
   - Rerun the job

3. **Check model path in Phase 2 logs:**
   - Open Phase 2 job logs
   - Find the "Model saved to:" message
   - Use that exact path for Phase 2b

### Problem: Registration fails with MLflow error

**Common causes:**
- Unity Catalog schema doesn't exist
- Permissions issue
- Model format incompatible

**Solutions:**

1. **Ensure schema exists:**
   ```sql
   CREATE SCHEMA IF NOT EXISTS laurent_prat_entity_matching_dev.models;
   ```

2. **Check permissions:**
   - Verify you have CREATE MODEL permission on the schema
   - Check run_as user has proper access

3. **Retry registration:**
   ```bash
   # Phase 2b is safe to rerun
   ./deploy-phase.sh 2b dev
   ```

## Benefits of Split Approach

### Time Savings
- **Before:** 4 hours to retry registration (full retrain)
- **After:** 1 hour to retry registration (no retrain)

### Cost Savings
- Training uses GPU compute (~$5-10/hour)
- Registration uses serverless (~$0.10-0.50)
- Savings: ~$15-40 per retry

### Development Speed
- Fix registration bugs quickly
- Iterate on MLflow logging
- Test different model versions
- Experiment with Champion alias strategy

### Flexibility
- Register same trained model multiple times
- Try different registration approaches
- Test model serving configurations
- Develop registration logic independently

## Advanced Usage

### Register Specific Training Run

If you have multiple training runs and want to register a specific one:

```python
# In Phase 2b notebook, add parameter:
custom_model_path = "/Workspace/Users/{user}/.bundle/entity_matching/dev/training_data/models/ditto_matcher_20260126"
```

### Register to Different Catalog

```python
# Modify catalog_name parameter:
catalog_name = "entity_matching_staging"
```

### Skip Evaluation

Comment out evaluation section in `02b_register_evaluate_model.py` if you only want registration.

## Next Steps After Phase 2b

Once Phase 2b completes successfully:

1. **Verify Champion alias:**
   ```bash
   databricks models get {catalog}.models.entity_matching_ditto --profile LPT_FREE_EDITION
   ```

2. **Deploy serving endpoint:**
   ```bash
   ./deploy-phase.sh 3 dev
   ```

3. **Test endpoint:**
   ```python
   import mlflow.deployments
   client = mlflow.deployments.get_deploy_client("databricks")
   response = client.predict(
       endpoint="ditto-em-dev",
       inputs={"dataframe_split": {...}}
   )
   ```

4. **Run production pipeline:**
   ```bash
   ./deploy-phase.sh 4 dev
   ```

## Summary

The Phase 2 split provides:

✅ **Independent execution** - Train once, register many times
✅ **Faster iteration** - 1 hour vs 4 hours for registration fixes
✅ **Cost efficient** - No GPU costs for registration retries
✅ **Development friendly** - Easy to test registration changes
✅ **Production ready** - Same workflow for all environments

---

**Questions or Issues?**

Check the logs in Databricks UI for detailed error messages. The notebooks include comprehensive error handling and helpful messages to guide you through any issues.
