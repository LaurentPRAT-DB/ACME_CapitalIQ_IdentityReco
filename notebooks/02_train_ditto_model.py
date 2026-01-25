# Databricks notebook source
# MAGIC %md
# MAGIC # Train Ditto Entity Matching Model
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Generate training data from S&P 500 gold standard
# MAGIC 2. Fine-tune Ditto model
# MAGIC 3. Evaluate model performance
# MAGIC 4. Deploy to MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install --upgrade transformers>=4.40.0 torch>=2.1.0 sentence-transformers>=2.3.0 scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

# Get parameters from job (set by DABs)
dbutils.widgets.text("workspace_path", "/Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev/files")
dbutils.widgets.text("catalog_name", "entity_matching")
dbutils.widgets.text("num_positive_pairs", "1000")
dbutils.widgets.text("num_negative_pairs", "1000")
dbutils.widgets.text("output_path", "/Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev/training_data")

workspace_path = dbutils.widgets.get("workspace_path")
catalog_name = dbutils.widgets.get("catalog_name")
num_positive_pairs = int(dbutils.widgets.get("num_positive_pairs"))
num_negative_pairs = int(dbutils.widgets.get("num_negative_pairs"))
output_path = dbutils.widgets.get("output_path")

print(f"Using catalog: {catalog_name}")
print(f"Workspace path: {workspace_path}")
print(f"Output path: {output_path}")

# Debug: Check if workspace_path exists
print(f"\nChecking workspace path...")
try:
    # Use os.listdir for workspace paths, not dbutils.fs.ls
    if os.path.exists(workspace_path):
        files = os.listdir(workspace_path)
        print(f"Files in workspace_path: {files[:10]}")
    else:
        print(f"Warning: workspace_path does not exist: {workspace_path}")
except Exception as e:
    print(f"Error listing workspace_path: {e}")

# Debug: Check sys.path
print(f"\nCurrent sys.path: {sys.path[:5]}")

# Add the workspace path to sys.path so we can import from src
sys.path.append(workspace_path)
print(f"Added to sys.path: {workspace_path}")

# Debug: Check if src exists
src_path = os.path.join(workspace_path, "src")
print(f"\nChecking if src exists at: {src_path}")
print(f"Path exists: {os.path.exists(src_path)}")
if os.path.exists(src_path):
    print(f"Contents: {os.listdir(src_path)[:10]}")

# Verify transformers version
import transformers
print(f"Transformers version: {transformers.__version__}")
if tuple(map(int, transformers.__version__.split('.')[:2])) < (4, 40):
    raise RuntimeError(f"transformers version {transformers.__version__} is too old. Need >= 4.40.0")

from src.data.loader import DataLoader
from src.data.training_generator import TrainingDataGenerator
from src.models.ditto_matcher import DittoMatcher
import mlflow

# Initialize data loader
loader = DataLoader()

print("\n✅ Successfully imported all modules")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Reference Data

# COMMAND ----------

# Load S&P Capital IQ reference data from Unity Catalog
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference")

print(f"Loaded {reference_df.count()} reference entities from {catalog_name}.bronze.spglobal_reference")
display(reference_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Training Data

# COMMAND ----------

# Initialize training data generator
generator = TrainingDataGenerator(seed=42)

# Generate training pairs from S&P 500
# Convert Spark DataFrame to pandas for the generator
reference_pdf = reference_df.toPandas()

training_df = generator.generate_from_sp500(
    reference_df=reference_pdf,
    num_positive_pairs=num_positive_pairs,
    num_negative_pairs=num_negative_pairs
)

print(f"\nTraining data shape: {training_df.shape}")
display(training_df.head(10))

# COMMAND ----------

# Optionally augment training data
training_df_augmented = generator.augment_training_data(
    training_df,
    augmentation_factor=0.2
)

print(f"Augmented training data: {len(training_df)} -> {len(training_df_augmented)} pairs")

# COMMAND ----------

# Save training data using the output_path from DABs
training_data_path = f"{output_path}/ditto_training_data.csv"
print(f"Saving training data to: {training_data_path}")
loader.save_training_data(training_df_augmented, training_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train Ditto Model

# COMMAND ----------

# Initialize Ditto matcher
ditto = DittoMatcher(
    base_model="distilbert-base-uncased",
    max_length=256
)

# COMMAND ----------

# Setup MLflow experiment
# Use a simple experiment path under the user's workspace
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_name = f"{catalog_name}-ditto-model-training"
experiment_path = f"/Users/{username}/{experiment_name}"

# Set or create the experiment
mlflow.set_experiment(experiment_path)
print(f"Using MLflow experiment: {experiment_path}")

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name="ditto-entity-matcher"):
    # Log parameters
    mlflow.log_param("base_model", "distilbert-base-uncased")
    mlflow.log_param("max_length", 256)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 3e-5)
    mlflow.log_param("training_pairs", len(training_df_augmented))

    # Train model - use a subfolder under output_path for the model
    model_output_path = f"{output_path}/models/ditto_matcher"
    print(f"Training model, will save to: {model_output_path}")

    ditto.train(
        training_data_path=training_data_path,
        output_path=model_output_path,
        epochs=1,
        batch_size=64,
        learning_rate=3e-5,
        val_split=0.2
    )

    # Log model using MLflow's native transformers flavor
    # This properly handles both model and tokenizer
    components = {
        "model": ditto.model,
        "tokenizer": ditto.tokenizer
    }

    # Create signature for model serving
    import pandas as pd
    from mlflow.models import infer_signature

    # Create sample input for signature inference
    sample_input = pd.DataFrame({
        "left_entity": ["COL name VAL Apple Inc. COL ticker VAL AAPL"],
        "right_entity": ["COL name VAL Apple Computer COL ticker VAL AAPL"]
    })

    # Log using transformers flavor
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="ditto-model",
        registered_model_name="entity_matching_ditto",
        task="text-classification",  # Specify the task type
        pip_requirements=[
            "transformers>=4.40.0",
            "torch>=2.1.0",
            "sentencepiece",  # Required for some tokenizers
        ]
    )

    print(f"Model saved to {model_output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate Model

# COMMAND ----------

# Create test set (separate from training)
test_df = generator.generate_from_sp500(
    reference_df=reference_df,
    num_positive_pairs=100,
    num_negative_pairs=100
)

test_data_path = f"{output_path}/ditto_test_data.csv"
print(f"Saving test data to: {test_data_path}")
# Ensure parent directory exists
from pathlib import Path
Path(test_data_path).parent.mkdir(parents=True, exist_ok=True)
test_df.to_csv(test_data_path, index=False)

# COMMAND ----------

# Evaluate on test set
metrics = ditto.evaluate(test_data_path)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
print("=" * 60)

# Log metrics to MLflow (create new run for evaluation)
with mlflow.start_run(run_name="ditto-entity-matcher-evaluation"):
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Model on Sample Pairs

# COMMAND ----------

# Test on specific examples
test_pairs = [
    (
        "COL name VAL Apple Inc. COL ticker VAL AAPL COL industry VAL Technology Hardware",
        "COL name VAL Apple Computer Inc. COL ticker VAL AAPL COL industry VAL Consumer Electronics"
    ),
    (
        "COL name VAL Microsoft Corporation COL ticker VAL MSFT",
        "COL name VAL Apple Inc. COL ticker VAL AAPL"
    ),
    (
        "COL name VAL Amazon.com Inc COL ticker VAL AMZN",
        "COL name VAL Amazon COL industry VAL E-commerce"
    )
]

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i, (left, right) in enumerate(test_pairs, 1):
    prediction, confidence = ditto.predict(left, right)
    match_label = "MATCH" if prediction == 1 else "NO MATCH"

    print(f"\nPair {i}:")
    print(f"  Left:  {left[:80]}...")
    print(f"  Right: {right[:80]}...")
    print(f"  Prediction: {match_label} (confidence: {confidence:.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Create serving endpoint
endpoint_name = "ditto-entity-matcher"

w.serving_endpoints.create(
    name=endpoint_name,
    config={
        "served_models": [{
            "model_name": "entity_matching_ditto",
            "model_version": "1",
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }]
    }
)

print(f"Serving endpoint '{endpoint_name}' created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Test deployed endpoint**: Query the serving endpoint with sample data
# MAGIC 2. **Integrate with pipeline**: Update `HybridMatchingPipeline` with Ditto model path
# MAGIC 3. **Monitor performance**: Track inference latency and accuracy
# MAGIC 4. **Retrain periodically**: Update model as new training data becomes available
