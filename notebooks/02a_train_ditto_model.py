# Databricks notebook source
# MAGIC %md
# MAGIC # Train Ditto Entity Matching Model - Part 1: Training
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Generates training data from S&P 500 gold standard
# MAGIC 2. Fine-tunes Ditto model
# MAGIC 3. Saves trained model to workspace storage
# MAGIC
# MAGIC **Note:** This is separated from model registration to avoid retraining when registration fails

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install --upgrade transformers>=4.40.0 torch>=2.1.0 sentence-transformers>=2.3.0 scikit-learn mlflow databricks-sdk huggingface-hub

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
dbutils.widgets.text("bundle_target", "dev")

workspace_path = dbutils.widgets.get("workspace_path")
catalog_name = dbutils.widgets.get("catalog_name")
num_positive_pairs = int(dbutils.widgets.get("num_positive_pairs"))
num_negative_pairs = int(dbutils.widgets.get("num_negative_pairs"))
output_path = dbutils.widgets.get("output_path")
bundle_target = dbutils.widgets.get("bundle_target")

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
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

experiment_name = f"{catalog_name}-ditto-model-training"
experiment_path = f"/Users/{username}/{experiment_name}"

# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Ensure the 'models' schema exists in Unity Catalog
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.models")
    print(f"✅ Schema {catalog_name}.models is ready")
except Exception as e:
    print(f"Warning: Could not create schema {catalog_name}.models: {e}")

# Set or create the experiment
mlflow.set_experiment(experiment_path)
print(f"Using MLflow experiment: {experiment_path}")
print(f"Using Unity Catalog for model registry")

# COMMAND ----------

# Start MLflow run for training
with mlflow.start_run(run_name="ditto-entity-matcher-training"):
    # Log parameters
    mlflow.log_param("base_model", "distilbert-base-uncased")
    mlflow.log_param("max_length", 256)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", 3e-5)
    mlflow.log_param("training_pairs", len(training_df_augmented))

    # Train model - use a subfolder under output_path for the model
    model_output_path = f"{output_path}/models/ditto_matcher"
    print(f"Training model, will save to: {model_output_path}")

    ditto.train(
        training_data_path=training_data_path,
        output_path=model_output_path,
        epochs=1,
        batch_size=16,
        learning_rate=3e-5,
        val_split=0.2
    )

    # Log the model output path for the next notebook
    mlflow.log_param("model_output_path", model_output_path)

    run_id = mlflow.active_run().info.run_id
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {model_output_path}")
    print(f"   MLflow Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Register model**: Run notebook 02b_register_evaluate_model.py to register the trained model to Unity Catalog
# MAGIC 2. **Deploy serving endpoint**: Run Phase 3 DAB deployment after registration
# MAGIC    ```bash
# MAGIC    ./deploy-phase.sh 3 dev
# MAGIC    ```

# COMMAND ----------

print(f"✅ Training notebook complete!")
print(f"   Trained model location: {model_output_path}")
print(f"   Training data location: {training_data_path}")
print(f"   Proceed to notebook 02b for model registration and evaluation")
