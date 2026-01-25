# Databricks notebook source
# MAGIC %md
# MAGIC # Register Ditto Model to Unity Catalog
# MAGIC
# MAGIC Registers the trained Ditto model to Unity Catalog for model serving.

# COMMAND ----------

from __future__ import annotations

# COMMAND ----------

dbutils.widgets.text("model_path", "", "Model Path")
dbutils.widgets.text("model_name", "", "Model Name")

model_path = dbutils.widgets.get("model_path")
model_name = dbutils.widgets.get("model_name")

print(f"Model Path: {model_path}")
print(f"Model Name: {model_name}")

# COMMAND ----------

# MAGIC %md ## Install Required Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade transformers>=4.40.0 torch>=2.1.0 mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-get parameters after Python restart
model_path = dbutils.widgets.get("model_path")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md ## Register Model with MLflow

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print(f"Registering model to Unity Catalog: {model_name}")
print(f"Loading model from: {model_path}")

# COMMAND ----------

# MAGIC %md ### Load Trained Model

# COMMAND ----------

# Load the trained Ditto model
print("\nLoading tokenizer and model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("✓ Model loaded successfully")

    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Model type: {model.__class__.__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# COMMAND ----------

# MAGIC %md ### Register to Unity Catalog

# COMMAND ----------

# Create a signature for the model
import mlflow.pyfunc
from mlflow.models import infer_signature
import pandas as pd

# Create sample input/output for signature
sample_input = pd.DataFrame({
    "left_entity": ["COL name VAL Apple Inc. COL ticker VAL AAPL"],
    "right_entity": ["COL name VAL Apple Inc COL ticker VAL AAPL"]
})
sample_output = pd.DataFrame({
    "prediction": [1],
    "confidence": [0.99]
})

signature = infer_signature(sample_input, sample_output)
print("✓ Model signature created")

# COMMAND ----------

# MAGIC %md ### Log and Register Model

# COMMAND ----------

# Start an MLflow run to log the model
with mlflow.start_run(run_name="model-registration") as run:
    # Log the model with transformers flavor
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="ditto_model",
        task="text-classification",
        signature=signature,
        registered_model_name=model_name
    )

    # Log model metadata
    mlflow.log_param("model_type", "ditto_matcher")
    mlflow.log_param("base_model", "distilbert-base-uncased")
    mlflow.log_param("num_parameters", num_params)
    mlflow.log_param("task", "entity_matching")

    run_id = run.info.run_id
    print(f"✓ Model logged to MLflow run: {run_id}")

# COMMAND ----------

# MAGIC %md ### Verify Registration

# COMMAND ----------

# Verify model is registered in Unity Catalog
client = MlflowClient()

try:
    # Get registered model details
    registered_model = client.get_registered_model(model_name)
    print(f"\n✓ Model registered successfully!")
    print(f"  Name: {registered_model.name}")
    print(f"  Description: {registered_model.description}")

    # Get latest version
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0]
        print(f"  Latest Version: {latest_version.version}")
        print(f"  Status: {latest_version.status}")
        print(f"  Run ID: {latest_version.run_id}")

        # Add model description
        client.update_registered_model(
            name=model_name,
            description="Fine-tuned Ditto model for entity matching with S&P Capital IQ data"
        )

        # Add version description
        client.update_model_version(
            name=model_name,
            version=latest_version.version,
            description=f"Ditto entity matcher trained on S&P 500 data. Base model: distilbert-base-uncased. Parameters: {num_params:,}"
        )
        print("  ✓ Model descriptions updated")
    else:
        print("  ⚠ No versions found (unexpected)")

except Exception as e:
    print(f"❌ Error verifying model registration: {e}")
    raise

# COMMAND ----------

# MAGIC %md ## Set Model Alias (Optional)

# COMMAND ----------

# Set 'champion' alias for the latest version (used for serving)
try:
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version.version
    )
    print(f"✓ Set 'champion' alias to version {latest_version.version}")
except Exception as e:
    print(f"⚠ Could not set alias (might not be supported): {e}")

# COMMAND ----------

print(f"\n✅ Model registration complete!")
print(f"   Model: {model_name}")
print(f"   Version: {latest_version.version}")
print(f"   Ready for model serving in Phase 3")
