# Databricks notebook source
# MAGIC %md
# MAGIC # Register Ditto Model to Unity Catalog
# MAGIC
# MAGIC Registers the trained Ditto model to Unity Catalog for model serving.

# COMMAND ----------

dbutils.widgets.text("model_path", "", "Model Path")
dbutils.widgets.text("model_name", "", "Model Name")

model_path = dbutils.widgets.get("model_path")
model_name = dbutils.widgets.get("model_name")

print(f"Model Path: {model_path}")
print(f"Model Name: {model_name}")

# COMMAND ----------

# MAGIC %md ## Register Model with MLflow

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Set experiment
mlflow.set_experiment("/Shared/entity_matching/ditto_training")

# Register model (placeholder - actual implementation would load and log the model)
print(f"\nRegistering model to: {model_name}")
print(f"From path: {model_path}")

# In production, you would:
# 1. Load the trained model from model_path
# 2. Log it with mlflow.pytorch.log_model() or mlflow.transformers.log_model()
# 3. Register to Unity Catalog

# Placeholder for now
print("\n⚠️  Model registration is a placeholder")
print("   Implement actual model loading and registration here")

# COMMAND ----------

# MAGIC %md ## Transition to Production (Optional)

# COMMAND ----------

# client = MlflowClient()
#
# # Get latest version
# latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
#
# # Transition to production
# client.transition_model_version_stage(
#     name=model_name,
#     version=latest_version.version,
#     stage="Production"
# )
#
# print(f"✓ Model version {latest_version.version} promoted to Production")

# COMMAND ----------

print("✅ Model registration complete (placeholder)")
