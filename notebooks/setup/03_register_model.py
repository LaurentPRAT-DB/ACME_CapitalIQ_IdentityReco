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

# MAGIC %md ### Create PyFunc Wrapper for Model Serving

# COMMAND ----------

# Create a custom PyFunc wrapper for model serving
class DittoModelWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for Ditto entity matcher
    Enables model serving with custom inference logic
    """

    def load_context(self, context):
        """Load model and tokenizer from artifacts"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Load model artifacts
        model_path = context.artifacts["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.max_length = 256

    def predict(self, context, model_input):
        """
        Predict entity matches

        Args:
            model_input: DataFrame with columns 'left_entity' and 'right_entity'

        Returns:
            DataFrame with columns 'prediction' (0/1) and 'confidence' (float)
        """
        import torch
        import pandas as pd

        # Extract entity pairs
        left_entities = model_input["left_entity"].tolist()
        right_entities = model_input["right_entity"].tolist()

        # Tokenize
        encodings = self.tokenizer(
            left_entities,
            right_entities,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = probs.cpu().numpy()

            # Extract confidence for predicted class
            result_confidence = [confidences[i][predictions[i]] for i in range(len(predictions))]

        return pd.DataFrame({
            "prediction": predictions.astype(int),
            "confidence": result_confidence
        })

print("✓ PyFunc wrapper defined")

# COMMAND ----------

# MAGIC %md ### Log and Register Model

# COMMAND ----------

# Start an MLflow run to log the model
with mlflow.start_run(run_name="model-registration") as run:
    # Log the model with PyFunc wrapper (for model serving compatibility)
    mlflow.pyfunc.log_model(
        artifact_path="ditto_model",
        python_model=DittoModelWrapper(),
        artifacts={"model_path": model_path},
        signature=signature,
        registered_model_name=model_name,
        pip_requirements=[
            "transformers>=4.40.0",
            "torch>=2.1.0",
            "pandas>=1.5.0"
        ]
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
