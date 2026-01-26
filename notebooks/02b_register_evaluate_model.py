# Databricks notebook source
# MAGIC %md
# MAGIC # Register and Evaluate Ditto Model - Part 2: Registration & Evaluation
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the trained Ditto model from workspace storage
# MAGIC 2. Registers the model to Unity Catalog with Champion alias
# MAGIC 3. Evaluates model performance on test set
# MAGIC 4. Tests model on sample pairs
# MAGIC
# MAGIC **Note:** Run this after 02a_train_ditto_model.py completes

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
import pandas as pd

# Get parameters from job (set by DABs)
dbutils.widgets.text("workspace_path", "/Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev/files")
dbutils.widgets.text("catalog_name", "entity_matching")
dbutils.widgets.text("output_path", "/Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev/training_data")
dbutils.widgets.text("bundle_target", "dev")

workspace_path = dbutils.widgets.get("workspace_path")
catalog_name = dbutils.widgets.get("catalog_name")
output_path = dbutils.widgets.get("output_path")
bundle_target = dbutils.widgets.get("bundle_target")

print(f"Using catalog: {catalog_name}")
print(f"Output path: {output_path}")

# Add workspace path to sys.path
sys.path.append(workspace_path)

from src.data.loader import DataLoader
from src.data.training_generator import TrainingDataGenerator
from src.models.ditto_matcher import DittoMatcher
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print("\n✅ Successfully imported all modules")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Trained Model

# COMMAND ----------

# Load the trained model from workspace storage
# The model should have been saved by Phase 2 (02a_train_ditto_model.py)
model_output_path = f"{output_path}/models/ditto_matcher"

# Check if model exists
import os
if not os.path.exists(model_output_path):
    print(f"⚠ Model not found at: {model_output_path}")
    print(f"\nPlease ensure Phase 2 (training) has completed successfully.")
    print(f"The model should be saved at: {model_output_path}")
    print(f"\nYou can:")
    print(f"  1. Run Phase 2 first: ./deploy-phase.sh 2 {bundle_target}")
    print(f"  2. Or provide a custom model path below")

    # Allow override for custom model path
    dbutils.widgets.text("custom_model_path", "", "Custom Model Path (optional)")
    custom_path = dbutils.widgets.get("custom_model_path")

    if custom_path:
        model_output_path = custom_path
        print(f"\nUsing custom model path: {model_output_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_output_path} and no custom path provided")
else:
    print(f"✓ Found trained model at: {model_output_path}")

# Initialize Ditto matcher and load trained weights
print(f"Loading model...")
ditto = DittoMatcher(
    base_model="distilbert-base-uncased",
    max_length=256
)

# Load the trained model
ditto.load_model(model_output_path)
print(f"✅ Model loaded successfully from {model_output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register Model to Unity Catalog

# COMMAND ----------

# Setup MLflow experiment
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

experiment_name = f"{catalog_name}-ditto-model-registration"
experiment_path = f"/Users/{username}/{experiment_name}"

# Set or create the experiment
mlflow.set_experiment(experiment_path)
print(f"Using MLflow experiment: {experiment_path}")

# COMMAND ----------

# Register model to Unity Catalog with custom PyFunc wrapper
import mlflow.pyfunc
import tempfile

# Create custom wrapper to handle model loading
class DittoModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load model directly using transformers"""
        import os
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Set environment variable to disable warnings
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

        # Load model and tokenizer from saved path
        model_path = context.artifacts["model"]
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict(self, context, model_input):
        """Make predictions using the loaded model"""
        import torch
        import torch.nn.functional as F
        import pandas as pd

        # Handle both DataFrame and dict inputs
        if isinstance(model_input, pd.DataFrame):
            left = model_input["left_entity"].tolist()
            right = model_input["right_entity"].tolist()
        else:
            left = model_input["left_entity"]
            right = model_input["right_entity"]

        # Combine inputs for text classification
        texts = [f"{l} [SEP] {r}" for l, r in zip(left, right)]

        # Get predictions
        results = []
        with torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

                # Get model output
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get prediction and confidence
                probs = F.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()

                results.append({
                    'prediction': pred_class,
                    'confidence': confidence
                })

        return pd.DataFrame(results)

# COMMAND ----------

# Register model with MLflow using the already-saved model
print(f"Registering model from: {model_output_path}")

with mlflow.start_run(run_name="ditto-model-registration"):
    # Create sample input for signature inference
    sample_input = pd.DataFrame({
        "left_entity": ["COL name VAL Apple Inc. COL ticker VAL AAPL"],
        "right_entity": ["COL name VAL Apple Computer COL ticker VAL AAPL"]
    })

    # Infer signature using the loaded model
    wrapper = DittoModelWrapper()

    # Mock context for signature inference
    class MockContext:
        def __init__(self, model_path):
            self.artifacts = {"model": model_path}

    wrapper.load_context(MockContext(model_output_path))
    sample_output = wrapper.predict(None, sample_input)
    signature = infer_signature(sample_input, sample_output)

    # Log using pyfunc flavor with Unity Catalog
    # Use the already-saved model directory directly as artifact
    registered_model_name = f"{catalog_name}.models.entity_matching_ditto"

    print(f"Logging model to MLflow...")
    model_info = mlflow.pyfunc.log_model(
        artifact_path="ditto-model",
        python_model=DittoModelWrapper(),
        artifacts={"model": model_output_path},
        registered_model_name=registered_model_name,
        signature=signature,
        pip_requirements=[
            "transformers>=4.40.0",
            "torch>=2.1.0",
            "sentencepiece",
            "mlflow>=2.10.0",
            "huggingface-hub"
        ]
    )

    print(f"✅ Model registered in Unity Catalog as: {registered_model_name}")
    print(f"   Model URI: {model_info.model_uri}")
    print(f"   Run ID: {mlflow.active_run().info.run_id}")

    # Set Champion alias
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{registered_model_name}'")

    if model_versions:
        latest_version = max([int(mv.version) for mv in model_versions])
        print(f"   Model Version: {latest_version}")

        # Set Champion alias
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Champion",
            version=str(latest_version)
        )
        print(f"   ✅ Alias 'Champion' set to version {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate Model Performance

# COMMAND ----------

# Load reference data for test generation
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference")
reference_pdf = reference_df.toPandas()

# Initialize generator for test data
generator = TrainingDataGenerator(seed=42)

# Create test set (separate from training)
test_df = generator.generate_from_sp500(
    reference_df=reference_pdf,
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

# Log metrics to MLflow
with mlflow.start_run(run_name="ditto-model-evaluation"):
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test Model on Sample Pairs

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
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Deploy serving endpoint**: Run Phase 3 DAB deployment to create/update the serving endpoint
# MAGIC    ```bash
# MAGIC    ./deploy-phase.sh 3 dev
# MAGIC    ```
# MAGIC 2. **Test deployed endpoint**: Query the serving endpoint with sample data
# MAGIC 3. **Run production pipeline**: Execute Phase 4 to process entities using the full hybrid pipeline
# MAGIC 4. **Monitor performance**: Track inference latency and accuracy
# MAGIC 5. **Retrain periodically**: Re-run 02a when new training data becomes available

# COMMAND ----------

print("✅ Model registration and evaluation complete!")
print(f"   Model: {registered_model_name}@Champion")
print(f"   Ready for Phase 3 deployment")
