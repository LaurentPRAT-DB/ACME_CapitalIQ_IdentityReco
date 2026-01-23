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

# MAGIC %pip install transformers torch scikit-learn mlflow

# COMMAND ----------

import sys
sys.path.append("/Workspace/entity-matching")

from src.data.loader import DataLoader
from src.data.training_generator import TrainingDataGenerator
from src.models.ditto_matcher import DittoMatcher
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Reference Data

# COMMAND ----------

# Load S&P Capital IQ reference data
loader = DataLoader()
reference_df = loader.load_reference_data()

print(f"Loaded {len(reference_df)} reference entities")
display(reference_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Training Data

# COMMAND ----------

# Initialize training data generator
generator = TrainingDataGenerator(seed=42)

# Generate training pairs from S&P 500
training_df = generator.generate_from_sp500(
    reference_df=reference_df,
    num_positive_pairs=500,
    num_negative_pairs=500
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

# Save training data
training_data_path = "/dbfs/entity_matching/ditto_training_data.csv"
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

# Start MLflow run
with mlflow.start_run(run_name="ditto-entity-matcher"):
    # Log parameters
    mlflow.log_param("base_model", "distilbert-base-uncased")
    mlflow.log_param("max_length", 256)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 3e-5)
    mlflow.log_param("training_pairs", len(training_df_augmented))

    # Train model
    output_path = "/dbfs/entity_matching/models/ditto_matcher"

    ditto.train(
        training_data_path=training_data_path,
        output_path=output_path,
        epochs=20,
        batch_size=64,
        learning_rate=3e-5,
        val_split=0.2
    )

    # Log model
    mlflow.pytorch.log_model(
        ditto.model,
        "ditto-model",
        registered_model_name="entity_matching_ditto"
    )

    print(f"Model saved to {output_path}")

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

test_data_path = "/dbfs/entity_matching/ditto_test_data.csv"
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
with mlflow.start_run(nested=True):
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
