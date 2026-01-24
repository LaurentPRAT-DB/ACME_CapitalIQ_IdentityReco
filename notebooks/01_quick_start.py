# Databricks notebook source
# MAGIC %md
# MAGIC # Entity Matching Quick Start
# MAGIC
# MAGIC This notebook demonstrates the end-to-end entity matching pipeline using the hybrid approach.
# MAGIC
# MAGIC ## Setup
# MAGIC 1. Install required libraries
# MAGIC 2. Load sample data
# MAGIC 3. Run matching pipeline
# MAGIC 4. Evaluate results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install transformers>=4.36.0 sentence-transformers>=2.3.0 torch>=2.1.0 faiss-cpu scikit-learn tqdm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Get parameters from job (set by DABs) or use defaults for interactive mode
dbutils.widgets.text("workspace_path", "")
dbutils.widgets.text("catalog_name", "entity_matching")

workspace_path = dbutils.widgets.get("workspace_path")
catalog_name = dbutils.widgets.get("catalog_name")

# Import required modules
import sys
import os

# Only add workspace_path if provided (from DABs deployment)
if workspace_path:
    print(f"Using workspace path from DABs: {workspace_path}")
    sys.path.append(workspace_path)
else:
    # For interactive development, try to find the src module
    # Assuming notebook is in notebooks/ and src is at project root
    print("No workspace_path provided, using interactive mode")
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(project_root)
    print(f"Using project root: {project_root}")

print(f"sys.path: {sys.path[:3]}")

from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from src.evaluation.metrics import print_metrics, calculate_pipeline_metrics

print(f"Using catalog: {catalog_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Sample Data

# COMMAND ----------

# Load sample reference data (S&P Capital IQ) from Unity Catalog
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference").toPandas()
print(f"Loaded {len(reference_df)} reference entities from {catalog_name}.bronze.spglobal_reference")
display(reference_df.head())

# COMMAND ----------

# Load sample source entities from Unity Catalog
source_entities_df = spark.table(f"{catalog_name}.bronze.source_entities").toPandas()
source_entities = source_entities_df.to_dict('records')
print(f"Loaded {len(source_entities)} source entities to match from {catalog_name}.bronze.source_entities")

for entity in source_entities[:3]:
    print(f"\n{entity}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Pipeline
# MAGIC
# MAGIC The hybrid pipeline includes:
# MAGIC - Stage 1: Exact matching (LEI, CUSIP, ISIN)
# MAGIC - Stage 2: Vector search with BGE embeddings
# MAGIC - Stage 3: Ditto fine-tuned matcher (requires trained model)
# MAGIC - Stage 4: Foundation Model fallback

# COMMAND ----------

# Initialize pipeline (without Ditto for this demo)
pipeline = HybridMatchingPipeline(
    reference_df=reference_df,
    ditto_model_path=None,  # Set path to trained Ditto model
    enable_foundation_model=False  # Disable for demo
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Match Single Entity

# COMMAND ----------

# Match a single entity
entity = source_entities[0]
print(f"Matching entity: {entity['company_name']}")

result = pipeline.match(entity, return_candidates=True)

print(f"\n{'=' * 60}")
print(f"MATCH RESULT")
print(f"{'=' * 60}")
print(f"CIQ ID: {result['ciq_id']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Method: {result['match_method']}")
print(f"Stage: {result['stage_name']}")
print(f"Reasoning: {result['reasoning']}")

if 'candidates' in result:
    print(f"\nTop Candidates:")
    for i, candidate in enumerate(result['candidates'], 1):
        print(f"  {i}. {candidate['metadata']['company_name']} ({candidate['ciq_id']}) - Similarity: {candidate['similarity']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch Matching

# COMMAND ----------

# Match all source entities
results = pipeline.batch_match(source_entities, show_progress=True)

# Display results
import pandas as pd
results_df = pd.DataFrame(results)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pipeline Statistics

# COMMAND ----------

# Print pipeline statistics
pipeline.print_pipeline_stats(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluate Against Ground Truth (if available)

# COMMAND ----------

# Create ground truth for sample data (in practice, load from file)
ground_truth = {
    "CRM-001": "IQ24937",  # Apple Inc
    "CRM-002": "IQ4004",   # Microsoft Corporation
    "CRM-003": "IQ24937",  # Apple Computer Inc -> Apple Inc
    "TRD-001": "IQ4004",   # MSFT -> Microsoft
    "VND-001": "IQ112209"  # Amazon.com Inc
}

# Calculate metrics
metrics = calculate_pipeline_metrics(results, ground_truth)
print_metrics(metrics, title="Pipeline Evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export Results

# COMMAND ----------

# Save results to Delta table
from pyspark.sql import SparkSession
spark_df = spark.createDataFrame(results_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.gold.matched_entities")

print(f"Results saved to {catalog_name}.gold.matched_entities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Train Ditto Model**: See notebook `02_train_ditto.py`
# MAGIC 2. **Run Full Pipeline**: Enable Ditto and Foundation Model
# MAGIC 3. **Deploy to Production**: See notebook `05_production_deployment.py`
# MAGIC 4. **Monitor Performance**: Track accuracy and costs
