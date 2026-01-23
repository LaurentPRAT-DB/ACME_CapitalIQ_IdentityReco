# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Connect Example - Local to Remote Databricks
# MAGIC
# MAGIC This notebook demonstrates how to use Spark Connect to run code locally
# MAGIC that executes on a remote Databricks cluster.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Databricks workspace with a running cluster
# MAGIC 2. Personal access token
# MAGIC 3. Cluster ID
# MAGIC 4. Environment variables set (see .env.example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Required Modules

# COMMAND ----------

import sys
import os
from pathlib import Path

# Add project root to path (adjust as needed for your setup)
project_root = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(project_root))

from src.utils.spark_utils import get_spark_session, init_spark_connect
from src.config import config
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Spark Connect Session
# MAGIC
# MAGIC Three ways to initialize Spark Connect:

# COMMAND ----------

# Method 1: Auto-detect from environment variables
# This uses USE_SPARK_CONNECT, DATABRICKS_HOST, DATABRICKS_TOKEN, SPARK_CONNECT_CLUSTER_ID
spark = get_spark_session()

# COMMAND ----------

# Method 2: Explicit parameters (uncomment to use)
# spark = init_spark_connect(
#     databricks_host="dbc-xxxxx.cloud.databricks.com",
#     databricks_token="dapiXXXXXXXXXXXX",
#     cluster_id="1234-567890-abcdefgh"
# )

# COMMAND ----------

# Method 3: Force local Spark (no remote connection)
# spark = get_spark_session(force_local=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Remote Connection

# COMMAND ----------

# Verify connection
print(f"Spark Version: {spark.version}")
print(f"Application ID: {spark.sparkContext.applicationId}")

# Test with simple DataFrame
test_df = spark.createDataFrame([
    (1, "Apple Inc.", "AAPL"),
    (2, "Microsoft Corp", "MSFT"),
    (3, "Amazon.com Inc", "AMZN")
], ["id", "company_name", "ticker"])

print("\nTest DataFrame:")
test_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read from Unity Catalog (if available)

# COMMAND ----------

# Example: Read from Unity Catalog table
# Uncomment if you have Unity Catalog tables set up
"""
reference_df = spark.table("main.entity_matching.spglobal_reference")
print(f"Reference entities: {reference_df.count()}")
reference_df.show(5)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch Entity Matching with Spark Connect

# COMMAND ----------

from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from src.data.loader import DataLoader

# Load reference data
loader = DataLoader()
reference_df_pandas = loader.load_reference_data()

# Initialize matching pipeline
pipeline = HybridMatchingPipeline(
    reference_df=reference_df_pandas,
    ditto_model_path=None,  # Set to your model path
    enable_foundation_model=False
)

print("Pipeline initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Spark UDF for Entity Matching

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd

# Define return schema
result_schema = StructType([
    StructField("ciq_id", StringType(), True),
    StructField("confidence", DoubleType(), True),
    StructField("match_method", StringType(), True),
    StructField("stage_name", StringType(), True),
    StructField("reasoning", StringType(), True)
])

@pandas_udf(result_schema)
def match_entity_udf(company_names: pd.Series, tickers: pd.Series) -> pd.DataFrame:
    """
    Pandas UDF for matching entities
    This runs on the remote Databricks cluster
    """
    results = []

    for company_name, ticker in zip(company_names, tickers):
        entity = {
            "company_name": company_name,
            "ticker": ticker if pd.notna(ticker) else None
        }

        result = pipeline.match(entity, return_candidates=False)
        results.append({
            "ciq_id": result.get("ciq_id"),
            "confidence": result.get("confidence"),
            "match_method": result.get("match_method"),
            "stage_name": result.get("stage_name"),
            "reasoning": result.get("reasoning")
        })

    return pd.DataFrame(results)

print("UDF defined successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Apply Matching to Spark DataFrame

# COMMAND ----------

# Create sample source entities
source_df = spark.createDataFrame([
    ("CRM-001", "Apple Inc.", "AAPL"),
    ("CRM-002", "Microsoft Corporation", "MSFT"),
    ("CRM-003", "Apple Computer Inc.", "AAPL"),
    ("TRD-001", "MSFT", None),
    ("VND-001", "Amazon.com Inc", "AMZN")
], ["source_id", "company_name", "ticker"])

print("Source entities:")
source_df.show()

# Apply matching UDF
matched_df = source_df.withColumn(
    "match_result",
    match_entity_udf(F.col("company_name"), F.col("ticker"))
)

# Expand struct column
matched_expanded = matched_df.select(
    "source_id",
    "company_name",
    "ticker",
    F.col("match_result.ciq_id").alias("ciq_id"),
    F.col("match_result.confidence").alias("confidence"),
    F.col("match_result.match_method").alias("match_method"),
    F.col("match_result.stage_name").alias("stage"),
    F.col("match_result.reasoning").alias("reasoning")
)

print("\nMatched entities:")
matched_expanded.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write Results to Delta Table

# COMMAND ----------

# Write to Delta table in Unity Catalog
# Uncomment when ready to write
"""
matched_expanded.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.entity_matching.matched_entities")

print("Results saved to main.entity_matching.matched_entities")
"""

# Or write to DBFS path
output_path = "/tmp/entity_matches"
matched_expanded.write \
    .format("delta") \
    .mode("overwrite") \
    .save(output_path)

print(f"Results saved to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Read Results Back

# COMMAND ----------

# Read from Delta
results_df = spark.read.format("delta").load(output_path)
results_df.show()

# Calculate statistics
print("\nMatch Statistics:")
results_df.groupBy("stage").count().show()
results_df.groupBy("match_method").count().show()

print(f"\nAverage Confidence: {results_df.select(F.avg('confidence')).first()[0]:.2%}")
print(f"Matched Entities: {results_df.filter(F.col('ciq_id').isNotNull()).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Performance Comparison

# COMMAND ----------

import time

# Test with larger dataset
print("Creating test dataset...")
large_df = spark.range(0, 1000).select(
    F.col("id").cast("string").alias("source_id"),
    F.lit("Test Company ").alias("company_name"),
    F.lit("TEST").alias("ticker")
)

print("Running batch matching...")
start_time = time.time()

matched_large = large_df.withColumn(
    "match_result",
    match_entity_udf(F.col("company_name"), F.col("ticker"))
).cache()

# Force computation
count = matched_large.count()
elapsed = time.time() - start_time

print(f"\nMatched {count} entities in {elapsed:.2f} seconds")
print(f"Throughput: {count/elapsed:.1f} entities/second")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cleanup

# COMMAND ----------

# Stop Spark session when done
# spark.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC 1. Connecting to remote Databricks cluster via Spark Connect
# MAGIC 2. Running local code on remote cluster
# MAGIC 3. Using Pandas UDFs for entity matching
# MAGIC 4. Writing results to Delta tables
# MAGIC 5. Performance testing at scale
# MAGIC
# MAGIC ## Benefits of Spark Connect
# MAGIC - **Local Development**: Write and test code locally
# MAGIC - **Remote Execution**: Leverage cluster compute power
# MAGIC - **Unified API**: Same PySpark API as notebooks
# MAGIC - **Easy Debugging**: Use local IDE with breakpoints
# MAGIC - **Cost Efficient**: Only pay for compute when running
