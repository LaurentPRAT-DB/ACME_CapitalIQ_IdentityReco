# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 2 & 3: Vector Search + Ditto Matching
# MAGIC
# MAGIC Uses vector search to find candidates, then Ditto matcher to confirm matches.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
dbutils.widgets.text("ditto_endpoint", "ditto-em-dev", "Ditto Endpoint")
dbutils.widgets.text("vector_search_endpoint", "entity-matching-vs-dev", "Vector Search Endpoint")

catalog_name = dbutils.widgets.get("catalog_name")
ditto_endpoint = dbutils.widgets.get("ditto_endpoint")
vs_endpoint = dbutils.widgets.get("vector_search_endpoint")

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Load Unmatched Entities

# COMMAND ----------

from pyspark.sql.functions import col

# Get entities not matched in Stage 1
source_df = spark.table(f"{catalog_name}.bronze.source_entities")
exact_matches = spark.table(f"{catalog_name}.silver.exact_matches_temp")

unmatched_df = source_df.join(
    exact_matches.select("source_id", "source_system"),
    on=["source_id", "source_system"],
    how="left_anti"
)

unmatched_count = unmatched_df.count()
print(f"Entities to match with vector search: {unmatched_count}")

# COMMAND ----------

# MAGIC %md ## Vector Search + Ditto Matching (Placeholder)

# COMMAND ----------

from pyspark.sql.functions import lit, current_timestamp

# Placeholder: In production, this would use vector search and Ditto endpoint
# For now, create empty results
matched_df = spark.createDataFrame([], schema="""
    source_id STRING,
    source_system STRING,
    company_name STRING,
    matched_ciq_id STRING,
    match_confidence DOUBLE,
    match_method STRING,
    match_stage STRING,
    reasoning STRING,
    matched_company_name STRING,
    match_timestamp TIMESTAMP,
    processing_time_ms LONG,
    model_version STRING
""")

# Write to temp table
matched_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog_name}.silver.vector_ditto_matches_temp")

print(f"✓ Vector search + Ditto matches: {matched_df.count()}")
print("\nNote: This is a placeholder. Implement vector search and Ditto inference here.")

# COMMAND ----------

print("✅ Stage 2 & 3: Vector search + Ditto matching complete!")
