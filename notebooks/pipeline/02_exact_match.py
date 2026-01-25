# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 1: Exact Match
# MAGIC
# MAGIC Matches entities based on exact identifier matches (LEI, CUSIP, ISIN).

# COMMAND ----------

from __future__ import annotations

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

from pyspark.sql.functions import col, lit, current_timestamp

# Load source entities (from previous step)
source_df = spark.table(f"{catalog_name}.bronze.source_entities") \
    .filter("ingestion_timestamp >= current_timestamp() - interval 1 day")

# Load reference data
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference")

print(f"Source entities: {source_df.count()}")
print(f"Reference entities: {reference_df.count()}")

# COMMAND ----------

# MAGIC %md ## Exact Match on LEI

# COMMAND ----------

exact_matches_lei = source_df.alias("src") \
    .join(
        reference_df.alias("ref"),
        (col("src.lei").isNotNull()) & (col("src.lei") == col("ref.lei")),
        "inner"
    ) \
    .select(
        col("src.source_id"),
        col("src.source_system"),
        col("src.company_name"),
        col("ref.ciq_id").alias("matched_ciq_id"),
        lit(1.0).alias("match_confidence"),
        lit("exact_match").alias("match_method"),
        lit("Stage 1: Exact Match (LEI)").alias("match_stage"),
        lit("Exact LEI match").alias("reasoning"),
        col("ref.company_name").alias("matched_company_name"),
        current_timestamp().alias("match_timestamp"),
        lit(5).alias("processing_time_ms"),
        lit("v1.0").alias("model_version")
    )

lei_count = exact_matches_lei.count()
print(f"✓ LEI matches: {lei_count}")

# COMMAND ----------

# MAGIC %md ## Exact Match on CUSIP

# COMMAND ----------

exact_matches_cusip = source_df.alias("src") \
    .join(
        reference_df.alias("ref"),
        (col("src.cusip").isNotNull()) & (col("src.cusip") == col("ref.cusip")),
        "inner"
    ) \
    .select(
        col("src.source_id"),
        col("src.source_system"),
        col("src.company_name"),
        col("ref.ciq_id").alias("matched_ciq_id"),
        lit(1.0).alias("match_confidence"),
        lit("exact_match").alias("match_method"),
        lit("Stage 1: Exact Match (CUSIP)").alias("match_stage"),
        lit("Exact CUSIP match").alias("reasoning"),
        col("ref.company_name").alias("matched_company_name"),
        current_timestamp().alias("match_timestamp"),
        lit(5).alias("processing_time_ms"),
        lit("v1.0").alias("model_version")
    )

cusip_count = exact_matches_cusip.count()
print(f"✓ CUSIP matches: {cusip_count}")

# COMMAND ----------

# MAGIC %md ## Combine and Write Results

# COMMAND ----------

# Combine all exact matches
all_exact_matches = exact_matches_lei.union(exact_matches_cusip) \
    .dropDuplicates(["source_id", "source_system"])

# Write to temporary table for next stage
all_exact_matches.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog_name}.silver.exact_matches_temp")

total_exact = all_exact_matches.count()
print(f"\n✓ Total exact matches: {total_exact}")
print(f"  - LEI: {lei_count}")
print(f"  - CUSIP: {cusip_count}")

# COMMAND ----------

print("✅ Stage 1: Exact matching complete!")
