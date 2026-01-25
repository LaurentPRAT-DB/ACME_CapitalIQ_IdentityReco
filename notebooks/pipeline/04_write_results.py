# Databricks notebook source
# MAGIC %md
# MAGIC # Write Results to Gold Table
# MAGIC
# MAGIC Combines all matches and writes to the gold table.

# COMMAND ----------

from __future__ import annotations

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
dbutils.widgets.text("output_table", "", "Output Table (optional)")

catalog_name = dbutils.widgets.get("catalog_name")
output_table_param = dbutils.widgets.get("output_table")

output_table = output_table_param if output_table_param else f"{catalog_name}.gold.matched_entities"

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Combine All Matches

# COMMAND ----------

# Load matches from all stages
exact_matches = spark.table(f"{catalog_name}.silver.exact_matches_temp")
vector_ditto_matches = spark.table(f"{catalog_name}.silver.vector_ditto_matches_temp")

# Combine all matches
all_matches = exact_matches.union(vector_ditto_matches)

# Add auto_matched and needs_review flags based on confidence thresholds
from pyspark.sql.functions import when, col

all_matches = all_matches.withColumn(
    "auto_matched",
    when(col("match_confidence") >= 0.90, True).otherwise(False)
).withColumn(
    "needs_review",
    when(col("match_confidence") < 0.70, True).otherwise(False)
)

total_matches = all_matches.count()
print(f"Total matches to write: {total_matches}")

# COMMAND ----------

# MAGIC %md ## Write to Gold Table

# COMMAND ----------

# Write results
all_matches.write.format("delta") \
    .mode("append") \
    .saveAsTable(output_table)

print(f"✓ Written {total_matches} matches to {output_table}")

# COMMAND ----------

# MAGIC %md ## Display Summary

# COMMAND ----------

from pyspark.sql.functions import count

summary = all_matches.groupBy("match_method").agg(count("*").alias("count"))
summary.show()

# COMMAND ----------

print("✅ Results written to gold table successfully!")
