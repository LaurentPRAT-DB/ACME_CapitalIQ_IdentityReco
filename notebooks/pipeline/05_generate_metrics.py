# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Pipeline Metrics
# MAGIC
# MAGIC Calculates and displays pipeline performance metrics.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Calculate Metrics

# COMMAND ----------

from pyspark.sql.functions import count, avg, sum, when, col

# Get today's matches
today_matches = spark.sql(f"""
    SELECT *
    FROM {catalog_name}.gold.matched_entities
    WHERE date(match_timestamp) = current_date()
""")

total_count = today_matches.count()

if total_count == 0:
    print("No matches found for today")
    dbutils.notebook.exit("No data to analyze")

# COMMAND ----------

# MAGIC %md ## Pipeline Statistics

# COMMAND ----------

# Calculate statistics
stats = today_matches.agg(
    count("*").alias("total_entities"),
    avg("match_confidence").alias("avg_confidence"),
    sum(when(col("auto_matched"), 1).otherwise(0)).alias("auto_matched_count"),
    sum(when(col("needs_review"), 1).otherwise(0)).alias("review_count"),
    avg("processing_time_ms").alias("avg_latency_ms")
).collect()[0]

print("\n" + "="*80)
print("PIPELINE METRICS - Today")
print("="*80)
print(f"Total Entities: {stats.total_entities}")
print(f"Avg Confidence: {stats.avg_confidence:.2%}")
print(f"Auto-Matched: {stats.auto_matched_count} ({stats.auto_matched_count/stats.total_entities*100:.1f}%)")
print(f"Needs Review: {stats.review_count} ({stats.review_count/stats.total_entities*100:.1f}%)")
print(f"Avg Latency: {stats.avg_latency_ms:.1f}ms")

# COMMAND ----------

# MAGIC %md ## Matches by Method

# COMMAND ----------

by_method = today_matches.groupBy("match_method") \
    .agg(
        count("*").alias("count"),
        avg("match_confidence").alias("avg_confidence")
    ) \
    .orderBy("count", ascending=False)

print("\nMatches by Method:")
by_method.show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Cost Estimate

# COMMAND ----------

# Estimate cost based on method
cost_per_method = {
    "exact_match": 0.0000,
    "vector_search": 0.0001,
    "ditto_matcher": 0.001,
    "foundation_model": 0.05
}

method_counts = {row.match_method: row['count'] for row in by_method.collect()}

total_cost = sum(method_counts.get(method, 0) * cost for method, cost in cost_per_method.items())
cost_per_entity = total_cost / stats.total_entities if stats.total_entities > 0 else 0

print(f"\nEstimated Cost:")
print(f"  Total: ${total_cost:.2f}")
print(f"  Per Entity: ${cost_per_entity:.4f}")
print(f"  Target: $0.01/entity")

# COMMAND ----------

print("\n✅ Pipeline metrics generated successfully!")
