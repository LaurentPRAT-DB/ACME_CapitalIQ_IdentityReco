# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Pipeline Metrics
# MAGIC
# MAGIC Calculates and displays pipeline performance metrics.

# COMMAND ----------

from __future__ import annotations

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
# Check if auto_matched and needs_review columns exist (they should from write_results step)
has_auto_matched = "auto_matched" in today_matches.columns
has_needs_review = "needs_review" in today_matches.columns

# Build aggregation dynamically based on available columns
agg_exprs = [
    count("*").alias("total_entities"),
    avg("match_confidence").alias("avg_confidence"),
    avg("processing_time_ms").alias("avg_latency_ms")
]

if has_auto_matched:
    agg_exprs.append(sum(when(col("auto_matched"), 1).otherwise(0)).alias("auto_matched_count"))
else:
    # Fall back to confidence-based calculation
    agg_exprs.append(sum(when(col("match_confidence") >= 0.90, 1).otherwise(0)).alias("auto_matched_count"))

if has_needs_review:
    agg_exprs.append(sum(when(col("needs_review"), 1).otherwise(0)).alias("review_count"))
else:
    # Fall back to confidence-based calculation
    agg_exprs.append(sum(when(col("match_confidence") < 0.70, 1).otherwise(0)).alias("review_count"))

stats = today_matches.agg(*agg_exprs).collect()[0]

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
