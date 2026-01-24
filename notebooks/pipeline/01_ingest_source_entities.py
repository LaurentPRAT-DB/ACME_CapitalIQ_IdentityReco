# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Source Entities
# MAGIC
# MAGIC Loads source entities from external systems into the bronze layer.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
dbutils.widgets.text("source_table", "", "Source Table (optional)")

catalog_name = dbutils.widgets.get("catalog_name")
source_table = dbutils.widgets.get("source_table")

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Load Source Data
# MAGIC
# MAGIC This is a placeholder - replace with your actual data source

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# Example: Load from CSV/Parquet/Delta/External Table
# Replace this with your actual data ingestion logic

sample_sources = [
    ("CRM-001", "Salesforce", "Apple Computer Inc.", "AAPL", None, None, "US", "Technology"),
    ("CRM-002", "Salesforce", "Microsoft Corp", "MSFT", None, "594918104", "US", "Software"),
    ("TRD-001", "Bloomberg", "GOOGL", "GOOGL", None, None, "US", "Internet"),
    ("VND-001", "Reuters", "Amazon.com", "AMZN", "ZXTILKJKG63JELOEG630", None, "US", "Retail")
]

df = spark.createDataFrame(
    sample_sources,
    ["source_id", "source_system", "company_name", "ticker", "lei", "cusip", "country", "industry"]
)

df = df.withColumn("isin", lit(None).cast("string"))
df = df.withColumn("ingestion_timestamp", current_timestamp())

# COMMAND ----------

# MAGIC %md ## Write to Bronze

# COMMAND ----------

# Write to bronze table
df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(f"{catalog_name}.bronze.source_entities")

count = df.count()
print(f"✓ Ingested {count} source entities to {catalog_name}.bronze.source_entities")

# COMMAND ----------

print("✅ Source entity ingestion complete!")
