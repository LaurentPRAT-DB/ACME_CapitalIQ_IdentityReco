# Databricks notebook source
# MAGIC %md
# MAGIC # Load Large-Scale Test Dataset
# MAGIC
# MAGIC This notebook generates and loads large-scale test data:
# MAGIC - **1000+ reference entities** in `spglobal_reference`
# MAGIC - **3000+ source entities** in `source_entities`
# MAGIC
# MAGIC The data includes realistic variations for comprehensive testing.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
dbutils.widgets.text("num_reference_entities", "1000", "Number of Reference Entities")
dbutils.widgets.text("num_source_entities", "3000", "Number of Source Entities")
dbutils.widgets.text("match_ratio", "0.7", "Match Ratio (0.7 = 70% should match)")
dbutils.widgets.dropdown("mode", "append", ["append", "overwrite"], "Write Mode")

catalog_name = dbutils.widgets.get("catalog_name")
num_reference = int(dbutils.widgets.get("num_reference_entities"))
num_source = int(dbutils.widgets.get("num_source_entities"))
match_ratio = float(dbutils.widgets.get("match_ratio"))
mode = dbutils.widgets.get("mode")

print(f"Configuration:")
print(f"  Catalog: {catalog_name}")
print(f"  Reference entities: {num_reference}")
print(f"  Source entities: {num_source}")
print(f"  Match ratio: {match_ratio*100:.0f}%")
print(f"  Write mode: {mode}")

spark.sql(f"USE CATALOG `{catalog_name}`")

# COMMAND ----------

# MAGIC %md ## 2. Add Module to Path

# COMMAND ----------

import sys
import os

# Try to find the src module
possible_paths = [
    "/Workspace/Users/${workspace.current_user.userName}/.bundle/entity_matching/dev/files",
    os.path.abspath(os.path.join(os.getcwd(), "..")),
    os.path.abspath(os.path.join(os.getcwd(), "../.."))
]

for path in possible_paths:
    if os.path.exists(os.path.join(path, "src")):
        sys.path.insert(0, path)
        print(f"✓ Added to sys.path: {path}")
        break
else:
    print("⚠ Warning: Could not find src module. Will use inline generation.")

# COMMAND ----------

# MAGIC %md ## 3. Generate Reference Entities

# COMMAND ----------

from src.data.large_dataset_generator import LargeDatasetGenerator

print(f"Generating {num_reference} reference entities...")
generator = LargeDatasetGenerator(seed=42)

reference_df = generator.generate_reference_entities(num_entities=num_reference)

print(f"\nSample reference entities:")
display(reference_df.head(10))

print(f"\nReference entity statistics:")
print(f"  Total entities: {len(reference_df)}")
print(f"  Countries: {reference_df['country'].nunique()}")
print(f"  Industries: {reference_df['industry'].nunique()}")
print(f"  With LEI: {reference_df['lei'].notna().sum()}")
print(f"  With CUSIP: {reference_df['cusip'].notna().sum()}")

# COMMAND ----------

# MAGIC %md ## 4. Load Reference Entities to Delta

# COMMAND ----------

from pyspark.sql import SparkSession

# Convert to Spark DataFrame
reference_spark_df = spark.createDataFrame(reference_df)

# Write to Delta table
reference_spark_df.write \
    .format("delta") \
    .mode(mode) \
    .saveAsTable(f"`{catalog_name}`.bronze.spglobal_reference")

# Verify
count = spark.sql(f"SELECT COUNT(*) as cnt FROM `{catalog_name}`.bronze.spglobal_reference").collect()[0].cnt
print(f"✅ Loaded {len(reference_df)} reference entities")
print(f"✅ Total reference entities in table: {count}")

# COMMAND ----------

# MAGIC %md ## 5. Generate Source Entities

# COMMAND ----------

print(f"Generating {num_source} source entities with {match_ratio*100:.0f}% match ratio...")

source_df = generator.generate_source_entities(
    reference_df=reference_df,
    num_entities=num_source,
    match_ratio=match_ratio
)

print(f"\nSample source entities:")
display(source_df.head(10))

print(f"\nSource entity statistics:")
print(f"  Total entities: {len(source_df)}")
print(f"  Source systems: {source_df['source_system'].nunique()}")
print(f"  With ticker: {source_df['ticker'].notna().sum()}")
print(f"  With LEI: {source_df['lei'].notna().sum() if 'lei' in source_df.columns else 0}")
print(f"  With CUSIP: {source_df['cusip'].notna().sum() if 'cusip' in source_df.columns else 0}")

# COMMAND ----------

# MAGIC %md ## 6. Load Source Entities to Delta

# COMMAND ----------

# Convert to Spark DataFrame
source_spark_df = spark.createDataFrame(source_df)

# Write to Delta table
source_spark_df.write \
    .format("delta") \
    .mode(mode) \
    .saveAsTable(f"`{catalog_name}`.bronze.source_entities")

# Verify
count = spark.sql(f"SELECT COUNT(*) as cnt FROM `{catalog_name}`.bronze.source_entities").collect()[0].cnt
print(f"✅ Loaded {len(source_df)} source entities")
print(f"✅ Total source entities in table: {count}")

# COMMAND ----------

# MAGIC %md ## 7. Verify Data Quality

# COMMAND ----------

print("="*80)
print("DATA QUALITY CHECKS")
print("="*80)

# Check reference entities
ref_stats = spark.sql(f"""
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT ciq_id) as unique_ids,
        COUNT(DISTINCT ticker) as unique_tickers,
        SUM(CASE WHEN lei IS NOT NULL THEN 1 ELSE 0 END) as with_lei,
        SUM(CASE WHEN cusip IS NOT NULL THEN 1 ELSE 0 END) as with_cusip,
        COUNT(DISTINCT country) as num_countries,
        COUNT(DISTINCT industry) as num_industries
    FROM `{catalog_name}`.bronze.spglobal_reference
""").collect()[0]

print(f"\nReference Entities ({catalog_name}.bronze.spglobal_reference):")
print(f"  Total records: {ref_stats.total}")
print(f"  Unique CIQ IDs: {ref_stats.unique_ids}")
print(f"  Unique tickers: {ref_stats.unique_tickers}")
print(f"  With LEI: {ref_stats.with_lei} ({ref_stats.with_lei/ref_stats.total*100:.1f}%)")
print(f"  With CUSIP: {ref_stats.with_cusip} ({ref_stats.with_cusip/ref_stats.total*100:.1f}%)")
print(f"  Countries: {ref_stats.num_countries}")
print(f"  Industries: {ref_stats.num_industries}")

# Check source entities
source_stats = spark.sql(f"""
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT source_id, source_system) as unique_ids,
        COUNT(DISTINCT source_system) as num_systems,
        SUM(CASE WHEN ticker IS NOT NULL THEN 1 ELSE 0 END) as with_ticker,
        SUM(CASE WHEN lei IS NOT NULL THEN 1 ELSE 0 END) as with_lei,
        COUNT(DISTINCT country) as num_countries
    FROM `{catalog_name}`.bronze.source_entities
""").collect()[0]

print(f"\nSource Entities ({catalog_name}.bronze.source_entities):")
print(f"  Total records: {source_stats.total}")
print(f"  Unique IDs: {source_stats.unique_ids}")
print(f"  Source systems: {source_stats.num_systems}")
print(f"  With ticker: {source_stats.with_ticker} ({source_stats.with_ticker/source_stats.total*100:.1f}%)")
print(f"  With LEI: {source_stats.with_lei} ({source_stats.with_lei/source_stats.total*100:.1f}%)")
print(f"  Countries: {source_stats.num_countries}")

print("="*80)

# COMMAND ----------

# MAGIC %md ## 8. Sample Data Preview

# COMMAND ----------

print("Sample Reference Entities:")
display(
    spark.sql(f"""
        SELECT ciq_id, company_name, ticker, country, industry
        FROM `{catalog_name}`.bronze.spglobal_reference
        LIMIT 10
    """)
)

print("\nSample Source Entities:")
display(
    spark.sql(f"""
        SELECT source_id, source_system, company_name, ticker, country, industry
        FROM `{catalog_name}`.bronze.source_entities
        LIMIT 10
    """)
)

# COMMAND ----------

# MAGIC %md ## 9. Create Sample Ground Truth (Optional)

# COMMAND ----------

# MAGIC %md
# MAGIC This section creates a ground truth mapping for evaluation purposes.
# MAGIC It tracks which source entities should match which reference entities.

# COMMAND ----------

# Create a ground truth table by analyzing the generation metadata
# Note: This would require the generator to track which source entities were created from which reference entities
# For now, we'll create a sample based on exact ticker matches

spark.sql(f"""
    CREATE OR REPLACE TABLE `{catalog_name}`.bronze.gold_standard AS
    SELECT
        s.source_id,
        s.source_system,
        s.company_name as source_name,
        r.ciq_id,
        r.company_name as reference_name,
        r.ticker,
        1 as is_match,
        'exact_ticker_match' as match_type,
        current_timestamp() as created_at
    FROM `{catalog_name}`.bronze.source_entities s
    INNER JOIN `{catalog_name}`.bronze.spglobal_reference r
        ON s.ticker = r.ticker
        AND s.ticker IS NOT NULL
    LIMIT 500
""")

gold_count = spark.sql(f"SELECT COUNT(*) as cnt FROM `{catalog_name}`.bronze.gold_standard").collect()[0].cnt
print(f"✅ Created ground truth with {gold_count} known matches")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("\n" + "="*80)
print("✅ LARGE-SCALE TEST DATASET LOADED SUCCESSFULLY")
print("="*80)
print(f"\nCatalog: {catalog_name}")
print(f"\nTables Created/Updated:")
print(f"  • bronze.spglobal_reference: {ref_stats.total} reference entities")
print(f"  • bronze.source_entities: {source_stats.total} source entities")
print(f"  • bronze.gold_standard: {gold_count} ground truth matches")
print(f"\nTest Coverage:")
print(f"  • Expected match rate: {match_ratio*100:.0f}%")
print(f"  • Expected matches: ~{int(source_stats.total * match_ratio)}")
print(f"  • Expected non-matches: ~{int(source_stats.total * (1-match_ratio))}")
print(f"\nNext Steps:")
print(f"  1. Run Phase 4 pipeline to match entities")
print(f"  2. Compare results against ground truth")
print(f"  3. Analyze match rates and accuracy")
print("="*80)

# COMMAND ----------

# MAGIC %md ## Optional: Export to CSV

# COMMAND ----------

# Uncomment to export data to CSV for backup or analysis
# reference_df.to_csv("/tmp/reference_entities.csv", index=False)
# source_df.to_csv("/tmp/source_entities.csv", index=False)
# print("✓ Exported data to CSV files in /tmp/")
