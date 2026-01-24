# Databricks notebook source
# MAGIC %md
# MAGIC # Create Reference Tables for Entity Matching
# MAGIC
# MAGIC Creates the Bronze, Silver, and Gold tables for the entity matching pipeline.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Bronze Tables

# COMMAND ----------

# S&P Capital IQ reference data
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {catalog_name}.bronze.spglobal_reference (
    ciq_id STRING NOT NULL COMMENT 'S&P Capital IQ identifier',
    company_name STRING NOT NULL COMMENT 'Company legal name',
    ticker STRING COMMENT 'Stock ticker symbol',
    lei STRING COMMENT 'Legal Entity Identifier',
    cusip STRING COMMENT 'CUSIP identifier',
    isin STRING COMMENT 'ISIN identifier',
    country STRING COMMENT 'Country of incorporation',
    industry STRING COMMENT 'Industry classification',
    sector STRING COMMENT 'Sector classification',
    market_cap DOUBLE COMMENT 'Market capitalization in USD',
    last_updated TIMESTAMP COMMENT 'Last update timestamp',
    CONSTRAINT pk_ciq PRIMARY KEY (ciq_id)
  )
  USING DELTA
  COMMENT 'S&P Capital IQ reference data'
  TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
  )
""")

print(f"✓ Created: {catalog_name}.bronze.spglobal_reference")

# COMMAND ----------

# Source entities table
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {catalog_name}.bronze.source_entities (
    source_id STRING NOT NULL COMMENT 'Source system entity ID',
    source_system STRING NOT NULL COMMENT 'Source system name',
    company_name STRING NOT NULL COMMENT 'Company name as in source',
    ticker STRING COMMENT 'Ticker symbol',
    lei STRING COMMENT 'LEI if available',
    cusip STRING COMMENT 'CUSIP if available',
    isin STRING COMMENT 'ISIN if available',
    country STRING COMMENT 'Country',
    industry STRING COMMENT 'Industry',
    ingestion_timestamp TIMESTAMP DEFAULT current_timestamp() COMMENT 'Ingestion time',
    CONSTRAINT pk_source PRIMARY KEY (source_id, source_system)
  )
  USING DELTA
  PARTITIONED BY (source_system)
  COMMENT 'Source entities to be matched'
  TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
  )
""")

print(f"✓ Created: {catalog_name}.bronze.source_entities")

# COMMAND ----------

# MAGIC %md ## Gold Tables

# COMMAND ----------

# Matched entities with results
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {catalog_name}.gold.matched_entities (
    source_id STRING NOT NULL,
    source_system STRING NOT NULL,
    company_name STRING,

    -- Match results
    matched_ciq_id STRING COMMENT 'Matched S&P CIQ ID',
    match_confidence DOUBLE COMMENT 'Match confidence score (0-1)',
    match_method STRING COMMENT 'Matching method used',
    match_stage STRING COMMENT 'Pipeline stage that matched',
    reasoning STRING COMMENT 'Explanation for the match',

    -- Metadata
    matched_company_name STRING COMMENT 'Matched company name from reference',
    match_timestamp TIMESTAMP DEFAULT current_timestamp(),
    processing_time_ms LONG COMMENT 'Processing time in milliseconds',
    model_version STRING COMMENT 'Model version used',

    -- Computed columns
    needs_review BOOLEAN GENERATED ALWAYS AS (match_confidence < 0.90),
    auto_matched BOOLEAN GENERATED ALWAYS AS (match_confidence >= 0.90),

    CONSTRAINT pk_matched PRIMARY KEY (source_id, source_system)
  )
  USING DELTA
  COMMENT 'Matched entities with CIQ IDs and confidence scores'
  TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.enableChangeDataFeed' = 'true'
  )
""")

print(f"✓ Created: {catalog_name}.gold.matched_entities")

# COMMAND ----------

# MAGIC %md ## Views

# COMMAND ----------

# Review queue view
spark.sql(f"""
  CREATE OR REPLACE VIEW {catalog_name}.gold.review_queue AS
  SELECT
    source_id,
    source_system,
    company_name,
    matched_ciq_id,
    match_confidence,
    match_method,
    reasoning,
    match_timestamp
  FROM {catalog_name}.gold.matched_entities
  WHERE needs_review = true
  ORDER BY match_confidence ASC
  COMMENT 'Entities requiring manual review (confidence < 90%)'
""")

print(f"✓ Created view: {catalog_name}.gold.review_queue")

# COMMAND ----------

# Daily statistics view
spark.sql(f"""
  CREATE OR REPLACE VIEW {catalog_name}.gold.daily_stats AS
  SELECT
    date(match_timestamp) as match_date,
    match_method,
    match_stage,
    COUNT(*) as match_count,
    AVG(match_confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_latency_ms,
    SUM(CASE WHEN auto_matched THEN 1 ELSE 0 END) as auto_matched_count,
    SUM(CASE WHEN needs_review THEN 1 ELSE 0 END) as review_count
  FROM {catalog_name}.gold.matched_entities
  GROUP BY match_date, match_method, match_stage
  ORDER BY match_date DESC
  COMMENT 'Daily matching statistics'
""")

print(f"✓ Created view: {catalog_name}.gold.daily_stats")

# COMMAND ----------

# MAGIC %md ## Sample Data (Optional)

# COMMAND ----------

# Load sample reference data if table is empty
count = spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog_name}.bronze.spglobal_reference").collect()[0].cnt

if count == 0:
    print("Loading sample reference data...")

    # Sample S&P 500 data
    sample_data = [
        ("IQ24937", "Apple Inc.", "AAPL", "HWUPKR0MPOU8FGXBT394", "037833100", "US0378331005", "United States", "Technology Hardware", "Technology"),
        ("IQ4004", "Microsoft Corporation", "MSFT", "INF2BRHRIO1PQHVMMP85", "594918104", "US5949181045", "United States", "Software", "Technology"),
        ("IQ5976767", "Alphabet Inc.", "GOOGL", "5493006MHB84DD0ZWV18", "02079K305", "US02079K3059", "United States", "Interactive Media", "Communication Services"),
        ("IQ3630", "Amazon.com Inc.", "AMZN", "ZXTILKJKG63JELOEG630", "023135106", "US0231351067", "United States", "Internet Retail", "Consumer Discretionary"),
        ("IQ181351", "NVIDIA Corporation", "NVDA", "549300S4KLFTLO7GSQ80", "67066G104", "US67066G1040", "United States", "Semiconductors", "Technology")
    ]

    df = spark.createDataFrame(
        sample_data,
        ["ciq_id", "company_name", "ticker", "lei", "cusip", "isin", "country", "industry", "sector"]
    )

    from pyspark.sql.functions import current_timestamp
    df = df.withColumn("market_cap", lit(None).cast("double"))
    df = df.withColumn("last_updated", current_timestamp())

    df.write.format("delta").mode("append").saveAsTable(f"{catalog_name}.bronze.spglobal_reference")

    print(f"✓ Loaded {len(sample_data)} sample reference entities")
else:
    print(f"✓ Reference table already has {count} entities")

# COMMAND ----------

print("\n" + "="*80)
print("✅ Reference tables created successfully!")
print("="*80)
print(f"\nBronze Tables:")
print(f"  - {catalog_name}.bronze.spglobal_reference")
print(f"  - {catalog_name}.bronze.source_entities")
print(f"\nGold Tables:")
print(f"  - {catalog_name}.gold.matched_entities")
print(f"\nViews:")
print(f"  - {catalog_name}.gold.review_queue")
print(f"  - {catalog_name}.gold.daily_stats")
