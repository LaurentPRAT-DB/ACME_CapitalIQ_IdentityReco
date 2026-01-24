# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Unity Catalog for Entity Matching
# MAGIC
# MAGIC Creates the catalog, schemas, and tables for the entity matching pipeline.

# COMMAND ----------

# MAGIC %md ## Get Parameters

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")

print(f"Catalog: {catalog_name}")

# COMMAND ----------

# MAGIC %md ## Create Catalog

# COMMAND ----------

# Create catalog if not exists
spark.sql(f"""
  CREATE CATALOG IF NOT EXISTS {catalog_name}
  COMMENT 'Entity matching to S&P Capital IQ identifiers'
""")

spark.sql(f"USE CATALOG {catalog_name}")

print(f"✓ Catalog '{catalog_name}' created/verified")

# COMMAND ----------

# MAGIC %md ## Create Schemas

# COMMAND ----------

# Bronze schema - raw data
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS {catalog_name}.bronze
  COMMENT 'Raw source data and S&P reference data'
""")

# Silver schema - cleaned and normalized
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS {catalog_name}.silver
  COMMENT 'Cleaned and normalized entities'
""")

# Gold schema - matched entities with results
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS {catalog_name}.gold
  COMMENT 'Matched entities with CIQ IDs and metrics'
""")

# Models schema - ML models and embeddings
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS {catalog_name}.models
  COMMENT 'ML models, embeddings, and model artifacts'
""")

print(f"✓ Schemas created:")
print(f"  - {catalog_name}.bronze")
print(f"  - {catalog_name}.silver")
print(f"  - {catalog_name}.gold")
print(f"  - {catalog_name}.models")

# COMMAND ----------

# MAGIC %md ## Grant Permissions

# COMMAND ----------

# Grant usage on catalog
spark.sql(f"""
  GRANT USE CATALOG ON CATALOG {catalog_name} TO `account users`
""")

# Grant usage and select on schemas
for schema in ["bronze", "silver", "gold", "models"]:
    spark.sql(f"""
      GRANT USE SCHEMA, SELECT ON SCHEMA {catalog_name}.{schema} TO `account users`
    """)

print("✓ Permissions granted")

# COMMAND ----------

# MAGIC %md ## Verify Setup

# COMMAND ----------

# List schemas
schemas = spark.sql(f"SHOW SCHEMAS IN {catalog_name}").collect()
print(f"\n✓ Verified {len(schemas)} schemas in catalog '{catalog_name}':")
for schema in schemas:
    print(f"  - {schema.databaseName}")

# COMMAND ----------

print("\n" + "="*80)
print("✅ Unity Catalog setup complete!")
print("="*80)
print(f"Catalog: {catalog_name}")
print(f"Schemas: bronze, silver, gold, models")
