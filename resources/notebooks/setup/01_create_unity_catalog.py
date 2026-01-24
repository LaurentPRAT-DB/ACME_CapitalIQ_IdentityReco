# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Setup Unity Catalog Schemas
# MAGIC
# MAGIC This notebook creates the schema structure for entity matching.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog must exist (created in Phase 0)
# MAGIC - User must have CREATE SCHEMA privileges on the catalog

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog_name", "laurent_prat_entity_matching_dev", "Catalog Name")
catalog_name = dbutils.widgets.get("catalog_name")

print(f"Setting up schemas in catalog: {catalog_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Catalog Exists

# COMMAND ----------

# Verify catalog exists
try:
    spark.sql(f"USE CATALOG `{catalog_name}`")
    print(f"✓ Using catalog '{catalog_name}'")
except Exception as e:
    print(f"✗ Error: Catalog '{catalog_name}' does not exist or is not accessible")
    print(f"   Please run Phase 0 first: python scripts/create_catalog.py --catalog-name {catalog_name} --owner <your-email>")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schemas

# COMMAND ----------

# Create bronze schema (raw/landing data)
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.bronze
  COMMENT 'Bronze layer - raw source data'
""")
print(f"✓ Schema '{catalog_name}.bronze' created/verified")

# COMMAND ----------

# Create silver schema (cleansed/standardized data)
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.silver
  COMMENT 'Silver layer - cleansed and standardized data'
""")
print(f"✓ Schema '{catalog_name}.silver' created/verified")

# COMMAND ----------

# Create gold schema (business-level aggregates)
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.gold
  COMMENT 'Gold layer - business-level aggregates and analytics'
""")
print(f"✓ Schema '{catalog_name}.gold' created/verified")

# COMMAND ----------

# Create models schema (ML models)
spark.sql(f"""
  CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.models
  COMMENT 'Models schema - Machine learning models'
""")
print(f"✓ Schema '{catalog_name}.models' created/verified")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# List all schemas
schemas = spark.sql(f"SHOW SCHEMAS IN `{catalog_name}`").collect()

print("\n" + "=" * 80)
print(f"Schemas in catalog '{catalog_name}':")
print("=" * 80)
for schema in schemas:
    print(f"  - {schema.databaseName}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"""
✓ Unity Catalog setup completed successfully!

Catalog: {catalog_name}
Schemas created:
  - {catalog_name}.bronze  (raw source data)
  - {catalog_name}.silver  (cleansed data)
  - {catalog_name}.gold    (business aggregates)
  - {catalog_name}.models  (ML models)

Next steps:
1. Run the create_reference_tables job
2. Proceed with model training
""")
