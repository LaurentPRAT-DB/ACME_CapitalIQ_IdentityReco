# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 2 & 3: Vector Search + Ditto Matching
# MAGIC
# MAGIC Uses vector search to find candidates, then Ditto matcher to confirm matches.

# COMMAND ----------

# MAGIC %md ## Install Required Libraries
# MAGIC
# MAGIC **IMPORTANT:** Install libraries and restart Python FIRST, before any widget definitions or data loading

# COMMAND ----------

# MAGIC %pip install --upgrade transformers>=4.40.0 sentence-transformers>=2.3.0 torch>=2.1.0 faiss-cpu scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

# COMMAND ----------

dbutils.widgets.text("catalog_name", "entity_matching", "Catalog Name")
dbutils.widgets.text("ditto_endpoint", "ditto-em-dev", "Ditto Endpoint")
dbutils.widgets.text("vector_search_endpoint", "entity-matching-vs-dev", "Vector Search Endpoint")
dbutils.widgets.text("workspace_path", "", "Workspace Path")
dbutils.widgets.dropdown("embeddings_provider", "huggingface", ["huggingface", "databricks"], "Embeddings Provider")
dbutils.widgets.text("embeddings_model_name", "", "Embeddings Model (optional)")

catalog_name = dbutils.widgets.get("catalog_name")
ditto_endpoint = dbutils.widgets.get("ditto_endpoint")
vs_endpoint = dbutils.widgets.get("vector_search_endpoint")
workspace_path = dbutils.widgets.get("workspace_path")
embeddings_provider = dbutils.widgets.get("embeddings_provider")
embeddings_model_name = dbutils.widgets.get("embeddings_model_name") or None

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

# MAGIC %md ## Vector Search + Ditto Matching Implementation

# COMMAND ----------

import time
import pandas as pd
from pyspark.sql.functions import lit, current_timestamp
from databricks.sdk import WorkspaceClient

# Convert unmatched entities to pandas for processing
unmatched_pandas = unmatched_df.toPandas()
print(f"Processing {len(unmatched_pandas)} unmatched entities")

# COMMAND ----------

# MAGIC %md ### 1. Initialize Models

# COMMAND ----------

# Add workspace path for imports if provided
if workspace_path:
    import sys
    sys.path.append(workspace_path)
    print(f"Added to sys.path: {workspace_path}")

from src.models.embeddings import create_embeddings_model
from src.models.ditto_matcher import DittoMatcher
from src.models.vector_search import VectorSearchIndex
from src.data.preprocessor import create_entity_features

# Initialize embeddings model (Hugging Face or Databricks)
print(f"Loading embeddings model (provider: {embeddings_provider})...")
if embeddings_model_name:
    print(f"Using custom model: {embeddings_model_name}")

embeddings_model = create_embeddings_model(
    provider=embeddings_provider,
    model_name=embeddings_model_name,
    databricks_client=WorkspaceClient() if embeddings_provider == "databricks" else None
)

# Initialize Ditto matcher - use serving endpoint for production
print(f"Connecting to Ditto serving endpoint: {ditto_endpoint}")

# Import MLflow deployments for endpoint queries
import mlflow.deployments

try:
    # Initialize deployments client for endpoint queries
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    # Test endpoint connection
    endpoint_info = deploy_client.get_endpoint(ditto_endpoint)
    print(f"✓ Connected to Ditto endpoint: {ditto_endpoint}")
    print(f"  Endpoint state: {endpoint_info.get('state', {}).get('ready', 'unknown')}")
    use_endpoint = True
except Exception as e:
    print(f"⚠ Could not connect to Ditto endpoint: {e}")
    print("Falling back to loading model directly from Unity Catalog")
    use_endpoint = False

    # Fallback: Load model from UC
    ditto_matcher = DittoMatcher()
    ditto_model_path = f"models:/{catalog_name}.models.entity_matching_ditto@Champion"
    try:
        ditto_matcher.load_model(ditto_model_path)
        print(f"✓ Loaded Ditto model from UC: {ditto_model_path}")
    except Exception as e2:
        print(f"⚠ Could not load Ditto model from UC: {e2}")
        print("Continuing without Ditto matcher - will use vector search only")
        ditto_matcher = None

# Helper function to predict using endpoint or direct model
def predict_ditto(left_text, right_text):
    """Query Ditto model via serving endpoint or direct model"""
    if use_endpoint:
        try:
            response = deploy_client.predict(
                endpoint=ditto_endpoint,
                inputs={
                    "dataframe_split": {
                        "columns": ["left_entity", "right_entity"],
                        "data": [[left_text, right_text]]
                    }
                }
            )
            # Parse response - predictions is a list
            predictions = response.get("predictions", [0])
            prediction = predictions[0] if predictions else 0
            # Confidence may be in response or need to be extracted
            confidence = response.get("confidence", [0.5])[0] if "confidence" in response else 0.5
            return prediction, confidence
        except Exception as e:
            print(f"⚠ Endpoint query failed: {e}, falling back to direct model")
            if ditto_matcher:
                return ditto_matcher.predict(left_text, right_text)
            return 0, 0.0
    elif ditto_matcher:
        return ditto_matcher.predict(left_text, right_text)
    else:
        return 0, 0.0

# COMMAND ----------

# MAGIC %md ### 2. Build Vector Search Index from Reference Data

# COMMAND ----------

# Load reference data
reference_df = spark.table(f"{catalog_name}.bronze.spglobal_reference").toPandas()
print(f"Loaded {len(reference_df)} reference entities")

# Generate embeddings for reference data
print("Generating embeddings for reference data...")
reference_texts = []
for _, entity in reference_df.iterrows():
    text = create_entity_features(entity.to_dict())
    reference_texts.append(text)

reference_embeddings = embeddings_model.encode(
    reference_texts,
    batch_size=32,
    show_progress_bar=True
)
print(f"Generated {len(reference_embeddings)} reference embeddings")

# Build vector search index
print("Building vector search index...")
vector_index = VectorSearchIndex(embedding_dim=embeddings_model.embedding_dim)
vector_index.build_index(
    embeddings=reference_embeddings,
    ids=reference_df["ciq_id"].tolist(),
    metadata=reference_df.to_dict('records')
)
print("✓ Vector search index ready")

# COMMAND ----------

# MAGIC %md ### 3. Match Unmatched Entities

# COMMAND ----------

# Match each unmatched entity
matched_results = []
start_time = time.time()

for idx, entity in unmatched_pandas.iterrows():
    entity_start = time.time()

    # Generate embedding for source entity
    entity_text = create_entity_features(entity.to_dict())
    entity_embedding = embeddings_model.encode(entity_text)

    # Find top candidates using vector search
    candidates = vector_index.search(entity_embedding, top_k=5)

    # Use Ditto matcher to verify top candidate if available
    best_match = None
    best_confidence = 0.0
    best_reasoning = "No match found"
    match_method = "vector_search"

    if candidates and (use_endpoint or ditto_matcher):
        # Try Ditto on top candidates (using endpoint or direct model)
        for candidate in candidates[:3]:  # Check top 3
            candidate_entity = candidate["metadata"]
            candidate_text = create_entity_features(candidate_entity)

            # Predict with Ditto (via endpoint or direct model)
            prediction, confidence = predict_ditto(entity_text, candidate_text)

            if prediction == 1 and confidence > best_confidence:
                best_match = candidate
                best_confidence = confidence
                method_type = "endpoint" if use_endpoint else "direct model"
                best_reasoning = f"Ditto match via {method_type} (confidence: {confidence:.2%}, vector similarity: {candidate['similarity']:.2%})"
                match_method = "ditto_matcher"

    elif candidates:
        # Fall back to vector search only if no Ditto
        best_match = candidates[0]
        best_confidence = candidates[0]["similarity"]
        best_reasoning = f"Vector search (similarity: {best_confidence:.2%})"
        match_method = "vector_search"

    # Record match if confidence is sufficient
    if best_match and best_confidence >= 0.70:
        processing_time = int((time.time() - entity_start) * 1000)

        matched_results.append({
            "source_id": entity["source_id"],
            "source_system": entity["source_system"],
            "company_name": entity["company_name"],
            "matched_ciq_id": best_match["ciq_id"],
            "match_confidence": best_confidence,
            "match_method": match_method,
            "match_stage": "Stage 2 & 3: Vector Search + Ditto",
            "reasoning": best_reasoning,
            "matched_company_name": best_match["metadata"].get("company_name", ""),
            "match_timestamp": pd.Timestamp.now(),
            "processing_time_ms": processing_time,
            "model_version": "v1.0"
        })

    # Progress update
    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {idx + 1}/{len(unmatched_pandas)} entities ({elapsed:.1f}s)")

total_time = time.time() - start_time
print(f"\n✓ Matched {len(matched_results)}/{len(unmatched_pandas)} entities in {total_time:.1f}s")
print(f"  Match rate: {len(matched_results)/len(unmatched_pandas)*100:.1f}%")
print(f"  Avg time per entity: {total_time/len(unmatched_pandas)*1000:.1f}ms")

# COMMAND ----------

# MAGIC %md ### 4. Write Results

# COMMAND ----------

# Convert to Spark DataFrame
if matched_results:
    matched_df = spark.createDataFrame(matched_results)
else:
    # Create empty DataFrame with correct schema
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

print(f"✓ Written {matched_df.count()} vector search + Ditto matches to {catalog_name}.silver.vector_ditto_matches_temp")

# COMMAND ----------

print("✅ Stage 2 & 3: Vector search + Ditto matching complete!")
