# POC: GenAI-Powered Identity Reconciliation to S&P Capital IQ Standard
## Cost-Optimized Databricks-Native Approach

## Executive Summary

This Proof of Concept demonstrates the application of Generative AI to automate the reconciliation of entity identifiers from multiple disparate data sources to S&P Capital IQ standard identifiers. **This optimized approach leverages Databricks-native capabilities and specialized open-source entity matching models to deliver superior accuracy at significantly lower cost.**

**Key Innovation:** Recent research (2024-2025) shows that specialized entity matching models like **Ditto achieve 96.5% F1 score** (vs 85-90% for general-purpose LLMs) while reducing inference costs by **80-98%**. This POC combines the best of both worlds: Ditto's high-accuracy matching with Databricks Foundation Models for edge cases.

**Key Objectives:**
- Automate entity matching across multiple source databases
- Map entities to S&P Capital IQ identifiers (CIQ IDs)
- **Achieve 93-95% matching accuracy** with hybrid Ditto + Foundation Model approach
- Reduce manual reconciliation effort by 60%+
- Minimize costs: **$0.01/entity** (vs $0.30 for GPT-4, $0.05 for DBRX-only)

**Expected Outcomes:**
- Proof of technical feasibility with specialized models + Databricks Foundation Models
- Accuracy benchmarks: 96%+ F1 with Ditto fine-tuning
- Cost-benefit analysis: **$10K/year production cost** (500K entities/year)
- Production-ready implementation roadmap

**POC Investment:**
- **Timeline**: 6 weeks (includes Ditto fine-tuning)
- **Cost**: $50,500 (43% reduction vs traditional approach)
- **ROI**: 53% cost reduction in production ($213K annual savings)

---

## Problem Statement

### Current Challenges

1. **Data Fragmentation**
   - Entity data scattered across multiple databases (CRM, ERP, trading systems, vendor data)
   - Inconsistent naming conventions (e.g., "Apple Inc.", "Apple Computer", "AAPL")
   - Missing or incomplete identifying information

2. **Manual Reconciliation Burden**
   - Time-intensive manual matching process
   - High error rates due to human oversight
   - Scalability limitations
   - Delayed reporting and analytics

3. **Business Impact**
   - Inaccurate entity relationships
   - Duplicate records
   - Incomplete risk assessments
   - Compliance and audit challenges

### Target Use Case

Reconcile company entities from the following source systems:
- Internal CRM database
- Trading system reference data
- Third-party vendor feeds (Bloomberg, Reuters, etc.)
- Legacy systems with historical data

**Example Reconciliation:**
```
Source 1: "Apple Computer Inc." (CRM)
Source 2: "AAPL" (Trading System)
Source 3: "Apple Inc" (Vendor Feed)
→ S&P Capital IQ ID: IQ12345678 (standardized identifier)
```

---

## Proposed Solution

### GenAI Approach

Leverage Large Language Models (LLMs) with the following capabilities:

1. **Semantic Understanding**
   - Understand entity relationships and context
   - Handle abbreviations, acronyms, and variations
   - Process unstructured text descriptions

2. **Fuzzy Matching Enhancement**
   - Go beyond simple string matching
   - Consider industry, location, business description
   - Handle mergers, acquisitions, and name changes

3. **Multi-Attribute Reconciliation**
   - Combine multiple data points (name, address, identifiers, website, industry)
   - Weighted scoring based on field confidence
   - Explainable matching decisions

### Architecture (Databricks-Native with Specialized Models)

```
┌───────────────────────────────────────────────────┐
│              Data Sources (2-3 Pilot)             │
├────────────────┬──────────────┬───────────────────┤
│   CRM DB       │  Trading DB  │  Vendor Feed      │
└────────┬───────┴──────┬───────┴───────┬───────────┘
         │              │               │
         └──────────────┴───────────────┘
                        │
              ┌─────────▼──────────┐
              │ Delta Lake Tables  │
              │  (Bronze Layer)    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Data Normalization │
              │  (Silver Layer)    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Feature Store      │
              │ (Entity Features)  │
              └─────────┬──────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
┌────────▼─────────┐         ┌────────▼────────┐
│ BGE Embeddings   │         │ S&P Capital IQ  │
│ (bge-large-en)   │         │ Reference Data  │
└────────┬─────────┘         │ (Delta Cache)   │
         │                   └────────┬────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
          ┌─────────▼──────────┐
          │ Vector Search      │
          │ (Top-10 Candidates)│
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │ *** DITTO ***      │
          │ Fine-Tuned Matcher │
          │ (BERT-based)       │
          │ 96%+ F1 Score      │
          │ (Model Serving)    │
          └─────────┬──────────┘
                    │
              (90%+ matched)
                    │
              ┌─────┴─────┐
              │           │
      (High Conf)   (Low Conf <80%)
              │           │
              │     ┌─────▼──────────┐
              │     │ Foundation     │
              │     │ Model Fallback │
              │     │ - DBRX/Llama   │
              │     └─────┬──────────┘
              │           │
              └─────┬─────┘
                    │
          ┌─────────▼──────────┐
          │ MLflow Tracking    │
          │ (Confidence Score) │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │ Gold Layer         │
          │ (Matched Entities) │
          └────────────────────┘
```

**Key Improvement:** Ditto fine-tuned matcher handles 90%+ of matches with 96% accuracy, reducing Foundation Model API costs by 80%.

### Specialized Models for Entity Matching (Research Findings)

Recent research (2024-2025) shows several **specialized open-source models** trained specifically for entity matching/resolution that outperform general-purpose LLMs:

#### 1. **Ditto** - Deep Entity Matching with Pre-Trained Language Models
**Best for: Entity matching with structured/semi-structured data**

- **Performance**: Achieves **96.5% F1 score** on company dataset matching (789K vs 412K records)
- **Approach**: Fine-tunes BERT/RoBERTa/DistilBERT for sequence-pair classification
- **Key Features**:
  - Domain knowledge injection (highlight important fields)
  - String summarization for long text
  - Data augmentation
- **Benchmark**: Outperforms prior SOTA by up to **29% F1 score**
- **Repository**: [megagonlabs/ditto](https://github.com/megagonlabs/ditto)
- **When to use**: When you have labeled training data (even small amounts) and structured entity records

#### 2. **GLiNER** - Zero-Shot Entity Recognition
**Best for: Extracting company names and entities from unstructured text**

- **Performance**: Outperforms ChatGPT and fine-tuned LLMs in zero-shot NER benchmarks
- **Approach**: Bidirectional transformer treating NER as entity-span matching
- **Key Features**:
  - No training data required (true zero-shot)
  - Specify entity types at runtime (e.g., "company", "financial institution")
  - 80x cheaper than large LLMs
- **Models on Hugging Face**:
  - `tomaarsen/gliner_medium-v2.1` (general purpose)
  - `numind/NuNER_Zero` (GLiNER architecture)
- **When to use**: For extracting company mentions from unstructured descriptions, news, or text fields

#### 3. **Fine-Tuned Llama 3.1** - Recent Breakthrough
**Best for: Entity matching with instruction tuning**

- **Performance**: Fine-tuned Llama3.1 **exceeds GPT-4 zero-shot by 1-10% F1** on 4/6 datasets
- **Approach**: Instruction tuning on entity matching tasks
- **Key Insight**: With only a few training examples, matches performance of PLMs trained on thousands of examples
- **When to use**: When you need strong performance with minimal training data and want to avoid external APIs

#### 4. **ReLiK** - Lightweight Entity Linking
**Best for: Linking entities to knowledge bases (e.g., S&P Capital IQ identifiers)**

- **Model**: `sapienzanlp/relik-entity-linking-base` on Hugging Face
- **Approach**: Fast and lightweight retrieval + linking
- **When to use**: For mapping extracted entities to standard identifiers in reference databases

#### 5. **BGE Embeddings** + **Vector Search** (Current POC Approach)
**Best for: Semantic similarity matching at scale**

- **Model**: `BAAI/bge-large-en-v1.5` (1024 dimensions)
- **Performance**: State-of-the-art embedding model for semantic search
- **Cost**: Free (open-source)
- **When to use**: For candidate retrieval before final matching decision

---

### Recommended Model Strategy for S&P Capital IQ Reconciliation

Based on the research, here's the **optimal multi-stage approach**:

```
Stage 1: Exact/Fuzzy Matching (Traditional SQL)
  ├─ Match on identifiers (LEI, CUSIP, ISIN) → 30-40% coverage
  └─ Exact company name matches → 10-15% coverage

Stage 2: GLiNER Entity Extraction (if unstructured data exists)
  ├─ Extract company names from descriptions/text
  └─ Normalize entities for matching

Stage 3: BGE Embeddings + Vector Search
  ├─ Generate embeddings for remaining entities
  ├─ Retrieve top-10 candidates from S&P Capital IQ
  └─ 40-50% coverage with high-confidence matches

Stage 4: Ditto Fine-Tuned Matcher (Recommended Alternative to DBRX)
  ├─ Fine-tune Ditto on 500-1000 labeled entity pairs
  ├─ Binary classification: match/no-match for candidate pairs
  ├─ Expected: 96%+ F1 score on company matching
  └─ Cost: One-time training, then inference is cheap

Stage 5: Foundation Model Review (DBRX/Llama)
  ├─ Only for low-confidence cases from Stage 3-4
  └─ Reduces LLM API costs by 70-80%
```

### Updated Cost-Benefit Analysis: Ditto vs Foundation Models

| Approach | Training Cost | Inference Cost/Entity | F1 Score | Pros | Cons |
|----------|---------------|----------------------|----------|------|------|
| **DBRX Foundation Model** | $0 | $0.05 | 85-90% | No training needed, flexible | Higher per-entity cost |
| **Ditto (Fine-Tuned)** | $200-500 | $0.001 | 96%+ | Superior accuracy, very cheap inference | Requires labeled data |
| **GLiNER (Zero-Shot)** | $0 | $0.0001 | 80-85% | Fast, no training, good for NER | Lower accuracy for matching |
| **Hybrid (Recommended)** | $200-500 | $0.01 | 93-95% | Best of both worlds | More complex pipeline |

### Three Approaches Compared

This POC evaluates three approaches. **Our recommendation: Hybrid (Option 3)**.

#### Option 1: External LLM APIs (Baseline - Not Recommended)
- **Models**: GPT-4 or Claude via external APIs
- **Timeline**: 4 weeks (fastest setup)
- **POC Cost**: $65,000
- **Production Cost**: $150,000+/year (500K entities)
- **Accuracy**: 88-92% F1
- **Cost/Entity**: $0.30
- **Pros**: No training, high accuracy out-of-box, flexible
- **Cons**: Very expensive, external dependency, data privacy concerns, rate limits

#### Option 2: Databricks Foundation Models Only (Fast Alternative)
- **Models**: DBRX Instruct / Llama 3.1 70B
- **Timeline**: 5 weeks
- **POC Cost**: $50,000
- **Production Cost**: $187,000/year
- **Accuracy**: 85-90% F1
- **Cost/Entity**: $0.05
- **Pros**: No training, Databricks-native, good cost vs external APIs
- **Cons**: Lower accuracy, higher per-entity cost than specialized models

#### Option 3: Hybrid with Ditto (RECOMMENDED)
- **Models**: Ditto (primary) + DBRX (fallback) + BGE embeddings
- **Timeline**: 6 weeks (+1 week for training)
- **POC Cost**: $59,400 (+$9,400 vs Option 2)
- **Production Cost**: $167,500/year (-$19,500 vs Option 2)
- **Accuracy**: 93-95% F1 (+8% vs Option 2)
- **Cost/Entity**: $0.01 (-80% vs Option 2)
- **Pros**: Highest accuracy, lowest production cost, explainable, research-backed
- **Cons**: Requires training data, +1 week timeline, slightly higher POC cost

**Decision Matrix:**

| Factor | Weight | Option 1 (External) | Option 2 (Foundation) | Option 3 (Hybrid) | Winner |
|--------|--------|---------------------|----------------------|-------------------|--------|
| Accuracy | 30% | 9/10 | 7/10 | 9.5/10 | **Option 3** |
| Production Cost | 25% | 2/10 | 6/10 | 8/10 | **Option 3** |
| Time to Deploy | 15% | 10/10 | 8/10 | 7/10 | Option 1 |
| Explainability | 15% | 6/10 | 7/10 | 9/10 | **Option 3** |
| Data Privacy | 10% | 4/10 | 10/10 | 10/10 | Options 2 & 3 |
| Maintenance | 5% | 8/10 | 9/10 | 7/10 | Option 2 |
| **Weighted Score** | | **6.35** | **7.35** | **8.60** | **Option 3** |

**Recommendation:** **Proceed with Option 3 (Hybrid with Ditto)** for best long-term value.

- **Why**: Superior accuracy (93-95% vs 85-90%) and lowest production cost ($167K vs $187K)
- **ROI**: Additional $9.4K POC investment pays back in 6 months via production savings
- **Risk**: Low - if Ditto training fails, fall back to Option 2 (Foundation Model only)

### Why Databricks-Native Approach?

**Cost Advantages:**
1. **No External API Fees**: Foundation Models charged at ~$0.50/M tokens (vs $2-15/M for OpenAI/Anthropic)
2. **Unified Billing**: Single platform eliminates integration costs and data egress fees
3. **Serverless Scaling**: Model Serving auto-scales to zero between runs (no idle compute)
4. **Open-Source Models**: BGE embeddings are free (vs $0.13/M tokens for OpenAI)

**Technical Advantages:**
1. **Co-Located Processing**: Data and compute in same environment (faster, no transfer costs)
2. **Native Vector Search**: Built-in similarity search (no separate Pinecone/Weaviate subscription)
3. **Unified Governance**: Unity Catalog manages data lineage, access control, audit logs
4. **Integrated MLOps**: MLflow experiment tracking and model versioning included

**Speed Advantages:**
1. **Fewer Integrations**: No external API setup, authentication, rate limit management
2. **Batch Optimization**: Process thousands of entities in parallel with Spark
3. **Pre-Trained Models**: Foundation Models ready to use (no fine-tuning required for POC)

### Technical Components (Databricks-Native Stack)

#### 1. Medallion Architecture (Bronze/Silver/Gold)
- **Bronze Layer**: Raw data ingestion from source systems (Delta Lake)
- **Silver Layer**: Cleansed and normalized entity data
- **Gold Layer**: Matched entities with CIQ IDs (business-ready)
- **Feature Store**: Versioned entity features for ML reproducibility

#### 2. Embedding Generation (Open-Source)
- **BGE-Large-EN**: BAAI's open-source embedding model (1024 dimensions)
- Cost-effective alternative to OpenAI embeddings
- Deployed via Databricks Model Serving
- Batch processing for efficiency (~1000 entities/minute)

#### 3. Vector Search & Candidate Retrieval
- **Databricks Vector Search**: Native vector similarity search
- Index S&P Capital IQ reference data (cached in Delta)
- Retrieve top-10 candidate matches per entity
- Sub-second retrieval times

#### 4. Foundation Model Matching
- **Primary: DBRX Instruct** (Databricks' open model, cost-effective)
- **Fallback: Llama 3.1 70B** (for complex cases)
- Deployed via Databricks Model Serving (serverless)
- Structured prompts for matching decisions + confidence scores

#### 5. MLflow Experiment Tracking
- Track matching experiments and hyperparameters
- Log confidence scores and accuracy metrics
- A/B test different prompts and thresholds
- Model versioning and lineage

#### 6. S&P Capital IQ Integration
- One-time bulk download of reference data
- Stored in Delta Lake (no real-time API costs during POC)
- Refresh cadence: weekly or monthly

#### 7. Human Review Queue
- Low-confidence matches (<80%) flagged for review
- Databricks SQL dashboard for review workflow
- Feedback loop for model improvement

---

## POC Scope

### In-Scope (Focused POC)

1. **Data Sources** (2-3 pilot sources - reduced for faster delivery)
   - Internal CRM (sample: 5,000 entities)
   - Trading system (sample: 3,000 entities)
   - Optional: One vendor feed (sample: 2,000 entities)

2. **Entity Types**
   - **Public companies only** (simplifies validation)
   - S&P 500 constituents prioritized for gold-standard testing

3. **Matching Scenarios**
   - Exact and fuzzy name matches
   - Identifier-based matches (CUSIP, ISIN, LEI)
   - Semantic matching via embeddings

4. **Success Metrics**
   - Matching accuracy (precision/recall)
   - Processing time per entity
   - Confidence score distribution
   - Manual review reduction rate
   - Cost per entity matched

### Out-of-Scope (Future Phases)

- Real-time streaming reconciliation
- Non-company entities (funds, indices, bonds)
- Multi-language support
- Historical entity lineage tracking
- Integration with all enterprise systems

### Timeline (Hybrid Approach with Ditto)

| Phase | Duration | Activities |
|-------|----------|-----------|
| **Phase 1: Setup & Integration** | 1 week | Databricks environment setup, data extraction, S&P reference data load, create training dataset |
| **Phase 2: Development** | 2.5 weeks | Embedding pipeline, vector search, **Ditto fine-tuning**, Foundation Model deployment |
| **Phase 3: Testing & Tuning** | 1.5 weeks | Accuracy testing on gold-standard set, threshold optimization, A/B testing |
| **Phase 4: Analysis & Reporting** | 1 week | Results documentation, cost analysis, executive presentation |

**Total Duration:** 6 weeks

**Comparison:**
- Traditional approach: 8 weeks
- Foundation Model only: 5 weeks
- **Hybrid with Ditto: 6 weeks** (25% faster than traditional, +1 week for 6% accuracy gain)

---

## Implementation Approach (Databricks-Native)

### Step 1: Data Preparation & Setup (Week 1)

1. **Databricks Workspace Setup**
   - Unity Catalog configuration (shared metastore)
   - Cluster setup (Standard_DS4_v2, 2-8 workers autoscaling)
   - Volume storage for reference data

2. **Extract sample datasets** from source systems
   ```sql
   -- Example: Extract CRM entities to Bronze layer
   CREATE OR REPLACE TABLE bronze.crm_entities AS
   SELECT
     entity_id,
     company_name,
     address,
     city,
     country,
     website,
     industry,
     ticker_symbol,
     current_timestamp() as ingestion_time
   FROM crm.companies
   WHERE status = 'ACTIVE'
   LIMIT 5000;
   ```

3. **Normalize to Silver layer** (PySpark)
   ```python
   from pyspark.sql.functions import upper, trim, regexp_replace

   # Standardize company names
   silver_df = bronze_df.withColumn(
       "company_name_normalized",
       upper(trim(regexp_replace("company_name", r"[,\.]", "")))
   )

   silver_df.write.format("delta").mode("overwrite") \
       .saveAsTable("silver.entities_normalized")
   ```

4. **S&P Capital IQ Reference Data**
   - One-time bulk download (API or file drop)
   - Load into Delta table for offline processing
   ```python
   # Cache S&P reference data
   spglobal_df = spark.read.parquet("s3://data/spglobal_universe.parquet")
   spglobal_df.write.format("delta").mode("overwrite") \
       .saveAsTable("reference.spglobal_entities")
   ```

### Step 2: Ditto Training Dataset Creation (Week 1.5-2)

**Critical Step:** Create high-quality training data for Ditto fine-tuning

1. **Generate training pairs from S&P 500 gold standard**
   ```python
   import pandas as pd
   import random

   # Load S&P 500 companies from S&P Capital IQ reference
   sp500 = spark.table("reference.spglobal_entities") \
       .filter(col("index") == "S&P 500") \
       .collect()

   training_pairs = []

   # Positive pairs (same company)
   for company in sp500:
       # Official name vs aliases
       for alias in company.aliases:
           training_pairs.append({
               "left_entity": company.company_name,
               "right_entity": alias,
               "label": 1
           })

       # Name vs ticker
       if company.ticker:
           training_pairs.append({
               "left_entity": company.company_name,
               "right_entity": company.ticker,
               "label": 1
           })

       # Name vs LEI (if available)
       if company.lei:
           training_pairs.append({
               "left_entity": company.company_name,
               "right_entity": f"{company.lei}",
               "label": 1
           })

   # Negative pairs (different companies)
   for i in range(len(training_pairs)):  # Match positive/negative ratio
       c1, c2 = random.sample(sp500, 2)
       training_pairs.append({
           "left_entity": c1.company_name,
           "right_entity": c2.company_name,
           "label": 0
       })

   # Save training data
   train_df = pd.DataFrame(training_pairs)
   train_df.to_csv("/dbfs/entity_matching/ditto_train.csv", index=False)
   print(f"Created {len(training_pairs)} training pairs")
   ```

2. **Manual labeling for edge cases** (optional but recommended)
   - Export 200-300 difficult pairs (low embedding similarity)
   - Human review and labeling via Databricks SQL dashboard
   - Focus on: recent M&A, subsidiaries, name changes

### Step 3: Ditto Fine-Tuning (Week 2)

1. **Set up Ditto environment**
   ```bash
   # Install Ditto on Databricks cluster
   %sh
   git clone https://github.com/megagonlabs/ditto
   cd ditto
   pip install -r requirements.txt
   pip install torch transformers
   ```

2. **Fine-tune Ditto model**
   ```python
   # Configure Ditto training
   !python train_ditto.py \
       --task entity_matching \
       --input_path /dbfs/entity_matching/ditto_train.csv \
       --output_path /dbfs/entity_matching/ditto_model \
       --lm distilbert \
       --epochs 20 \
       --batch_size 64 \
       --lr 3e-5 \
       --max_len 256 \
       --summarize  # Enable string summarization
   ```

3. **Log to MLflow and deploy**
   ```python
   import mlflow
   import torch

   # Log Ditto model
   with mlflow.start_run(run_name="ditto-entity-matcher"):
       mlflow.log_param("model", "distilbert")
       mlflow.log_param("epochs", 20)
       mlflow.log_param("training_pairs", len(training_pairs))

       # Log model
       mlflow.pytorch.log_model(
           ditto_model,
           "ditto-model",
           registered_model_name="ditto-entity-matcher"
       )

   # Deploy to Model Serving
   from databricks.sdk import WorkspaceClient

   w = WorkspaceClient()
   w.serving_endpoints.create(
       name="ditto-matcher-endpoint",
       config={
           "served_models": [{
               "model_name": "ditto-entity-matcher",
               "model_version": "1",
               "scale_to_zero_enabled": True,
               "workload_size": "Small"
           }]
       }
   )
   ```

### Step 4: GenAI Matching Pipeline (Week 2-3)

#### A. Deploy BGE Embedding Model (Databricks Model Serving)

1. **Register BGE model from Hugging Face**
   ```python
   import mlflow
   from mlflow.models import infer_signature

   # Load BGE-large-en model
   model = mlflow.transformers.load_model(
       "BAAI/bge-large-en-v1.5",
       task="feature-extraction"
   )

   # Log to MLflow
   with mlflow.start_run():
       mlflow.transformers.log_model(
           model,
           "bge-embeddings",
           signature=infer_signature(["sample text"], model("sample text"))
       )
   ```

2. **Deploy to Model Serving**
   ```python
   # Deploy via Databricks API
   from databricks.sdk import WorkspaceClient

   w = WorkspaceClient()
   w.serving_endpoints.create(
       name="bge-embeddings-endpoint",
       config={
           "served_models": [{
               "model_name": "bge-embeddings",
               "model_version": "1",
               "scale_to_zero_enabled": True
           }]
       }
   )
   ```

3. **Generate embeddings for all entities (batch)**
   ```python
   from databricks.vector_search.client import VectorSearchClient

   # Create embedding UDF
   @pandas_udf("array<float>")
   def get_embedding(texts: pd.Series) -> pd.Series:
       # Call model serving endpoint
       return texts.apply(lambda x: call_bge_endpoint(x))

   # Apply to all entities
   entities_with_embeddings = silver_df.withColumn(
       "embedding",
       get_embedding(concat_ws(" ", col("company_name"), col("industry")))
   )
   ```

#### B. Set Up Vector Search Index

```python
vsc = VectorSearchClient()

# Create vector search index for S&P reference data
vsc.create_endpoint(name="entity-matching-endpoint")

vsc.create_direct_access_index(
    endpoint_name="entity-matching-endpoint",
    index_name="spglobal_embeddings_index",
    primary_key="ciq_id",
    embedding_dimension=1024,
    embedding_vector_column="embedding",
    schema={
        "ciq_id": "string",
        "company_name": "string",
        "embedding": "array<float>"
    }
)
```

#### C. Deploy Foundation Model for Final Matching

1. **Choose Databricks Foundation Model** (DBRX Instruct - cost-effective)
   ```python
   from databricks.sdk import WorkspaceClient

   w = WorkspaceClient()

   # Use Foundation Model API (pay-per-token, no deployment needed)
   def match_with_dbrx(source_entity, candidates):
       prompt = f"""Given source entity:
       Name: {source_entity['name']}
       Ticker: {source_entity['ticker']}
       Industry: {source_entity['industry']}

       Candidate matches from S&P Capital IQ:
       {format_candidates(candidates)}

       Return JSON: {{"ciq_id": "best_match_id", "confidence": 0-100, "reason": "..."}}
       """

       response = w.serving_endpoints.query(
           name="databricks-dbrx-instruct",
           inputs=[{"prompt": prompt}]
       )
       return parse_json(response.predictions[0])
   ```

#### D. Hybrid Matching Pipeline (RECOMMENDED)

**This approach combines the best of all models for optimal accuracy and cost:**

**Stage 1: Rule-Based Exact Matching** (30-40% coverage, <1ms/entity)
- Match on LEI, CUSIP, ISIN identifiers
- Exact company name matches (normalized)
- Cost: $0 (SQL only)

**Stage 2: Vector Search Candidate Retrieval** (All remaining entities)
- Generate BGE embeddings for source entities
- Retrieve top-10 candidates from S&P Capital IQ via Vector Search
- Sub-second retrieval time
- Cost: $0.0001/entity (open-source embeddings)

**Stage 3: Ditto Fine-Tuned Classification** (90%+ of remaining entities)
- Binary classification: match/no-match for each candidate pair
- Select best match with highest confidence score
- Threshold: >90% confidence = auto-accept
- Performance: 96%+ F1 score
- Cost: $0.001/entity
- Latency: <100ms/entity

**Stage 4: Foundation Model Reasoning** (10% of entities with low Ditto confidence)
- Only for Ditto confidence <80%
- DBRX/Llama provides reasoning for edge cases
- Manual review queue for confidence 70-80%
- Cost: $0.05/entity (but only 10% of entities)

**Overall Pipeline Performance:**
- **Combined accuracy: 93-95% F1**
- **Average cost: $0.01/entity** (90% at $0.001, 10% at $0.05)
- **Average latency: <500ms/entity** (batch processing)
- **Auto-match rate: 85%+** (15% flagged for review)

```python
# Example: Hybrid Pipeline with Ditto
def match_entity_hybrid(source_entity, spglobal_index):
    # Stage 1: Exact match on identifiers
    exact_match = check_exact_match(source_entity)
    if exact_match:
        return exact_match, 1.0  # 100% confidence

    # Stage 2: Vector search for candidates
    embedding = get_bge_embedding(source_entity)
    candidates = vector_search(spglobal_index, embedding, top_k=10)

    # Stage 3: Ditto fine-tuned matcher
    for candidate in candidates:
        ditto_score = ditto_predict(source_entity, candidate)
        if ditto_score > 0.90:  # High confidence
            return candidate, ditto_score

    # Stage 4: Foundation Model for edge cases
    if max([c.score for c in candidates]) > 0.70:
        llm_result = dbrx_match(source_entity, candidates)
        return llm_result

    return None, 0.0  # No match
```

### Step 3: Validation & Tuning (Week 3.5-5)

1. **Create gold-standard test set**
   - Manually verified matches (300-500 entities from S&P 500)
   - Include edge cases (recent M&A, name changes)

2. **MLflow Experiment Tracking**
   ```python
   import mlflow

   with mlflow.start_run(run_name="entity_matching_experiment"):
       # Log parameters
       mlflow.log_param("embedding_model", "bge-large-en")
       mlflow.log_param("foundation_model", "dbrx-instruct")
       mlflow.log_param("confidence_threshold", 0.85)

       # Log metrics
       mlflow.log_metric("precision", precision)
       mlflow.log_metric("recall", recall)
       mlflow.log_metric("f1_score", f1_score)
       mlflow.log_metric("cost_per_entity", avg_cost)
   ```

3. **Hyperparameter tuning**
   - Optimize confidence thresholds (balance precision/recall)
   - Test different prompts for Foundation Model
   - Tune vector search top_k parameter
   - A/B test DBRX vs Llama 3.1

### Step 4: Analysis & Documentation (Week 5)

1. **Performance Summary Dashboard** (Databricks SQL)
   - Match accuracy by confidence bucket
   - Cost breakdown (compute + model serving)
   - Processing latency distribution

2. **Cost Analysis**
   - Per-entity cost calculation
   - Comparison: current manual process vs automated
   - Projected annual costs at scale

3. **Executive Presentation**
   - POC results and recommendations
   - Go/no-go decision criteria
   - Production implementation roadmap

---

## Success Metrics

### Primary KPIs (Hybrid Approach with Ditto)

| Metric | Target (Hybrid) | Foundation Only | Improvement |
|--------|-----------------|-----------------|-------------|
| **Match Accuracy (F1)** | ≥93% | ≥85% | +8% |
| **Precision** | ≥95% | ≥88% | +7% |
| **Recall Rate** | ≥90% | ≥82% | +8% |
| **Auto-Match Rate** | ≥85% | ≥70% | +15% |
| **Processing Time** | <2 sec/entity | <3 sec/entity | 33% faster |
| **Cost Efficiency** | **$0.01/entity** | $0.05/entity | **80% cheaper** |

### Secondary Metrics

| Metric | Target (Hybrid) | Foundation Only |
|--------|-----------------|-----------------|
| False positive rate | <2% | <5% |
| False negative rate | <5% | <10% |
| Human review time reduction | >70% | >60% |
| Confidence score calibration | ±5% accuracy | ±10% accuracy |
| Total POC cost | <$60K | <$50K |

**Key Insight:** The hybrid approach delivers **superior accuracy at 80% lower cost** compared to Foundation Model-only, justifying the additional $9.4K POC investment and +1 week timeline.

---

## Risk Assessment (Hybrid Approach with Ditto)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Ditto training data quality** | High | Medium | Use S&P 500 gold standard for high-quality pairs; manual review of 200-300 edge cases |
| **Ditto overfitting on training set** | Medium | Low | Use validation split (80/20); test on unseen companies outside S&P 500 |
| **Lower accuracy vs GPT-4** | Low | Low | Research shows Ditto achieves 96.5% F1 (better than GPT-4 zero-shot at 88-92%) |
| **S&P data licensing for ML** | Medium | Low | Confirm license permits ML training; document approval before POC start |
| **Ditto fine-tuning compute costs** | Low | Low | One-time 2-4 hour GPU training (<$100); amortized across all future inferences |
| **Model deployment complexity** | Medium | Low | Ditto deploys to standard Model Serving; test inference before scaling |
| **Data quality issues** | High | High | Data profiling first week; cleanse in Silver layer; validate identifiers |
| **Vector search cold start** | Low | Medium | Pre-warm index; acceptable for batch processing (not real-time) |
| **Timeline extension (+1 week)** | Low | Low | Ditto training parallelized with other development; well-documented process |
| **Limited test data** | Low | Low | S&P 500 provides 500+ companies with known variations and aliases |

**Key Risk Mitigation - Ditto Specific:**
- **Fallback Strategy**: If Ditto F1 <90% after training, fall back to Foundation Model-only approach (already developed)
- **Progressive Rollout**: Test Ditto on 10% of entities first, validate accuracy, then scale to 100%
- **A/B Testing**: Run Ditto and DBRX in parallel for first 1000 entities, compare results
- **Model Monitoring**: MLflow tracking of Ditto confidence scores vs actual accuracy; retrain if drift detected

---

## Cost Estimate (Optimized for Databricks-Native)

### Development (POC Phase - 6 Weeks with Ditto)

| Item | Cost | Notes |
|------|------|-------|
| Databricks compute (6 weeks) | $1,400 | Standard cluster with autoscaling (2-8 workers) |
| S&P Capital IQ API access | $3,000 | One-time bulk data pull (not ongoing API) |
| **Ditto fine-tuning** | $500 | Training compute + labeling 500-1000 entity pairs |
| Foundation Model API (DBRX/Llama) | $100 | Reduced to 10% usage (edge cases only) |
| Model Serving (BGE + Ditto) | $400 | Serverless, auto-scale to zero |
| Development team (1.5 engineers, 6 weeks) | $54,000 | Includes Ditto integration |
| **Total POC Cost** | **$59,400** | **Hybrid approach for superior accuracy** |

**Cost Breakdown:**
- **Infrastructure**: $1,900 (3.2% of budget)
- **Data/APIs**: $3,000 (5.0% of budget)
- **Personnel**: $54,000 (90.9% of budget)
- **Ditto Training**: $500 (0.8% of budget)

**Cost Comparison:**
- Original approach (8 weeks, external APIs): $88,500
- Foundation Model only (5 weeks): $50,000
- **Hybrid with Ditto (6 weeks): $59,400** (33% reduction, 6% accuracy gain)

### Production (Annual Estimate - Hybrid Approach)

| Item | Annual Cost | Notes |
|------|-------------|-------|
| Databricks production cluster | $18,000 | Job cluster (runs only during batch processing) |
| S&P Capital IQ subscription | $60,000 | Quarterly reference data refresh |
| **Ditto Model Serving** | $3,000 | Handles 90% of matches @ $0.001/entity |
| **Foundation Model API (DBRX)** | $2,500 | Only 10% of entities (edge cases) @ $0.05/entity |
| Model Serving (BGE embeddings) | $6,000 | Serverless BGE embeddings |
| Storage (Delta Lake) | $3,000 | Vector indices + matched entities |
| Maintenance & support (0.5 FTE) | $75,000 | Part-time oversight and improvements |
| **Total Annual Cost** | **$167,500** | **37% lower than external LLM, 10% lower than DBRX-only** |

**Cost Comparison (500K entities/year):**
| Approach | Annual Cost | Cost/Entity | F1 Score |
|----------|-------------|-------------|----------|
| GPT-4 API | $150,000 | $0.30 | 88-92% |
| DBRX Foundation Model only | $187,000 | $0.05 | 85-90% |
| **Hybrid (Ditto + DBRX)** | **$167,500** | **$0.01** | **93-95%** |

**ROI Analysis:**
- Manual reconciliation cost (current): ~$400,000/year (1.5 FTE @ $150K + error costs)
- **Automated solution cost: $167,500/year**
- **Net savings: $232,500/year (58% cost reduction)**
- **Payback period: 3 months** (including POC investment)
- **Superior accuracy**: 93-95% vs 85-90% (Foundation Model only)

---

## Next Steps

### Immediate Actions (Pre-POC - Hybrid Approach)

1. [ ] Provision Databricks workspace (Unity Catalog, Vector Search, Model Serving enabled)
2. [ ] Secure S&P Capital IQ bulk data access (confirm ML training license permits)
3. [ ] Identify and extract sample datasets from 2-3 source systems
4. [ ] **Prepare Ditto training environment** (install dependencies, clone repo)
5. [ ] **Create training dataset** (500-1000 entity pairs from S&P 500 gold standard)
6. [ ] Enable Foundation Model APIs (DBRX Instruct, Llama 3.1) for fallback
7. [ ] Create gold-standard test set (300-500 S&P 500 entities for validation)
8. [ ] Assign development team (1.5 FTE for 6 weeks) and stakeholders
9. [ ] Set up MLflow experiment tracking workspace
10. [ ] Allocate GPU cluster for Ditto fine-tuning (one-time, 2-4 hours)

### POC Deliverables

1. Working prototype demonstrating GenAI matching pipeline
2. Performance report with accuracy, cost, and timing metrics
3. Sample output: matched entity dataset with confidence scores
4. Code repository and documentation
5. Production implementation roadmap
6. Executive presentation and recommendations

### Decision Points

**Go/No-Go Criteria (Hybrid Approach with Ditto):**
- Achieve **≥93% F1 score** on gold-standard test set (S&P 500 entities)
- Ditto matcher achieves **≥95% precision** on matched pairs
- Process entities in <1 second average (batch mode)
- Demonstrate **≥70% reduction in manual effort** (85%+ auto-match rate)
- Validate average cost **<$0.02/entity** (target: $0.01)
- Confirm Ditto + Databricks stack outperforms external APIs on accuracy

**Success Indicators:**
- ✅ Ditto F1 score ≥96% on training set validation
- ✅ Hybrid pipeline accuracy ≥93% (8% better than Foundation Model only)
- ✅ 80% cost reduction vs Foundation Model-only approach
- ✅ Production cost projection <$200K/year (500K entities)

---

## Appendix

### A. Sample Data Schema

#### Source Entity Schema
```json
{
  "source_id": "CRM-12345",
  "source_system": "SALESFORCE",
  "company_name": "Apple Inc.",
  "address": "One Apple Park Way",
  "city": "Cupertino",
  "state": "CA",
  "country": "United States",
  "postal_code": "95014",
  "website": "www.apple.com",
  "ticker_symbol": "AAPL",
  "lei": "HWUPKR0MPOU8FGXBT394",
  "industry": "Technology Hardware",
  "description": "Designs and manufactures consumer electronics..."
}
```

#### S&P Capital IQ Reference Schema
```json
{
  "ciq_id": "IQ24937",
  "company_name": "Apple Inc.",
  "primary_ticker": "AAPL",
  "exchange": "NASDAQ",
  "country": "United States",
  "industry_code": "45202010",
  "identifiers": {
    "cusip": "037833100",
    "isin": "US0378331005",
    "lei": "HWUPKR0MPOU8FGXBT394"
  },
  "aliases": ["Apple Computer Inc.", "Apple Computer", "AAPL"]
}
```

#### Match Output Schema
```json
{
  "source_id": "CRM-12345",
  "ciq_id": "IQ24937",
  "match_confidence": 98.5,
  "match_method": "hybrid",
  "match_reason": "Exact LEI match + name similarity + location match",
  "review_required": false,
  "matched_timestamp": "2026-01-22T10:30:00Z"
}
```

### B. Technology Stack (Databricks-Native)

| Component | Technology | Cost Model |
|-----------|------------|------------|
| Data Platform | Databricks (Spark, Delta Lake, Unity Catalog) | DBU-based (job cluster) |
| Lakehouse Storage | Delta Lake (Bronze/Silver/Gold) | Storage only |
| Vector Database | Databricks Vector Search | Included with workspace |
| Embedding Model | BGE-Large-EN (BAAI/bge-large-en-v1.5) | Open-source via Model Serving |
| Foundation Model | DBRX Instruct (primary), Llama 3.1 70B (fallback) | Pay-per-token (~$0.50/M tokens) |
| Model Serving | Databricks Model Serving (Serverless) | Auto-scale, pay-per-request |
| Experiment Tracking | MLflow | Included with workspace |
| Orchestration | Databricks Workflows | Included with workspace |
| Monitoring | Lakehouse Monitoring | Included with workspace |
| Notebooks | Databricks Notebooks (Python/SQL) | Included with workspace |
| Version Control | Git integration (GitHub/GitLab) | Free |

**Why Databricks-Native:**
- **Single platform**: No external API dependencies or integration complexity
- **Cost efficiency**: Open-source models, serverless scaling, unified billing
- **Governance**: Unity Catalog for data lineage, access control, audit logs
- **Performance**: Co-located compute and storage (no data egress fees)

### C. Quick-Start Code Examples

#### Using Ditto for Entity Matching

```python
# Install Ditto
# git clone https://github.com/megagonlabs/ditto
# cd ditto && pip install -r requirements.txt

# Fine-tune on your entity pairs
from ditto import train_model

# Prepare training data (CSV format)
# left_entity,right_entity,label
# "Apple Inc.","Apple Computer Inc.",1
# "Microsoft Corp","Google LLC",0

train_model(
    data_path="entity_pairs_train.csv",
    model="distilbert-base-uncased",
    epochs=20,
    batch_size=64
)

# Inference on new pairs
from ditto import predict_matches

predictions = predict_matches(
    model_path="output/model.pt",
    pairs=[
        ("Apple Inc.", "AAPL"),
        ("Microsoft Corp", "MSFT")
    ]
)
# Returns: [(1, 0.98), (1, 0.95)] # (label, confidence)
```

#### Using GLiNER for Company Name Extraction

```python
from gliner import GLiNER

# Load model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Extract company names from text
text = "Apple Inc. and Microsoft Corporation announced a partnership."
labels = ["company", "organization", "financial institution"]

entities = model.predict_entities(text, labels)
# Returns: [
#   {"text": "Apple Inc.", "label": "company", "score": 0.95},
#   {"text": "Microsoft Corporation", "label": "company", "score": 0.93}
# ]

# Use in Databricks
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("array<struct<text:string,label:string,score:double>>")
def extract_companies_udf(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: model.predict_entities(x, ["company"]))

# Apply to DataFrame
df_with_entities = df.withColumn(
    "extracted_companies",
    extract_companies_udf(col("description"))
)
```

#### Using ReLiK for Entity Linking

```python
from relik import Relik

# Load model from Hugging Face
relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-base")

# Link entities to knowledge base
text = "Apple announced new products"
annotations = relik(text)

# Returns entities with KB identifiers
# [{"text": "Apple", "entity_id": "Q312", "label": "Apple Inc."}]
```

### D. References

**Research Papers (2024-2025):**
- [Entity Matching using Large Language Models (2023, updated 2025)](https://arxiv.org/abs/2310.11244)
- [Deep Entity Matching with Pre-Trained Language Models (Ditto)](https://arxiv.org/abs/2004.00584)
- [Effective entity matching with transformers](https://link.springer.com/article/10.1007/s00778-023-00779-z)
- [Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching](https://aclanthology.org/2025.coling-main.8/)
- [Multi-Agent RAG Framework for Entity Resolution (Dec 2025)](https://www.mdpi.com/2073-431X/14/12/525)
- [GLiNER: Generalist Model for Named Entity Recognition (NAACL 2024)](https://aclanthology.org/2024.naacl-long.300.pdf)

**Model Repositories:**
- [Ditto - Entity Matching](https://github.com/megagonlabs/ditto)
- [GLiNER - Zero-Shot NER](https://github.com/urchade/GLiNER)
- [DeepMatcher - Entity Matching Framework](https://github.com/anhaidgroup/deepmatcher)
- [ReLiK Entity Linking - Hugging Face](https://huggingface.co/sapienzanlp/relik-entity-linking-base)

**Hugging Face Models:**
- [GLiNER-medium-v2.1](https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1)
- [BAAI BGE-Large-EN Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [NuNER Zero (GLiNER architecture)](https://huggingface.co/numind/NuNER_Zero)

**Case Studies:**
- [CFM Financial Entity Recognition with Hugging Face](https://huggingface.co/blog/cfm-case-study) - Achieved 6.4% accuracy improvement and 80x cost reduction using open-source models

**Databricks Documentation:**
- Databricks Foundation Model APIs Documentation
- Databricks Vector Search Guide
- Databricks Feature Store Documentation
- MLflow Experiment Tracking Guide

**Standards:**
- Entity Resolution Best Practices (NIST)
- S&P Capital IQ API Documentation

---

## Document Summary

This POC document presents a **hybrid approach combining specialized entity matching models with Databricks Foundation Models** for optimal accuracy and cost-effectiveness:

| Aspect | Original | Foundation Only | Hybrid (Ditto + DBRX) | Improvement |
|--------|----------|-----------------|----------------------|-------------|
| **Timeline** | 8 weeks | 5 weeks | **6 weeks** | 25% faster |
| **POC Cost** | $88,500 | $50,000 | **$59,400** | 33% reduction |
| **Accuracy (F1)** | 88-92% | 85-90% | **93-95%** | +5-8% |
| **Data Sources** | 3-5 sources | 2-3 sources | **2-3 sources** | Focused scope |
| **Cost/Entity** | $0.30 | $0.05 | **$0.01** | 97% reduction |
| **Annual Prod Cost** | $266,000 | $187,000 | **$167,500** | 37% reduction |

**Key Technology Choices (Research-Backed):**
- **Primary Matcher**: Ditto fine-tuned on 500-1000 entity pairs (96%+ F1 score, $0.001/entity)
- **Embedding Model**: BGE-Large-EN (open-source, state-of-the-art)
- **Candidate Retrieval**: Databricks Vector Search (sub-second lookups)
- **Fallback**: DBRX Instruct / Llama 3.1 70B (10% of entities, edge cases only)
- **Infrastructure**: Databricks-native (Unity Catalog, Model Serving, MLflow)

**Why This Approach:**
1. **Superior Accuracy**: 93-95% F1 vs 85-90% (Foundation Model only)
2. **Lower Cost**: $0.01/entity vs $0.05 (80% savings on inference)
3. **Proven Performance**: Ditto achieves 96.5% F1 on company matching benchmarks
4. **Research-Backed**: 2024-2025 studies show specialized models outperform general LLMs for entity matching
5. **Production-Ready**: Explainable predictions, confidence scores, audit trails

**Investment vs Return:**
- **Additional POC Investment**: +$9,400 and +1 week
- **Annual Production Savings**: $19,500/year vs Foundation Model only
- **Accuracy Gain**: +5-8% F1 score
- **Payback Period**: <6 months

---

**Document Version:** 3.0 (Hybrid Approach with Specialized Models)
**Last Updated:** 2026-01-22
**Owner:** [Your Team/Department]
**Status:** Ready for Review & Approval

**Recommendation:** Proceed with hybrid approach for superior accuracy and long-term cost savings.
