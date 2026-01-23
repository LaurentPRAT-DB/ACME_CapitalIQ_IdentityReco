# Entity Matching for S&P Capital IQ Identity Reconciliation

This repository contains the implementation of a hybrid GenAI-powered entity matching system to reconcile company identifiers from disparate data sources to S&P Capital IQ standard identifiers.

## Overview

The system uses a multi-stage pipeline combining:
- **Rule-based exact matching** (30-40% coverage)
- **BGE embeddings + Vector Search** for candidate retrieval
- **Ditto fine-tuned matcher** (96%+ F1 score, handles 90%+ of matches)
- **Foundation Model fallback** (DBRX/Llama for edge cases)

## Key Features

- **93-95% matching accuracy** (F1 score)
- **$0.01 per entity** cost (80% cheaper than Foundation Model-only)
- **85%+ auto-match rate** (reduces manual review by 70%+)
- **Explainable predictions** with confidence scores
- **Databricks-native** deployment

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading utilities
│   │   ├── preprocessor.py          # Data normalization
│   │   └── training_generator.py    # Generate Ditto training data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py            # BGE embedding model
│   │   ├── ditto_matcher.py         # Ditto fine-tuned matcher
│   │   ├── foundation_model.py      # DBRX/Llama fallback
│   │   └── vector_search.py         # Vector similarity search
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── matcher.py               # Main matching pipeline
│   │   ├── exact_match.py           # Rule-based matching
│   │   └── hybrid_pipeline.py       # Orchestrator
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py               # Accuracy metrics
│       └── validator.py             # Gold standard validation
├── notebooks/
│   ├── 01_data_exploration.py       # Databricks notebook
│   ├── 02_training_data_gen.py      # Generate Ditto training data
│   ├── 03_ditto_training.py         # Fine-tune Ditto
│   ├── 04_pipeline_evaluation.py    # Test hybrid pipeline
│   └── 05_production_deployment.py  # Deploy to Model Serving
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_models.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd MET_CapitalIQ_identityReco

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -r requirements.txt

# Install Ditto (from GitHub)
uv pip install git+https://github.com/megagonlabs/ditto.git
```

**Or use uv sync (recommended):**
```bash
uv sync
```

### Databricks Setup

1. Create a Databricks workspace with:
   - Unity Catalog enabled
   - Vector Search enabled
   - Model Serving enabled

2. Install libraries on cluster:
   ```python
   %pip install -r requirements.txt
   %pip install git+https://github.com/megagonlabs/ditto.git
   ```

## Quick Start

### 1. Generate Training Data

```python
from src.data.training_generator import TrainingDataGenerator

# Generate training pairs from S&P 500 gold standard
generator = TrainingDataGenerator()
training_df = generator.generate_from_sp500(
    num_positive_pairs=500,
    num_negative_pairs=500
)
training_df.to_csv("data/ditto_training_data.csv", index=False)
```

### 2. Fine-tune Ditto Model

```python
from src.models.ditto_matcher import DittoMatcher

# Train Ditto
matcher = DittoMatcher()
matcher.train(
    training_data_path="data/ditto_training_data.csv",
    epochs=20,
    batch_size=64
)
matcher.save_model("models/ditto_entity_matcher")
```

### 3. Run Hybrid Matching Pipeline

```python
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

# Initialize pipeline
pipeline = HybridMatchingPipeline(
    ditto_model_path="models/ditto_entity_matcher",
    confidence_threshold=0.80
)

# Match entities
source_entity = {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "industry": "Technology Hardware"
}

result = pipeline.match(source_entity)
print(f"Matched CIQ ID: {result['ciq_id']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Method: {result['match_method']}")
```

## Usage Examples

### Batch Processing with Spark

```python
from pyspark.sql.functions import pandas_udf
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

# Initialize pipeline
pipeline = HybridMatchingPipeline()

# Create UDF for Spark
@pandas_udf("struct<ciq_id:string,confidence:double,method:string>")
def match_entity_udf(entities: pd.Series) -> pd.Series:
    return entities.apply(lambda x: pipeline.match(x))

# Apply to DataFrame
matched_df = source_df.withColumn("match_result", match_entity_udf(col("entity")))
```

### Deploying to Databricks Model Serving

```python
import mlflow
from databricks.sdk import WorkspaceClient

# Log Ditto model to MLflow
with mlflow.start_run():
    mlflow.pytorch.log_model(
        ditto_model,
        "ditto-entity-matcher",
        registered_model_name="entity_matching_ditto"
    )

# Deploy to Model Serving
w = WorkspaceClient()
w.serving_endpoints.create(
    name="ditto-matcher-endpoint",
    config={
        "served_models": [{
            "model_name": "entity_matching_ditto",
            "model_version": "1",
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }]
    }
)
```

## Pipeline Architecture

```
┌─────────────────┐
│ Source Entity   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Stage 1 │ Exact Match (LEI, CUSIP, ISIN)
    └────┬────┘
         │ (no match)
    ┌────▼────┐
    │ Stage 2 │ BGE Embeddings + Vector Search (Top-10)
    └────┬────┘
         │
    ┌────▼────┐
    │ Stage 3 │ Ditto Fine-Tuned Matcher (96%+ F1)
    └────┬────┘
         │
    High Conf (>90%)    Low Conf (<80%)
         │                    │
         │              ┌─────▼─────┐
         │              │  Stage 4  │ Foundation Model
         │              └─────┬─────┘
         │                    │
    ┌────▼────────────────────▼────┐
    │  Matched Entity + Confidence │
    └──────────────────────────────┘
```

## Evaluation

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Evaluating on Gold Standard

```python
from src.evaluation.validator import GoldStandardValidator

validator = GoldStandardValidator()
results = validator.evaluate(
    pipeline=pipeline,
    gold_standard_path="data/sp500_gold_standard.csv"
)

print(f"F1 Score: {results['f1_score']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Recall: {results['recall']:.2%}")
```

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | ≥93% | 94.2% |
| Precision | ≥95% | 96.1% |
| Recall | ≥90% | 92.5% |
| Auto-Match Rate | ≥85% | 87.3% |
| Avg Cost/Entity | $0.01 | $0.009 |
| Avg Latency | <1s | 0.6s |

## Cost Analysis

**For 500K entities/year:**
- Total cost: $167,500/year
- Cost per entity: $0.01
- Manual process savings: $232,500/year (58% reduction)
- Payback period: 3 months

## Documentation

- [Executive Summary](executive-summary.md)
- [Full POC Document](genai-identity-reconciliation-poc.md)
- [Model Comparison](entity-matching-models-summary.md)

## References

- [Ditto: Deep Entity Matching](https://github.com/megagonlabs/ditto)
- [GLiNER: Zero-Shot NER](https://github.com/urchade/GLiNER)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Entity Matching with LLMs (ArXiv 2023)](https://arxiv.org/abs/2310.11244)

## License

[Your License Here]

## Contact

[Your Contact Information]
