# Getting Started Guide

**Get up and running with entity matching in 5 minutes**

This guide will help you quickly install, configure, and test the entity matching pipeline locally with sample data.

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd MET_CapitalIQ_identityReco

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Verify installation**:
```bash
python -c "import torch; import sentence_transformers; print('✅ All dependencies installed')"
```

### Step 2: Run Example (1 minute)

```bash
# Run local example with sample data
python example.py
```

**Expected Output**:
```
================================================================================
Entity Matching for S&P Capital IQ - Quick Example
================================================================================

1. Loading data...
   - Reference entities: 500
   - Source entities: 50

2. Initializing pipeline...
   ✓ Pipeline initialized (Stages 1-2: Exact Match + Vector Search)

3. Matching single entity...
   Source Entity:
   - Name: Apple Inc.
   - Ticker: AAPL
   - LEI: HWUPKR0MPOU8FGXBT394

   Match Result:
   - CIQ ID: IQ24937
   - Confidence: 98.50%
   - Method: exact_match
   - Stage: Stage 1: Exact Match (LEI)
   - Reasoning: Exact LEI match

4. Batch matching all entities...
   ✓ Matched 50 entities in 2.3 seconds

5. Pipeline Statistics:
   - Total Entities: 50
   - Matched: 47 (94.0%)
   - Unmatched: 3 (6.0%)
   - Avg Confidence: 93.2%

   Matches by Stage:
     exact_match: 18 (36.0%)
     vector_search: 24 (48.0%)
     ditto_matcher: 5 (10.0%)

6. Generating training data for Ditto...
   ✓ Generated 200 training pairs
   - Positive pairs: 100
   - Negative pairs: 100
   - Saved to: data/ditto_training_sample.csv

================================================================================
Example completed successfully!
================================================================================
```

### Step 3: Verify Results (30 seconds)

```bash
# Check generated training data
ls -lh data/ditto_training_sample.csv

# Preview training data
head -n 5 data/ditto_training_sample.csv
```

---

## 🎯 What You Just Did

1. **Installed** all required Python packages
2. **Ran** the hybrid matching pipeline with sample S&P 500 data
3. **Generated** training data for Ditto fine-tuning

The example demonstrates:
- **Stage 1**: Exact matching on identifiers (LEI, CUSIP, ISIN)
- **Stage 2**: Vector search for semantic similarity (BGE embeddings)
- **Training data generation**: For fine-tuning Ditto matcher

---

## 📊 Understanding the Results

### Match Result Fields

| Field | Description | Example |
|-------|-------------|---------|
| `ciq_id` | S&P Capital IQ identifier | `IQ24937` |
| `confidence` | Match confidence (0-1) | `0.985` (98.5%) |
| `match_method` | Matching stage used | `exact_match`, `vector_search`, `ditto_matcher` |
| `stage_name` | Stage description | `Stage 1: Exact Match (LEI)` |
| `reasoning` | Explanation | `Exact LEI match` |

### Confidence Thresholds

| Confidence | Action | Description |
|------------|--------|-------------|
| ≥ 90% | Auto-match | High confidence, no review needed |
| 70-89% | Review queue | Medium confidence, flag for review |
| < 70% | No match | Low confidence, manual research needed |

**Target**: 85%+ auto-match rate (confidence ≥ 90%)

---

## 🔍 Next Steps

### Option A: Continue Local Testing

→ See [TESTING_GUIDE.md](TESTING_GUIDE.md) for:
- Spark Connect setup (test with remote Databricks)
- Comprehensive testing scenarios
- Troubleshooting guide

### Option B: Train Ditto Model

```bash
# Use the generated training data to fine-tune Ditto
python -m src.models.ditto_matcher \
    --training-data data/ditto_training_sample.csv \
    --output models/ditto_matcher \
    --epochs 20
```

→ See [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) for Databricks training

### Option C: Deploy to Production

→ See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for:
- Unity Catalog setup
- Model Serving deployment
- Vector Search configuration
- Scheduled jobs

---

## 🧪 Try These Examples

### Example 1: Match a Specific Entity

```python
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

# Load data
loader = DataLoader()
reference_df = loader.load_reference_data()

# Initialize pipeline
pipeline = HybridMatchingPipeline(reference_df=reference_df)

# Match Microsoft
entity = {
    "company_name": "Microsoft Corporation",
    "ticker": "MSFT",
    "cusip": "594918104"
}

result = pipeline.match(entity)
print(f"Matched: {result['ciq_id']} ({result['confidence']:.1%} confidence)")
print(f"Method: {result['match_method']}")
print(f"Reasoning: {result['reasoning']}")
```

### Example 2: Batch Match Multiple Entities

```python
# Match multiple entities
entities = [
    {"company_name": "Apple Inc.", "ticker": "AAPL"},
    {"company_name": "Alphabet Inc.", "ticker": "GOOGL"},
    {"company_name": "Amazon.com Inc.", "ticker": "AMZN"}
]

results = pipeline.batch_match(entities)

for i, result in enumerate(results):
    print(f"{i+1}. {entities[i]['company_name']}")
    print(f"   → CIQ ID: {result['ciq_id']} ({result['confidence']:.1%})")
```

### Example 3: Get Pipeline Statistics

```python
# Get detailed statistics
stats = pipeline.get_pipeline_stats(results)

print(f"Match Rate: {stats['match_rate']:.1%}")
print(f"Avg Confidence: {stats['avg_confidence']:.1%}")
print(f"Auto-Match Rate: {stats['auto_match_rate']:.1%}")

print("\nMatches by Stage:")
for method, count in stats['by_method'].items():
    print(f"  {method}: {count} ({count/len(results)*100:.1%})")
```

---

## 📁 Sample Data

The example uses synthetic S&P 500 company data with:

### Reference Data (500 companies)
- Company name
- Ticker symbol
- LEI (Legal Entity Identifier)
- CUSIP (Committee on Uniform Securities Identification Procedures)
- ISIN (International Securities Identification Number)
- Industry classification
- Country

### Source Data (50 test entities)
- Variations of company names (e.g., "Apple Computer Inc." vs "Apple Inc.")
- Missing identifiers (to test semantic matching)
- Abbreviations and acronyms

---

## 🔧 Configuration

### Default Configuration

The example runs with these defaults:
```python
# src/config.py
EXACT_MATCH_FIELDS = ["lei", "cusip", "isin"]
VECTOR_SEARCH_TOP_K = 10
DITTO_CONFIDENCE_THRESHOLD = 0.80
AUTO_MATCH_THRESHOLD = 0.90
FOUNDATION_MODEL_THRESHOLD = 0.80
```

### Customize Configuration

Create a `config.yaml` file:
```yaml
# config.yaml
pipeline:
  exact_match_fields:
    - lei
    - cusip
    - isin
    - ticker  # Add ticker to exact match

  vector_search:
    top_k: 20  # Retrieve more candidates
    model: "BAAI/bge-large-en-v1.5"

  ditto:
    confidence_threshold: 0.75  # Lower threshold
    model_path: "models/ditto_entity_matcher"

  foundation_model:
    enabled: false  # Disable for testing
    model: "databricks-dbrx-instruct"
```

Load custom config:
```python
from src.config import Config

config = Config.from_yaml("config.yaml")
pipeline = HybridMatchingPipeline(config=config)
```

---

## 🐛 Troubleshooting

### Issue: "Module not found: torch"

```bash
# Install PyTorch
pip install torch==2.1.0

# Or install all dependencies
pip install -r requirements.txt
```

### Issue: "Module not found: sentence_transformers"

```bash
pip install sentence-transformers==2.2.2
```

### Issue: "No such file or directory: data/"

```bash
# Create data directory
mkdir -p data

# Run example again
python example.py
```

### Issue: Example runs slowly

The first run downloads BGE embeddings (~1.2GB). Subsequent runs are faster:
- First run: ~30 seconds
- Subsequent runs: ~3 seconds

**Speed up downloads**:
```bash
# Pre-download embeddings
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
```

### Issue: Out of memory

If running on a machine with <8GB RAM:
```python
# Use smaller embedding model
from src.config import Config

config = Config(embedding_model="BAAI/bge-small-en-v1.5")  # 133MB vs 1.2GB
pipeline = HybridMatchingPipeline(config=config)
```

---

## ✅ Success Checklist

You're ready to move forward when:

- [ ] `python example.py` runs successfully
- [ ] Training data generated in `data/ditto_training_sample.csv`
- [ ] Match confidence ≥ 90% for most entities
- [ ] Understand pipeline stages and confidence thresholds

---

## 🎓 Learn More

### Documentation
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing with Spark Connect
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Deploy to Databricks
- [README.md](README.md) - Full project overview

### Business Case
- [executive-summary.md](executive-summary.md) - ROI and business value
- [genai-identity-reconciliation-poc.md](genai-identity-reconciliation-poc.md) - POC specification

### Models & Research
- [entity-matching-models-summary.md](entity-matching-models-summary.md) - Model comparison
- [Ditto Paper](https://arxiv.org/abs/2004.00584) - Deep entity matching research

### Notebooks (Databricks)
- [notebooks/01_quick_start.py](notebooks/01_quick_start.py) - Databricks quick start
- [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) - Train Ditto
- [notebooks/03_full_pipeline_example.py](notebooks/03_full_pipeline_example.py) - Production pipeline

---

## 💡 Tips

1. **Start with exact matches**: Test with entities that have LEI/CUSIP for high confidence
2. **Review training data**: Inspect `data/ditto_training_sample.csv` to understand positive/negative pairs
3. **Tune thresholds**: Adjust `AUTO_MATCH_THRESHOLD` based on your precision requirements
4. **Monitor performance**: Track match rates and confidence distributions

---

## 🎯 Quick Commands Reference

```bash
# Install
pip install -r requirements.txt

# Run example
python example.py

# Generate more training data
python -c "from src.data.training_generator import TrainingDataGenerator; \
          gen = TrainingDataGenerator(); \
          df = gen.generate_from_sp500(num_positive_pairs=500, num_negative_pairs=500); \
          df.to_csv('data/training_1000.csv', index=False)"

# Train Ditto (after generating training data)
python -m src.models.ditto_matcher --training-data data/training_1000.csv

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

---

**Ready for production?** → [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)

**Need more testing?** → [TESTING_GUIDE.md](TESTING_GUIDE.md)

**Target: 93-95% F1 Score | $0.01/entity | 85%+ Auto-Match Rate**
