# Testing Guide

**Comprehensive guide for local development and testing with Spark Connect**

This guide covers testing the entity matching pipeline locally, with remote Databricks execution via Spark Connect, and preparing for production deployment.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Testing](#quick-start-testing)
3. [Local Testing (Pandas Only)](#local-testing-pandas-only)
4. [Spark Connect Testing (Remote Databricks)](#spark-connect-testing-remote-databricks)
5. [Unit Tests](#unit-tests)
6. [Performance Testing](#performance-testing)
7. [Troubleshooting](#troubleshooting)
8. [Testing Cheatsheet](#testing-cheatsheet)

---

## Prerequisites

### Required

- ✅ Python 3.9+ installed
- ✅ Virtual environment activated
- ✅ Dependencies installed (`pip install -r requirements.txt`)

### For Spark Connect (Remote Databricks)

- ✅ Databricks workspace access
- ✅ Databricks CLI installed and configured
- ✅ Running Databricks cluster or serverless warehouse
- ✅ Personal Access Token (PAT)

### Optional

- Git (for version control)
- Docker (for containerized testing)
- pytest (for unit tests)

---

## Quick Start Testing

### 1. Local Pandas Test (2 minutes)

Test the pipeline with local sample data (no Databricks required):

```bash
# Run example
python example.py
```

**What this tests:**
- Data loading and preprocessing
- Exact matching (Stage 1)
- Vector search (Stage 2)
- Training data generation
- Pipeline orchestration

**Expected output**: 94% match rate, 93.2% average confidence

### 2. Spark Connect Test (5 minutes)

Test with remote Databricks execution:

```bash
# Configure Databricks CLI
databricks configure --profile DEFAULT

# Set cluster ID
echo "SPARK_CONNECT_CLUSTER_ID=your-cluster-id" >> .env

# Test connection
python test_spark_connect.py

# Run Spark Connect example
python example_spark_connect.py
```

**What this tests:**
- Databricks authentication
- Spark Connect connection
- Remote DataFrame operations
- Pandas UDF execution
- Delta table read/write

---

## Local Testing (Pandas Only)

### Setup

```bash
# Activate environment
source .venv/bin/activate

# Verify dependencies
python -c "import pandas, torch, sentence_transformers; print('✅ Ready')"
```

### Test 1: Single Entity Matching

```python
# test_single_match.py
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

# Load data
loader = DataLoader()
reference_df = loader.load_reference_data()

# Initialize pipeline
pipeline = HybridMatchingPipeline(
    reference_df=reference_df,
    ditto_model_path=None,  # No Ditto for quick test
    enable_foundation_model=False
)

# Test cases
test_entities = [
    {
        "name": "Test 1: Exact LEI Match",
        "entity": {"company_name": "Apple Inc.", "lei": "HWUPKR0MPOU8FGXBT394"},
        "expected_method": "exact_match",
        "expected_confidence": 1.0
    },
    {
        "name": "Test 2: Exact CUSIP Match",
        "entity": {"company_name": "Microsoft Corporation", "cusip": "594918104"},
        "expected_method": "exact_match",
        "expected_confidence": 1.0
    },
    {
        "name": "Test 3: Vector Search Match",
        "entity": {"company_name": "Alphabet Inc", "ticker": "GOOGL"},
        "expected_method": "vector_search",
        "expected_confidence": 0.85  # Minimum
    }
]

# Run tests
for test in test_entities:
    print(f"\n{test['name']}")
    result = pipeline.match(test['entity'])

    # Validate
    assert result['match_method'] == test['expected_method'], \
        f"Expected {test['expected_method']}, got {result['match_method']}"
    assert result['confidence'] >= test['expected_confidence'], \
        f"Expected confidence >= {test['expected_confidence']}, got {result['confidence']}"

    print(f"  ✅ Passed: {result['ciq_id']} ({result['confidence']:.1%})")

print("\n✅ All tests passed!")
```

Run:
```bash
python test_single_match.py
```

### Test 2: Batch Processing

```python
# test_batch_processing.py
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
import time

loader = DataLoader()
reference_df = loader.load_reference_data()
pipeline = HybridMatchingPipeline(reference_df=reference_df)

# Load source entities
source_entities = loader.load_sample_entities()[:100]  # Test with 100 entities

# Benchmark
start_time = time.time()
results = pipeline.batch_match(source_entities)
elapsed = time.time() - start_time

# Statistics
matched = sum(1 for r in results if r['ciq_id'] is not None)
avg_confidence = sum(r['confidence'] for r in results if r['ciq_id']) / matched
auto_matched = sum(1 for r in results if r['confidence'] >= 0.90)

print(f"\n=== Batch Processing Results ===")
print(f"Total Entities: {len(source_entities)}")
print(f"Matched: {matched} ({matched/len(source_entities)*100:.1%})")
print(f"Avg Confidence: {avg_confidence:.1%}")
print(f"Auto-Match Rate: {auto_matched/len(source_entities)*100:.1%}")
print(f"Processing Time: {elapsed:.2f}s")
print(f"Throughput: {len(source_entities)/elapsed:.1f} entities/sec")

# Validate targets
assert matched/len(source_entities) >= 0.85, "Match rate below 85% target"
assert avg_confidence >= 0.90, "Avg confidence below 90% target"
assert auto_matched/len(source_entities) >= 0.85, "Auto-match rate below 85% target"

print("\n✅ All targets met!")
```

### Test 3: Training Data Generation

```python
# test_training_data.py
from src.data.training_generator import TrainingDataGenerator
from src.data.loader import DataLoader

loader = DataLoader()
reference_df = loader.load_reference_data()

generator = TrainingDataGenerator()
training_df = generator.generate_from_sp500(
    reference_df=reference_df,
    num_positive_pairs=500,
    num_negative_pairs=500
)

print(f"\n=== Training Data Generated ===")
print(f"Total Pairs: {len(training_df)}")
print(f"Positive Pairs: {(training_df['label'] == 1).sum()}")
print(f"Negative Pairs: {(training_df['label'] == 0).sum()}")

# Validate
assert len(training_df) == 1000, "Expected 1000 training pairs"
assert (training_df['label'] == 1).sum() == 500, "Expected 500 positive pairs"
assert (training_df['label'] == 0).sum() == 500, "Expected 500 negative pairs"
assert 'entity1_name' in training_df.columns, "Missing entity1_name column"
assert 'entity2_name' in training_df.columns, "Missing entity2_name column"

# Save
training_df.to_csv("data/test_training_data.csv", index=False)
print("✅ Saved to data/test_training_data.csv")
```

---

## Spark Connect Testing (Remote Databricks)

### Setup (One-Time)

#### Step 1: Install Databricks CLI

```bash
pip install databricks-cli
```

#### Step 2: Configure Authentication

```bash
databricks configure --profile DEFAULT
```

You'll be prompted for:
- **Host**: `https://your-workspace.cloud.databricks.com`
- **Token**: Get from Databricks UI → User Settings → Developer → Access Tokens

**Create a token:**
1. Click user icon (top right)
2. Settings → Developer
3. Access Tokens → Generate New Token
4. Copy token (you won't see it again!)

#### Step 3: Get Cluster ID

**Option A: From Databricks UI**
1. Compute → Select your cluster
2. Configuration tab → Copy Cluster ID
3. Format: `1234-567890-abcdefgh`

**Option B: Using CLI**
```bash
databricks clusters list
```

**Option C: Create New Cluster**
```bash
databricks clusters create --json '{
  "cluster_name": "entity-matching-dev",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "num_workers": 2,
  "autotermination_minutes": 30
}'
```

#### Step 4: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env
```

Add:
```bash
DATABRICKS_PROFILE=DEFAULT
SPARK_CONNECT_CLUSTER_ID=your-cluster-id
USE_SPARK_CONNECT=true
```

### Test Connection

```bash
python test_spark_connect.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════════════╗
║                  Spark Connect Configuration Test                    ║
╚══════════════════════════════════════════════════════════════════════╝

Testing Databricks CLI Configuration
✓ Databricks CLI is configured
✓ DATABRICKS_HOST=your-workspace.cloud.databricks.com
✓ DATABRICKS_TOKEN=dapi****** (hidden)

Testing Environment Variables
✓ SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
✓ DATABRICKS_PROFILE=DEFAULT

Testing Spark Connect
✓ Successfully connected to Databricks via Spark Connect
Spark version: 3.5.0
✓ Created test DataFrame with 10 rows
✓ SQL query successful: Hello from Databricks!

TEST SUMMARY
  Databricks CLI:        ✓ PASS
  Environment Variables: ✓ PASS
  Spark Connect:         ✓ PASS

🎉 SUCCESS! Your Spark Connect is configured correctly!
```

### Run Spark Connect Example

```bash
python example_spark_connect.py
```

**What this does:**
1. Connects to Databricks cluster
2. Loads reference data
3. Creates sample source entities in Spark DataFrame
4. Applies matching using Pandas UDF
5. Writes results to Delta table
6. Shows statistics

### Advanced Spark Connect Tests

#### Test 1: DataFrame Operations

```python
# test_spark_dataframe.py
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()
spark = get_spark_session()

# Test 1: Range
print("Test 1: Creating DataFrame...")
df = spark.range(1000)
assert df.count() == 1000, "Count mismatch"
print("✅ Passed")

# Test 2: SQL Query
print("Test 2: SQL Query...")
result = spark.sql("SELECT 1 + 1 as result").collect()[0][0]
assert result == 2, "SQL query failed"
print("✅ Passed")

# Test 3: DataFrame transformations
print("Test 3: Transformations...")
from pyspark.sql.functions import col
df2 = df.withColumn("doubled", col("id") * 2)
assert df2.columns == ["id", "doubled"], "Column mismatch"
print("✅ Passed")

# Test 4: Aggregations
print("Test 4: Aggregations...")
total = df.selectExpr("sum(id) as total").collect()[0][0]
expected = sum(range(1000))
assert total == expected, f"Expected {expected}, got {total}"
print("✅ Passed")

print("\n✅ All Spark Connect tests passed!")
spark.stop()
```

#### Test 2: Pandas UDF

```python
# test_pandas_udf.py
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd

load_dotenv()
spark = get_spark_session()

# Define Pandas UDF
@pandas_udf(DoubleType())
def square_udf(values: pd.Series) -> pd.Series:
    return values ** 2

# Test
df = spark.range(10)
result_df = df.withColumn("squared", square_udf("id"))

# Verify
results = result_df.select("id", "squared").collect()
for row in results:
    assert row.squared == row.id ** 2, f"UDF failed for {row.id}"

print("✅ Pandas UDF test passed!")
spark.stop()
```

#### Test 3: Delta Table Operations

```python
# test_delta_table.py
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()
spark = get_spark_session()

# Create test DataFrame
test_data = [
    ("TEST-001", "Apple Inc.", "AAPL"),
    ("TEST-002", "Microsoft Corporation", "MSFT")
]
df = spark.createDataFrame(test_data, ["id", "name", "ticker"])

# Write to Delta (use /tmp for testing)
table_path = "/tmp/entity_matching_test"
df.write.format("delta").mode("overwrite").save(table_path)
print(f"✅ Written to {table_path}")

# Read back
df_read = spark.read.format("delta").load(table_path)
assert df_read.count() == 2, "Count mismatch after read"
print("✅ Read from Delta table")

# Update
from delta.tables import DeltaTable
delta_table = DeltaTable.forPath(spark, table_path)
delta_table.update(
    condition="ticker = 'AAPL'",
    set={"name": "'Apple Computer Inc.'"}
)
print("✅ Updated Delta table")

# Verify update
df_updated = spark.read.format("delta").load(table_path)
apple_row = df_updated.filter("ticker = 'AAPL'").collect()[0]
assert apple_row.name == "Apple Computer Inc.", "Update failed"
print("✅ Update verified")

print("\n✅ All Delta table tests passed!")
spark.stop()
```

---

## Unit Tests

### Setup pytest

```bash
pip install pytest pytest-cov
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Structure

```
tests/
├── __init__.py
├── test_pipeline.py         # Pipeline tests
├── test_exact_match.py       # Stage 1 tests
├── test_vector_search.py     # Stage 2 tests
├── test_ditto_matcher.py     # Stage 3 tests
├── test_preprocessor.py      # Data preprocessing tests
└── test_metrics.py           # Evaluation metrics tests
```

### Example Test

```python
# tests/test_exact_match.py
import pytest
from src.pipeline.exact_match import ExactMatcher
from src.data.loader import DataLoader

@pytest.fixture
def reference_data():
    loader = DataLoader()
    return loader.load_reference_data()

@pytest.fixture
def matcher(reference_data):
    return ExactMatcher(reference_data)

def test_lei_exact_match(matcher):
    entity = {"lei": "HWUPKR0MPOU8FGXBT394"}
    result = matcher.match(entity)

    assert result is not None, "LEI match should not be None"
    assert result['confidence'] == 1.0, "Exact match should have 100% confidence"
    assert result['match_method'] == 'exact_match', "Method should be exact_match"

def test_cusip_exact_match(matcher):
    entity = {"cusip": "594918104"}
    result = matcher.match(entity)

    assert result is not None
    assert result['confidence'] == 1.0
    assert 'reasoning' in result

def test_no_identifiers(matcher):
    entity = {"company_name": "Unknown Company Inc."}
    result = matcher.match(entity)

    assert result is None, "Should return None when no identifiers match"

def test_invalid_identifier(matcher):
    entity = {"lei": "INVALID_LEI_123"}
    result = matcher.match(entity)

    assert result is None, "Invalid identifier should not match"
```

---

## Performance Testing

### Test 1: Throughput Benchmark

```python
# test_performance_throughput.py
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
import time

loader = DataLoader()
reference_df = loader.load_reference_data()
pipeline = HybridMatchingPipeline(reference_df=reference_df)

# Test with increasing batch sizes
batch_sizes = [10, 50, 100, 500, 1000]

print("\n=== Throughput Benchmark ===")
print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput (entities/s)':<25}")
print("-" * 50)

for size in batch_sizes:
    entities = loader.load_sample_entities()[:size]

    start = time.time()
    results = pipeline.batch_match(entities)
    elapsed = time.time() - start

    throughput = size / elapsed
    print(f"{size:<12} {elapsed:<10.2f} {throughput:<25.1f}")

# Target: >100 entities/second (local), >1000 entities/second (Spark)
```

### Test 2: Latency Benchmark

```python
# test_performance_latency.py
import time
import statistics

# Test latency for different match methods
test_cases = {
    "Exact Match (LEI)": {"lei": "HWUPKR0MPOU8FGXBT394"},
    "Exact Match (CUSIP)": {"cusip": "594918104"},
    "Vector Search": {"company_name": "Apple Computer Inc."},
    "Complex Match": {"company_name": "MSFT"}
}

print("\n=== Latency Benchmark ===")
print(f"{'Method':<25} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
print("-" * 65)

for method_name, entity in test_cases.items():
    latencies = []

    for _ in range(100):  # 100 iterations
        start = time.time()
        result = pipeline.match(entity)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]

    print(f"{method_name:<25} {avg:<12.1f} {p50:<12.1f} {p95:<12.1f}")

# Target: <100ms P95 latency for exact match, <500ms for vector search
```

### Test 3: Memory Usage

```python
# test_performance_memory.py
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("\n=== Memory Usage Benchmark ===")
baseline = get_memory_mb()
print(f"Baseline: {baseline:.1f} MB")

# Load reference data
from src.data.loader import DataLoader
loader = DataLoader()
reference_df = loader.load_reference_data()
after_load = get_memory_mb()
print(f"After loading reference: {after_load:.1f} MB (+{after_load - baseline:.1f} MB)")

# Initialize pipeline
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
pipeline = HybridMatchingPipeline(reference_df=reference_df)
after_init = get_memory_mb()
print(f"After pipeline init: {after_init:.1f} MB (+{after_init - after_load:.1f} MB)")

# Process batch
entities = loader.load_sample_entities()[:1000]
results = pipeline.batch_match(entities)
after_batch = get_memory_mb()
print(f"After batch processing: {after_batch:.1f} MB (+{after_batch - after_init:.1f} MB)")

print(f"\nTotal memory usage: {after_batch:.1f} MB")
# Target: <4GB for 500K entity reference dataset
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Databricks CLI not configured** | Run `databricks configure --profile DEFAULT` |
| **Cluster ID required** | Add `SPARK_CONNECT_CLUSTER_ID` to `.env` |
| **Connection refused** | Check cluster is running in Databricks UI |
| **Authentication failed** | Regenerate PAT token and reconfigure CLI |
| **Module not found** | Run `pip install -r requirements.txt` |
| **Import error: torch** | Install PyTorch: `pip install torch==2.1.0` |
| **Slow first run** | BGE embeddings downloading (~1.2GB), subsequent runs faster |
| **Out of memory** | Use smaller embedding model or reduce batch size |

### Detailed Troubleshooting

#### Issue: Databricks CLI Not Configured

```bash
# Check CLI installation
databricks --version

# Install if missing
pip install databricks-cli

# Configure
databricks configure --profile DEFAULT

# Verify
databricks workspace ls /
```

#### Issue: Spark Connect Connection Fails

```bash
# Check environment variables
cat .env | grep SPARK_CONNECT

# Verify cluster is running
databricks clusters list | grep "RUNNING"

# Test connection
python test_spark_connect.py
```

#### Issue: Low Match Rate

```python
# Debug pipeline statistics
results = pipeline.batch_match(entities)
stats = pipeline.get_pipeline_stats(results)

print("Match rate by stage:")
for method, count in stats['by_method'].items():
    print(f"  {method}: {count}")

# Check reference data quality
print(f"Reference entities: {len(reference_df)}")
print(f"Entities with LEI: {reference_df['lei'].notna().sum()}")
print(f"Entities with CUSIP: {reference_df['cusip'].notna().sum()}")
```

#### Issue: High Latency

```python
# Profile pipeline stages
import time

entity = {"company_name": "Apple Inc."}

start = time.time()
result = pipeline.match(entity)
total_time = time.time() - start

print(f"Total time: {total_time*1000:.1f}ms")
print(f"Method: {result['match_method']}")

# Check stage times in result
if 'stage_times' in result:
    for stage, time_ms in result['stage_times'].items():
        print(f"  {stage}: {time_ms:.1f}ms")
```

---

## Testing Cheatsheet

### Daily Commands

```bash
# Activate environment
source .venv/bin/activate

# Run quick test
python example.py

# Test Spark Connect
python test_spark_connect.py

# Run unit tests
pytest tests/ -v

# Full test suite
pytest tests/ --cov=src --cov-report=html
```

### Quick Tests

```bash
# Test 1: Spark Connect
python -c "from dotenv import load_dotenv; from src.utils.spark_utils import get_spark_session; load_dotenv(); spark = get_spark_session(); print(f'✅ Spark {spark.version}')"

# Test 2: Dependencies
python -c "import torch, sentence_transformers, pandas; print('✅ All deps OK')"

# Test 3: Data loading
python -c "from src.data.loader import DataLoader; loader = DataLoader(); ref = loader.load_reference_data(); print(f'✅ Loaded {len(ref)} entities')"

# Test 4: Pipeline init
python -c "from src.pipeline.hybrid_pipeline import HybridMatchingPipeline; from src.data.loader import DataLoader; loader = DataLoader(); ref = loader.load_reference_data(); pipeline = HybridMatchingPipeline(ref); print('✅ Pipeline ready')"
```

### Verification Checklist

Before moving to production:

- [ ] `python example.py` runs successfully
- [ ] `python test_spark_connect.py` passes all tests
- [ ] Unit tests pass: `pytest tests/ -v`
- [ ] Match rate ≥ 85%
- [ ] Auto-match rate ≥ 85%
- [ ] Avg confidence ≥ 90%
- [ ] Throughput ≥ 100 entities/second (local)
- [ ] P95 latency < 1 second

---

## Next Steps

### Ready for Production?

→ See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for:
- Unity Catalog setup
- Model Serving deployment
- Vector Search configuration
- Scheduled jobs
- Monitoring and alerts

### Need to Train Ditto?

→ See [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) for:
- Training data preparation
- Model fine-tuning
- Evaluation
- MLflow tracking

### Want to Understand the Business Case?

→ See [executive-summary.md](executive-summary.md) for:
- ROI analysis
- Cost breakdown
- Success metrics

---

**Target: 93-95% F1 Score | $0.01/entity | 85%+ Auto-Match Rate**
