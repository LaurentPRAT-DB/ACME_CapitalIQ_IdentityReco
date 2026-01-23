# Local Testing Guide

Complete instructions for testing the Entity Matching pipeline locally with Spark Connect.

## 📋 Prerequisites

Before you start, ensure you have:

- ✅ Python 3.9+ installed
- ✅ Access to a Databricks workspace
- ✅ A running Databricks cluster (or ability to create one)
- ✅ Databricks personal access token
- ✅ Git installed (optional, for cloning)

## 🚀 Step-by-Step Setup

### Step 1: Clone and Setup Environment

```bash
# Navigate to project directory
cd /Users/laurent.prat/Documents/lpdev/claude_code_training/MET_CapitalIQ_identityReco

# Create virtual environment (if not already done)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
pip install uv
uv pip install -r requirements.txt
```

### Step 2: Configure Databricks CLI

```bash
# Install Databricks CLI (if not in requirements.txt)
pip install databricks-cli

# Configure authentication
databricks configure --profile DEFAULT

# You'll be prompted for:
# - Databricks Host: https://dbc-xxxxx-xxxx.cloud.databricks.com
# - Token: dapi... (get from User Settings > Developer > Access Tokens)

# Verify configuration
databricks workspace ls /
databricks auth env --profile DEFAULT
```

**Creating a Personal Access Token:**
1. Go to your Databricks workspace
2. Click your user icon (top right) → Settings
3. Go to Developer → Access Tokens
4. Click "Generate New Token"
5. Give it a name (e.g., "local-dev")
6. Set lifetime (e.g., 90 days)
7. Copy the token (you won't see it again!)

### Step 3: Get Your Cluster ID

**Option A: From Databricks UI**
1. Go to Compute in left sidebar
2. Click on your cluster
3. Look at the URL: `https://...cloud.databricks.com/#compute/clusters/<cluster-id>`
4. Or go to Configuration tab and find "Cluster ID"

**Option B: Using CLI**
```bash
databricks clusters list
# Copy the cluster ID from the output
```

**Don't have a cluster? Create one:**
```bash
# Create a small cluster for testing
databricks clusters create --json '{
  "cluster_name": "entity-matching-dev",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "num_workers": 2,
  "autotermination_minutes": 30
}'
```

### Step 4: Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your favorite editor
```

**Minimum required configuration:**
```bash
# Databricks CLI profile (already configured in Step 2)
DATABRICKS_PROFILE=DEFAULT

# Your cluster ID from Step 3
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# Spark Connect is enabled by default, no need to set USE_SPARK_CONNECT=true
```

**Optional configurations:**
```bash
# Use a different profile
DATABRICKS_PROFILE=dev

# MLflow tracking
MLFLOW_TRACKING_URI=databricks

# Custom Spark app name
SPARK_APP_NAME=entity-matching-local-test

# Disable Spark Connect (use local Spark instead)
# USE_SPARK_CONNECT=false
```

### Step 5: Test Spark Connect Connection

```bash
# Run the comprehensive test script
python test_spark_connect.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════════════╗
║                  Spark Connect Configuration Test                    ║
╚══════════════════════════════════════════════════════════════════════╝

Testing Databricks CLI Configuration
✓ Databricks CLI is configured
✓ DATABRICKS_HOST=dbc-xxxxx.cloud.databricks.com
✓ DATABRICKS_TOKEN=dapi****** (hidden)

Testing Environment Variables
✓ SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
✓ DATABRICKS_PROFILE=DEFAULT

Testing Spark Connect
Using Databricks CLI profile: DEFAULT
Using Spark Connect to remote Databricks cluster (default behavior)
Connecting to Databricks cluster 1234-567890-abcdefgh via Spark Connect...
Workspace: dbc-xxxxx.cloud.databricks.com
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

**If you see errors, see Troubleshooting section below.**

### Step 6: Run Basic Example

```bash
# Quick test with minimal data
python -c "
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()
spark = get_spark_session()

# Simple test
df = spark.range(100)
print(f'Count: {df.count()}')
print('✓ Spark Connect working!')

spark.stop()
"
```

### Step 7: Run Full Entity Matching Example

```bash
# Run the comprehensive example
python example_spark_connect.py
```

**What this does:**
1. Connects to Databricks cluster via Spark Connect
2. Loads sample reference data (S&P Capital IQ entities)
3. Initializes entity matching pipeline
4. Creates sample source entities in Spark
5. Applies matching using Pandas UDFs
6. Shows match results and statistics
7. Writes results to Delta table

**Expected output:**
```
================================================================================
Entity Matching with Spark Connect
================================================================================

1. Initializing Spark Connect session...
   Using Databricks CLI profile: DEFAULT
   ✓ Connected to Databricks cluster
   ✓ Spark version: 3.5.0

2. Loading reference data...
   ✓ Loaded 5 reference entities

3. Initializing matching pipeline...
   ✓ Pipeline initialized

4. Creating sample source entities...
   ✓ Created 7 source entities

   Sample entities:
   +----------+------------------------+------+-------------+
   |source_id |company_name            |ticker|country      |
   +----------+------------------------+------+-------------+
   |CRM-001   |Apple Inc.              |AAPL  |United States|
   |CRM-002   |Microsoft Corporation   |MSFT  |United States|
   +----------+------------------------+------+-------------+

6. Running entity matching on Databricks cluster...
   (This computation runs remotely on your Databricks cluster)
   ✓ Matched 7 entities

7. Match Results:
   [Shows matched entities with CIQ IDs and confidence scores]

8. Pipeline Statistics:
   Average Confidence: 95%
   Match Rate: 7/7 (100%)

✓ Results saved to /tmp/entity_matches_demo

SUCCESS! Entity matching completed with Spark Connect
```

### Step 8: Test with Sample Data

```bash
# Run quick start example
python example.py
```

This tests the pipeline with local pandas (no Spark), useful for:
- Quick testing of matching logic
- Debugging pipeline components
- Generating training data

## 🧪 Additional Tests

### Test 1: Vector Search (Requires Setup)

```python
python -c "
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

loader = DataLoader()
reference_df = loader.load_reference_data()

pipeline = HybridMatchingPipeline(
    reference_df=reference_df,
    enable_foundation_model=False
)

entity = {
    'company_name': 'Apple Inc.',
    'ticker': 'AAPL'
}

result = pipeline.match(entity)
print(f'Matched: {result[\"ciq_id\"]} (confidence: {result[\"confidence\"]:.2%})')
"
```

### Test 2: Local Spark (No Remote Connection)

```bash
# Test with local Spark
USE_SPARK_CONNECT=false python example.py

# Or in code
python -c "
from src.utils.spark_utils import get_spark_session

spark = get_spark_session(force_local=True)
print(f'Local Spark: {spark.version}')
spark.range(10).show()
"
```

### Test 3: Different Profiles

```bash
# Configure multiple profiles
databricks configure --profile dev
databricks configure --profile prod

# Test with dev profile
DATABRICKS_PROFILE=dev python test_spark_connect.py

# Test with prod profile
DATABRICKS_PROFILE=prod python test_spark_connect.py
```

## 🔍 Verification Checklist

After setup, verify:

- [ ] Virtual environment is activated
- [ ] All dependencies installed (`pip list | grep pyspark`)
- [ ] Databricks CLI configured (`databricks workspace ls /`)
- [ ] .env file exists with SPARK_CONNECT_CLUSTER_ID
- [ ] Cluster is running (check Databricks UI)
- [ ] test_spark_connect.py passes all tests
- [ ] example_spark_connect.py runs successfully
- [ ] Can read/write Delta tables

## ⚙️ Configuration Options

### Minimal Configuration (Default)
```bash
# .env
DATABRICKS_PROFILE=DEFAULT
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
```

### Full Configuration
```bash
# .env
# Databricks authentication
DATABRICKS_PROFILE=DEFAULT

# Spark Connect
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
SPARK_APP_NAME=entity-matching-dev

# MLflow
MLFLOW_TRACKING_URI=databricks

# Model paths
DITTO_MODEL_PATH=models/ditto_entity_matcher

# To use local Spark instead
# USE_SPARK_CONNECT=false
```

### Alternative: Environment Variables
```bash
# .env
# Instead of CLI profile, use explicit credentials
DATABRICKS_HOST=dbc-xxxxx.cloud.databricks.com
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
```

## 🐛 Troubleshooting

### Error: "Databricks CLI is not installed"

```bash
pip install databricks-cli
```

### Error: "Failed to get Databricks profile"

```bash
# Configure the profile
databricks configure --profile DEFAULT

# Verify
databricks auth env --profile DEFAULT
```

### Error: "Cluster ID is required"

Edit `.env` and add:
```bash
SPARK_CONNECT_CLUSTER_ID=your-cluster-id
```

Get cluster ID:
```bash
databricks clusters list
```

### Error: "Connection refused"

**Possible causes:**
1. Cluster is not running
   - Check Databricks UI → Compute
   - Start the cluster if stopped

2. Wrong cluster ID
   - Verify ID in Databricks UI
   - Update SPARK_CONNECT_CLUSTER_ID in .env

3. Network issues
   - Check VPN if required
   - Verify workspace URL is correct

### Error: "Authentication failed"

```bash
# Regenerate token
# 1. Go to Databricks → Settings → Developer → Access Tokens
# 2. Generate new token
# 3. Reconfigure CLI
databricks configure --profile DEFAULT
```

### Error: "Module not found"

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep pyspark
pip list | grep databricks
```

### Spark Connect is slow

**Tips:**
- Use a larger cluster
- Enable auto-scaling
- Cache frequently accessed data
- Use broadcast joins for small tables

```python
# Cache expensive operations
reference_df = spark.table("reference").cache()
```

### Want to use local Spark

```bash
# Add to .env
USE_SPARK_CONNECT=false

# Or in code
spark = get_spark_session(force_local=True)
```

## 📊 Performance Testing

### Test with Larger Dataset

```python
# test_performance.py
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session
import time

load_dotenv()
spark = get_spark_session()

# Create test dataset
sizes = [1000, 10000, 100000]

for size in sizes:
    print(f"\nTesting with {size:,} rows...")

    start = time.time()
    df = spark.range(size)
    count = df.count()
    elapsed = time.time() - start

    print(f"  Count: {count:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {count/elapsed:,.0f} rows/sec")

spark.stop()
```

Run:
```bash
python test_performance.py
```

## 🎓 Next Steps

Once local testing works:

1. **Train Ditto Model**
   ```bash
   python notebooks/02_train_ditto_model.py
   ```

2. **Run Full Pipeline**
   ```bash
   python notebooks/03_full_pipeline_example.py
   ```

3. **Deploy to Production**
   - See `notebooks/05_production_deployment.py`
   - Set up MLflow model serving
   - Configure Unity Catalog

4. **Scale Up**
   - Use larger clusters
   - Enable auto-scaling
   - Optimize Spark configurations

## 📚 Useful Commands

```bash
# Check Python version
python --version

# List installed packages
pip list | grep -E "pyspark|databricks|spark"

# Check Databricks CLI version
databricks --version

# List available clusters
databricks clusters list

# Get cluster info
databricks clusters get --cluster-id 1234-567890-abcdefgh

# Test authentication
databricks workspace ls /Users

# Check environment variables
cat .env

# Activate virtual environment
source .venv/bin/activate

# Deactivate virtual environment
deactivate
```

## 🔗 Resources

- **Test Script**: `test_spark_connect.py`
- **Example**: `example_spark_connect.py`
- **Quick Start**: `example.py`
- **Setup Guide**: `SPARK_CONNECT_GUIDE.md`
- **Troubleshooting**: `SPARK_CONNECT_GUIDE.md#troubleshooting`

## 💡 Tips

1. **Start Small**: Test with small data first
2. **Use Auto-Termination**: Set cluster to auto-terminate after 30 min
3. **Monitor Costs**: Check Databricks workspace for cluster costs
4. **Cache Data**: Cache frequently accessed DataFrames
5. **Use Profiles**: Set up separate dev/prod profiles

## ✅ Success Indicators

You're ready to develop when:

- ✅ `test_spark_connect.py` passes all tests
- ✅ `example_spark_connect.py` completes successfully
- ✅ Can run Spark SQL queries remotely
- ✅ Can read/write Delta tables
- ✅ Pipeline processes entities correctly

---

**Questions?** Check `SPARK_CONNECT_GUIDE.md` or run `python test_spark_connect.py`
