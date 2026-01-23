# ⚡ Quick Start - Local Testing

**5-minute setup to test entity matching with Spark Connect**

## Prerequisites
- Python 3.9+
- Databricks workspace access
- Running Databricks cluster

## Setup (One-Time)

```bash
# 1. Install dependencies
cd /Users/laurent.prat/Documents/lpdev/claude_code_training/MET_CapitalIQ_identityReco
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure Databricks CLI
databricks configure --profile DEFAULT
# Enter: https://dbc-xxxxx.cloud.databricks.com
# Enter: dapi... (your token)

# 3. Create .env file
cp .env.example .env

# 4. Edit .env - add your cluster ID
echo "SPARK_CONNECT_CLUSTER_ID=your-cluster-id" >> .env
```

## Get Cluster ID

**Option 1: Databricks UI**
- Go to Compute → Select cluster → Copy ID from URL

**Option 2: CLI**
```bash
databricks clusters list
```

## Test

```bash
# Test connection
python test_spark_connect.py

# Run example
python example_spark_connect.py
```

## Quick Test Commands

```bash
# Test 1: Basic Spark Connect
python -c "
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session
load_dotenv()
spark = get_spark_session()
print(f'✓ Connected! Spark {spark.version}')
spark.range(10).show()
"

# Test 2: Entity Matching
python example.py

# Test 3: Full Pipeline
python example_spark_connect.py
```

## Common Issues

### "Databricks CLI not configured"
```bash
databricks configure --profile DEFAULT
```

### "Cluster ID required"
```bash
# Add to .env
echo "SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh" >> .env
```

### "Connection refused"
- Check cluster is running in Databricks UI
- Verify cluster ID is correct

### Use Local Spark Instead
```bash
# Add to .env
echo "USE_SPARK_CONNECT=false" >> .env
```

## What's Next?

- ✅ Test passed? → See `notebooks/` for examples
- ❌ Issues? → Read `LOCAL_TESTING_GUIDE.md`
- 📚 Learn more? → Read `SPARK_CONNECT_GUIDE.md`

## Essential Commands

```bash
# Activate environment
source .venv/bin/activate

# Test connection
python test_spark_connect.py

# Run example
python example_spark_connect.py

# Check cluster status
databricks clusters get --cluster-id <your-id>

# List clusters
databricks clusters list

# Verify config
databricks auth env --profile DEFAULT
```

---

**Full Guide**: `LOCAL_TESTING_GUIDE.md`
