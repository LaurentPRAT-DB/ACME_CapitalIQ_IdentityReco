# 🚀 Testing Cheatsheet

## One-Time Setup

```bash
# Install
pip install -r requirements.txt
databricks configure --profile DEFAULT

# Configure
cp .env.example .env
# Edit .env: Add SPARK_CONNECT_CLUSTER_ID=your-cluster-id
```

## Daily Testing

```bash
# Activate environment
source .venv/bin/activate

# Test connection
python test_spark_connect.py

# Run example
python example_spark_connect.py
```

## Essential Commands

| Command | Purpose |
|---------|---------|
| `databricks configure --profile DEFAULT` | Setup auth |
| `databricks clusters list` | Get cluster ID |
| `databricks auth env --profile DEFAULT` | Verify config |
| `python test_spark_connect.py` | Test connection |
| `python example_spark_connect.py` | Full example |
| `python example.py` | Local pandas test |

## Quick Tests

```bash
# Test 1: Spark Connect
python -c "from dotenv import load_dotenv; from src.utils.spark_utils import get_spark_session; load_dotenv(); spark = get_spark_session(); print(f'✓ Spark {spark.version}')"

# Test 2: SQL Query
python -c "from dotenv import load_dotenv; from src.utils.spark_utils import get_spark_session; load_dotenv(); spark = get_spark_session(); spark.sql('SELECT 1 as test').show()"

# Test 3: Count Test
python -c "from dotenv import load_dotenv; from src.utils.spark_utils import get_spark_session; load_dotenv(); spark = get_spark_session(); print(f'Count: {spark.range(100).count()}')"
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| CLI not installed | `pip install databricks-cli` |
| Not configured | `databricks configure --profile DEFAULT` |
| Cluster ID missing | Add to `.env`: `SPARK_CONNECT_CLUSTER_ID=...` |
| Connection refused | Check cluster is running |
| Want local Spark | Add to `.env`: `USE_SPARK_CONNECT=false` |

## File Locations

- **Test script**: `test_spark_connect.py`
- **Example**: `example_spark_connect.py`
- **Config**: `.env`
- **Credentials**: `~/.databrickscfg`
- **Full guide**: `LOCAL_TESTING_GUIDE.md`

## Getting Cluster ID

```bash
# Method 1: UI
# Databricks → Compute → Select cluster → Copy from URL

# Method 2: CLI
databricks clusters list

# Method 3: Create new
databricks clusters create --json '{
  "cluster_name": "test-cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "num_workers": 2
}'
```

## Environment Variables

```bash
# Minimum (.env)
DATABRICKS_PROFILE=DEFAULT
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# Optional
MLFLOW_TRACKING_URI=databricks
SPARK_APP_NAME=my-app

# Disable Spark Connect
USE_SPARK_CONNECT=false
```

## Success Indicators

- ✅ `test_spark_connect.py` shows all green checks
- ✅ Can run `spark.sql("SELECT 1").show()`
- ✅ Example script completes successfully
- ✅ See "✓ Connected to Databricks cluster"

## Next Steps

1. **Working?** → Explore `notebooks/`
2. **Issues?** → Read `LOCAL_TESTING_GUIDE.md`
3. **Production?** → See deployment docs

---

**Quick Start**: `QUICK_START.md` | **Full Guide**: `LOCAL_TESTING_GUIDE.md`
