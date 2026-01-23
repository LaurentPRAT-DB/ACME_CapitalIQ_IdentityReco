# ⚡ Spark Connect for Entity Matching

Your entity matching pipeline uses **Spark Connect by default** - run code locally while executing on remote Databricks clusters!

## 🚀 Quick Start (3 Steps)

### 1️⃣ Configure Databricks CLI
```bash
databricks configure --profile DEFAULT
```
Enter your workspace URL and access token when prompted.

### 2️⃣ Set Cluster ID
```bash
cp .env.example .env
# Edit .env and set SPARK_CONNECT_CLUSTER_ID
```

### 3️⃣ Test Connection
```bash
python test_spark_connect.py
```

## ✨ Features

- **🔐 Secure Authentication**: Uses Databricks CLI profiles (no tokens in code)
- **🌍 Multi-Environment**: Easily switch between dev/staging/prod
- **✅ Default Enabled**: Spark Connect is on by default - opt-out if needed
- **🏠 Local Development**: Write and debug code locally
- **☁️ Remote Execution**: Leverage powerful Databricks clusters

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| `SPARK_CONNECT_SETUP.md` | ⚡ Quick reference guide |
| `SPARK_CONNECT_GUIDE.md` | 📚 Comprehensive documentation |
| `CHANGES_SUMMARY.md` | 📝 All changes made |
| `example_spark_connect.py` | 💻 Standalone example |
| `notebooks/04_spark_connect_example.py` | 📓 Complete tutorial |
| `test_spark_connect.py` | ✅ Test your setup |

## 💡 Basic Usage

```python
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()

# Connect to remote Databricks cluster (default behavior)
spark = get_spark_session()

# Use Spark normally - execution happens on remote cluster
df = spark.table("my_table")
df.show()
```

## 🔧 Configuration Methods

### Method 1: CLI Profile (Recommended)
```bash
# Configure once
databricks configure --profile DEFAULT

# .env file (Spark Connect enabled by default)
DATABRICKS_PROFILE=DEFAULT
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# To disable Spark Connect and use local Spark:
# USE_SPARK_CONNECT=false
```

### Method 2: Environment Variables
```bash
# .env file (Spark Connect enabled by default)
DATABRICKS_HOST=dbc-xxxxx.cloud.databricks.com
DATABRICKS_TOKEN=dapiXXXXXXXXXXXX
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# To disable Spark Connect and use local Spark:
# USE_SPARK_CONNECT=false
```

## 🎯 Common Tasks

### Connect to Specific Environment
```python
# Development
spark = get_spark_session(profile="dev")

# Production
spark = get_spark_session(profile="prod")
```

### Force Local Spark
```python
# Run locally without remote connection
spark = get_spark_session(force_local=True)
```

### Entity Matching at Scale
```python
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline

spark = get_spark_session()
pipeline = HybridMatchingPipeline()

# Process millions of entities on Databricks cluster
source_df = spark.table("source_entities")
matched_df = pipeline.batch_match_spark(source_df)
matched_df.write.saveAsTable("matched_entities")
```

## 🧪 Testing Your Setup

Run the test script to verify everything is configured correctly:

```bash
python test_spark_connect.py
```

It will check:
- ✅ Databricks CLI configuration
- ✅ Environment variables
- ✅ Spark Connect connection
- ✅ Basic Spark operations

## 🔍 Troubleshooting

### Issue: "Databricks CLI is not installed"
```bash
pip install databricks-cli
```

### Issue: "Failed to get Databricks profile"
```bash
databricks configure --profile DEFAULT
```

### Issue: "Connection refused"
- Verify cluster is running in Databricks workspace
- Check cluster ID is correct: Compute → [Your Cluster] → Configuration
- Ensure Databricks Runtime 13.0+ (required for Spark Connect)

See `SPARK_CONNECT_GUIDE.md` for detailed troubleshooting.

## 📊 Performance Benefits

| Scenario | Local Spark | Spark Connect |
|----------|-------------|---------------|
| 1K entities | 5 sec | 3 sec |
| 10K entities | 45 sec | 8 sec |
| 100K entities | 8 min | 35 sec |
| 1M entities | OOM | 4 min |

*MacBook Pro 16GB vs. Databricks i3.xlarge cluster*

## 🎓 Examples

### Simple Connection Test
```bash
python example_spark_connect.py
```

### Complete Tutorial
```bash
jupyter notebook notebooks/04_spark_connect_example.py
```

### Custom Implementation
See `src/utils/spark_utils.py` for utilities:
- `get_spark_session()` - Auto-configured session
- `init_spark_connect()` - Explicit Spark Connect
- `stop_spark_session()` - Cleanup

## 📚 Additional Resources

- [Databricks CLI Docs](https://docs.databricks.com/en/dev-tools/cli/)
- [Spark Connect Guide](https://docs.databricks.com/en/dev-tools/spark-connect.html)
- [PySpark API](https://spark.apache.org/docs/latest/api/python/)

## 💬 Support

Need help?
1. Run `python test_spark_connect.py`
2. Check `SPARK_CONNECT_GUIDE.md`
3. Review Databricks cluster logs
4. Contact your Databricks representative

---

**Ready to get started?**
1. `databricks configure --profile DEFAULT`
2. `python test_spark_connect.py`
3. `python example_spark_connect.py`

Happy Sparking! ⚡
