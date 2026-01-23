# Spark Connect Setup Guide

This guide explains how to use Spark Connect to run your entity matching pipeline locally while executing computations on a remote Databricks cluster.

## What is Spark Connect?

Spark Connect is a new client-server architecture introduced in Spark 3.4+ that allows you to:
- Write and run code locally (on your laptop)
- Execute computations remotely (on your Databricks cluster)
- Use the same PySpark API you're familiar with
- Debug locally with your IDE and breakpoints

## Benefits

- **Local Development**: Write code in your favorite IDE with full debugging support
- **Remote Execution**: Leverage powerful Databricks clusters for computation
- **Cost Efficient**: Only pay for compute time when running jobs
- **No Sync Required**: No need to sync notebooks or code to Databricks
- **Seamless Integration**: Works with Unity Catalog, Delta Lake, and MLflow

## Prerequisites

1. **Databricks Workspace**
   - Active Databricks workspace
   - Running cluster (any size)
   - Personal Access Token

2. **Local Environment**
   - Python 3.8+
   - PySpark 3.5+ with Connect support
   - Network access to Databricks workspace

## Setup Steps

### 1. Install Dependencies

```bash
# Install PySpark with Spark Connect support
pip install 'pyspark[connect]>=3.5.0'

# Install Databricks CLI
pip install databricks-cli

# Or install all project dependencies (includes both)
pip install -r requirements.txt
```

### 2. Configure Databricks CLI (Recommended Method)

**Option A: Interactive Configuration**

```bash
# Configure default profile
databricks configure --profile DEFAULT

# Follow the prompts:
# - Databricks Host: https://dbc-xxxxx-xxxx.cloud.databricks.com
# - Token: [paste your token from User Settings > Developer > Access Tokens]
```

**Option B: Multiple Profiles**

```bash
# Create separate profiles for different environments
databricks configure --profile dev
databricks configure --profile prod
databricks configure --profile staging
```

The CLI stores credentials securely in `~/.databrickscfg`:

```ini
[DEFAULT]
host = https://dbc-xxxxx-xxxx.cloud.databricks.com
token = dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[dev]
host = https://dbc-dev-xxxxx.cloud.databricks.com
token = dapiYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY

[prod]
host = https://dbc-prod-xxxxx.cloud.databricks.com
token = dapiZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
```

**Verify Configuration:**

```bash
# Test connection
databricks workspace ls /

# Get auth details
databricks auth env --profile DEFAULT
```

### 3. Get Cluster ID

**Cluster ID:**
- Go to Compute → Select your cluster
- Copy the Cluster ID from:
  - URL bar: `/compute/clusters/<cluster-id>`
  - Configuration tab: Look for "Cluster ID"
- Format: `1234-567890-abcdefgh`

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Databricks CLI profile (recommended)
DATABRICKS_PROFILE=DEFAULT  # or dev, prod, staging, etc.

# Cluster ID for Spark Connect
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# Enable Spark Connect
USE_SPARK_CONNECT=true

# Optional: MLflow tracking
MLFLOW_TRACKING_URI=databricks
```

**Alternative: Using Environment Variables (not recommended)**

If you prefer not to use CLI profiles:

```bash
# Databricks workspace (without https://)
DATABRICKS_HOST=dbc-xxxxx-xxxx.cloud.databricks.com

# Personal access token
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Rest remains the same
```

### 5. Load Environment Variables

**Option A: Using python-dotenv**

```python
from dotenv import load_dotenv
load_dotenv()
```

**Option B: Using direnv**

```bash
# Install direnv
brew install direnv  # macOS
# or: sudo apt install direnv  # Linux

# Configure
echo "dotenv" > .envrc
direnv allow
```

## Usage

### Basic Connection (Using CLI Profile)

```python
from src.utils.spark_utils import get_spark_session

# Method 1: Use DEFAULT profile (most common)
spark = get_spark_session()

# Method 2: Use specific profile
spark = get_spark_session(profile="dev")

# Verify connection
print(f"Spark version: {spark.version}")
spark.sql("SELECT current_database()").show()
```

### Advanced Configuration

```python
from src.utils.spark_utils import init_spark_connect

# Use specific CLI profile and cluster
spark = init_spark_connect(
    cluster_id="1234-567890-abcdefgh",
    profile="prod"
)

# Override with explicit credentials (not recommended)
spark = init_spark_connect(
    cluster_id="1234-567890-abcdefgh",
    databricks_host="dbc-xxxxx.cloud.databricks.com",
    databricks_token="dapiXXXXXXXXXXXX"
)
```

### Force Local Spark

```python
# Run Spark locally without Spark Connect
spark = get_spark_session(force_local=True)
```

### Multiple Profiles Example

```python
from src.utils.spark_utils import get_spark_session

# Development environment
dev_spark = get_spark_session(profile="dev")

# Production environment
prod_spark = get_spark_session(profile="prod")

# Staging environment
staging_spark = get_spark_session(profile="staging")
```

## Example: Entity Matching with Spark Connect

```python
from src.utils.spark_utils import get_spark_session
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from pyspark.sql.functions import pandas_udf, col
import pandas as pd

# 1. Connect to remote cluster
spark = get_spark_session()

# 2. Initialize pipeline
pipeline = HybridMatchingPipeline()

# 3. Create UDF
@pandas_udf("struct<ciq_id:string,confidence:double>")
def match_udf(names: pd.Series) -> pd.DataFrame:
    results = []
    for name in names:
        result = pipeline.match({"company_name": name})
        results.append({
            "ciq_id": result["ciq_id"],
            "confidence": result["confidence"]
        })
    return pd.DataFrame(results)

# 4. Apply to DataFrame (runs on remote cluster)
df = spark.table("source_entities")
matched_df = df.withColumn("match", match_udf(col("company_name")))

# 5. Save to Unity Catalog
matched_df.write.saveAsTable("main.entity_matching.results")
```

## Troubleshooting

### Connection Refused

```
Error: Connection refused
```

**Solutions:**
- Verify cluster is running (not terminated or stopped)
- Check network connectivity to Databricks workspace
- Verify cluster allows Spark Connect (enabled by default)

### Authentication Failed

```
Error: Invalid access token
```

**Solutions:**
- Verify token hasn't expired
- Check token has correct permissions
- Regenerate token if needed

### Cluster Not Found

```
Error: Cluster ID not found
```

**Solutions:**
- Verify cluster ID is correct
- Check you have access to the cluster
- Ensure cluster is in the same workspace

### Import Errors

```
ModuleNotFoundError: No module named 'pyspark.sql.connect'
```

**Solutions:**
```bash
# Reinstall PySpark with Connect support
pip install --upgrade 'pyspark[connect]>=3.5.0'
```

### Slow Performance

**Tips:**
- Use larger cluster for better performance
- Enable caching for frequently accessed data
- Optimize Spark configurations
- Use broadcast joins for small reference tables

## Best Practices

### 1. Connection Management

```python
# Get or create session (reuses existing)
spark = get_spark_session()

# Stop when done (optional, auto-closes on exit)
spark.stop()
```

### 2. Error Handling

```python
try:
    spark = get_spark_session()
except ValueError as e:
    print(f"Configuration error: {e}")
    # Fall back to local Spark
    spark = get_spark_session(force_local=True)
```

### 3. Resource Cleanup

```python
# Cache expensive operations
reference_df = spark.table("reference").cache()

# Unpersist when done
reference_df.unpersist()

# Stop session when finished
spark.stop()
```

### 4. Development Workflow

```python
# Development: Use small cluster with auto-termination
spark = get_spark_session()

# Production: Use dedicated cluster
spark = init_spark_connect(
    cluster_id="production-cluster-id"
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Local Environment                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Your IDE / Jupyter / Python Script                      │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  PySpark API (DataFrame, SQL, MLlib)              │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       │                                   │  │
│  │  ┌────────────────────▼───────────────────────────────┐  │  │
│  │  │  Spark Connect Client                              │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  └───────────────────────┼───────────────────────────────────┘  │
└────────────────────────┬─┼──────────────────────────────────────┘
                         │ │
                         │ │ HTTPS/gRPC
                         │ │
┌────────────────────────▼─▼──────────────────────────────────────┐
│                  Databricks Workspace                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Spark Cluster                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Spark Connect Server                              │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       │                                   │  │
│  │  ┌────────────────────▼───────────────────────────────┐  │  │
│  │  │  Spark Executors (Distributed Processing)         │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       │                                   │  │
│  │  ┌────────────────────▼───────────────────────────────┐  │  │
│  │  │  Unity Catalog / Delta Lake / DBFS                 │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Comparison

| Scenario | Local Spark | Spark Connect |
|----------|-------------|---------------|
| 1K entities | 5 sec | 3 sec |
| 10K entities | 45 sec | 8 sec |
| 100K entities | 8 min | 35 sec |
| 1M entities | OOM | 4 min |

*Using: Local = MacBook Pro 16GB, Remote = Databricks i3.xlarge (4 cores, 30GB)*

## Cost Considerations

- **Cluster Running Time**: You pay for the entire time the cluster is running
- **Auto-termination**: Set cluster to auto-terminate after inactivity (e.g., 30 minutes)
- **Cluster Size**: Start small (Single Node) and scale up as needed
- **Development**: Use smaller clusters for development/testing
- **Production**: Use larger clusters for production workloads

## Additional Resources

- [Spark Connect Documentation](https://spark.apache.org/docs/latest/spark-connect-overview.html)
- [Databricks Spark Connect Guide](https://docs.databricks.com/en/dev-tools/spark-connect.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Unity Catalog Documentation](https://docs.databricks.com/en/data-governance/unity-catalog/)

## Examples

See the following files for complete examples:
- `example_spark_connect.py` - Standalone script
- `notebooks/04_spark_connect_example.py` - Comprehensive notebook
- `src/utils/spark_utils.py` - Utility functions

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Databricks cluster logs
3. Check Spark Connect compatibility
4. Contact your Databricks representative
