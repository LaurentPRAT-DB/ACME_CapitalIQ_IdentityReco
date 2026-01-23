# Spark Connect Configuration Summary

This document provides a quick reference for setting up and using Spark Connect with Databricks CLI authentication.

## Quick Start

### 1. Configure Databricks CLI

```bash
# Install Databricks CLI (if not already installed)
pip install databricks-cli

# Configure authentication
databricks configure --profile DEFAULT

# When prompted, enter:
# - Host: https://dbc-xxxxx-xxxx.cloud.databricks.com
# - Token: [your personal access token]

# Verify configuration
databricks workspace ls /
databricks auth env --profile DEFAULT
```

### 2. Set Environment Variables

Create `.env` file:

```bash
# Use Databricks CLI profile for auth (recommended)
DATABRICKS_PROFILE=DEFAULT

# Your cluster ID
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh

# Enable Spark Connect
USE_SPARK_CONNECT=true
```

### 3. Test Connection

```python
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()
spark = get_spark_session()
print(f"Connected! Spark version: {spark.version}")
```

## Configuration Methods

The system supports multiple authentication methods (in priority order):

### 1. Databricks CLI Profile (Recommended)

```python
# Use DEFAULT profile
spark = get_spark_session()

# Use specific profile
spark = get_spark_session(profile="dev")
```

**Advantages:**
- ✓ Secure credential storage
- ✓ Multiple environment support
- ✓ No tokens in code or .env
- ✓ Easy to rotate credentials

### 2. Environment Variables

```bash
DATABRICKS_HOST=dbc-xxxxx.cloud.databricks.com
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
USE_SPARK_CONNECT=true
```

```python
spark = get_spark_session()
```

### 3. Explicit Parameters

```python
spark = init_spark_connect(
    cluster_id="1234-567890-abcdefgh",
    databricks_host="dbc-xxxxx.cloud.databricks.com",
    databricks_token="dapiXXXXXXXXXXXX"
)
```

## Multiple Environments

### Configure Profiles

```bash
# Development
databricks configure --profile dev

# Production
databricks configure --profile prod

# Staging
databricks configure --profile staging
```

### Use in Code

```python
from src.utils.spark_utils import get_spark_session

# Development
dev_spark = get_spark_session(profile="dev")

# Production
prod_spark = get_spark_session(profile="prod")

# Staging
staging_spark = get_spark_session(profile="staging")
```

### Environment-Specific .env

```bash
# Development
DATABRICKS_PROFILE=dev
SPARK_CONNECT_CLUSTER_ID=dev-cluster-id

# Production
DATABRICKS_PROFILE=prod
SPARK_CONNECT_CLUSTER_ID=prod-cluster-id
```

## Files Modified

### New Files Created:
1. `src/utils/spark_utils.py` - Spark Connect utilities with CLI profile support
2. `src/utils/__init__.py` - Utils module initialization
3. `.env.example` - Environment variable template
4. `SPARK_CONNECT_GUIDE.md` - Comprehensive setup guide
5. `SPARK_CONNECT_SETUP.md` - This quick reference
6. `example_spark_connect.py` - Standalone example script
7. `notebooks/04_spark_connect_example.py` - Complete notebook example

### Modified Files:
1. `src/config.py` - Added SparkConfig class with profile support
2. `requirements.txt` - Added pyspark[connect]>=3.5.0
3. `README.md` - Updated with Spark Connect CLI setup instructions

## Key Features

### Authentication Priority
1. Explicit parameters (if provided)
2. Environment variables (if set)
3. Databricks CLI profile (default fallback)

### Auto-Detection
```python
# Automatically uses CLI profile if no other auth provided
spark = get_spark_session()
```

### Profile Selection
```python
# Use specific profile from ~/.databrickscfg
spark = get_spark_session(profile="prod")
```

### Local Override
```python
# Force local Spark (no remote connection)
spark = get_spark_session(force_local=True)
```

## Common Use Cases

### 1. Development with CLI Profile

```python
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()

# Use DEFAULT profile
spark = get_spark_session()

# Run your code
df = spark.table("my_table")
df.show()
```

### 2. CI/CD with Environment Variables

```bash
# In CI/CD pipeline, set environment variables
export DATABRICKS_HOST="dbc-xxxxx.cloud.databricks.com"
export DATABRICKS_TOKEN="${DATABRICKS_TOKEN_SECRET}"
export SPARK_CONNECT_CLUSTER_ID="ci-cluster-id"
export USE_SPARK_CONNECT=true
```

```python
# Code works without changes
spark = get_spark_session()
```

### 3. Multi-Environment Testing

```python
# Test against multiple environments
environments = ["dev", "staging", "prod"]

for env in environments:
    spark = get_spark_session(profile=env)
    result = spark.sql("SELECT COUNT(*) FROM my_table").first()[0]
    print(f"{env}: {result} rows")
    spark.stop()
```

## Troubleshooting

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

Set in `.env`:
```bash
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
```

### Error: "Connection refused"

- Verify cluster is running in Databricks workspace
- Check cluster ID is correct
- Ensure cluster supports Spark Connect (Databricks Runtime 13.0+)

### Slow Performance

- Use a larger cluster
- Enable auto-scaling
- Cache frequently accessed data
- Use broadcast joins for small tables

## Best Practices

1. **Use CLI Profiles**: Store credentials securely in ~/.databrickscfg
2. **Separate Environments**: Use different profiles for dev/staging/prod
3. **Auto-Termination**: Set cluster to auto-terminate after inactivity
4. **Resource Cleanup**: Stop Spark sessions when done
5. **Error Handling**: Handle connection failures gracefully

```python
try:
    spark = get_spark_session()
except ValueError as e:
    print(f"Connection failed: {e}")
    # Fall back to local Spark
    spark = get_spark_session(force_local=True)
```

## Next Steps

1. **Run Examples**: Try `example_spark_connect.py`
2. **Read Guide**: See `SPARK_CONNECT_GUIDE.md` for details
3. **Test Notebook**: Explore `notebooks/04_spark_connect_example.py`
4. **Configure Clusters**: Set up dev/prod clusters
5. **Deploy Pipeline**: Use for batch entity matching

## Support

- Databricks CLI: https://docs.databricks.com/en/dev-tools/cli/
- Spark Connect: https://docs.databricks.com/en/dev-tools/spark-connect.html
- Project Issues: [Your GitHub Issues Link]
