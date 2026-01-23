# Spark Connect Implementation Summary

## Overview

The project has been updated to support Spark Connect, allowing you to run code locally while executing computations on a remote Databricks cluster. Authentication is handled via the Databricks CLI with profile support, making it secure and easy to manage multiple environments.

## Changes Made

### 1. New Files Created

#### Core Implementation
- **`src/utils/spark_utils.py`** - Main Spark Connect utilities
  - `init_spark_connect()` - Initialize Spark with Connect to remote cluster
  - `get_spark_session()` - Auto-detect and create appropriate Spark session
  - `_get_databricks_config_from_profile()` - Read credentials from CLI profiles
  - `stop_spark_session()` - Cleanup utility

- **`src/utils/__init__.py`** - Utils module initialization

#### Configuration
- **`.env.example`** - Environment variable template with CLI profile support
  - Documents both CLI profile and environment variable auth methods
  - Includes all required Spark Connect settings

#### Documentation
- **`SPARK_CONNECT_GUIDE.md`** - Comprehensive setup and usage guide
  - Detailed prerequisites and setup steps
  - Multiple authentication methods
  - Troubleshooting section
  - Best practices and performance tips

- **`SPARK_CONNECT_SETUP.md`** - Quick reference guide
  - Fast setup instructions
  - Common use cases
  - Files modified summary

- **`CHANGES_SUMMARY.md`** - This file

#### Examples
- **`example_spark_connect.py`** - Standalone example script
  - Complete entity matching workflow
  - CLI profile authentication
  - Batch processing with Pandas UDFs
  - Delta table operations
  - Statistics and metrics

- **`notebooks/04_spark_connect_example.py`** - Comprehensive notebook
  - Step-by-step tutorial
  - Multiple connection methods
  - UDF creation and usage
  - Performance testing
  - Production patterns

- **`test_spark_connect.py`** - Configuration test script
  - Validates CLI setup
  - Tests environment variables
  - Verifies Spark Connect connection
  - Provides troubleshooting guidance

### 2. Modified Files

#### Configuration
- **`src/config.py`**
  - Added `SparkConfig` dataclass with:
    - `use_spark_connect` - Enable/disable Spark Connect
    - `spark_remote` - Spark Connect URL
    - `spark_connect_cluster_id` - Target cluster ID
    - `spark_app_name` - Application name
    - `spark_master` - Local Spark master
  - Updated `Config.from_env()` to load Spark Connect settings
  - Added support for `DATABRICKS_PROFILE` environment variable

#### Dependencies
- **`requirements.txt`**
  - Updated PySpark to >=3.5.0 (required for Spark Connect)
  - Added `pyspark[connect]>=3.5.0` for Connect support

#### Documentation
- **`README.md`**
  - Added "Spark Connect Setup" section
  - CLI profile configuration instructions
  - Multiple environment setup examples
  - Updated batch processing examples to show Spark Connect usage

## Key Features

### 1. Databricks CLI Authentication (Recommended)

Uses Databricks CLI profiles stored in `~/.databrickscfg` for secure credential management:

```bash
# Configure once
databricks configure --profile DEFAULT

# Use in code
spark = get_spark_session()  # Uses DEFAULT profile
spark = get_spark_session(profile="dev")  # Uses dev profile
```

**Benefits:**
- ✓ Secure credential storage
- ✓ No tokens in code or environment files
- ✓ Easy to manage multiple environments
- ✓ Simple credential rotation

### 2. Flexible Authentication

Supports three authentication methods (in priority order):

1. **Explicit Parameters** (highest priority)
```python
spark = init_spark_connect(
    cluster_id="xxx",
    databricks_host="xxx",
    databricks_token="xxx"
)
```

2. **Environment Variables**
```bash
DATABRICKS_HOST=xxx
DATABRICKS_TOKEN=xxx
SPARK_CONNECT_CLUSTER_ID=xxx
```

3. **CLI Profile** (default fallback)
```bash
DATABRICKS_PROFILE=DEFAULT
SPARK_CONNECT_CLUSTER_ID=xxx
```

### 3. Auto-Detection

Automatically determines whether to use Spark Connect or local Spark:

```python
# Auto-detect from USE_SPARK_CONNECT env var
spark = get_spark_session()

# Force local Spark
spark = get_spark_session(force_local=True)
```

### 4. Multiple Environment Support

Easy switching between dev/staging/prod:

```bash
# Configure each environment
databricks configure --profile dev
databricks configure --profile prod

# Use in code
dev_spark = get_spark_session(profile="dev")
prod_spark = get_spark_session(profile="prod")
```

## Usage Examples

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Databricks CLI
databricks configure --profile DEFAULT

# 3. Create .env file
cp .env.example .env
# Edit: Set SPARK_CONNECT_CLUSTER_ID

# 4. Test connection
python test_spark_connect.py

# 5. Run example
python example_spark_connect.py
```

### In Your Code

```python
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

# Load environment
load_dotenv()

# Connect to Databricks
spark = get_spark_session()

# Use Spark normally
df = spark.table("my_table")
df.show()

# Write results
df.write.saveAsTable("results_table")
```

### Entity Matching Pipeline

```python
from src.utils.spark_utils import get_spark_session
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from pyspark.sql.functions import pandas_udf

# Connect to cluster
spark = get_spark_session(profile="prod")

# Initialize pipeline
pipeline = HybridMatchingPipeline()

# Create UDF
@pandas_udf("struct<ciq_id:string,confidence:double>")
def match_udf(names):
    # Match entities
    return pipeline.batch_match(names)

# Process at scale
source_df = spark.table("source_entities")
matched_df = source_df.withColumn("match", match_udf("company_name"))
matched_df.write.saveAsTable("matched_entities")
```

## Architecture

```
┌─────────────────────┐
│   Local Machine     │
│                     │
│  Python Script      │
│  or Jupyter         │
│        │            │
│        ▼            │
│  Spark Connect      │
│  Client             │
└─────────┬───────────┘
          │
          │ HTTPS/gRPC
          │
┌─────────▼───────────┐
│  Databricks         │
│  Workspace          │
│                     │
│  ┌───────────────┐  │
│  │ Spark Connect │  │
│  │ Server        │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Spark Cluster │  │
│  │ (Executors)   │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Unity Catalog │  │
│  │ Delta Lake    │  │
│  └───────────────┘  │
└─────────────────────┘
```

## Benefits

1. **Local Development**
   - Write code in your IDE with full debugging support
   - No need to sync notebooks or files
   - Use local version control

2. **Remote Execution**
   - Leverage powerful Databricks clusters
   - Access Unity Catalog and Delta Lake
   - Scale to millions of entities

3. **Cost Efficiency**
   - Only pay for compute when running
   - Auto-terminate idle clusters
   - No local resource constraints

4. **Unified API**
   - Same PySpark API everywhere
   - Easy to switch between local and remote
   - No code changes needed

## Testing

Run the test script to verify your setup:

```bash
python test_spark_connect.py
```

This will check:
- ✓ Databricks CLI configuration
- ✓ Environment variables
- ✓ Spark Connect connection
- ✓ Basic Spark operations

## Troubleshooting

### Common Issues

1. **"Databricks CLI is not installed"**
   ```bash
   pip install databricks-cli
   ```

2. **"Failed to get Databricks profile"**
   ```bash
   databricks configure --profile DEFAULT
   ```

3. **"Cluster ID is required"**
   - Set `SPARK_CONNECT_CLUSTER_ID` in .env file

4. **"Connection refused"**
   - Verify cluster is running
   - Check cluster ID is correct
   - Ensure DBR 13.0+ (Spark Connect requirement)

See `SPARK_CONNECT_GUIDE.md` for detailed troubleshooting.

## Next Steps

1. **Configure CLI**: `databricks configure --profile DEFAULT`
2. **Test Setup**: `python test_spark_connect.py`
3. **Run Example**: `python example_spark_connect.py`
4. **Explore Notebook**: `notebooks/04_spark_connect_example.py`
5. **Read Guide**: `SPARK_CONNECT_GUIDE.md`

## Resources

- **Setup Guide**: `SPARK_CONNECT_GUIDE.md` - Comprehensive documentation
- **Quick Reference**: `SPARK_CONNECT_SETUP.md` - Common patterns
- **Example Script**: `example_spark_connect.py` - Complete workflow
- **Test Script**: `test_spark_connect.py` - Verify configuration
- **Notebook**: `notebooks/04_spark_connect_example.py` - Tutorial

## Support

For issues or questions:
1. Run test script: `python test_spark_connect.py`
2. Check guide: `SPARK_CONNECT_GUIDE.md`
3. Review Databricks logs
4. Contact Databricks support

---

**Note**: All changes are backward compatible. Existing code continues to work unchanged. Spark Connect is opt-in via `USE_SPARK_CONNECT=true` environment variable.
