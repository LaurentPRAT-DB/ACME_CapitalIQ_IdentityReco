# ⚠️ Migration Notice: Spark Connect Now Default

## What Changed?

**Spark Connect is now enabled by default** for local development. Your code will execute on remote Databricks clusters unless explicitly configured otherwise.

### Before (Previous Behavior)
```python
# Required explicit configuration to use Spark Connect
USE_SPARK_CONNECT=true  # in .env file

spark = get_spark_session()  # Would use local Spark by default
```

### After (New Default Behavior)
```python
# Spark Connect is enabled by default
# USE_SPARK_CONNECT=true  # No longer needed - it's the default!

spark = get_spark_session()  # Now uses Spark Connect (remote execution)
```

## Why This Change?

1. **Better Development Experience**: Write and debug locally, execute on powerful clusters
2. **Cost Efficiency**: No need for local Spark resources
3. **Scalability**: Handle large datasets without local memory constraints
4. **Consistency**: Same execution environment for dev/staging/prod

## Migration Guide

### If You Want to Keep Using Spark Connect (No Action Needed)

Your existing setup will continue to work. Just ensure:

1. Databricks CLI is configured: `databricks configure --profile DEFAULT`
2. Cluster ID is set in `.env`: `SPARK_CONNECT_CLUSTER_ID=your-cluster-id`
3. Remove `USE_SPARK_CONNECT=true` from `.env` (it's redundant now)

### If You Want to Use Local Spark (Opt-Out)

**Option 1: Set Environment Variable**

Add to your `.env` file:
```bash
USE_SPARK_CONNECT=false
```

**Option 2: Code Override**

```python
# Force local Spark in code
spark = get_spark_session(force_local=True)
```

**Option 3: Update Config**

```python
from src.config import Config

config = Config.from_env()
config.spark.use_spark_connect = False
spark = get_spark_session(config_obj=config)
```

## Configuration Changes

### Updated .env.example

**Old:**
```bash
# Enable Spark Connect
USE_SPARK_CONNECT=true
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
```

**New:**
```bash
# Spark Connect is enabled by default
# To disable: USE_SPARK_CONNECT=false
SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
```

### Updated src/config.py

**Old:**
```python
use_spark_connect: bool = False  # Disabled by default
```

**New:**
```python
use_spark_connect: bool = True  # Enabled by default
```

## Testing Your Setup

Verify your configuration is working correctly:

```bash
# Test Spark Connect connection
python test_spark_connect.py

# Or test in code
python -c "
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session

load_dotenv()
spark = get_spark_session()
print(f'✓ Connected! Spark version: {spark.version}')
"
```

## Common Scenarios

### Scenario 1: I don't have Databricks CLI configured yet

**What happens:**
- Code will try to use Spark Connect by default
- Will fail with authentication error

**Solution:**
```bash
# Configure Databricks CLI
databricks configure --profile DEFAULT

# Or disable Spark Connect temporarily
echo "USE_SPARK_CONNECT=false" >> .env
```

### Scenario 2: I don't have a Databricks cluster

**What happens:**
- Code will try to connect but cluster ID is missing
- Will fail with configuration error

**Solution:**
```bash
# Either disable Spark Connect
echo "USE_SPARK_CONNECT=false" >> .env

# Or get a cluster ID from your Databricks workspace
```

### Scenario 3: I want to use Spark Connect sometimes, not always

**Solution:**
```python
# In your code, decide dynamically
use_remote = os.getenv("ENVIRONMENT") == "production"

if use_remote:
    spark = get_spark_session()  # Uses Spark Connect
else:
    spark = get_spark_session(force_local=True)  # Local Spark
```

### Scenario 4: I'm running in Databricks notebook

**What happens:**
- Spark Connect is not needed (you're already on Databricks)
- Code should automatically detect this

**Solution:**
- No action needed - the code handles this automatically
- Or explicitly: `spark = get_spark_session(force_local=True)`

## Rollback Instructions

If you need to revert to the old behavior:

1. **Update src/config.py:**
```python
use_spark_connect: bool = False  # Back to opt-in
```

2. **Update .env:**
```bash
USE_SPARK_CONNECT=true  # Explicitly enable when needed
```

3. **Restart your application**

## Questions?

- **Documentation**: See `SPARK_CONNECT_README.md`
- **Detailed Guide**: See `SPARK_CONNECT_GUIDE.md`
- **Test Script**: Run `python test_spark_connect.py`
- **Examples**: Check `example_spark_connect.py`

## Timeline

- **Change Made**: 2026-01-23
- **Default Behavior**: Spark Connect enabled
- **Old Behavior**: Set `USE_SPARK_CONNECT=false` to restore

---

**Summary**: Spark Connect is now the default. To opt-out, set `USE_SPARK_CONNECT=false` in your `.env` file.
