# Spark Connect: Default Behavior Update

## Summary of Changes

Spark Connect has been changed from **opt-in** to **enabled by default**.

## Files Modified

### 1. Core Configuration
- **`src/config.py`**
  ```python
  # CHANGED: from False to True
  use_spark_connect: bool = True  # Enabled by default
  ```

### 2. Environment Template
- **`.env.example`**
  ```bash
  # CHANGED: Comment out USE_SPARK_CONNECT=true (now default)
  # Spark Connect is ENABLED BY DEFAULT
  # To disable: USE_SPARK_CONNECT=false
  SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh  # REQUIRED
  ```

### 3. Documentation Updates
- **`README.md`** - Updated setup instructions
- **`SPARK_CONNECT_README.md`** - Reflected new default
- **`SPARK_CONNECT_GUIDE.md`** - Updated configuration examples
- **`SPARK_CONNECT_SETUP.md`** - Updated quick reference
- **`CHANGES_SUMMARY.md`** - Documented default behavior

### 4. Utility Functions
- **`src/utils/spark_utils.py`**
  - Updated docstrings to emphasize default behavior
  - Improved log messages to show when Spark Connect is active
  - Added helpful tip when using local Spark

### 5. New Documentation
- **`MIGRATION_NOTICE.md`** - Migration guide for users
- **`DEFAULT_BEHAVIOR_UPDATE.md`** - This file

## New Default Behavior

### What Happens Now

```python
from src.utils.spark_utils import get_spark_session

# By default, uses Spark Connect (remote execution)
spark = get_spark_session()
```

**Result:**
- ✓ Connects to remote Databricks cluster
- ✓ Uses Databricks CLI profile authentication
- ✓ Executes code on cluster resources
- ✓ Requires: `SPARK_CONNECT_CLUSTER_ID` in `.env`

### How to Opt-Out (Use Local Spark)

**Method 1: Environment Variable**
```bash
# .env file
USE_SPARK_CONNECT=false
```

**Method 2: Code Parameter**
```python
spark = get_spark_session(force_local=True)
```

**Method 3: Config Object**
```python
from src.config import Config
config = Config.from_env()
config.spark.use_spark_connect = False
```

## Required Configuration

### Minimum Setup for Spark Connect (Default)

1. **Databricks CLI configured:**
   ```bash
   databricks configure --profile DEFAULT
   ```

2. **Cluster ID in .env:**
   ```bash
   SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
   ```

3. **That's it!** Spark Connect will be used automatically.

### Minimum Setup for Local Spark (Opt-Out)

1. **Disable Spark Connect in .env:**
   ```bash
   USE_SPARK_CONNECT=false
   ```

2. **That's it!** Local Spark will be used.

## Benefits of Default Spark Connect

1. **✅ Better Performance**: Cluster resources vs. local machine
2. **✅ Scalability**: Handle datasets larger than local memory
3. **✅ Consistency**: Same execution environment across dev/prod
4. **✅ Cost Efficiency**: No need for powerful local hardware
5. **✅ Easy Debugging**: Write/debug locally, execute remotely

## Breaking Changes?

**No breaking changes!** The code gracefully handles:

1. **Missing Configuration**: Falls back to local Spark with helpful message
2. **Authentication Errors**: Clear error messages with troubleshooting steps
3. **Cluster Not Running**: Fails fast with actionable error

## Quick Reference

| Scenario | Configuration | Result |
|----------|--------------|--------|
| Default (new) | No `USE_SPARK_CONNECT` set | Spark Connect (remote) |
| Explicit enable | `USE_SPARK_CONNECT=true` | Spark Connect (remote) |
| Explicit disable | `USE_SPARK_CONNECT=false` | Local Spark |
| Force in code | `force_local=True` | Local Spark |
| Missing cluster ID | Default + no cluster ID | Error with helpful message |

## Testing

Run the test script to verify your setup:

```bash
python test_spark_connect.py
```

**Expected Output (Default Behavior):**
```
Testing Databricks CLI Configuration
✓ Databricks CLI is configured
✓ DATABRICKS_HOST=...
✓ DATABRICKS_TOKEN=dapi****** (hidden)

Testing Environment Variables
✓ USE_SPARK_CONNECT=true (or not set - defaults to true)
✓ SPARK_CONNECT_CLUSTER_ID=1234-567890-abcdefgh
✓ DATABRICKS_PROFILE=DEFAULT

Testing Spark Connect
Using Spark Connect to remote Databricks cluster (default behavior)
✓ Connected successfully!
✓ Spark version: 3.5.0
```

## Migration Checklist

- [ ] Read `MIGRATION_NOTICE.md`
- [ ] Run `python test_spark_connect.py`
- [ ] Verify `.env` has `SPARK_CONNECT_CLUSTER_ID` (if using Spark Connect)
- [ ] Add `USE_SPARK_CONNECT=false` (if opting out to local Spark)
- [ ] Test your application
- [ ] Update CI/CD pipelines if needed
- [ ] Update team documentation

## Support

- **Quick Start**: `SPARK_CONNECT_README.md`
- **Full Guide**: `SPARK_CONNECT_GUIDE.md`
- **Migration**: `MIGRATION_NOTICE.md`
- **Test**: `python test_spark_connect.py`
- **Examples**: `example_spark_connect.py`

## Rollback

To revert to opt-in behavior:

1. Edit `src/config.py`:
   ```python
   use_spark_connect: bool = False
   ```

2. Add to `.env`:
   ```bash
   USE_SPARK_CONNECT=true  # When you want to use Spark Connect
   ```

---

**Date**: 2026-01-23
**Change**: Spark Connect enabled by default
**Impact**: Opt-out instead of opt-in for remote execution
