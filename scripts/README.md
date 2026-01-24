# Scripts Directory

This directory contains utility scripts for the Entity Matching project deployment.

## Scripts

### `create_catalog.py`

**Purpose**: Creates a Unity Catalog using the Databricks SDK.

**Usage**:
```bash
python scripts/create_catalog.py \
    --catalog-name <catalog_name> \
    --owner <user_email> \
    [--comment "<description>"] \
    [--grant-to <additional_user_or_group>]
```

**Examples**:
```bash
# Create dev catalog
python scripts/create_catalog.py \
    --catalog-name laurent_prat_entity_matching_dev \
    --owner laurent.prat@databricks.com

# Create with additional permissions
python scripts/create_catalog.py \
    --catalog-name entity_matching_staging \
    --owner laurent.prat@databricks.com \
    --grant-to data-engineers \
    --grant-to data-analysts
```

**Features**:
- Idempotent - safe to run multiple times
- Creates catalog if it doesn't exist
- Updates ownership
- Grants permissions to additional principals
- Validates catalog accessibility

**Requirements**:
- Python 3.7+
- `databricks-sdk` package
- Databricks CLI configured with valid credentials
- Permission to create catalogs in the workspace

## Helper Script

The `deploy-phase0.sh` script in the project root wraps `create_catalog.py` for easier usage:

```bash
# Instead of the full python command, use:
./deploy-phase0.sh dev
./deploy-phase0.sh staging
./deploy-phase0.sh prod
```

## Troubleshooting

### Import Error: databricks.sdk
```bash
pip install databricks-sdk
```

### Permission Denied
Ensure your Databricks user/service principal has:
- `CREATE CATALOG` privilege on the metastore
- Account admin or workspace admin role

### Catalog Already Exists
The script will detect existing catalogs and skip creation:
```
✓ Catalog 'laurent_prat_entity_matching_dev' already exists
```

This is normal and indicates the catalog is ready for use.
