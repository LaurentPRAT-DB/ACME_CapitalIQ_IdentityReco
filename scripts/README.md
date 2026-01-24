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

## Deployment Script

The project now uses a unified `deploy-phase.sh` script that handles all deployment phases (0-4):

```bash
# Phase 0: Catalog Setup (creates catalog and schemas via Databricks jobs)
./deploy-phase.sh 0 dev
./deploy-phase.sh 0 staging
./deploy-phase.sh 0 prod

# Subsequent phases
./deploy-phase.sh 1 dev  # Data load
./deploy-phase.sh 2 dev  # Model training
./deploy-phase.sh 3 dev  # Model deployment
./deploy-phase.sh 4 dev  # Production pipeline
```

**Note**: The `create_catalog.py` script is kept for reference but is no longer used in the standard deployment workflow. Catalog creation is now handled through Databricks jobs in Phase 0.

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
