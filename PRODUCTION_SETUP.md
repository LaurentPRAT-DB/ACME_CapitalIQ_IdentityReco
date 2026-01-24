# Production Deployment Setup Guide

This guide explains how to configure and deploy the Entity Matching pipeline to staging and production environments using service principals.

## Overview

Production deployments (staging and prod) should use **service principals** instead of user accounts for:
- Better security and access control
- Automated deployments in CI/CD pipelines
- Separation of duties
- Long-term stability (not tied to individual users)

## Prerequisites

- Workspace admin access to create service principals
- Databricks CLI configured
- Access to staging and production workspaces

---

## Step 1: Create Service Principals

### Create Staging Service Principal

1. Navigate to your Databricks workspace
2. Go to **Workspace Settings** → **Identity and Access** → **Service Principals**
3. Click **Add Service Principal**
4. Enter details:
   - **Display Name**: `entity-matching-staging`
   - **Description**: `Service principal for Entity Matching staging environment`
5. Click **Create**
6. **Copy the Application ID** (UUID format) - this is your `staging_service_principal_id`
7. Grant permissions:
   - Workspace access: **User** or **Admin** (depending on needs)
   - Unity Catalog: **USE CATALOG** on staging catalog
   - Clusters: **Can Attach To** on job clusters

### Create Production Service Principal

Repeat the same process for production:
- **Display Name**: `entity-matching-production`
- **Description**: `Service principal for Entity Matching production environment`
- Copy the Application ID - this is your `prod_service_principal_id`

### Example Service Principal IDs

```
Staging:  12345678-1234-1234-1234-123456789abc
Production: 87654321-4321-4321-4321-cba987654321
```

---

## Step 2: Configure Service Principal IDs

You have three options to configure the service principal IDs:

### Option 1: Environment Variables (Recommended for Local Deployments)

```bash
# Set environment variables
export STAGING_SERVICE_PRINCIPAL_ID="12345678-1234-1234-1234-123456789abc"
export PROD_SERVICE_PRINCIPAL_ID="87654321-4321-4321-4321-cba987654321"

# Deploy to staging
./deploy-phase.sh 0 staging
```

### Option 2: CLI Variable Override (For One-Time Deployments)

```bash
# Deploy to staging with inline variable
databricks bundle deploy -t staging \
  --var="staging_service_principal_id=12345678-1234-1234-1234-123456789abc"

# Deploy to production with inline variable
databricks bundle deploy -t prod \
  --var="prod_service_principal_id=87654321-4321-4321-4321-cba987654321"
```

### Option 3: Local Configuration File (For Team Deployments)

```bash
# Copy the template
cp production-config.yml production-config.local.yml

# Edit the file and replace placeholders with actual IDs
nano production-config.local.yml

# Update these lines:
staging:
  service_principal_id: "12345678-1234-1234-1234-123456789abc"

production:
  service_principal_id: "87654321-4321-4321-4321-cba987654321"
```

**Note:** `production-config.local.yml` is gitignored and will not be committed.

---

## Step 3: Grant Service Principal Permissions

### Unity Catalog Permissions

```sql
-- For Staging Service Principal
GRANT USE CATALOG ON CATALOG entity_matching_staging
  TO `12345678-1234-1234-1234-123456789abc`;

GRANT USE SCHEMA ON SCHEMA entity_matching_staging.bronze
  TO `12345678-1234-1234-1234-123456789abc`;

GRANT USE SCHEMA ON SCHEMA entity_matching_staging.silver
  TO `12345678-1234-1234-1234-123456789abc`;

GRANT USE SCHEMA ON SCHEMA entity_matching_staging.gold
  TO `12345678-1234-1234-1234-123456789abc`;

GRANT USE SCHEMA ON SCHEMA entity_matching_staging.models
  TO `12345678-1234-1234-1234-123456789abc`;

-- For Production Service Principal
GRANT USE CATALOG ON CATALOG entity_matching
  TO `87654321-4321-4321-4321-cba987654321`;

GRANT USE SCHEMA ON SCHEMA entity_matching.bronze
  TO `87654321-4321-4321-4321-cba987654321`;

GRANT USE SCHEMA ON SCHEMA entity_matching.silver
  TO `87654321-4321-4321-4321-cba987654321`;

GRANT USE SCHEMA ON SCHEMA entity_matching.gold
  TO `87654321-4321-4321-4321-cba987654321`;

GRANT USE SCHEMA ON SCHEMA entity_matching.models
  TO `87654321-4321-4321-4321-cba987654321`;
```

### Model Serving Permissions

The service principals need access to model serving endpoints. This is handled automatically by the bundle deployment, but you can verify:

```bash
# Check model serving endpoint permissions
databricks serving-endpoints get ditto-em-staging
databricks serving-endpoints get ditto-em-prod
```

---

## Step 4: Deploy to Staging

```bash
# Set environment variables
export STAGING_SERVICE_PRINCIPAL_ID="your-staging-sp-id"

# Deploy all phases to staging
./deploy-phase.sh 0 staging  # Catalog Setup
./deploy-phase.sh 1 staging  # Data Load
./deploy-phase.sh 2 staging  # Model Training
./deploy-phase.sh 3 staging  # Model Deployment
./deploy-phase.sh 4 staging  # Production Pipeline
```

### Verify Staging Deployment

```bash
# Check deployed jobs
databricks jobs list --profile DEFAULT | grep staging

# Check model serving endpoint
databricks serving-endpoints get ditto-em-staging

# Verify catalog
databricks catalogs get entity_matching_staging
```

---

## Step 5: Deploy to Production

```bash
# Set environment variables
export PROD_SERVICE_PRINCIPAL_ID="your-prod-sp-id"

# Deploy all phases to production
./deploy-phase.sh 0 prod  # Catalog Setup
./deploy-phase.sh 1 prod  # Data Load
./deploy-phase.sh 2 prod  # Model Training
./deploy-phase.sh 3 prod  # Model Deployment
./deploy-phase.sh 4 prod  # Production Pipeline
```

### Verify Production Deployment

```bash
# Check deployed jobs
databricks jobs list --profile DEFAULT | grep prod

# Check model serving endpoint
databricks serving-endpoints get ditto-em-prod

# Verify catalog
databricks catalogs get entity_matching
```

---

## Security Best Practices

### 1. Service Principal Secrets

- **Never commit service principal IDs to git**
- Use environment variables or secure secret management
- Rotate secrets regularly

### 2. Least Privilege Access

Grant only the minimum permissions needed:
- ✅ **Do**: Grant USE CATALOG, USE SCHEMA
- ✅ **Do**: Grant CAN_MANAGE on bundle resources
- ❌ **Don't**: Grant workspace admin to service principals
- ❌ **Don't**: Grant CREATE CATALOG or other elevated permissions

### 3. Separate Service Principals

- Use **different** service principals for staging and production
- Never share service principals across environments
- Use descriptive names for easy identification

### 4. Audit and Monitoring

- Regularly audit service principal permissions
- Monitor service principal activity in audit logs
- Set up alerts for unusual service principal behavior

### 5. Secret Management

For CI/CD pipelines, use:
- **GitHub Secrets** for GitHub Actions
- **Azure Key Vault** for Azure Pipelines
- **AWS Secrets Manager** for AWS environments
- **Databricks Secrets** for Databricks-native workflows

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Databricks CLI
        run: pip install databricks-cli

      - name: Configure Databricks CLI
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks configure --token <<EOF
          $DATABRICKS_HOST
          $DATABRICKS_TOKEN
          EOF

      - name: Deploy to Production
        env:
          PROD_SERVICE_PRINCIPAL_ID: ${{ secrets.PROD_SERVICE_PRINCIPAL_ID }}
        run: |
          export prod_service_principal_id=$PROD_SERVICE_PRINCIPAL_ID
          ./deploy-phase.sh 0 prod
          ./deploy-phase.sh 1 prod
          ./deploy-phase.sh 2 prod
          ./deploy-phase.sh 3 prod
          ./deploy-phase.sh 4 prod
```

---

## Troubleshooting

### Issue: Service Principal Not Found

**Error:** `Service principal '...' does not exist`

**Solution:**
1. Verify the service principal ID is correct (UUID format)
2. Check the service principal exists: `databricks service-principals get <id>`
3. Ensure you're in the correct workspace

### Issue: Permission Denied

**Error:** `User does not have permission to ...`

**Solution:**
1. Grant required Unity Catalog permissions (see Step 3)
2. Grant workspace access to the service principal
3. Check bundle permissions in databricks-phase*.yml

### Issue: Environment Variable Not Set

**Error:** `variable "staging_service_principal_id" not found`

**Solution:**
```bash
# Check if variable is set
echo $STAGING_SERVICE_PRINCIPAL_ID

# Set the variable
export STAGING_SERVICE_PRINCIPAL_ID="your-id"

# Or use CLI override
databricks bundle deploy -t staging --var="staging_service_principal_id=your-id"
```

---

## Summary

✅ **Created** service principals for staging and production
✅ **Configured** service principal IDs via environment variables
✅ **Granted** Unity Catalog and workspace permissions
✅ **Deployed** all phases to staging environment
✅ **Deployed** all phases to production environment
✅ **Verified** deployments are working

Your production deployment is now secure and ready! 🎉

---

## Next Steps

- Set up monitoring and alerting for production jobs
- Configure backup and disaster recovery
- Implement CI/CD pipelines for automated deployments
- Document runbooks for common operational tasks
- Schedule regular reviews of service principal permissions
