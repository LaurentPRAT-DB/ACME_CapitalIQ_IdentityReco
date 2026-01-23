#!/bin/bash
# Refresh Databricks Token and Test Spark Connect

echo "=================================="
echo "Databricks Token Refresh"
echo "=================================="
echo ""
echo "Steps:"
echo "1. Go to: https://e2-demo-field-eng.cloud.databricks.com"
echo "2. Click Profile (top right) → Settings → Developer → Access Tokens"
echo "3. Click 'Generate New Token'"
echo "4. Copy the token"
echo ""
echo "Then run: databricks configure --profile DEFAULT"
echo "Or paste token below:"
echo ""

# Activate venv
source .venv/bin/activate

# Run configuration
databricks configure --profile DEFAULT

echo ""
echo "=================================="
echo "Testing Connection..."
echo "=================================="
echo ""

# Test the connection
python /tmp/test_serverless.py

echo ""
echo "=================================="
echo "If successful, you can now run:"
echo "  python test_spark_connect.py"
echo "  python example_spark_connect.py"
echo "=================================="
