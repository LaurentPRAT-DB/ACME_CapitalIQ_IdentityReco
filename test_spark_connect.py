#!/usr/bin/env python3
"""
Quick test script to verify Spark Connect configuration

This script tests:
1. Databricks CLI profile authentication
2. Spark Connect connection
3. Basic Spark operations

Run: python test_spark_connect.py
"""
import os
import sys
from dotenv import load_dotenv


def test_databricks_cli():
    """Test Databricks CLI configuration"""
    import subprocess

    print("=" * 80)
    print("Testing Databricks CLI Configuration")
    print("=" * 80)

    try:
        # Get profile from environment or use DEFAULT
        profile = os.getenv("DATABRICKS_PROFILE", "DEFAULT")
        print(f"\n1. Testing profile: {profile}")

        # Test databricks CLI
        result = subprocess.run(
            ["databricks", "auth", "env", "--profile", profile],
            capture_output=True,
            text=True,
            check=True
        )

        print(f"   ✓ Databricks CLI is configured")

        # Parse and display (without showing full token)
        for line in result.stdout.strip().split('\n'):
            if 'DATABRICKS_HOST' in line:
                print(f"   ✓ {line}")
            elif 'DATABRICKS_TOKEN' in line:
                print(f"   ✓ DATABRICKS_TOKEN=dapi****** (hidden)")

        return True

    except FileNotFoundError:
        print("   ✗ Databricks CLI is not installed")
        print("\n   Install with:")
        print("     pip install databricks-cli")
        return False

    except subprocess.CalledProcessError as e:
        print(f"   ✗ Databricks CLI configuration failed")
        print(f"\n   Configure with:")
        print(f"     databricks configure --profile {profile}")
        return False


def test_environment_variables():
    """Test required environment variables"""
    print("\n" + "=" * 80)
    print("Testing Environment Variables")
    print("=" * 80)

    load_dotenv()

    required_vars = {
        "USE_SPARK_CONNECT": os.getenv("USE_SPARK_CONNECT"),
        "SPARK_CONNECT_CLUSTER_ID": os.getenv("SPARK_CONNECT_CLUSTER_ID"),
        "DATABRICKS_PROFILE": os.getenv("DATABRICKS_PROFILE", "DEFAULT")
    }

    all_set = True
    for var, value in required_vars.items():
        if value:
            if var == "SPARK_CONNECT_CLUSTER_ID":
                print(f"   ✓ {var}={value}")
            else:
                print(f"   ✓ {var}={value}")
        else:
            print(f"   ✗ {var} is not set")
            all_set = False

    if not all_set:
        print("\n   Create a .env file with:")
        print("     cp .env.example .env")
        print("   Then edit .env with your configuration")

    return all_set


def test_spark_connect():
    """Test Spark Connect connection"""
    print("\n" + "=" * 80)
    print("Testing Spark Connect")
    print("=" * 80)

    try:
        from src.utils.spark_utils import get_spark_session

        print("\n2. Connecting to Databricks cluster...")
        spark = get_spark_session()

        print(f"   ✓ Connected successfully!")
        print(f"   ✓ Spark version: {spark.version}")
        print(f"   ✓ Application ID: {spark.sparkContext.applicationId}")

        # Test basic operation
        print("\n3. Testing basic Spark operation...")
        test_df = spark.range(10)
        count = test_df.count()
        print(f"   ✓ Created test DataFrame with {count} rows")

        # Test SQL
        print("\n4. Testing Spark SQL...")
        result = spark.sql("SELECT 'Hello from Databricks!' as message, current_timestamp() as timestamp")
        row = result.first()
        print(f"   ✓ SQL query successful: {row['message']}")
        print(f"   ✓ Timestamp: {row['timestamp']}")

        # Show Spark UI URL if available
        try:
            ui_url = spark.sparkContext.uiWebUrl
            if ui_url:
                print(f"\n   Spark UI: {ui_url}")
        except:
            pass

        return True

    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        print("\n   Install dependencies with:")
        print("     pip install -r requirements.txt")
        return False

    except ValueError as e:
        print(f"   ✗ Configuration error: {e}")
        return False

    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("\n   Troubleshooting:")
        print("   - Verify cluster is running in Databricks workspace")
        print("   - Check SPARK_CONNECT_CLUSTER_ID is correct")
        print("   - Ensure cluster supports Spark Connect (DBR 13.0+)")
        return False


def main():
    """Run all tests"""
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Spark Connect Configuration Test" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: Databricks CLI
    cli_ok = test_databricks_cli()

    # Test 2: Environment Variables
    env_ok = test_environment_variables()

    # Only test Spark Connect if CLI and env are OK
    if cli_ok and env_ok:
        spark_ok = test_spark_connect()
    else:
        print("\n" + "=" * 80)
        print("Skipping Spark Connect test (prerequisites not met)")
        print("=" * 80)
        spark_ok = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Databricks CLI:       {'✓ PASS' if cli_ok else '✗ FAIL'}")
    print(f"  Environment Variables: {'✓ PASS' if env_ok else '✗ FAIL'}")
    print(f"  Spark Connect:        {'✓ PASS' if spark_ok else '✗ FAIL'}")
    print("=" * 80)

    if cli_ok and env_ok and spark_ok:
        print("\n🎉 SUCCESS! Your Spark Connect is configured correctly!")
        print("\nNext steps:")
        print("  - Run example: python example_spark_connect.py")
        print("  - Explore notebook: notebooks/04_spark_connect_example.py")
        print("  - Read guide: SPARK_CONNECT_GUIDE.md")
        return 0
    else:
        print("\n❌ Configuration incomplete. Please fix the issues above.")
        print("\nQuick setup:")
        print("  1. databricks configure --profile DEFAULT")
        print("  2. cp .env.example .env")
        print("  3. Edit .env with your SPARK_CONNECT_CLUSTER_ID")
        print("  4. python test_spark_connect.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
