"""
Example: Using Spark Connect to run entity matching on remote Databricks cluster

This script demonstrates how to:
1. Connect to a remote Databricks cluster via Spark Connect using CLI profiles
2. Run entity matching pipeline with distributed processing
3. Write results to Delta tables

Prerequisites:
- Configure Databricks CLI: databricks configure --profile DEFAULT
- Set SPARK_CONNECT_CLUSTER_ID in .env file (see .env.example)
- Ensure Databricks cluster is running
- Install dependencies: pip install -r requirements.txt

Setup:
  1. Configure Databricks CLI:
     $ databricks configure --profile DEFAULT
     Enter host: https://dbc-xxxxx-xxxx.cloud.databricks.com
     Enter token: dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

  2. Create .env file:
     $ cp .env.example .env
     Edit SPARK_CONNECT_CLUSTER_ID with your cluster ID

  3. Run:
     $ python example_spark_connect.py
"""
import os
from dotenv import load_dotenv
from src.utils.spark_utils import get_spark_session
from src.data.loader import DataLoader
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd


def main():
    """Run entity matching with Spark Connect"""

    # Load environment variables
    load_dotenv()

    print("=" * 80)
    print("Entity Matching with Spark Connect")
    print("=" * 80)

    # 1. Initialize Spark Connect session
    print("\n1. Initializing Spark Connect session...")
    print("   Using Databricks CLI profile for authentication")
    try:
        spark = get_spark_session()
        print(f"   ✓ Connected to Databricks cluster")
        print(f"   ✓ Spark version: {spark.version}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\n   Troubleshooting:")
        print("   1. Configure Databricks CLI:")
        print("      $ databricks configure --profile DEFAULT")
        print("   2. Set environment variables in .env:")
        print("      - DATABRICKS_PROFILE=DEFAULT")
        print("      - SPARK_CONNECT_CLUSTER_ID=<your-cluster-id>")
        print("      - USE_SPARK_CONNECT=true")
        print("   3. Ensure your Databricks cluster is running")
        return

    # 2. Load reference data
    print("\n2. Loading reference data...")
    loader = DataLoader()
    reference_df = loader.load_reference_data()
    print(f"   ✓ Loaded {len(reference_df)} reference entities")

    # 3. Initialize matching pipeline
    print("\n3. Initializing matching pipeline...")
    pipeline = HybridMatchingPipeline(
        reference_df=reference_df,
        ditto_model_path=None,  # Set to your trained model path
        enable_foundation_model=False  # Disable for demo
    )
    print("   ✓ Pipeline initialized")

    # 4. Create sample data in Spark
    print("\n4. Creating sample source entities...")
    source_data = [
        ("CRM-001", "Apple Inc.", "AAPL", "United States"),
        ("CRM-002", "Microsoft Corporation", "MSFT", "United States"),
        ("CRM-003", "Apple Computer Inc.", "AAPL", "USA"),
        ("TRD-001", "MSFT", None, "United States"),
        ("VND-001", "Amazon.com Inc", "AMZN", "United States"),
        ("VND-002", "Tesla Inc", "TSLA", "United States"),
        ("VND-003", "Alphabet Inc", "GOOGL", "United States"),
    ]

    source_df = spark.createDataFrame(
        source_data,
        ["source_id", "company_name", "ticker", "country"]
    )

    print(f"   ✓ Created {source_df.count()} source entities")
    print("\n   Sample entities:")
    source_df.show(5, truncate=False)

    # 5. Define Pandas UDF for entity matching
    print("\n5. Defining entity matching UDF...")

    result_schema = StructType([
        StructField("ciq_id", StringType(), True),
        StructField("confidence", DoubleType(), True),
        StructField("match_method", StringType(), True),
        StructField("stage_name", StringType(), True),
        StructField("reasoning", StringType(), True)
    ])

    @pandas_udf(result_schema)
    def match_entity_udf(
        company_names: pd.Series,
        tickers: pd.Series,
        countries: pd.Series
    ) -> pd.DataFrame:
        """
        Match entities using the hybrid pipeline
        This UDF runs on the remote Databricks cluster
        """
        results = []

        for name, ticker, country in zip(company_names, tickers, countries):
            entity = {
                "company_name": name,
                "ticker": ticker if pd.notna(ticker) else None,
                "country": country if pd.notna(country) else None
            }

            try:
                result = pipeline.match(entity, return_candidates=False)
                results.append({
                    "ciq_id": result.get("ciq_id"),
                    "confidence": result.get("confidence"),
                    "match_method": result.get("match_method"),
                    "stage_name": result.get("stage_name"),
                    "reasoning": result.get("reasoning", "")
                })
            except Exception as e:
                results.append({
                    "ciq_id": None,
                    "confidence": 0.0,
                    "match_method": "error",
                    "stage_name": "error",
                    "reasoning": str(e)
                })

        return pd.DataFrame(results)

    print("   ✓ UDF defined")

    # 6. Apply matching to Spark DataFrame
    print("\n6. Running entity matching on Databricks cluster...")
    print("   (This computation runs remotely on your Databricks cluster)")

    matched_df = source_df.withColumn(
        "match_result",
        match_entity_udf(
            col("company_name"),
            col("ticker"),
            col("country")
        )
    )

    # Expand the struct column
    matched_expanded = matched_df.select(
        "source_id",
        "company_name",
        "ticker",
        "country",
        col("match_result.ciq_id").alias("ciq_id"),
        col("match_result.confidence").alias("confidence"),
        col("match_result.match_method").alias("match_method"),
        col("match_result.stage_name").alias("stage"),
        col("match_result.reasoning").alias("reasoning")
    )

    # Cache results for multiple operations
    matched_expanded.cache()

    print(f"   ✓ Matched {matched_expanded.count()} entities")

    # 7. Display results
    print("\n7. Match Results:")
    matched_expanded.show(truncate=False)

    # 8. Calculate statistics
    print("\n8. Pipeline Statistics:")

    # Matches by stage
    print("\n   Matches by Stage:")
    matched_expanded.groupBy("stage").count().orderBy("count", ascending=False).show()

    # Matches by method
    print("   Matches by Method:")
    matched_expanded.groupBy("match_method").count().orderBy("count", ascending=False).show()

    # Average confidence
    avg_conf = matched_expanded.select(col("confidence")).agg({"confidence": "avg"}).first()[0]
    print(f"\n   Average Confidence: {avg_conf:.2%}")

    # Match rate
    matched_count = matched_expanded.filter(col("ciq_id").isNotNull()).count()
    total_count = matched_expanded.count()
    print(f"   Match Rate: {matched_count}/{total_count} ({matched_count/total_count:.1%})")

    # 9. Save results to Delta table
    print("\n9. Saving results...")

    # Option 1: Save to temporary location
    output_path = "/tmp/entity_matches_demo"
    matched_expanded.write \
        .format("delta") \
        .mode("overwrite") \
        .save(output_path)

    print(f"   ✓ Results saved to {output_path}")

    # Option 2: Save to Unity Catalog (uncomment if available)
    # catalog = os.getenv("UNITY_CATALOG", "main")
    # schema = os.getenv("UNITY_SCHEMA", "entity_matching")
    # table = f"{catalog}.{schema}.matched_entities"
    #
    # matched_expanded.write \
    #     .format("delta") \
    #     .mode("overwrite") \
    #     .option("overwriteSchema", "true") \
    #     .saveAsTable(table)
    #
    # print(f"   ✓ Results saved to {table}")

    # 10. Verify saved data
    print("\n10. Verifying saved data...")
    saved_df = spark.read.format("delta").load(output_path)
    print(f"    ✓ Read back {saved_df.count()} records from Delta table")

    print("\n" + "=" * 80)
    print("SUCCESS! Entity matching completed with Spark Connect")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("✓ Code ran locally but executed on remote Databricks cluster")
    print("✓ Leveraged cluster compute power for distributed processing")
    print("✓ Used familiar PySpark API with Spark Connect")
    print("✓ Results saved to Delta table for downstream consumption")

    print("\nNext Steps:")
    print("1. Scale up to millions of entities")
    print("2. Enable Ditto model for higher accuracy")
    print("3. Deploy to production with Unity Catalog")
    print("4. Set up monitoring and alerting")

    # Cleanup
    # spark.stop()  # Uncomment to stop Spark session


if __name__ == "__main__":
    main()
