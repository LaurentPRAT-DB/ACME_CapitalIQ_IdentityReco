#!/usr/bin/env python3
"""
Generate large-scale test data and export to CSV files

This script generates test data locally and saves it as CSV files
that can be uploaded to Databricks or used for testing.

Usage:
    python scripts/generate_test_data_csv.py --reference 1000 --source 3000 --output data/test/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.large_dataset_generator import LargeDatasetGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale test data for entity matching"
    )
    parser.add_argument(
        "--reference",
        type=int,
        default=1000,
        help="Number of reference entities to generate (default: 1000)"
    )
    parser.add_argument(
        "--source",
        type=int,
        default=3000,
        help="Number of source entities to generate (default: 3000)"
    )
    parser.add_argument(
        "--match-ratio",
        type=float,
        default=0.7,
        help="Ratio of source entities that should match (default: 0.7 = 70%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test",
        help="Output directory for CSV files (default: data/test)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.reference < 1:
        print("❌ Error: --reference must be >= 1")
        sys.exit(1)

    if args.source < 1:
        print("❌ Error: --source must be >= 1")
        sys.exit(1)

    if not 0 <= args.match_ratio <= 1:
        print("❌ Error: --match-ratio must be between 0 and 1")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LARGE-SCALE TEST DATA GENERATOR")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Reference entities: {args.reference}")
    print(f"  Source entities: {args.source}")
    print(f"  Match ratio: {args.match_ratio*100:.0f}%")
    print(f"  Output directory: {output_dir}")
    print(f"  Random seed: {args.seed}")
    print()

    # Initialize generator
    print("Initializing generator...")
    generator = LargeDatasetGenerator(seed=args.seed)

    # Generate reference entities
    print(f"\nGenerating {args.reference} reference entities...")
    reference_df = generator.generate_reference_entities(num_entities=args.reference)

    # Save reference entities
    reference_file = output_dir / "reference_entities.csv"
    reference_df.to_csv(reference_file, index=False)
    print(f"✅ Saved reference entities to: {reference_file}")

    # Print reference stats
    print(f"\nReference Entity Statistics:")
    print(f"  Total: {len(reference_df)}")
    print(f"  Countries: {reference_df['country'].nunique()}")
    print(f"  Industries: {reference_df['industry'].nunique()}")
    print(f"  With LEI: {reference_df['lei'].notna().sum()} ({reference_df['lei'].notna().sum()/len(reference_df)*100:.1f}%)")
    print(f"  With CUSIP: {reference_df['cusip'].notna().sum()} ({reference_df['cusip'].notna().sum()/len(reference_df)*100:.1f}%)")

    # Generate source entities
    print(f"\nGenerating {args.source} source entities with {args.match_ratio*100:.0f}% match ratio...")
    source_df = generator.generate_source_entities(
        reference_df=reference_df,
        num_entities=args.source,
        match_ratio=args.match_ratio
    )

    # Save source entities
    source_file = output_dir / "source_entities.csv"
    source_df.to_csv(source_file, index=False)
    print(f"✅ Saved source entities to: {source_file}")

    # Print source stats
    print(f"\nSource Entity Statistics:")
    print(f"  Total: {len(source_df)}")
    print(f"  Source systems: {source_df['source_system'].nunique()}")
    print(f"  With ticker: {source_df['ticker'].notna().sum()} ({source_df['ticker'].notna().sum()/len(source_df)*100:.1f}%)")
    if 'lei' in source_df.columns:
        print(f"  With LEI: {source_df['lei'].notna().sum()} ({source_df['lei'].notna().sum()/len(source_df)*100:.1f}%)")

    # Print sample data
    print(f"\n--- Sample Reference Entities ---")
    print(reference_df[['ciq_id', 'company_name', 'ticker', 'country', 'industry']].head(5).to_string(index=False))

    print(f"\n--- Sample Source Entities ---")
    print(source_df[['source_id', 'source_system', 'company_name', 'ticker', 'country']].head(5).to_string(index=False))

    # Summary
    print("\n" + "="*80)
    print("✅ TEST DATA GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  📄 {reference_file} ({len(reference_df)} entities)")
    print(f"  📄 {source_file} ({len(source_df)} entities)")
    print(f"\nExpected matches: ~{int(len(source_df) * args.match_ratio)}")
    print(f"Expected non-matches: ~{int(len(source_df) * (1 - args.match_ratio))}")
    print(f"\nNext steps:")
    print(f"  1. Upload CSV files to Databricks")
    print(f"  2. Load into Delta tables:")
    print(f"     spark.read.csv('{reference_file}', header=True, inferSchema=True)")
    print(f"     .write.format('delta').mode('overwrite')")
    print(f"     .saveAsTable('catalog.bronze.spglobal_reference')")
    print(f"  3. Run entity matching pipeline")
    print("="*80)


if __name__ == "__main__":
    main()
