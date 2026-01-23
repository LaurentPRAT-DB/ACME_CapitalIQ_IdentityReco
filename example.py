"""
Simple example demonstrating entity matching with the hybrid pipeline
"""
from src.data.loader import DataLoader
from src.data.training_generator import TrainingDataGenerator
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline


def main():
    """Run entity matching example"""
    print("=" * 80)
    print("Entity Matching for S&P Capital IQ - Quick Example")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    reference_df = loader.load_reference_data()
    source_entities = loader.load_sample_entities()

    print(f"   - Reference entities: {len(reference_df)}")
    print(f"   - Source entities: {len(source_entities)}")

    # 2. Initialize pipeline (without Ditto for quick demo)
    print("\n2. Initializing pipeline...")
    pipeline = HybridMatchingPipeline(
        reference_df=reference_df,
        ditto_model_path=None,  # Set to your trained model path
        enable_foundation_model=False  # Disable for demo
    )

    # 3. Match a single entity
    print("\n3. Matching single entity...")
    entity = source_entities[0]
    print(f"\n   Source Entity:")
    print(f"   - Name: {entity['company_name']}")
    print(f"   - Ticker: {entity.get('ticker', 'N/A')}")
    print(f"   - LEI: {entity.get('lei', 'N/A')}")

    result = pipeline.match(entity, return_candidates=True)

    print(f"\n   Match Result:")
    print(f"   - CIQ ID: {result['ciq_id']}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"   - Method: {result['match_method']}")
    print(f"   - Stage: {result['stage_name']}")
    print(f"   - Reasoning: {result['reasoning']}")

    # 4. Batch matching
    print("\n4. Batch matching all entities...")
    results = pipeline.batch_match(source_entities, show_progress=False)

    # 5. Display statistics
    print("\n5. Pipeline Statistics:")
    stats = pipeline.get_pipeline_stats(results)

    print(f"   - Total Entities: {stats['total_entities']}")
    print(f"   - Matched: {stats['matched']} ({stats['match_rate']:.1%})")
    print(f"   - Avg Confidence: {stats['avg_confidence']:.1%}")

    print(f"\n   Matches by Stage:")
    for stage, count in stats['stages'].items():
        pct = count / stats['total_entities'] * 100
        print(f"     {stage}: {count} ({pct:.1f}%)")

    # 6. Training data generation example
    print("\n6. Generating training data for Ditto...")
    generator = TrainingDataGenerator()
    training_df = generator.generate_from_sp500(
        reference_df=reference_df,
        num_positive_pairs=100,
        num_negative_pairs=100
    )

    print(f"   - Generated {len(training_df)} training pairs")
    print(f"   - Positive pairs: {len(training_df[training_df['label'] == 1])}")
    print(f"   - Negative pairs: {len(training_df[training_df['label'] == 0])}")

    # Save training data
    output_path = "data/ditto_training_sample.csv"
    training_df.to_csv(output_path, index=False)
    print(f"   - Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Train Ditto model: python -m src.models.ditto_matcher")
    print("2. Run full pipeline with Ditto enabled")
    print("3. Deploy to Databricks for production use")
    print("=" * 80)


if __name__ == "__main__":
    main()
