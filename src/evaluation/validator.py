"""
Gold standard validator for evaluating pipeline accuracy
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, List
from .metrics import calculate_pipeline_metrics, print_metrics


class GoldStandardValidator:
    """Validate pipeline against gold standard test set"""

    def __init__(self):
        pass

    def load_gold_standard(self, filepath: str) -> Dict[str, str]:
        """
        Load gold standard mappings

        Expected CSV format:
        source_id, true_ciq_id

        Args:
            filepath: Path to gold standard CSV

        Returns:
            Dictionary mapping source_id to true_ciq_id
        """
        df = pd.read_csv(filepath)

        if not all(col in df.columns for col in ["source_id", "true_ciq_id"]):
            raise ValueError("Gold standard must have columns: source_id, true_ciq_id")

        ground_truth = dict(zip(df["source_id"], df["true_ciq_id"]))
        print(f"Loaded {len(ground_truth)} gold standard mappings")

        return ground_truth

    def evaluate(
        self,
        pipeline,
        test_entities: List[Dict],
        ground_truth: Dict[str, str]
    ) -> Dict:
        """
        Evaluate pipeline on test entities

        Args:
            pipeline: HybridMatchingPipeline instance
            test_entities: List of test entities
            ground_truth: Dictionary mapping source_id to true_ciq_id

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating pipeline on {len(test_entities)} entities...")

        # Run pipeline
        results = pipeline.batch_match(test_entities, show_progress=True)

        # Calculate metrics
        metrics = calculate_pipeline_metrics(results, ground_truth)

        # Print results
        print_metrics(metrics, title="Pipeline Evaluation Results")

        # Add detailed results
        metrics["results"] = results
        metrics["ground_truth"] = ground_truth

        return metrics

    def analyze_errors(
        self,
        results: List[Dict],
        ground_truth: Dict[str, str],
        top_n: int = 10
    ):
        """
        Analyze matching errors

        Args:
            results: List of match results
            ground_truth: Dictionary of ground truth mappings
            top_n: Number of errors to show
        """
        errors = []

        for result in results:
            source_id = result.get("source_id")
            if source_id not in ground_truth:
                continue

            true_ciq_id = ground_truth[source_id]
            pred_ciq_id = result.get("ciq_id")

            # False negative (should have matched but didn't)
            if true_ciq_id and not pred_ciq_id:
                errors.append({
                    "type": "false_negative",
                    "source_id": source_id,
                    "true_ciq_id": true_ciq_id,
                    "pred_ciq_id": None,
                    "confidence": result.get("confidence", 0),
                    "stage": result.get("stage_name", "unknown"),
                    "reasoning": result.get("reasoning", "")
                })

            # False positive (matched but wrong)
            elif true_ciq_id != pred_ciq_id and pred_ciq_id:
                errors.append({
                    "type": "false_positive",
                    "source_id": source_id,
                    "true_ciq_id": true_ciq_id,
                    "pred_ciq_id": pred_ciq_id,
                    "confidence": result.get("confidence", 0),
                    "stage": result.get("stage_name", "unknown"),
                    "reasoning": result.get("reasoning", "")
                })

        print(f"\n{'=' * 80}")
        print(f"ERROR ANALYSIS (Top {top_n})")
        print(f"{'=' * 80}")
        print(f"Total Errors: {len(errors)}")

        # Group by error type
        fn_errors = [e for e in errors if e["type"] == "false_negative"]
        fp_errors = [e for e in errors if e["type"] == "false_positive"]

        print(f"  False Negatives: {len(fn_errors)} (should have matched)")
        print(f"  False Positives: {len(fp_errors)} (wrong match)")

        # Show top false negatives
        if fn_errors:
            print(f"\nTop False Negatives:")
            for i, error in enumerate(fn_errors[:top_n], 1):
                print(f"\n  {i}. Source ID: {error['source_id']}")
                print(f"     True CIQ ID: {error['true_ciq_id']}")
                print(f"     Stage: {error['stage']}")
                print(f"     Reasoning: {error['reasoning'][:100]}...")

        # Show top false positives
        if fp_errors:
            print(f"\nTop False Positives:")
            for i, error in enumerate(fp_errors[:top_n], 1):
                print(f"\n  {i}. Source ID: {error['source_id']}")
                print(f"     True CIQ ID: {error['true_ciq_id']}")
                print(f"     Predicted CIQ ID: {error['pred_ciq_id']}")
                print(f"     Confidence: {error['confidence']:.2%}")
                print(f"     Stage: {error['stage']}")
                print(f"     Reasoning: {error['reasoning'][:100]}...")

        print(f"{'=' * 80}")

        return errors

    def create_gold_standard(
        self,
        source_entities: List[Dict],
        reference_df: pd.DataFrame,
        output_path: str
    ):
        """
        Create gold standard dataset from known mappings (e.g., S&P 500)

        Args:
            source_entities: List of source entities with known CIQ IDs
            reference_df: Reference data for validation
            output_path: Path to save gold standard CSV
        """
        gold_standard = []

        for entity in source_entities:
            source_id = entity.get("source_id")
            true_ciq_id = entity.get("ciq_id") or entity.get("true_ciq_id")

            if source_id and true_ciq_id:
                # Validate CIQ ID exists in reference
                if true_ciq_id in reference_df["ciq_id"].values:
                    gold_standard.append({
                        "source_id": source_id,
                        "true_ciq_id": true_ciq_id,
                        "company_name": entity.get("company_name", "")
                    })

        df = pd.DataFrame(gold_standard)
        df.to_csv(output_path, index=False)

        print(f"Created gold standard with {len(df)} entities")
        print(f"Saved to: {output_path}")

        return df
