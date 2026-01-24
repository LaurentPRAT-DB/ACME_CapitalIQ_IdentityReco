"""
Evaluation metrics for entity matching
"""
from __future__ import annotations

from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np


def calculate_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float] = None
) -> Dict[str, float]:
    """
    Calculate matching metrics

    Args:
        true_labels: Ground truth labels (0 or 1)
        pred_labels: Predicted labels (0 or 1)
        confidences: Prediction confidence scores

    Returns:
        Dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "f1_score": f1_score(true_labels, pred_labels, zero_division=0)
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # Additional metrics
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Confidence metrics
    if confidences:
        metrics["avg_confidence"] = np.mean(confidences)
        metrics["median_confidence"] = np.median(confidences)
        metrics["min_confidence"] = np.min(confidences)
        metrics["max_confidence"] = np.max(confidences)

    return metrics


def calculate_pipeline_metrics(results: List[Dict], ground_truth: Dict[str, str]) -> Dict:
    """
    Calculate metrics for pipeline results

    Args:
        results: List of match results from pipeline
        ground_truth: Dictionary mapping source_id to true ciq_id

    Returns:
        Dictionary with metrics
    """
    true_labels = []
    pred_labels = []
    confidences = []

    for result in results:
        source_id = result.get("source_id")
        if source_id not in ground_truth:
            continue

        true_ciq_id = ground_truth[source_id]
        pred_ciq_id = result.get("ciq_id")

        # Convert to binary labels
        true_labels.append(1 if true_ciq_id else 0)
        pred_labels.append(1 if pred_ciq_id else 0)

        # Check if match is correct
        if true_ciq_id and pred_ciq_id:
            if true_ciq_id == pred_ciq_id:
                confidences.append(result.get("confidence", 0))
            else:
                # Wrong match - treat as low confidence
                confidences.append(0.0)
        else:
            confidences.append(result.get("confidence", 0))

    metrics = calculate_metrics(true_labels, pred_labels, confidences)

    # Add pipeline-specific metrics
    metrics["total_entities"] = len(results)
    metrics["matched_entities"] = len([r for r in results if r.get("ciq_id")])
    metrics["match_rate"] = metrics["matched_entities"] / metrics["total_entities"]

    # Accuracy by stage
    stage_accuracy = {}
    for result in results:
        stage = result.get("stage_name", "unknown")
        if stage not in stage_accuracy:
            stage_accuracy[stage] = {"correct": 0, "total": 0}

        stage_accuracy[stage]["total"] += 1

        source_id = result.get("source_id")
        if source_id in ground_truth:
            true_ciq_id = ground_truth[source_id]
            pred_ciq_id = result.get("ciq_id")
            if true_ciq_id == pred_ciq_id:
                stage_accuracy[stage]["correct"] += 1

    metrics["stage_accuracy"] = {
        stage: data["correct"] / data["total"] if data["total"] > 0 else 0
        for stage, data in stage_accuracy.items()
    }

    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    """Print formatted metrics"""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

    # Main metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")

    # Confidence metrics
    if "avg_confidence" in metrics:
        print(f"\nConfidence Scores:")
        print(f"  Average: {metrics['avg_confidence']:.2%}")
        print(f"  Median:  {metrics['median_confidence']:.2%}")
        print(f"  Range:   {metrics['min_confidence']:.2%} - {metrics['max_confidence']:.2%}")

    # Pipeline metrics
    if "match_rate" in metrics:
        print(f"\nPipeline Metrics:")
        print(f"  Total Entities: {metrics['total_entities']}")
        print(f"  Matched: {metrics['matched_entities']}")
        print(f"  Match Rate: {metrics['match_rate']:.2%}")

    if "stage_accuracy" in metrics:
        print(f"\nAccuracy by Stage:")
        for stage, accuracy in metrics["stage_accuracy"].items():
            print(f"  {stage}: {accuracy:.2%}")

    print("=" * 60)
