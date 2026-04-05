"""
Evaluation Metrics.

Computes TOP-K accuracy at leaf, L2, and L1 category levels,
generates confusion matrices and classification reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_topk_accuracy(predictions: list[dict], ground_truth: list[dict],
                          k_values: list[int] = None) -> dict:
    """
    Compute TOP-K accuracy at leaf, L2, and L1 levels.

    Args:
        predictions: list of pipeline outputs, each with "top_candidates"
        ground_truth: list of dicts with true category_leaf, category_l2, category_l1
        k_values: list of K values to compute (default [1, 3, 5])

    Returns:
        dict of metrics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    metrics = {}

    for k in k_values:
        leaf_correct = 0
        l2_correct = 0
        l1_correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truth):
            candidates = pred.get("top_candidates", [])[:k]
            pred_leaves = [c["category_leaf"] for c in candidates]
            pred_l2s = [c["category_l2"] for c in candidates]
            pred_l1s = [c["category_l1"] for c in candidates]

            if gt["category_leaf"] in pred_leaves:
                leaf_correct += 1
            if gt["category_l2"] in pred_l2s:
                l2_correct += 1
            if gt["category_l1"] in pred_l1s:
                l1_correct += 1

        metrics[f"leaf_top{k}"] = leaf_correct / total if total > 0 else 0.0
        metrics[f"l2_top{k}"] = l2_correct / total if total > 0 else 0.0
        metrics[f"l1_top{k}"] = l1_correct / total if total > 0 else 0.0

    return metrics


def generate_classification_report(predictions: list[dict],
                                    ground_truth: list[dict],
                                    level: str = "leaf") -> str:
    """Generate sklearn classification report at the specified level."""
    key_map = {"leaf": "category_leaf", "l2": "category_l2", "l1": "category_l1"}
    pred_key = f"predicted_{level}"
    gt_key = key_map[level]

    y_true = [gt[gt_key] for gt in ground_truth]
    y_pred = [pred[pred_key] for pred in predictions]

    return classification_report(y_true, y_pred, zero_division=0)


def plot_confusion_matrix(predictions: list[dict], ground_truth: list[dict],
                          level: str, output_path: Path, top_n_classes: int = 20):
    """
    Plot confusion matrix for the specified category level.

    Only shows top_n_classes most frequent classes for readability.
    """
    key_map = {"leaf": "category_leaf", "l2": "category_l2", "l1": "category_l1"}
    pred_key = f"predicted_{level}"
    gt_key = key_map[level]

    y_true = [gt[gt_key] for gt in ground_truth]
    y_pred = [pred[pred_key] for pred in predictions]

    # Get most frequent classes
    from collections import Counter
    freq = Counter(y_true)
    top_classes = [c for c, _ in freq.most_common(top_n_classes)]

    # Filter to top classes
    mask = [yt in top_classes for yt in y_true]
    y_true_filtered = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_filtered = [yp for yp, m in zip(y_pred, mask) if m]

    labels = sorted(set(y_true_filtered))
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)

    # Normalize
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5),
                                     max(10, len(labels) * 0.4)))
    sns.heatmap(cm_normalized, annot=False, xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {level.upper()} Level (Top {top_n_classes})")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {output_path}")


def save_evaluation_results(metrics: dict, output_dir: Path,
                            predictions: list[dict] = None,
                            ground_truth: list[dict] = None):
    """Save all evaluation results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Print metrics summary
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f} ({value:.1%})")
    print("=" * 50)

    if predictions is not None and ground_truth is not None:
        # Classification reports
        for level in ["l1", "l2"]:
            report = generate_classification_report(predictions, ground_truth, level)
            report_path = output_dir / f"classification_report_{level}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Classification report ({level}): {report_path}")

        # Confusion matrices
        for level in ["l1", "l2"]:
            n_classes = 20 if level == "l2" else 15
            plot_confusion_matrix(
                predictions, ground_truth, level,
                output_dir / f"confusion_matrix_{level}.png",
                top_n_classes=n_classes,
            )
