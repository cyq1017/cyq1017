"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import compute_topk_accuracy


class TestTopKAccuracy:
    """Test TopK accuracy computation at all category levels."""

    @pytest.fixture
    def perfect_predictions(self):
        """Predictions where top1 is always correct."""
        predictions = [
            {
                "predicted_leaf": "T恤",
                "predicted_l2": "男装",
                "predicted_l1": "服装鞋帽",
                "confidence": 0.95,
                "top_candidates": [
                    {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽"},
                ],
            },
            {
                "predicted_leaf": "智能手机",
                "predicted_l2": "手机",
                "predicted_l1": "数码电器",
                "confidence": 0.9,
                "top_candidates": [
                    {"category_leaf": "智能手机", "category_l2": "手机", "category_l1": "数码电器"},
                ],
            },
        ]
        ground_truth = [
            {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽"},
            {"category_leaf": "智能手机", "category_l2": "手机", "category_l1": "数码电器"},
        ]
        return predictions, ground_truth

    @pytest.fixture
    def partial_predictions(self):
        """Predictions where correct answer is in top3 but not top1."""
        predictions = [
            {
                "predicted_leaf": "衬衫",
                "top_candidates": [
                    {"category_leaf": "衬衫", "category_l2": "男装", "category_l1": "服装鞋帽"},
                    {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽"},
                    {"category_leaf": "连衣裙", "category_l2": "女装", "category_l1": "服装鞋帽"},
                ],
            },
        ]
        ground_truth = [
            {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽"},
        ]
        return predictions, ground_truth

    def test_perfect_top1(self, perfect_predictions):
        """Perfect predictions should give 100% top1 accuracy."""
        preds, gt = perfect_predictions
        metrics = compute_topk_accuracy(preds, gt, k_values=[1])

        assert metrics["leaf_top1"] == 1.0
        assert metrics["l2_top1"] == 1.0
        assert metrics["l1_top1"] == 1.0

    def test_partial_top1_vs_top3(self, partial_predictions):
        """Correct at top3 but not top1."""
        preds, gt = partial_predictions
        metrics = compute_topk_accuracy(preds, gt, k_values=[1, 3])

        assert metrics["leaf_top1"] == 0.0  # T恤 not in top1
        assert metrics["leaf_top3"] == 1.0  # T恤 is in top3
        assert metrics["l1_top1"] == 1.0    # L1 is correct even at top1

    def test_empty_predictions(self):
        """Empty predictions should return 0 accuracy."""
        metrics = compute_topk_accuracy([], [], k_values=[1])
        assert metrics["leaf_top1"] == 0.0

    def test_default_k_values(self, perfect_predictions):
        """Default k_values should be [1, 3, 5]."""
        preds, gt = perfect_predictions
        metrics = compute_topk_accuracy(preds, gt)

        assert "leaf_top1" in metrics
        assert "leaf_top3" in metrics
        assert "leaf_top5" in metrics
