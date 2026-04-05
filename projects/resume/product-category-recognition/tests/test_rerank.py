"""Tests for RerankStage."""

import pytest

from src.online.rerank import RerankStage


class TestRerankStage:
    """Test rerank stage: score fusion logic."""

    @pytest.fixture
    def recall_results(self):
        return [
            {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽",
             "recall_score": 0.9, "num_anchors": 5},
            {"category_leaf": "衬衫", "category_l2": "男装", "category_l1": "服装鞋帽",
             "recall_score": 0.7, "num_anchors": 3},
            {"category_leaf": "连衣裙", "category_l2": "女装", "category_l1": "服装鞋帽",
             "recall_score": 0.3, "num_anchors": 1},
        ]

    @pytest.fixture
    def rank_results(self):
        return [
            {"category_leaf": "T恤", "category_l2": "男装", "category_l1": "服装鞋帽",
             "rank_score": 0.85},
            {"category_leaf": "连衣裙", "category_l2": "女装", "category_l1": "服装鞋帽",
             "rank_score": 0.6},
            {"category_leaf": "智能手机", "category_l2": "手机", "category_l1": "数码电器",
             "rank_score": 0.4},
        ]

    def test_rerank_returns_results(self, recall_results, rank_results):
        """Rerank should return non-empty results."""
        stage = RerankStage(recall_weight=0.45, rank_weight=0.55)
        results = stage.rerank(recall_results, rank_results)
        assert len(results) > 0

    def test_rerank_result_structure(self, recall_results, rank_results):
        """Each rerank result should have required keys."""
        stage = RerankStage()
        results = stage.rerank(recall_results, rank_results)

        required_keys = {"category_leaf", "category_l2", "category_l1",
                         "final_score", "recall_score", "rank_score"}
        for result in results:
            assert required_keys.issubset(result.keys())

    def test_rerank_sorted_descending(self, recall_results, rank_results):
        """Results should be sorted by final_score descending."""
        stage = RerankStage()
        results = stage.rerank(recall_results, rank_results)

        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_union_of_candidates(self, recall_results, rank_results):
        """Rerank should include union of recall and rank candidates."""
        stage = RerankStage()
        results = stage.rerank(recall_results, rank_results)

        result_leaves = {r["category_leaf"] for r in results}
        recall_leaves = {r["category_leaf"] for r in recall_results}
        rank_leaves = {r["category_leaf"] for r in rank_results}

        assert result_leaves == recall_leaves | rank_leaves

    def test_rerank_weight_influence(self):
        """Higher recall_weight should favor recall scores more."""
        recall = [
            {"category_leaf": "A", "category_l2": "X", "category_l1": "Y",
             "recall_score": 1.0, "num_anchors": 5},
            {"category_leaf": "B", "category_l2": "X", "category_l1": "Y",
             "recall_score": 0.0, "num_anchors": 1},
        ]
        rank = [
            {"category_leaf": "A", "category_l2": "X", "category_l1": "Y",
             "rank_score": 0.0},
            {"category_leaf": "B", "category_l2": "X", "category_l1": "Y",
             "rank_score": 1.0},
        ]

        # Heavy recall weight → A should win
        stage_recall = RerankStage(recall_weight=0.9, rank_weight=0.1)
        results_recall = stage_recall.rerank(recall, rank)
        assert results_recall[0]["category_leaf"] == "A"

        # Heavy rank weight → B should win
        stage_rank = RerankStage(recall_weight=0.1, rank_weight=0.9)
        results_rank = stage_rank.rerank(recall, rank)
        assert results_rank[0]["category_leaf"] == "B"

    def test_rerank_empty_inputs(self):
        """Should handle empty recall or rank gracefully."""
        stage = RerankStage()

        assert stage.rerank([], []) == []

    def test_rerank_only_recall(self):
        """Should work with rank_results empty."""
        stage = RerankStage()
        recall = [
            {"category_leaf": "A", "category_l2": "X", "category_l1": "Y",
             "recall_score": 0.8, "num_anchors": 3},
        ]
        results = stage.rerank(recall, [])
        assert len(results) == 1
        assert results[0]["category_leaf"] == "A"
