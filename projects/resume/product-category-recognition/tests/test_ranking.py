"""Tests for RankingStage."""

import pytest
import torch

from src.online.ranking import RankingStage


class TestRankingStage:
    """Test ranking stage: image-text similarity scoring."""

    def test_rank_returns_results(self, sample_category_embeddings,
                                   sample_category_names,
                                   sample_category_hierarchy, embedding_dim):
        """Ranking should return non-empty results."""
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=5,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query)
        assert len(results) > 0

    def test_rank_result_structure(self, sample_category_embeddings,
                                    sample_category_names,
                                    sample_category_hierarchy, embedding_dim):
        """Each ranking result should have required keys."""
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=5,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query)

        required_keys = {"category_leaf", "category_l2", "category_l1", "rank_score"}
        for result in results:
            assert required_keys.issubset(result.keys())

    def test_rank_respects_top_n(self, sample_category_embeddings,
                                  sample_category_names,
                                  sample_category_hierarchy, embedding_dim):
        """Should return at most top_n results."""
        top_n = 3
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=top_n,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query)
        assert len(results) <= top_n

    def test_rank_sorted_descending(self, sample_category_embeddings,
                                     sample_category_names,
                                     sample_category_hierarchy, embedding_dim):
        """Results should be sorted by rank_score descending."""
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=8,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query)

        scores = [r["rank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_with_candidate_filter(self, sample_category_embeddings,
                                         sample_category_names,
                                         sample_category_hierarchy, embedding_dim):
        """When candidate_leaves is provided, only those categories appear."""
        candidates = sample_category_names[:3]
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=10,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query, candidate_leaves=candidates)

        result_leaves = {r["category_leaf"] for r in results}
        assert result_leaves.issubset(set(candidates))

    def test_rank_with_empty_candidates(self, sample_category_embeddings,
                                         sample_category_names,
                                         sample_category_hierarchy, embedding_dim):
        """Empty candidate list should return empty results."""
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=10,
        )
        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.rank(query, candidate_leaves=[])
        assert results == []

    def test_rank_matching_embedding_scores_highest(
            self, sample_category_embeddings, sample_category_names,
            sample_category_hierarchy):
        """Query identical to a category embedding should rank that category first."""
        stage = RankingStage(
            sample_category_embeddings, sample_category_names,
            sample_category_hierarchy, top_n=8,
        )
        # Use the first category's embedding as query
        query = sample_category_embeddings[0]
        expected_leaf = sample_category_names[0]

        results = stage.rank(query)
        assert results[0]["category_leaf"] == expected_leaf
