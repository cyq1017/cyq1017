"""Tests for RecallStage."""

import faiss
import numpy as np
import pytest
import torch

from src.online.recall import RecallStage


class TestRecallStage:
    """Test recall stage: FAISS search + category aggregation."""

    def _build_index(self, embeddings: torch.Tensor) -> faiss.Index:
        """Helper to build a FAISS index from embeddings."""
        emb_np = embeddings.numpy().astype(np.float32)
        index = faiss.IndexFlatIP(emb_np.shape[1])
        index.add(emb_np)
        return index

    def test_recall_returns_results(self, sample_anchor_embeddings,
                                     sample_anchor_metadata, embedding_dim):
        """Recall should return non-empty results for a valid query."""
        index = self._build_index(sample_anchor_embeddings)
        stage = RecallStage(index, sample_anchor_metadata, top_n=10)

        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.recall(query)

        assert len(results) > 0

    def test_recall_result_structure(self, sample_anchor_embeddings,
                                      sample_anchor_metadata, embedding_dim):
        """Each recall result should have required keys."""
        index = self._build_index(sample_anchor_embeddings)
        stage = RecallStage(index, sample_anchor_metadata, top_n=10)

        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.recall(query)

        required_keys = {"category_leaf", "category_l2", "category_l1",
                         "recall_score", "num_anchors"}
        for result in results:
            assert required_keys.issubset(result.keys())

    def test_recall_sorted_descending(self, sample_anchor_embeddings,
                                       sample_anchor_metadata, embedding_dim):
        """Results should be sorted by recall_score descending."""
        index = self._build_index(sample_anchor_embeddings)
        stage = RecallStage(index, sample_anchor_metadata, top_n=20)

        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.recall(query)

        scores = [r["recall_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_aggregates_by_leaf(self, sample_anchor_embeddings,
                                        sample_anchor_metadata, embedding_dim):
        """Each leaf category should appear at most once."""
        index = self._build_index(sample_anchor_embeddings)
        stage = RecallStage(index, sample_anchor_metadata, top_n=50)

        query = torch.nn.functional.normalize(torch.randn(embedding_dim), dim=-1)
        results = stage.recall(query)

        leaves = [r["category_leaf"] for r in results]
        assert len(leaves) == len(set(leaves)), "Duplicate leaf categories in recall results"

    def test_recall_similar_query_returns_correct_category(
            self, sample_anchor_embeddings, sample_anchor_metadata, embedding_dim):
        """Query similar to a category's anchors should recall that category first."""
        index = self._build_index(sample_anchor_embeddings)
        stage = RecallStage(index, sample_anchor_metadata, top_n=50)

        # Use the first anchor's embedding as query (should match its own category)
        query = sample_anchor_embeddings[0]
        expected_leaf = sample_anchor_metadata[0]["category_leaf"]

        results = stage.recall(query)
        assert results[0]["category_leaf"] == expected_leaf
