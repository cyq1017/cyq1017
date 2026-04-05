"""
Recall Stage.

Uses multimodal embedding similarity to recall TopN anchor products,
then aggregates by leaf category with weighted similarity scores.
"""

from __future__ import annotations

from collections import defaultdict

import faiss
import numpy as np
import torch


class RecallStage:
    """
    Recall stage: find similar anchor products and aggregate by category.

    1. Query FAISS index with new product's multimodal embedding
    2. Get TopN most similar anchor products
    3. Aggregate scores by leaf category (weighted average)
    4. Return candidate categories with scores
    """

    def __init__(self, anchor_index: faiss.Index, anchor_metadata: list[dict],
                 top_n: int = 50):
        self.anchor_index = anchor_index
        self.anchor_metadata = anchor_metadata
        self.top_n = top_n

    def recall(self, query_embedding: torch.Tensor) -> list[dict]:
        """
        Recall candidate categories for a query product.

        Args:
            query_embedding: [D] normalized multimodal embedding

        Returns:
            List of {category_leaf, category_l2, category_l1, score}
            sorted by score descending.
        """
        query_np = query_embedding.unsqueeze(0).numpy().astype(np.float32)
        scores, indices = self.anchor_index.search(query_np, self.top_n)

        scores = scores[0]   # [top_n]
        indices = indices[0]  # [top_n]

        # Aggregate by leaf category
        category_scores = defaultdict(list)
        category_info = {}

        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            meta = self.anchor_metadata[idx]
            leaf = meta["category_leaf"]
            category_scores[leaf].append(float(score))
            category_info[leaf] = {
                "category_l1": meta["category_l1"],
                "category_l2": meta["category_l2"],
            }

        # Weighted average: products with higher similarity contribute more
        results = []
        for leaf, leaf_scores in category_scores.items():
            weights = np.array(leaf_scores)
            # Use softmax-like weighting
            weighted_score = float(np.average(leaf_scores, weights=np.abs(weights)))
            info = category_info[leaf]
            results.append({
                "category_leaf": leaf,
                "category_l2": info["category_l2"],
                "category_l1": info["category_l1"],
                "recall_score": weighted_score,
                "num_anchors": len(leaf_scores),
            })

        results.sort(key=lambda x: x["recall_score"], reverse=True)
        return results
