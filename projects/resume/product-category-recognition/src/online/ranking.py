"""
Ranking Stage.

Uses image embedding vs category text embedding similarity
to rank candidate leaf categories.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RankingStage:
    """
    Ranking stage: score categories by image-text similarity.

    Computes cosine similarity between the query product's image embedding
    and each leaf category's text embedding.
    """

    def __init__(self, category_embeddings: torch.Tensor,
                 category_names: list[str],
                 category_hierarchy: dict,
                 top_n: int = 10):
        self.category_embeddings = category_embeddings  # [num_cats, D]
        self.category_names = category_names
        self.category_hierarchy = category_hierarchy
        self.top_n = top_n

        # Build name-to-index mapping
        self.name_to_idx = {name: i for i, name in enumerate(category_names)}

    def rank(self, image_embedding: torch.Tensor,
             candidate_leaves: list[str] | None = None) -> list[dict]:
        """
        Rank leaf categories by image-text similarity.

        Args:
            image_embedding: [D] normalized image embedding
            candidate_leaves: optional list of leaf names to restrict ranking to.
                            If None, ranks against all categories.

        Returns:
            List of {category_leaf, category_l2, category_l1, rank_score}
            sorted by score descending, limited to top_n.
        """
        if candidate_leaves is not None:
            indices = [self.name_to_idx[leaf] for leaf in candidate_leaves
                       if leaf in self.name_to_idx]
            if not indices:
                return []
            cat_embs = self.category_embeddings[indices]
            cat_names = [self.category_names[i] for i in indices]
        else:
            cat_embs = self.category_embeddings
            cat_names = self.category_names

        # Cosine similarity (vectors are already normalized)
        similarities = F.cosine_similarity(
            image_embedding.unsqueeze(0),  # [1, D]
            cat_embs,                       # [K, D]
        )  # [K]

        # Sort and take top_n
        top_k = min(self.top_n, len(cat_names))
        top_scores, top_indices = similarities.topk(top_k)

        results = []
        for score, idx in zip(top_scores, top_indices):
            leaf = cat_names[idx]
            hierarchy = self.category_hierarchy.get(leaf, ("unknown", "unknown"))
            results.append({
                "category_leaf": leaf,
                "category_l2": hierarchy[1] if isinstance(hierarchy, tuple) else hierarchy.get("l2", "unknown"),
                "category_l1": hierarchy[0] if isinstance(hierarchy, tuple) else hierarchy.get("l1", "unknown"),
                "rank_score": float(score),
            })

        return results
