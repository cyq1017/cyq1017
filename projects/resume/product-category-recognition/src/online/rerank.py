"""
Rerank Stage.

Fuses recall scores and ranking scores to produce the final category prediction.
"""

from __future__ import annotations


class RerankStage:
    """
    Rerank stage: fuse recall and ranking scores.

    Combines scores from recall (multimodal similarity) and ranking
    (image-text similarity) stages using weighted fusion.
    """

    def __init__(self, recall_weight: float = 0.45, rank_weight: float = 0.55):
        self.recall_weight = recall_weight
        self.rank_weight = rank_weight

    def rerank(self, recall_results: list[dict],
               rank_results: list[dict]) -> list[dict]:
        """
        Fuse recall and ranking scores to produce final predictions.

        Args:
            recall_results: from RecallStage, each has "recall_score"
            rank_results: from RankingStage, each has "rank_score"

        Returns:
            List of {category_leaf, category_l2, category_l1, final_score,
                     recall_score, rank_score} sorted by final_score descending.
        """
        # Build score maps
        recall_map = {r["category_leaf"]: r for r in recall_results}
        rank_map = {r["category_leaf"]: r for r in rank_results}

        # Get union of all candidate categories
        all_leaves = set(recall_map.keys()) | set(rank_map.keys())

        # Normalize scores to [0, 1] range
        recall_scores = [r["recall_score"] for r in recall_results] if recall_results else [0]
        rank_scores = [r["rank_score"] for r in rank_results] if rank_results else [0]

        recall_min, recall_max = min(recall_scores), max(recall_scores)
        rank_min, rank_max = min(rank_scores), max(rank_scores)

        recall_range = recall_max - recall_min if recall_max != recall_min else 1.0
        rank_range = rank_max - rank_min if rank_max != rank_min else 1.0

        results = []
        for leaf in all_leaves:
            recall_info = recall_map.get(leaf, {})
            rank_info = rank_map.get(leaf, {})

            raw_recall = recall_info.get("recall_score", recall_min)
            raw_rank = rank_info.get("rank_score", rank_min)

            # Min-max normalize
            norm_recall = (raw_recall - recall_min) / recall_range
            norm_rank = (raw_rank - rank_min) / rank_range

            final_score = self.recall_weight * norm_recall + self.rank_weight * norm_rank

            # Get hierarchy info from whichever source has it
            info = recall_info or rank_info
            results.append({
                "category_leaf": leaf,
                "category_l2": info.get("category_l2", "unknown"),
                "category_l1": info.get("category_l1", "unknown"),
                "final_score": final_score,
                "recall_score": raw_recall,
                "rank_score": raw_rank,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
