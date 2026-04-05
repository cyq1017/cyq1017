"""
Online Three-Stage Pipeline.

Assembles Recall → Ranking → Rerank into a unified classification pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.model.blip2_encoder import BLIP2Encoder
from src.offline.index_builder import load_faiss_index
from src.online.recall import RecallStage
from src.online.ranking import RankingStage
from src.online.rerank import RerankStage


class ProductClassificationPipeline:
    """
    End-to-end product category prediction pipeline.

    Usage:
        pipeline = ProductClassificationPipeline.from_config(config)
        result = pipeline.predict(title, description, image)
    """

    def __init__(self, encoder: BLIP2Encoder, recall_stage: RecallStage,
                 ranking_stage: RankingStage, rerank_stage: RerankStage):
        self.encoder = encoder
        self.recall_stage = recall_stage
        self.ranking_stage = ranking_stage
        self.rerank_stage = rerank_stage

    @classmethod
    def from_config(cls, config: dict, encoder: BLIP2Encoder | None = None):
        """Build pipeline from config dict and pre-computed embeddings."""
        embedding_dir = Path(config["offline"]["embedding_dir"])
        index_dir = Path(config["offline"]["index_dir"])

        # Load encoder if not provided
        if encoder is None:
            encoder = BLIP2Encoder(
                model_name=config["model"]["name"],
                device=config["model"]["device"],
                dtype=config["model"]["dtype"],
            )

        # Load anchor index and metadata
        anchor_index = load_faiss_index(index_dir / "anchor_index.faiss")
        with open(embedding_dir / "anchor_metadata.json", "r", encoding="utf-8") as f:
            anchor_metadata = json.load(f)

        # Load category embeddings and names
        category_embeddings = torch.load(
            embedding_dir / "category_embeddings.pt", weights_only=True
        )
        with open(embedding_dir / "category_names.json", "r", encoding="utf-8") as f:
            category_names = json.load(f)
        with open(embedding_dir / "category_hierarchy.json", "r", encoding="utf-8") as f:
            category_hierarchy = json.load(f)

        # Convert hierarchy format for RankingStage
        hierarchy_tuples = {
            leaf: (info["l1"], info["l2"])
            for leaf, info in category_hierarchy.items()
        }

        # Build stages
        recall_cfg = config["online"]["recall"]
        rank_cfg = config["online"]["ranking"]
        rerank_cfg = config["online"]["rerank"]

        recall_stage = RecallStage(
            anchor_index=anchor_index,
            anchor_metadata=anchor_metadata,
            top_n=recall_cfg["top_n"],
        )
        ranking_stage = RankingStage(
            category_embeddings=category_embeddings,
            category_names=category_names,
            category_hierarchy=hierarchy_tuples,
            top_n=rank_cfg["top_n"],
        )
        rerank_stage = RerankStage(
            recall_weight=rerank_cfg["recall_weight"],
            rank_weight=rerank_cfg["rank_weight"],
        )

        return cls(encoder, recall_stage, ranking_stage, rerank_stage)

    def predict(self, title: str, description: str,
                image: Image.Image) -> dict:
        """
        Predict the category for a single product.

        Args:
            title: product title
            description: product description
            image: product image (PIL Image)

        Returns:
            dict with predicted categories and scores at all levels.
        """
        text = f"{title} {description}"

        # Extract embeddings
        multimodal_emb = self.encoder.get_multimodal_embedding(text, image)
        image_emb = self.encoder.get_image_embedding(image)

        # Stage 1: Recall
        recall_results = self.recall_stage.recall(multimodal_emb)

        # Stage 2: Ranking - restrict to recalled categories
        recalled_leaves = [r["category_leaf"] for r in recall_results]
        rank_results = self.ranking_stage.rank(image_emb, candidate_leaves=recalled_leaves)

        # Stage 3: Rerank
        final_results = self.rerank_stage.rerank(recall_results, rank_results)

        if not final_results:
            return {
                "predicted_leaf": "unknown",
                "predicted_l2": "unknown",
                "predicted_l1": "unknown",
                "confidence": 0.0,
                "top_candidates": [],
            }

        top = final_results[0]
        return {
            "predicted_leaf": top["category_leaf"],
            "predicted_l2": top["category_l2"],
            "predicted_l1": top["category_l1"],
            "confidence": top["final_score"],
            "top_candidates": final_results[:5],
        }

    def predict_batch(self, products: list[dict],
                      batch_size: int = 16) -> list[dict]:
        """
        Predict categories for a batch of products.

        Each product dict should have: title, description, image (PIL Image)
        """
        # Pre-compute embeddings in batch
        texts = [f"{p['title']} {p['description']}" for p in products]
        images = [p["image"] for p in products]

        multimodal_embs = self.encoder.get_multimodal_embeddings_batch(
            texts, images, batch_size=batch_size
        )
        image_embs = self.encoder.get_image_embeddings_batch(
            images, batch_size=batch_size
        )

        # Run pipeline per product
        results = []
        for i in range(len(products)):
            recall_results = self.recall_stage.recall(multimodal_embs[i])
            recalled_leaves = [r["category_leaf"] for r in recall_results]
            rank_results = self.ranking_stage.rank(image_embs[i],
                                                    candidate_leaves=recalled_leaves)
            final_results = self.rerank_stage.rerank(recall_results, rank_results)

            if final_results:
                top = final_results[0]
                results.append({
                    "predicted_leaf": top["category_leaf"],
                    "predicted_l2": top["category_l2"],
                    "predicted_l1": top["category_l1"],
                    "confidence": top["final_score"],
                    "top_candidates": final_results[:5],
                })
            else:
                results.append({
                    "predicted_leaf": "unknown",
                    "predicted_l2": "unknown",
                    "predicted_l1": "unknown",
                    "confidence": 0.0,
                    "top_candidates": [],
                })

        return results
