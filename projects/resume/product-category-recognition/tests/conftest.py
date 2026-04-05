"""
Shared test fixtures for product category recognition tests.

Provides mock BLIP2 encoder and sample data for testing pipeline logic
without requiring GPU or model downloads.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension for tests."""
    return 32


@pytest.fixture
def sample_categories():
    """Minimal category taxonomy for testing."""
    return {
        "服装鞋帽": {
            "男装": ["T恤", "衬衫"],
            "女装": ["连衣裙", "半身裙"],
        },
        "数码电器": {
            "手机": ["智能手机", "功能手机"],
            "电脑": ["笔记本", "台式机"],
        },
    }


@pytest.fixture
def leaf_categories(sample_categories):
    """Flat list of (l1, l2, leaf) tuples."""
    result = []
    for l1, l2_dict in sample_categories.items():
        for l2, leaves in l2_dict.items():
            for leaf in leaves:
                result.append((l1, l2, leaf))
    return result


@pytest.fixture
def mock_encoder(embedding_dim):
    """Mock BLIP2Encoder that returns deterministic embeddings."""
    encoder = MagicMock()

    def fake_text_embedding(text):
        torch.manual_seed(hash(text) % (2**31))
        emb = torch.randn(embedding_dim)
        return torch.nn.functional.normalize(emb, dim=-1)

    def fake_image_embedding(image):
        torch.manual_seed(42)
        emb = torch.randn(embedding_dim)
        return torch.nn.functional.normalize(emb, dim=-1)

    def fake_multimodal_embedding(text, image):
        torch.manual_seed(hash(text) % (2**31))
        emb = torch.randn(embedding_dim)
        return torch.nn.functional.normalize(emb, dim=-1)

    def fake_text_embeddings_batch(texts, batch_size=32):
        embs = torch.stack([fake_text_embedding(t) for t in texts])
        return embs

    encoder.get_text_embedding.side_effect = fake_text_embedding
    encoder.get_image_embedding.side_effect = fake_image_embedding
    encoder.get_multimodal_embedding.side_effect = fake_multimodal_embedding
    encoder.get_text_embeddings_batch.side_effect = fake_text_embeddings_batch

    return encoder


@pytest.fixture
def sample_anchor_metadata(leaf_categories):
    """Generate anchor metadata for 4 products per leaf."""
    metadata = []
    for l1, l2, leaf in leaf_categories:
        for i in range(4):
            metadata.append({
                "product_id": f"{leaf}_{i:04d}",
                "category_leaf": leaf,
                "category_l2": l2,
                "category_l1": l1,
            })
    return metadata


@pytest.fixture
def sample_anchor_embeddings(sample_anchor_metadata, embedding_dim):
    """Generate deterministic anchor embeddings matching metadata."""
    n = len(sample_anchor_metadata)
    embeddings = []
    for meta in sample_anchor_metadata:
        torch.manual_seed(hash(meta["category_leaf"]) % (2**31))
        emb = torch.randn(embedding_dim)
        # Add small per-product noise
        emb = emb + torch.randn(embedding_dim) * 0.1
        embeddings.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.stack(embeddings)


@pytest.fixture
def sample_category_embeddings(leaf_categories, embedding_dim):
    """Generate category text embeddings."""
    embeddings = []
    for l1, l2, leaf in leaf_categories:
        torch.manual_seed(hash(leaf) % (2**31))
        emb = torch.randn(embedding_dim)
        embeddings.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.stack(embeddings)


@pytest.fixture
def sample_category_names(leaf_categories):
    """Ordered list of leaf category names."""
    return [leaf for _, _, leaf in leaf_categories]


@pytest.fixture
def sample_category_hierarchy(leaf_categories):
    """Leaf -> (l1, l2) mapping."""
    return {leaf: (l1, l2) for l1, l2, leaf in leaf_categories}
