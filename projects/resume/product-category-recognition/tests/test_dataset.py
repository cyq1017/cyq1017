# tests/test_dataset.py
"""Tests for ITC training dataset."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock


class TestITCDataset:

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal dataset files."""
        import pandas as pd
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        rows = []
        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            path = img_dir / f"{i}.jpg"
            img.save(str(path))
            rows.append({
                "name": f"{i}.jpg", "class": i % 5,
                "image_path": str(path), "category_leaf": str(i % 5),
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "train.csv"
        df.to_csv(csv_path, index=False)

        descs = {str(i): f"Category {i} description" for i in range(5)}
        import json
        desc_path = tmp_path / "descriptions.json"
        with open(desc_path, "w") as f:
            json.dump(descs, f)

        return csv_path, desc_path

    def test_dataset_length(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        assert len(ds) == 20

    def test_dataset_returns_image_and_text(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        image, text, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(text, str)
        assert isinstance(label, int)

    def test_dataset_text_comes_from_descriptions(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        _, text, label = ds[0]
        assert f"Category {label}" in text
