# tests/test_prepare_v2_data.py
"""Tests for V2 data preparation."""
from __future__ import annotations
import pandas as pd
import pytest


def _make_fake_train_csv(n_skus=1000, imgs_per_sku=15):
    """Create a fake train.csv matching Products-10K format."""
    rows = []
    for sku in range(n_skus):
        for i in range(imgs_per_sku):
            rows.append({"name": f"{sku}_{i}.jpg", "class": sku, "group": sku % 10})
    return pd.DataFrame(rows)


class TestSelectTopCategories:

    def test_selects_correct_count(self):
        from scripts.prepare_v2_data import select_top_categories
        df = _make_fake_train_csv(n_skus=1000, imgs_per_sku=15)
        result = select_top_categories(df, top_n=500)
        assert result["class"].nunique() == 500

    def test_selects_highest_frequency(self):
        from scripts.prepare_v2_data import select_top_categories
        rows = []
        for sku in range(100):
            count = 50 if sku < 10 else 5
            for i in range(count):
                rows.append({"name": f"{sku}_{i}.jpg", "class": sku, "group": 0})
        df = pd.DataFrame(rows)
        result = select_top_categories(df, top_n=10)
        assert set(result["class"].unique()) == set(range(10))

    def test_respects_max_products(self):
        from scripts.prepare_v2_data import select_top_categories
        df = _make_fake_train_csv(n_skus=100, imgs_per_sku=100)
        result = select_top_categories(df, top_n=50, max_products=2000)
        assert len(result) <= 2000
