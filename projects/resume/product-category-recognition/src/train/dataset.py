# src/train/dataset.py
"""ITC training dataset for LoRA fine-tuning."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ITCDataset(Dataset):
    """Dataset returning (image, category_text, label) for ITC training."""

    def __init__(self, csv_path: str | Path, descriptions_path: str | Path,
                 image_size: int = 224):
        self.df = pd.read_csv(csv_path)
        with open(descriptions_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["class"])
        cat_key = str(row.get("category_leaf", label))
        text = self.descriptions.get(cat_key, cat_key)
        return image, text, label
