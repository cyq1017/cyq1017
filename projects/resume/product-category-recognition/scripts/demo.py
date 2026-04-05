"""
Demo Script.

Picks random test products and shows prediction results.
"""

import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.blip2_encoder import BLIP2Encoder
from src.online.pipeline import ProductClassificationPipeline


def main():
    random.seed(42)

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize
    encoder = BLIP2Encoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
    )
    pipeline = ProductClassificationPipeline.from_config(config, encoder=encoder)

    # Pick random test products
    test_file = PROJECT_ROOT / config["data"]["test_file"]
    test_df = pd.read_csv(test_file)
    sample_df = test_df.sample(n=min(10, len(test_df)), random_state=42)

    correct = {"leaf": 0, "l2": 0, "l1": 0}
    total = len(sample_df)

    print("=" * 70)
    print("DEMO: Product Category Recognition")
    print("=" * 70)

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        result = pipeline.predict(row["title"], row["description"], image)

        leaf_ok = result["predicted_leaf"] == row["category_leaf"]
        l2_ok = result["predicted_l2"] == row["category_l2"]
        l1_ok = result["predicted_l1"] == row["category_l1"]

        correct["leaf"] += int(leaf_ok)
        correct["l2"] += int(l2_ok)
        correct["l1"] += int(l1_ok)

        status = "OK" if leaf_ok else "MISS"
        print(f"\n[{i+1}/{total}] {status}")
        print(f"  Title: {row['title']}")
        print(f"  True:  {row['category_l1']}/{row['category_l2']}/{row['category_leaf']}")
        print(f"  Pred:  {result['predicted_l1']}/{result['predicted_l2']}/{result['predicted_leaf']}")
        print(f"  Score: {result['confidence']:.4f}")

    print("\n" + "=" * 70)
    print("Demo Summary:")
    print(f"  Leaf accuracy: {correct['leaf']}/{total} ({correct['leaf']/total:.1%})")
    print(f"  L2 accuracy:   {correct['l2']}/{total} ({correct['l2']/total:.1%})")
    print(f"  L1 accuracy:   {correct['l1']}/{total} ({correct['l1']/total:.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
