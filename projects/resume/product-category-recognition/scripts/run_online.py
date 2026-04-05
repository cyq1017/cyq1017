"""
Online Inference Entry Point.

Runs the three-stage pipeline on a single product or batch of products.
"""

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.blip2_encoder import BLIP2Encoder
from src.online.pipeline import ProductClassificationPipeline


def main():
    parser = argparse.ArgumentParser(description="Online product classification")
    parser.add_argument("--title", type=str, required=True, help="Product title")
    parser.add_argument("--description", type=str, default="", help="Product description")
    parser.add_argument("--image", type=str, required=True, help="Path to product image")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize
    encoder = BLIP2Encoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
    )
    pipeline = ProductClassificationPipeline.from_config(config, encoder=encoder)

    # Predict
    image = Image.open(args.image).convert("RGB")
    result = pipeline.predict(args.title, args.description, image)

    # Output
    print("\n" + "=" * 50)
    print("Product Classification Result")
    print("=" * 50)
    print(f"  Input: {args.title}")
    print(f"  Predicted L1: {result['predicted_l1']}")
    print(f"  Predicted L2: {result['predicted_l2']}")
    print(f"  Predicted Leaf: {result['predicted_leaf']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print("\n  Top 5 Candidates:")
    for i, c in enumerate(result["top_candidates"][:5]):
        print(f"    {i+1}. {c['category_l1']}/{c['category_l2']}/{c['category_leaf']} "
              f"(score={c['final_score']:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
