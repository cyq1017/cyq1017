"""
Evaluation Entry Point.

Runs the full online pipeline on test products and computes metrics.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.blip2_encoder import BLIP2Encoder
from src.online.pipeline import ProductClassificationPipeline
from src.evaluation.metrics import (
    compute_topk_accuracy,
    save_evaluation_results,
)


def main():
    # Load config
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_file = PROJECT_ROOT / config["data"]["test_file"]
    output_dir = PROJECT_ROOT / config["evaluation"]["output_dir"]
    k_values = config["evaluation"]["top_k"]

    # Initialize encoder and pipeline
    encoder = BLIP2Encoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
    )
    pipeline = ProductClassificationPipeline.from_config(config, encoder=encoder)

    # Load test data
    test_df = pd.read_csv(test_file)
    print(f"Test products: {len(test_df)}")

    # Run predictions
    predictions = []
    ground_truth = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        image = Image.open(row["image_path"]).convert("RGB")
        result = pipeline.predict(row["title"], row["description"], image)
        predictions.append(result)
        ground_truth.append({
            "category_leaf": row["category_leaf"],
            "category_l2": row["category_l2"],
            "category_l1": row["category_l1"],
        })

    # Compute metrics
    metrics = compute_topk_accuracy(predictions, ground_truth, k_values=k_values)

    # Save results
    save_evaluation_results(metrics, output_dir, predictions, ground_truth)


if __name__ == "__main__":
    main()
