"""
Offline Stage Entry Point.

Generates embeddings for anchor products and category descriptions,
then builds FAISS indexes for fast retrieval.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.blip2_encoder import BLIP2Encoder
from src.offline.embedding_generator import (
    generate_anchor_embeddings,
    generate_category_embeddings,
)
from src.offline.index_builder import build_faiss_index


def main():
    parser = argparse.ArgumentParser(description="Offline embedding generation")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to LoRA adapter checkpoint")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embedding_dir = PROJECT_ROOT / config["offline"]["embedding_dir"]
    index_dir = PROJECT_ROOT / config["offline"]["index_dir"]
    categories_file = PROJECT_ROOT / config["data"]["categories_file"]
    anchor_file = PROJECT_ROOT / config["data"]["anchor_file"]
    batch_size = config["offline"]["batch_size"]

    # Initialize encoder
    encoder = BLIP2Encoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        lora_adapter_path=args.lora_adapter,
    )

    # Step 1: Generate anchor product embeddings
    print("\n" + "=" * 60)
    print("Step 1: Generating anchor product embeddings")
    print("=" * 60)
    anchor_df = pd.read_csv(anchor_file)
    anchor_result = generate_anchor_embeddings(
        encoder, anchor_df, embedding_dir, batch_size=batch_size
    )

    # Step 2: Generate category text embeddings
    print("\n" + "=" * 60)
    print("Step 2: Generating category text embeddings")
    print("=" * 60)
    generate_category_embeddings(
        encoder, str(categories_file), embedding_dir, batch_size=batch_size
    )

    # Step 3: Build FAISS index
    print("\n" + "=" * 60)
    print("Step 3: Building FAISS index")
    print("=" * 60)
    build_faiss_index(
        anchor_result["embeddings"],
        index_dir / "anchor_index.faiss",
        index_type="flat",
    )

    print("\n" + "=" * 60)
    print("Offline stage complete!")
    print(f"  Embeddings: {embedding_dir}")
    print(f"  Indexes: {index_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
