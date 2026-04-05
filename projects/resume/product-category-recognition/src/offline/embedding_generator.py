"""
Offline Embedding Generator.

Generates and stores embeddings for:
1. Anchor products (multimodal: text + image)
2. Category descriptions (text only)
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.model.blip2_encoder import BLIP2Encoder


# Category description templates for richer text embeddings
CATEGORY_TEMPLATES = [
    "{leaf}",
    "{l1} {l2} {leaf}",
    "{leaf}，属于{l2}类，{l1}大类",
]


def build_category_texts(categories_file: str) -> dict:
    """
    Build enriched text descriptions for each leaf category.

    Returns:
        dict mapping leaf_name -> list of text descriptions
    """
    with open(categories_file, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    category_texts = {}
    category_hierarchy = {}  # leaf -> (l1, l2)

    for l1, l2_dict in taxonomy.items():
        for l2, leaves in l2_dict.items():
            for leaf in leaves:
                texts = []
                for template in CATEGORY_TEMPLATES:
                    texts.append(template.format(l1=l1, l2=l2, leaf=leaf))
                category_texts[leaf] = texts
                category_hierarchy[leaf] = (l1, l2)

    return category_texts, category_hierarchy


def generate_anchor_embeddings(
    encoder: BLIP2Encoder,
    anchor_df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 16,
) -> dict:
    """
    Generate multimodal embeddings for all anchor products.

    Stores:
      - anchor_embeddings.pt: tensor [N, D]
      - anchor_metadata.json: list of {product_id, category_leaf, category_l2, category_l1}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    images = []
    metadata = []

    print("Loading anchor product data...")
    for _, row in tqdm(anchor_df.iterrows(), total=len(anchor_df), desc="Loading"):
        text = f"{row['title']} {row['description']}"
        texts.append(text)
        img = Image.open(row["image_path"]).convert("RGB")
        images.append(img)
        metadata.append({
            "product_id": row["product_id"],
            "category_leaf": row["category_leaf"],
            "category_l2": row["category_l2"],
            "category_l1": row["category_l1"],
        })

    print(f"Generating multimodal embeddings for {len(texts)} anchor products...")
    embeddings = encoder.get_multimodal_embeddings_batch(texts, images, batch_size=batch_size)

    # Save
    torch.save(embeddings, output_dir / "anchor_embeddings.pt")
    with open(output_dir / "anchor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Anchor embeddings saved: shape={embeddings.shape}")
    return {"embeddings": embeddings, "metadata": metadata}


def generate_category_embeddings(
    encoder: BLIP2Encoder,
    categories_file: str,
    output_dir: Path,
    batch_size: int = 32,
) -> dict:
    """
    Generate text embeddings for all leaf category descriptions.

    For each leaf category, generates embeddings for multiple text descriptions
    and averages them into a single representative embedding.

    Stores:
      - category_embeddings.pt: tensor [num_leaves, D]
      - category_names.json: ordered list of leaf category names
      - category_hierarchy.json: leaf -> {l1, l2}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    category_texts, category_hierarchy = build_category_texts(categories_file)
    leaf_names = sorted(category_texts.keys())

    print(f"Generating text embeddings for {len(leaf_names)} leaf categories...")

    # Flatten all texts for batch processing
    all_texts = []
    text_to_leaf_idx = []
    for i, leaf in enumerate(leaf_names):
        for text in category_texts[leaf]:
            all_texts.append(text)
            text_to_leaf_idx.append(i)

    # Batch encode all texts
    all_embeddings = encoder.get_text_embeddings_batch(all_texts, batch_size=batch_size)

    # Average embeddings per leaf category
    category_embeddings = torch.zeros(len(leaf_names), all_embeddings.shape[1])
    counts = torch.zeros(len(leaf_names))
    for idx, leaf_idx in enumerate(text_to_leaf_idx):
        category_embeddings[leaf_idx] += all_embeddings[idx]
        counts[leaf_idx] += 1
    category_embeddings = category_embeddings / counts.unsqueeze(1)
    category_embeddings = torch.nn.functional.normalize(category_embeddings, dim=-1)

    # Save
    torch.save(category_embeddings, output_dir / "category_embeddings.pt")
    with open(output_dir / "category_names.json", "w", encoding="utf-8") as f:
        json.dump(leaf_names, f, ensure_ascii=False, indent=2)

    hierarchy_for_save = {leaf: {"l1": h[0], "l2": h[1]} for leaf, h in category_hierarchy.items()}
    with open(output_dir / "category_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy_for_save, f, ensure_ascii=False, indent=2)

    print(f"Category embeddings saved: shape={category_embeddings.shape}")
    return {
        "embeddings": category_embeddings,
        "names": leaf_names,
        "hierarchy": category_hierarchy,
    }
