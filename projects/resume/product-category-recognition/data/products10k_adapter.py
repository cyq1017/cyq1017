"""
Products-10K Dataset Adapter.

Converts the Products-10K dataset (Kaggle: warcoder/visual-product-recognition)
into the format expected by our pipeline (anchor/test split with category labels).

Products-10K has ~141,931 train images across 9,691 SKU classes.
The dataset only has flat SKU IDs — we construct a pseudo-hierarchy by grouping
SKUs with similar visual embeddings (or use a simple numeric partition as baseline).
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_products10k(dataset_dir: str) -> pd.DataFrame:
    """
    Load Products-10K dataset from extracted directory.

    Args:
        dataset_dir: path to extracted products-10k/ directory

    Returns:
        DataFrame with columns: product_id, image_path, category_leaf, category_l2, category_l1
    """
    dataset_dir = Path(dataset_dir)

    # Try common extraction structures
    train_csv = None
    for candidate in [
        dataset_dir / "train.csv",
        dataset_dir / "products-10k" / "train.csv",
    ]:
        if candidate.exists():
            train_csv = candidate
            break

    if train_csv is None:
        raise FileNotFoundError(
            f"train.csv not found in {dataset_dir}. "
            "Expected structure: products-10k/train.csv"
        )

    # Images may be in train/ or train/train/ (nested extraction)
    train_dir = train_csv.parent / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"train/ directory not found at {train_dir}")
    # Check for nested structure: train/train/*.jpg
    nested_dir = train_dir / "train"
    if nested_dir.exists() and any(nested_dir.iterdir()):
        train_dir = nested_dir

    df = pd.read_csv(train_csv)
    print(f"Loaded train.csv: {len(df)} rows, columns: {list(df.columns)}")

    # Columns: 'name' (filename), 'class' (numeric SKU ID), optionally 'group'
    df = df.rename(columns={"name": "filename", "class": "sku_id"})

    # Build image paths
    df["image_path"] = df["filename"].apply(lambda x: str(train_dir / x))

    # Filter out missing images
    exists_mask = df["image_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} images not found, filtering out")
        df = df[exists_mask].reset_index(drop=True)

    # Build pseudo-hierarchy from SKU IDs
    # Group SKUs into L2 (100 groups) and L1 (10 groups) by numeric ranges
    df = _build_hierarchy(df)

    # Product ID
    df["product_id"] = df.index.astype(str)

    # Rename for pipeline compatibility
    df = df.rename(columns={"sku_id": "category_leaf_id"})
    df["category_leaf"] = df["category_leaf_id"].astype(str)

    print(f"Dataset ready: {len(df)} products")
    print(f"  Leaf categories (SKU): {df['category_leaf'].nunique()}")
    print(f"  L2 categories: {df['category_l2'].nunique()}")
    print(f"  L1 categories: {df['category_l1'].nunique()}")

    return df[["product_id", "image_path", "category_leaf", "category_l2",
               "category_l1", "filename"]]


def _build_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pseudo-hierarchy by grouping SKU IDs.

    Strategy: sort unique SKUs by frequency, split into N_L2 groups of ~equal size,
    then group L2s into N_L1 super-groups.
    """
    n_l1 = 10
    n_l2 = 100

    # Get SKU frequencies and sort
    sku_counts = df["sku_id"].value_counts()
    sorted_skus = sku_counts.index.tolist()

    # Assign each SKU to an L2 group (round-robin for balance)
    sku_to_l2 = {}
    for i, sku in enumerate(sorted_skus):
        l2_id = i % n_l2
        sku_to_l2[sku] = f"L2_{l2_id:03d}"

    # Assign each L2 to an L1 group
    l2_to_l1 = {}
    for l2_id in range(n_l2):
        l1_id = l2_id % n_l1
        l2_name = f"L2_{l2_id:03d}"
        l2_to_l1[l2_name] = f"L1_{l1_id:02d}"

    df["category_l2"] = df["sku_id"].map(sku_to_l2)
    df["category_l1"] = df["category_l2"].map(l2_to_l1)

    return df


def split_anchor_test(df: pd.DataFrame, anchor_ratio: float = 0.7,
                      seed: int = 42, max_products: int | None = None) -> tuple:
    """
    Split products into anchor and test sets, stratified by leaf category.

    Args:
        df: full product DataFrame
        anchor_ratio: fraction for anchor set
        seed: random seed
        max_products: optional limit on total products (for faster dev iteration)

    Returns:
        (anchor_df, test_df)
    """
    random.seed(seed)

    if max_products is not None and len(df) > max_products:
        # Simple random sample maintaining approximate category distribution
        df = df.sample(n=max_products, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} products")

    anchor_dfs = []
    test_dfs = []

    for leaf, group in df.groupby("category_leaf"):
        if len(group) < 2:
            # Single-product categories go to anchor only
            anchor_dfs.append(group)
            continue

        n_anchor = max(1, int(len(group) * anchor_ratio))
        shuffled = group.sample(frac=1, random_state=seed)
        anchor_dfs.append(shuffled.iloc[:n_anchor])
        test_dfs.append(shuffled.iloc[n_anchor:])

    anchor_df = pd.concat(anchor_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    print(f"Split: {len(anchor_df)} anchor, {len(test_df)} test "
          f"({len(anchor_df)/(len(anchor_df)+len(test_df)):.1%}/{len(test_df)/(len(anchor_df)+len(test_df)):.1%})")

    return anchor_df, test_df


def build_categories_json(df: pd.DataFrame, output_path: Path):
    """Build categories.json hierarchy from the DataFrame."""
    taxonomy = defaultdict(lambda: defaultdict(list))

    for _, row in df[["category_l1", "category_l2", "category_leaf"]].drop_duplicates().iterrows():
        leaf = row["category_leaf"]
        l2 = row["category_l2"]
        l1 = row["category_l1"]
        if leaf not in taxonomy[l1][l2]:
            taxonomy[l1][l2].append(leaf)

    # Convert to regular dict
    taxonomy_dict = {l1: dict(l2_dict) for l1, l2_dict in taxonomy.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_dict, f, ensure_ascii=False, indent=2)

    print(f"Categories saved: {output_path}")
    print(f"  L1: {len(taxonomy_dict)}, L2: {sum(len(v) for v in taxonomy_dict.values())}, "
          f"Leaf: {sum(len(leaves) for l2 in taxonomy_dict.values() for leaves in l2.values())}")


def main():
    """Convert Products-10K to pipeline format."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to extracted Products-10K directory")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data"),
                        help="Output directory for converted data")
    parser.add_argument("--anchor-ratio", type=float, default=0.7)
    parser.add_argument("--max-products", type=int, default=None,
                        help="Limit total products (for dev iteration)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and convert
    df = load_products10k(args.dataset_dir)

    # Build categories
    build_categories_json(df, output_dir / "categories.json")

    # Split
    anchor_df, test_df = split_anchor_test(
        df, anchor_ratio=args.anchor_ratio, max_products=args.max_products
    )

    # Save CSVs (adding title/description columns for pipeline compatibility)
    for split_df, name in [(anchor_df, "anchor_products"), (test_df, "test_products"), (df, "products")]:
        # Generate minimal text from category info
        split_df = split_df.copy()
        split_df["title"] = split_df["category_leaf"]
        split_df["description"] = split_df.apply(
            lambda r: f"{r['category_l1']} {r['category_l2']} {r['category_leaf']}", axis=1
        )
        csv_path = output_dir / f"{name}.csv"
        split_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {csv_path} ({len(split_df)} rows)")

    print("\nDone! Ready to run: make offline && make evaluate")


if __name__ == "__main__":
    main()
