# scripts/prepare_v2_data.py
"""Select top categories from Products-10K and build V2 train/test split."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def select_top_categories(df: pd.DataFrame, top_n: int = 500,
                          max_products: int | None = None) -> pd.DataFrame:
    """Select top_n most frequent leaf categories."""
    sku_counts = df["class"].value_counts()
    top_skus = sku_counts.head(top_n).index.tolist()
    result = df[df["class"].isin(top_skus)].copy()

    if max_products is not None and len(result) > max_products:
        result = result.sample(n=max_products, random_state=42).reset_index(drop=True)

    return result


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by class."""
    train_dfs, test_dfs = [], []
    for cls, group in df.groupby("class"):
        n_train = max(1, int(len(group) * train_ratio))
        shuffled = group.sample(frac=1, random_state=seed)
        train_dfs.append(shuffled.iloc[:n_train])
        if n_train < len(group):
            test_dfs.append(shuffled.iloc[n_train:])
    train = pd.concat(train_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--max-products", type=int, default=20000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_csv = dataset_dir / "train.csv"
    train_img_dir = dataset_dir / "train"
    if (train_img_dir / "train").exists():
        train_img_dir = train_img_dir / "train"

    df = pd.read_csv(train_csv)
    subset = select_top_categories(df, top_n=args.top_n, max_products=args.max_products)
    subset["image_path"] = subset["name"].apply(lambda x: str(train_img_dir / x))
    subset["category_leaf"] = subset["class"].astype(str)

    train, test = split_train_test(subset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "v2_train.csv", index=False)
    test.to_csv(output_dir / "v2_test.csv", index=False)

    print(f"V2 data: {len(train)} train, {len(test)} test, "
          f"{subset['class'].nunique()} categories")


if __name__ == "__main__":
    main()
