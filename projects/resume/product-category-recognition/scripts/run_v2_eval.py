"""V2 evaluation: direct embedding + FAISS + metrics on 500-category subset."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import faiss
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.blip2_encoder import BLIP2Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/v2_eval.yaml")
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--tag", default="v1_baseline")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    image_dir = Path(config["data"]["image_dir"])

    print("Loading BLIP2...")
    encoder = BLIP2Encoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        lora_adapter_path=args.lora_adapter,
    )

    train_df = pd.read_csv(PROJECT_ROOT / config["data"]["anchor_file"])
    test_df = pd.read_csv(PROJECT_ROOT / config["data"]["test_file"])
    with open(PROJECT_ROOT / config["data"]["categories_file"]) as f:
        descriptions = json.load(f)

    # Fix paths
    for df in [train_df, test_df]:
        if not str(df["image_path"].iloc[0]).startswith("/"):
            df["image_path"] = df["name"].apply(lambda x: str(image_dir / x))

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Categories: {len(descriptions)}")

    # Step 1: Generate anchor image embeddings (image-only, like V1)
    print("\n=== Step 1: Anchor image embeddings ===")
    t0 = time.time()
    all_anchor_embs = []
    all_anchor_labels = []
    bs = 32
    for i in tqdm(range(0, len(train_df), bs)):
        batch = train_df.iloc[i:i+bs]
        images = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                images.append(img)
                all_anchor_labels.append(str(row["class"]))
            except Exception as e:
                continue
        if images:
            embs = encoder.get_image_embeddings_batch(images, batch_size=len(images))
            all_anchor_embs.append(embs)
    anchor_embs = torch.cat(all_anchor_embs, dim=0).numpy()
    print(f"Anchor embeddings: {anchor_embs.shape} in {time.time()-t0:.1f}s")

    # Step 2: Category text embeddings
    print("\n=== Step 2: Category text embeddings ===")
    t0 = time.time()
    cat_ids = sorted(descriptions.keys())
    cat_texts = [descriptions[c] for c in cat_ids]
    cat_embs = encoder.get_text_embeddings_batch(cat_texts, batch_size=32).numpy()
    print(f"Category embeddings: {cat_embs.shape} in {time.time()-t0:.1f}s")

    # Step 3: Build FAISS index
    print("\n=== Step 3: FAISS index ===")
    dim = anchor_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(anchor_embs)
    index.add(anchor_embs)
    print(f"FAISS index: {index.ntotal} vectors")

    # Step 4: Evaluate test set
    print(f"\n=== Step 4: Evaluate {len(test_df)} test samples ===")
    top_k_list = [1, 3, 5]
    correct = {f"{level}_top{k}": 0 for level in ["leaf", "l2", "l1"] for k in top_k_list}
    total = 0

    for i in tqdm(range(0, len(test_df), bs)):
        batch = test_df.iloc[i:i+bs]
        images = []
        labels = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                images.append(img)
                labels.append(str(row["class"]))
            except:
                continue
        if not images:
            continue

        # Image embeddings for test
        test_embs = encoder.get_image_embeddings_batch(images, batch_size=len(images)).numpy()
        faiss.normalize_L2(test_embs)

        # Recall: top 50 anchors
        D, I = index.search(test_embs, 50)

        # For each test sample
        for j in range(len(images)):
            true_label = labels[j]
            true_l2 = "L2_" + str(int(true_label) // 100)
            true_l1 = "L1_" + str(int(true_label) // 1000)

            # Recall stage: aggregate anchor labels
            recalled_labels = [all_anchor_labels[idx] for idx in I[j]]
            recalled_scores = {}
            for lbl, score in zip(recalled_labels, D[j]):
                recalled_scores[lbl] = recalled_scores.get(lbl, 0) + score

            # Ranking stage: image vs category text
            test_emb_tensor = torch.tensor(test_embs[j:j+1])
            cat_emb_tensor = torch.tensor(cat_embs)
            rank_scores = F.cosine_similarity(test_emb_tensor, cat_emb_tensor).numpy()
            rank_dict = {cat_ids[k]: float(rank_scores[k]) for k in range(len(cat_ids))}

            # Rerank: fuse recall + ranking
            all_candidates = set(recalled_scores.keys()) | set(rank_dict.keys())
            fused = {}
            for c in all_candidates:
                r_score = recalled_scores.get(c, 0)
                k_score = rank_dict.get(c, 0)
                # Normalize
                max_r = max(recalled_scores.values()) if recalled_scores else 1
                min_r = min(recalled_scores.values()) if recalled_scores else 0
                r_norm = (r_score - min_r) / (max_r - min_r + 1e-8)
                k_norm = (k_score + 1) / 2  # cosine [-1,1] -> [0,1]
                fused[c] = 0.45 * r_norm + 0.55 * k_norm

            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

            for k in top_k_list:
                top_preds = [p[0] for p in ranked[:k]]
                top_l2 = ["L2_" + str(int(p) // 100) for p in top_preds]
                top_l1 = ["L1_" + str(int(p) // 1000) for p in top_preds]
                if true_label in top_preds:
                    correct[f"leaf_top{k}"] += 1
                if true_l2 in top_l2:
                    correct[f"l2_top{k}"] += 1
                if true_l1 in top_l1:
                    correct[f"l1_top{k}"] += 1
            total += 1

    # Print results
    print(f"\n=== Results ({args.tag}, {total} samples) ===")
    metrics = {}
    for key, val in correct.items():
        acc = val / total if total > 0 else 0
        metrics[key] = round(acc * 100, 2)
        print(f"  {key}: {acc*100:.1f}%")

    results_dir = PROJECT_ROOT / config["evaluation"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {results_dir}/{args.tag}_metrics.json")


if __name__ == "__main__":
    main()
