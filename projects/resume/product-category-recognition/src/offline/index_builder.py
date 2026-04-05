"""
FAISS Index Builder.

Builds ANN indexes for fast similarity search during the online recall stage.
"""

from pathlib import Path

import faiss
import numpy as np
import torch


def build_faiss_index(embeddings: torch.Tensor, output_path: Path,
                      index_type: str = "flat") -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings: [N, D] tensor of normalized embeddings
        output_path: path to save the index
        index_type: "flat" for exact search, "ivf" for approximate

    Returns:
        FAISS index
    """
    embeddings_np = embeddings.numpy().astype(np.float32)
    d = embeddings_np.shape[1]

    if index_type == "ivf":
        nlist = min(100, embeddings_np.shape[0] // 10)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings_np)
    else:
        # Flat index with inner product (cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(d)

    index.add(embeddings_np)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"FAISS index saved: {output_path} ({index.ntotal} vectors, dim={d})")
    return index


def load_faiss_index(index_path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    index = faiss.read_index(str(index_path))
    print(f"FAISS index loaded: {index_path} ({index.ntotal} vectors)")
    return index
