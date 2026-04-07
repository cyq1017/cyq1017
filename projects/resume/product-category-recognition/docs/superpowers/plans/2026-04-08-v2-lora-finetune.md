# V2 LoRA Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LoRA fine-tune BLIP2 Q-Former on Products-10K top 500 categories with ITC loss, improving product category recognition accuracy over V1 zero-shot baseline.

**Architecture:** Select top 500 categories (~20K products) from Products-10K, generate rich category descriptions via LLM, train Q-Former LoRA adapters with ITC loss (image vs text contrastive), then evaluate through the existing three-stage pipeline (recall → ranking → rerank).

**Tech Stack:** PyTorch, Transformers 5.x, PEFT (LoRA), FAISS, MiniMax API (OpenAI-compatible)

**Spec:** `docs/superpowers/specs/2026-04-08-v2-lora-finetune-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|----------------|
| `scripts/prepare_v2_data.py` | Select top 500 categories, build V2 train/test CSV |
| `scripts/generate_descriptions.py` | Call MiniMax API to generate category descriptions |
| `scripts/run_train.py` | Training entry point |
| `src/train/__init__.py` | Package init |
| `src/train/dataset.py` | ITC training dataset (image + text pairs) |
| `src/train/lora_config.py` | LoRA configuration for Q-Former |
| `src/train/trainer.py` | Training loop with ITC loss |
| `config/train.yaml` | Training hyperparameters |
| `tests/test_dataset.py` | Tests for ITC dataset |
| `tests/test_lora_config.py` | Tests for LoRA config |
| `tests/test_trainer.py` | Tests for trainer (mock model) |

### Modified files
| File | Change |
|------|--------|
| `src/model/blip2_encoder.py` | Add `lora_adapter_path` param, fix `get_multimodal_embeddings_batch` |
| `scripts/run_offline.py` | Add `--lora-adapter` CLI flag |
| `requirements.txt` | Pin peft==0.14.0, add openai |

---

## Task 1: Data Preparation Script

**Files:**
- Create: `scripts/prepare_v2_data.py`
- Test: `tests/test_prepare_v2_data.py`

- [ ] **Step 1: Write failing test for category selection**

```python
# tests/test_prepare_v2_data.py
"""Tests for V2 data preparation."""
from __future__ import annotations
import pandas as pd
import pytest


def _make_fake_train_csv(n_skus=1000, imgs_per_sku=15):
    """Create a fake train.csv matching Products-10K format."""
    rows = []
    for sku in range(n_skus):
        for i in range(imgs_per_sku):
            rows.append({"name": f"{sku}_{i}.jpg", "class": sku, "group": sku % 10})
    return pd.DataFrame(rows)


class TestSelectTopCategories:

    def test_selects_correct_count(self):
        from scripts.prepare_v2_data import select_top_categories
        df = _make_fake_train_csv(n_skus=1000, imgs_per_sku=15)
        result = select_top_categories(df, top_n=500)
        assert result["class"].nunique() == 500

    def test_selects_highest_frequency(self):
        from scripts.prepare_v2_data import select_top_categories
        # Make some SKUs have more images
        rows = []
        for sku in range(100):
            count = 50 if sku < 10 else 5  # top 10 have 50 imgs each
            for i in range(count):
                rows.append({"name": f"{sku}_{i}.jpg", "class": sku, "group": 0})
        df = pd.DataFrame(rows)
        result = select_top_categories(df, top_n=10)
        assert set(result["class"].unique()) == set(range(10))

    def test_respects_max_products(self):
        from scripts.prepare_v2_data import select_top_categories
        df = _make_fake_train_csv(n_skus=100, imgs_per_sku=100)
        result = select_top_categories(df, top_n=50, max_products=2000)
        assert len(result) <= 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prepare_v2_data.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 3: Implement prepare_v2_data.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_prepare_v2_data.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_v2_data.py tests/test_prepare_v2_data.py
git commit -m "feat: V2 data preparation script with top-N category selection"
```

---

## Task 2: LLM Category Description Generator

**Files:**
- Create: `scripts/generate_descriptions.py`
- Test: `tests/test_generate_descriptions.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_generate_descriptions.py
"""Tests for LLM category description generator."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestGenerateDescription:

    def test_generates_description_for_category(self):
        from scripts.generate_descriptions import generate_description
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="运动鞋，属于男鞋类目，常见品牌包括Nike、Adidas"))]
        )
        result = generate_description(mock_client, "运动鞋", model="test-model")
        assert len(result) > 10
        assert "运动鞋" in result

    def test_cache_prevents_duplicate_calls(self):
        from scripts.generate_descriptions import generate_with_cache
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="描述"))]
        )
        cache = {"运动鞋": "已缓存描述"}
        result = generate_with_cache(mock_client, "运动鞋", cache, model="test")
        assert result == "已缓存描述"
        mock_client.chat.completions.create.assert_not_called()

    def test_fallback_on_api_error(self):
        from scripts.generate_descriptions import generate_with_cache
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        cache = {}
        result = generate_with_cache(mock_client, "运动鞋", cache, model="test")
        assert "运动鞋" in result  # fallback template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_generate_descriptions.py -v`
Expected: FAIL

- [ ] **Step 3: Implement generate_descriptions.py**

```python
# scripts/generate_descriptions.py
"""Generate rich category descriptions via LLM API."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_description(client: OpenAI, category_name: str,
                         model: str = "M2.7") -> str:
    """Generate a rich description for a single category."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"给出'{category_name}'商品类目的详细描述（50-100字），"
                       f"包括典型品类词、常见品牌、商品特征。只输出描述，不要标题。"
        }],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_with_cache(client: OpenAI, category_name: str,
                        cache: dict, model: str = "M2.7") -> str:
    """Generate description with cache, fallback on error."""
    if category_name in cache:
        return cache[category_name]
    try:
        desc = generate_description(client, category_name, model=model)
    except Exception as e:
        print(f"  API error for {category_name}: {e}, using fallback")
        desc = f"{category_name}，电商商品类目"
    cache[category_name] = desc
    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories-file", required=True,
                        help="Path to v2_train.csv or categories list")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/category_descriptions.json"))
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.minimax.io/v1")
    parser.add_argument("--model", default="M2.7")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    output_path = Path(args.output)

    # Load existing cache
    cache = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Get unique categories
    import pandas as pd
    df = pd.read_csv(args.categories_file)
    categories = sorted(df["category_leaf"].unique()) if "category_leaf" in df.columns \
        else sorted(df["class"].astype(str).unique())

    print(f"Generating descriptions for {len(categories)} categories "
          f"({len(cache)} cached)")

    for i, cat in enumerate(categories):
        generate_with_cache(client, str(cat), cache, model=args.model)
        if (i + 1) % 50 == 0:
            # Save periodically
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"  Progress: {i+1}/{len(categories)}")

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cache)} descriptions to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_generate_descriptions.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_descriptions.py tests/test_generate_descriptions.py
git commit -m "feat: LLM category description generator with cache and fallback"
```

---

## Task 3: ITC Training Dataset

**Files:**
- Create: `src/train/__init__.py`, `src/train/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_dataset.py
"""Tests for ITC training dataset."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock


class TestITCDataset:

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal dataset files."""
        import pandas as pd
        # Create fake images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        rows = []
        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            path = img_dir / f"{i}.jpg"
            img.save(str(path))
            rows.append({
                "name": f"{i}.jpg", "class": i % 5,
                "image_path": str(path), "category_leaf": str(i % 5),
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "train.csv"
        df.to_csv(csv_path, index=False)

        descs = {str(i): f"Category {i} description" for i in range(5)}
        import json
        desc_path = tmp_path / "descriptions.json"
        with open(desc_path, "w") as f:
            json.dump(descs, f)

        return csv_path, desc_path

    def test_dataset_length(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        assert len(ds) == 20

    def test_dataset_returns_image_and_text(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        image, text, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(text, str)
        assert isinstance(label, int)

    def test_dataset_text_comes_from_descriptions(self, sample_data):
        from src.train.dataset import ITCDataset
        csv_path, desc_path = sample_data
        ds = ITCDataset(csv_path, desc_path, image_size=224)
        _, text, label = ds[0]
        assert f"Category {label}" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dataset.py**

```python
# src/train/__init__.py
# (empty)

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
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_dataset.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/train/__init__.py src/train/dataset.py tests/test_dataset.py
git commit -m "feat: ITC training dataset for V2 LoRA fine-tuning"
```

---

## Task 4: LoRA Configuration

**Files:**
- Create: `src/train/lora_config.py`, `config/train.yaml`
- Test: `tests/test_lora_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_lora_config.py
"""Tests for LoRA configuration."""
from __future__ import annotations
import pytest


class TestLoRAConfig:

    def test_creates_valid_lora_config(self):
        from src.train.lora_config import create_lora_config
        config = create_lora_config(rank=8, alpha=16, dropout=0.1)
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1

    def test_targets_qformer_attention(self):
        from src.train.lora_config import create_lora_config
        config = create_lora_config()
        # Should target q, k, v in Q-Former
        for module_name in config.target_modules:
            assert any(proj in module_name for proj in ["query", "key", "value",
                                                         "q_proj", "k_proj", "v_proj"])

    def test_load_train_config(self):
        from src.train.lora_config import load_train_config
        config = load_train_config()
        assert "training" in config
        assert config["training"]["epochs"] > 0
        assert config["training"]["batch_size"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_lora_config.py -v`
Expected: FAIL

- [ ] **Step 3: Create config/train.yaml**

```yaml
# config/train.yaml — V2 LoRA training hyperparameters
training:
  epochs: 3
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  scheduler: cosine
  fp16: true
  gradient_accumulation_steps: 2

lora:
  rank: 8
  alpha: 16
  dropout: 0.1

data:
  top_n_categories: 500
  max_products: 20000
  train_ratio: 0.7
  image_size: 224

checkpointing:
  save_dir: checkpoints
  save_every_epoch: true
```

- [ ] **Step 4: Implement lora_config.py**

```python
# src/train/lora_config.py
"""LoRA configuration for Q-Former fine-tuning."""
from __future__ import annotations
from pathlib import Path

import yaml
from peft import LoraConfig, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_lora_config(rank: int = 8, alpha: int = 16,
                       dropout: float = 0.1) -> LoraConfig:
    """Create LoRA config targeting Q-Former attention layers."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def validate_target_modules(qformer_module, target_modules: list[str]) -> None:
    """Verify target modules exist in Q-Former. Raises if none found."""
    all_names = [name for name, _ in qformer_module.named_modules()]
    matched = [t for t in target_modules if any(t in n for n in all_names)]
    if not matched:
        raise ValueError(
            f"No target modules {target_modules} found in Q-Former. "
            f"Available: {[n for n in all_names if 'attention' in n.lower()][:20]}"
        )
    print(f"LoRA target modules validated: {matched}")


def load_train_config(config_path: str | Path | None = None) -> dict:
    """Load training config from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "train.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_lora_config.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/train/lora_config.py config/train.yaml tests/test_lora_config.py
git commit -m "feat: LoRA config for Q-Former + training hyperparameters"
```

---

## Task 5: ITC Trainer

**Files:**
- Create: `src/train/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trainer.py
"""Tests for ITC trainer."""
from __future__ import annotations
import torch
import pytest
from unittest.mock import MagicMock


class TestITCLoss:

    def test_loss_is_scalar(self):
        from src.train.trainer import compute_itc_loss
        image_embs = torch.randn(4, 256)
        text_embs = torch.randn(4, 256)
        loss = compute_itc_loss(image_embs, text_embs, temperature=0.07)
        assert loss.dim() == 0  # scalar

    def test_loss_decreases_for_aligned_pairs(self):
        from src.train.trainer import compute_itc_loss
        # Aligned: image_i matches text_i
        embs = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
        noise = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
        loss_aligned = compute_itc_loss(embs, embs, temperature=0.07)
        loss_random = compute_itc_loss(embs, noise, temperature=0.07)
        assert loss_aligned < loss_random

    def test_loss_is_positive(self):
        from src.train.trainer import compute_itc_loss
        a = torch.randn(4, 256)
        b = torch.randn(4, 256)
        loss = compute_itc_loss(a, b, temperature=0.07)
        assert loss.item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement trainer.py**

```python
# src/train/trainer.py
"""ITC training loop for LoRA fine-tuning."""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_itc_loss(image_embs: torch.Tensor, text_embs: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Compute symmetric Image-Text Contrastive loss.

    Args:
        image_embs: [B, D] normalized image embeddings
        text_embs: [B, D] normalized text embeddings
        temperature: scaling factor

    Returns:
        Scalar loss (average of image→text and text→image cross-entropy)
    """
    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    logits = image_embs @ text_embs.T / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


class ITCTrainer:
    """Trainer for ITC LoRA fine-tuning."""

    def __init__(self, model, processor, optimizer, scheduler=None,
                 device="cuda", fp16=True, grad_accum_steps=2,
                 checkpoint_dir="checkpoints"):
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.fp16 = fp16
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Image → Q-Former → language_projection → [B, D]"""
        dtype = torch.float16 if self.fp16 else torch.float32
        pixel_values = pixel_values.to(self.device, dtype=dtype)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attn = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn,
            return_dict=True,
        )
        query_output = qformer_outputs.last_hidden_state.to(dtype)
        projected = self.model.language_projection(query_output)
        return F.normalize(projected.mean(dim=1).float(), dim=-1)

    def _get_text_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Text → LM word_embeddings → mean-pool → [B, D] (frozen, same as V1)"""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        return F.normalize(text_embeds.mean(dim=1).float(), dim=-1)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for step, (images, texts, labels) in enumerate(dataloader):
            # Process inputs
            img_inputs = self.processor(images=list(images), return_tensors="pt")
            txt_inputs = self.processor(text=list(texts), return_tensors="pt",
                                        padding=True, truncation=True)

            image_embs = self._get_image_embedding(img_inputs["pixel_values"])
            text_embs = self._get_text_embedding(txt_inputs["input_ids"])

            loss = compute_itc_loss(image_embs, text_embs) / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_steps += 1

            if (step + 1) % 100 == 0:
                avg = total_loss / n_steps
                print(f"  Epoch {epoch} Step {step+1}: loss={avg:.4f}")

        return total_loss / max(n_steps, 1)

    def save_checkpoint(self, epoch: int, tag: str = ""):
        """Save LoRA adapter weights."""
        name = f"epoch{epoch}" if not tag else tag
        save_path = self.checkpoint_dir / name
        self.model.save_pretrained(str(save_path))
        print(f"Checkpoint saved: {save_path}")
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_trainer.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.py tests/test_trainer.py
git commit -m "feat: ITC trainer with symmetric contrastive loss"
```

---

## Task 6: Update BLIP2Encoder for LoRA Loading

**Files:**
- Modify: `src/model/blip2_encoder.py`
- Test: `tests/test_blip2_encoder_lora.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_blip2_encoder_lora.py
"""Tests for BLIP2Encoder LoRA adapter loading."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch


class TestBLIP2EncoderLoRA:

    def test_init_without_lora(self):
        """Default init should work without LoRA path."""
        # This tests the interface, not actual model loading
        from src.model.blip2_encoder import BLIP2Encoder
        # BLIP2Encoder.__init__ signature should accept lora_adapter_path
        import inspect
        sig = inspect.signature(BLIP2Encoder.__init__)
        assert "lora_adapter_path" in sig.parameters

    def test_fix_multimodal_uses_text(self):
        """get_multimodal_embeddings_batch should use text input."""
        from src.model.blip2_encoder import BLIP2Encoder
        import inspect
        src = inspect.getsource(BLIP2Encoder.get_multimodal_embeddings_batch)
        # Should reference batch_texts somewhere in the implementation
        assert "batch_texts" in src or "texts" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_blip2_encoder_lora.py -v`
Expected: FAIL (lora_adapter_path not in signature yet)

- [ ] **Step 3: Modify blip2_encoder.py**

Add `lora_adapter_path` parameter to `__init__` and fix `get_multimodal_embeddings_batch`.

Changes to `__init__`:
```python
def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
             device: str = "cuda", dtype: str = "float16",
             lora_adapter_path: str | None = None):
    # ... existing init code ...
    self.model.eval()

    if lora_adapter_path is not None:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        self.model.eval()

    print("BLIP2 model loaded.")
```

Fix `get_multimodal_embeddings_batch` to use text — weight image and text embeddings:
```python
@torch.no_grad()
def get_multimodal_embeddings_batch(self, texts, images, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_images = images[i:i + batch_size]

        # Image embedding via Q-Former
        inputs = self.processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        image_embeds = self._get_vision_embeds(pixel_values)
        projected = self._qformer_with_vision(image_embeds)
        img_emb = projected.mean(dim=1)  # [B, D]

        # Text embedding via LM word embeddings
        txt_inputs = self.processor(text=batch_texts, return_tensors="pt",
                                     padding=True, truncation=True)
        input_ids = txt_inputs["input_ids"].to(self.device)
        txt_emb = self.model.language_model.get_input_embeddings()(input_ids).mean(dim=1)

        # Weighted combination (image-dominant for multimodal)
        combined = 0.7 * img_emb.float() + 0.3 * txt_emb.float()
        embeddings = F.normalize(combined, dim=-1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)
```

Also add `--lora-adapter` flag to `scripts/run_offline.py`:
```python
# In run_offline.py main(), add argparse argument:
parser.add_argument("--lora-adapter", default=None, help="Path to LoRA adapter")
# Pass to BLIP2Encoder:
encoder = BLIP2Encoder(..., lora_adapter_path=args.lora_adapter)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_blip2_encoder_lora.py tests/ -v`
Expected: ALL PASS (including existing 23 tests)

- [ ] **Step 5: Commit**

```bash
git add src/model/blip2_encoder.py tests/test_blip2_encoder_lora.py
git commit -m "feat: add LoRA adapter loading to BLIP2Encoder, fix multimodal batch"
```

---

## Task 7: Training Entry Point

**Files:**
- Create: `scripts/run_train.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create run_train.py**

```python
# scripts/run_train.py
"""V2 LoRA training entry point."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.dataset import ITCDataset
from src.train.lora_config import create_lora_config, load_train_config
from src.train.trainer import ITCTrainer


def collate_fn(batch):
    images, texts, labels = zip(*batch)
    return list(images), list(texts), list(labels)


def main():
    config = load_train_config()
    tc = config["training"]
    lc = config["lora"]

    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).cuda()

    # Apply LoRA to Q-Former ONLY (not LM or vision encoder)
    lora_config = create_lora_config(rank=lc["rank"], alpha=lc["alpha"], dropout=lc["dropout"])
    # Validate target modules exist in Q-Former before applying
    from src.train.lora_config import validate_target_modules
    validate_target_modules(model.qformer, lora_config.target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    data_dir = PROJECT_ROOT / "data"
    dataset = ITCDataset(data_dir / "v2_train.csv",
                         data_dir / "category_descriptions.json")
    dataloader = DataLoader(dataset, batch_size=tc["batch_size"],
                            shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    total_steps = len(dataloader) * tc["epochs"] // tc["gradient_accumulation_steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Train
    trainer = ITCTrainer(model, processor, optimizer, scheduler,
                         fp16=tc["fp16"], grad_accum_steps=tc["gradient_accumulation_steps"])

    for epoch in range(1, tc["epochs"] + 1):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
        trainer.save_checkpoint(epoch)

    print("Training complete. Run evaluation with the saved adapter.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update requirements.txt**

Add `peft==0.14.0` and `openai` (if not present).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_train.py requirements.txt
git commit -m "feat: V2 LoRA training entry point"
```

---

## Task 8: Run V1 Baseline on 500-Category Subset (GPU)

- [ ] **Step 1: Prepare V2 data on server**

```bash
# On GPU server
python scripts/prepare_v2_data.py --dataset-dir /root/autodl-tmp/products-10k --top-n 500 --max-products 20000
```

- [ ] **Step 2: Generate category descriptions**

```bash
python scripts/generate_descriptions.py --categories-file data/v2_train.csv --api-key <KEY> --model M2.7
```

- [ ] **Step 3: Run V1 zero-shot on 500-category subset**

```bash
# Regenerate embeddings with V2 subset data, evaluate
python scripts/run_offline.py  # uses v2 anchor data
python scripts/evaluate.py     # uses v2 test data
```

Record V1 baseline metrics on 500-category subset.

- [ ] **Step 4: Commit baseline results**

```bash
git commit -m "docs: V1 baseline on 500-category subset for V2 comparison"
```

---

## Task 9: Run V2 LoRA Training (GPU)

- [ ] **Step 1: Inspect Q-Former module names**

```python
# On GPU server, find actual LoRA target module names
for name, _ in model.qformer.named_modules():
    if "attention" in name.lower():
        print(name)
```

Update `lora_config.py` target_modules if needed.

- [ ] **Step 2: Run training**

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_train.py
```

Expected: loss decreases over 3 epochs, ~60-90 min.

- [ ] **Step 3: Evaluate V2**

```bash
# Re-run offline embedding with LoRA adapter
python scripts/run_offline.py --lora-adapter checkpoints/epoch3
python scripts/evaluate.py
```

- [ ] **Step 4: Record and compare**

Compare V1 vs V2 metrics on same 500-category subset. Update docs/devlog.md.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: V2 LoRA training results"
git tag v2.0
```

---

## Task 10: Rank Comparison Experiment (GPU)

- [ ] **Step 1: Train rank=4**

Edit config/train.yaml: `rank: 4, alpha: 8`
Run: `python scripts/run_train.py`
Evaluate, record metrics.

- [ ] **Step 2: Train rank=16**

Edit config/train.yaml: `rank: 16, alpha: 32`
Run: `python scripts/run_train.py`
Evaluate, record metrics.

- [ ] **Step 3: Record comparison table**

| Rank | Trainable Params | Training Time | Loss | leaf_top1 | l2_top1 | l1_top1 |
|------|-----------------|---------------|------|-----------|---------|---------|
| 4    | ~1.2M           |               |      |           |         |         |
| 8    | ~2.4M           |               |      |           |         |         |
| 16   | ~4.8M           |               |      |           |         |         |

Update docs/devlog.md and docs/design.md.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: LoRA rank comparison experiment results"
```

---

## Task 11: Wrap-up

- [ ] **Step 1: Update docs/design.md with V2 results**
- [ ] **Step 2: Update CLAUDE.md current status**
- [ ] **Step 3: Update orbit project note**
- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "docs: V2 completion, update design and project status"
```

- [ ] **Step 5: Run /end-work**
