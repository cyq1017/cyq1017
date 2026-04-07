# src/train/lora_config.py
"""LoRA configuration for Q-Former fine-tuning."""
from __future__ import annotations

from pathlib import Path

import yaml

try:
    from peft import LoraConfig, TaskType
except ImportError:
    # Stub for environments without peft installed
    from dataclasses import dataclass, field
    from typing import Any

    class TaskType:
        FEATURE_EXTRACTION = "FEATURE_EXTRACTION"

    @dataclass
    class LoraConfig:
        r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.1
        target_modules: list[str] = field(default_factory=list)
        bias: str = "none"
        task_type: Any = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_lora_config(
    rank: int = 8, alpha: int = 16, dropout: float = 0.1
) -> LoraConfig:
    """Create LoRA config targeting Q-Former attention layers."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def validate_target_modules(
    qformer_module: object, target_modules: list[str]
) -> None:
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
