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
        for module_name in config.target_modules:
            assert any(proj in module_name for proj in ["query", "key", "value",
                                                         "q_proj", "k_proj", "v_proj"])

    def test_load_train_config(self):
        from src.train.lora_config import load_train_config
        config = load_train_config()
        assert "training" in config
        assert config["training"]["epochs"] > 0
        assert config["training"]["batch_size"] > 0
