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
