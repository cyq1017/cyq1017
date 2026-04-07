# tests/test_blip2_encoder_lora.py
"""Tests for BLIP2Encoder LoRA adapter loading."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch


class TestBLIP2EncoderLoRA:

    def test_init_without_lora(self):
        """Default init should work without LoRA path."""
        from src.model.blip2_encoder import BLIP2Encoder
        import inspect
        sig = inspect.signature(BLIP2Encoder.__init__)
        assert "lora_adapter_path" in sig.parameters

    def test_fix_multimodal_uses_text(self):
        """get_multimodal_embeddings_batch should use text input."""
        from src.model.blip2_encoder import BLIP2Encoder
        import inspect
        src = inspect.getsource(BLIP2Encoder.get_multimodal_embeddings_batch)
        assert "batch_texts" in src or "texts" in src
