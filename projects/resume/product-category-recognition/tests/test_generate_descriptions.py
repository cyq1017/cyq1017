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
