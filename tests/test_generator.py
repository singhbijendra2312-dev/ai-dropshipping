import pytest
from app.content.generator import generate_with_fallback
from app.llm.base import LLMError


class _FakeClient:
    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0

    def generate_content(self, product):
        self.calls += 1
        action = self._behaviors.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def test_happy_path(sample_input, sample_content):
    client = _FakeClient([sample_content])
    block, source = generate_with_fallback(client, sample_input, max_retries=1)
    assert source == "llm"
    assert block == sample_content
    assert client.calls == 1


def test_retry_then_success(sample_input, sample_content):
    client = _FakeClient([LLMError("transient"), sample_content])
    block, source = generate_with_fallback(client, sample_input, max_retries=1)
    assert source == "llm"
    assert block == sample_content
    assert client.calls == 2


def test_retry_exhausted_returns_fallback(sample_input):
    client = _FakeClient([LLMError("down"), LLMError("still down")])
    block, source = generate_with_fallback(client, sample_input, max_retries=1)
    assert source == "fallback"
    assert "Portable Blender" in block.product_title
    assert len(block.bullets) >= 3
    assert client.calls == 2


def test_fallback_when_features_empty(sample_input):
    no_features = sample_input.model_copy(update={"features": []})
    client = _FakeClient([LLMError("x"), LLMError("y")])
    block, source = generate_with_fallback(client, no_features, max_retries=1)
    assert source == "fallback"
    assert len(block.bullets) >= 3


def test_fallback_truncates_long_product_name(sample_input):
    long_input = sample_input.model_copy(
        update={"product_name": "A" * 200, "target_audience": "B" * 200}
    )
    client = _FakeClient([LLMError("x"), LLMError("y")])
    block, source = generate_with_fallback(client, long_input, max_retries=1)
    assert source == "fallback"
    assert len(block.product_title) <= 200
    assert len(block.description) <= 2000
    assert len(block.ad_copy) <= 500
