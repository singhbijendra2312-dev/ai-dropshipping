import pytest
from app.content.competitive import (
    run,
    DIFFERENTIATION_FALLBACKS,
    WEAKNESSES_FALLBACKS,
    GENERIC_DIFF_FALLBACK,
    GENERIC_WEAK_FALLBACK,
)
from app.llm.base import LLMError
from app.schemas import (
    Competitor,
    CompetitiveIntel,
    PriceBenchmarks,
)


class _FakeClient:
    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0

    async def generate_competitive_intel(self, product):
        self.calls += 1
        action = self._behaviors.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _full_intel():
    return CompetitiveIntel(
        price_benchmarks=PriceBenchmarks(
            low=15.0, median=22.5, high=30.0, sample_size=4
        ),
        competitors=[
            Competitor(
                name="Acme Blender",
                price=19.99,
                source_url="https://example.com/acme",
                key_feature="USB-C charging",
            ),
            Competitor(
                name="Zip Mini Blender",
                price=24.99,
                source_url="https://example.com/zip",
                key_feature="650 ml capacity",
            ),
        ],
        differentiation_suggestions=[
            "Faster blending cycle",
            "Quieter operation",
            "Larger battery",
        ],
        common_weaknesses=[
            "Leaks at the lid",
            "Loud motor",
            "Short battery life",
        ],
    )


async def test_happy_path_returns_full_intel(sample_input):
    intel = _full_intel()
    client = _FakeClient([intel])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "llm"
    assert out == intel
    assert client.calls == 1


async def test_retry_then_success(sample_input):
    intel = _full_intel()
    client = _FakeClient([LLMError("transient"), intel])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "llm"
    assert client.calls == 2


async def test_retry_exhausted_returns_kitchen_fallback(sample_input):
    # sample_input.category == "Kitchen"
    client = _FakeClient([LLMError("down"), LLMError("still down")])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "fallback"
    assert out.competitors == []
    assert out.price_benchmarks is None
    assert out.differentiation_suggestions == DIFFERENTIATION_FALLBACKS["kitchen"]
    assert out.common_weaknesses == WEAKNESSES_FALLBACKS["kitchen"]


async def test_unknown_category_returns_generic_fallback(sample_input):
    unknown = sample_input.model_copy(update={"category": "Toys"})
    client = _FakeClient([LLMError("x"), LLMError("y")])
    out, source = await run(client, unknown, max_retries=1)
    assert source == "fallback"
    assert out.differentiation_suggestions == GENERIC_DIFF_FALLBACK
    assert out.common_weaknesses == GENERIC_WEAK_FALLBACK


async def test_empty_competitors_still_classified_as_llm(sample_input):
    # Search ran successfully but found nothing usable. Honest empty result.
    intel = CompetitiveIntel(
        price_benchmarks=None,
        competitors=[],
        differentiation_suggestions=["a", "b", "c"],
        common_weaknesses=["x", "y", "z"],
    )
    client = _FakeClient([intel])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "llm"
    assert out.competitors == []
    assert out.price_benchmarks is None
    assert out == intel


async def test_llm_error_both_attempts_returns_fallback(sample_input):
    client = _FakeClient([LLMError("a"), LLMError("b")])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "fallback"
    assert client.calls == 2


async def test_fallback_dispatches_by_category_case_insensitive(sample_input):
    # "BEAUTY" should dispatch to beauty bank
    beauty = sample_input.model_copy(update={"category": "BEAUTY"})
    client = _FakeClient([LLMError("x"), LLMError("y")])
    out, source = await run(client, beauty, max_retries=1)
    assert source == "fallback"
    assert out.differentiation_suggestions == DIFFERENTIATION_FALLBACKS["beauty"]
    assert out.common_weaknesses == WEAKNESSES_FALLBACKS["beauty"]
