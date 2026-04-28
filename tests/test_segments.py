import pytest
from app.content.segments import run, GENERIC_FALLBACK, CATEGORY_FALLBACKS
from app.llm.base import LLMError
from app.schemas import AudienceSegment


def _segs(*tuples):
    return [
        AudienceSegment(
            name=n, description=d, pain_point=p, recommended_channel=c
        )
        for (n, d, p, c) in tuples
    ]


class _FakeClient:
    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0

    async def generate_segments(self, product):
        self.calls += 1
        action = self._behaviors.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


async def test_happy_path_returns_three_segments(sample_input):
    raw = _segs(
        ("A", "desc a", "pain a", "tiktok"),
        ("B", "desc b", "pain b", "instagram"),
        ("C", "desc c", "pain c", "facebook"),
    )
    client = _FakeClient([raw])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "llm"
    assert len(out) == 3
    assert client.calls == 1


async def test_wrong_count_triggers_retry(sample_input):
    two = _segs(
        ("A", "d", "p", "tiktok"),
        ("B", "d", "p", "instagram"),
    )
    three = _segs(
        ("A", "d", "p", "tiktok"),
        ("B", "d", "p", "instagram"),
        ("C", "d", "p", "facebook"),
    )
    client = _FakeClient([two, three])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "llm"
    assert client.calls == 2


async def test_wrong_count_both_attempts_returns_fallback(sample_input):
    two = _segs(("A", "d", "p", "tiktok"), ("B", "d", "p", "instagram"))
    client = _FakeClient([two, two])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "fallback"
    assert len(out) == 3


async def test_llm_error_returns_category_based_fallback(sample_input):
    # sample_input is Kitchen — should pick Kitchen overrides
    client = _FakeClient([LLMError("down"), LLMError("still down")])
    out, source = await run(client, sample_input, max_retries=1)
    assert source == "fallback"
    assert out == CATEGORY_FALLBACKS["kitchen"]


async def test_unknown_category_returns_generic_fallback(sample_input):
    unknown = sample_input.model_copy(update={"category": "Toys"})
    client = _FakeClient([LLMError("x"), LLMError("y")])
    out, source = await run(client, unknown, max_retries=1)
    assert source == "fallback"
    assert out == GENERIC_FALLBACK
