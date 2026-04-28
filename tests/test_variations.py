import pytest
from app.content.variations import run, FALLBACK_TEMPLATES
from app.llm.base import LLMError
from app.schemas import AdVariation


class _FakeClient:
    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0
        self.last_axes = None

    async def generate_variations(self, product, axes):
        self.calls += 1
        self.last_axes = list(axes)
        action = self._behaviors.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _vars(*pairs):
    return [AdVariation(axis=a, ad_copy=c) for a, c in pairs]


async def test_happy_path_returns_requested_axes_in_order(sample_input):
    raw = _vars(("humor", "Funny copy"), ("urgency", "Hurry copy"))
    client = _FakeClient([raw])
    out, source = await run(client, sample_input, ["urgency", "humor"], max_retries=1)
    assert source == "llm"
    assert [v.axis for v in out] == ["urgency", "humor"]
    assert client.calls == 1


async def test_dedupe_repeated_axes(sample_input):
    raw = _vars(("urgency", "u"))
    client = _FakeClient([raw])
    out, source = await run(client, sample_input, ["urgency", "urgency"], max_retries=1)
    assert source == "llm"
    assert len(out) == 1
    assert out[0].axis == "urgency"
    assert client.last_axes == ["urgency"]


async def test_extra_axes_filtered_to_requested(sample_input):
    raw = _vars(
        ("urgency", "u"),
        ("humor", "h"),
        ("aspirational", "a"),
    )
    client = _FakeClient([raw])
    out, source = await run(client, sample_input, ["urgency", "humor"], max_retries=1)
    assert source == "llm"
    assert [v.axis for v in out] == ["urgency", "humor"]


async def test_missing_axes_triggers_retry_then_fallback(sample_input):
    incomplete = _vars(("urgency", "u"))  # missing 'humor'
    complete = _vars(("urgency", "u"), ("humor", "h"))
    # First call returns incomplete -> retry returns complete
    client = _FakeClient([incomplete, complete])
    out, source = await run(client, sample_input, ["urgency", "humor"], max_retries=1)
    assert source == "llm"
    assert client.calls == 2


async def test_missing_axes_both_attempts_returns_fallback(sample_input):
    incomplete = _vars(("urgency", "u"))
    client = _FakeClient([incomplete, incomplete])
    out, source = await run(client, sample_input, ["urgency", "humor"], max_retries=1)
    assert source == "fallback"
    assert [v.axis for v in out] == ["urgency", "humor"]
    assert all(v.ad_copy for v in out)


async def test_llm_error_then_fallback(sample_input):
    client = _FakeClient([LLMError("down"), LLMError("still down")])
    out, source = await run(client, sample_input, ["urgency"], max_retries=1)
    assert source == "fallback"
    assert len(out) == 1
    assert out[0].axis == "urgency"
    assert out[0].ad_copy == FALLBACK_TEMPLATES["urgency"]


async def test_problem_solution_template_substitutes_name(sample_input):
    client = _FakeClient([LLMError("down"), LLMError("still down")])
    out, source = await run(client, sample_input, ["problem_solution"], max_retries=1)
    assert source == "fallback"
    assert "Portable Blender" in out[0].ad_copy
