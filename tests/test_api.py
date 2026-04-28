import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client
from app.config import Settings, get_settings
from app.llm.base import LLMError
from app.main import app


class _FakeClient:
    """Configurable async fake LLM client for API tests."""

    def __init__(
        self,
        content=None,
        variations=None,
        segments=None,
    ):
        self._content = content
        self._variations = variations
        self._segments = segments

    async def generate_content(self, product):
        if isinstance(self._content, Exception):
            raise self._content
        return self._content

    async def generate_variations(self, product, axes):
        if isinstance(self._variations, Exception):
            raise self._variations
        # Echo back only the requested axes from the configured payload
        by_axis = {v.axis: v for v in (self._variations or [])}
        return [by_axis[a] for a in axes if a in by_axis]

    async def generate_segments(self, product):
        if isinstance(self._segments, Exception):
            raise self._segments
        return self._segments or []


@pytest.fixture
def make_client():
    fake_settings = Settings(  # type: ignore[call-arg]
        anthropic_api_key="test-key-not-real",
    )

    def _make(**fake_kwargs):
        app.dependency_overrides[get_settings] = lambda: fake_settings
        app.dependency_overrides[get_llm_client] = lambda: _FakeClient(**fake_kwargs)
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_generate_product_includes_score(make_client, sample_content):
    client = make_client(content=sample_content)
    payload = {
        "product_name": "Portable Blender",
        "cost_price": 30,
        "category": "Kitchen",
        "features": ["a", "b", "c"],
        "target_audience": "Health-conscious users",
    }
    r = client.post("/generate-product", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "product_score" in body
    assert isinstance(body["product_score"], int)
    assert 0 <= body["product_score"] <= 100
    assert body["source"] == "llm"
    # No variations/segments requested
    assert body["ad_variations"] is None
    assert body["variations_source"] == "skipped"
    assert body["audience_segments"] is None
    assert body["segments_source"] == "skipped"


def test_generate_product_fallback_path(make_client):
    client = make_client(content=LLMError("provider down"))
    payload = {
        "product_name": "Mystery Gadget",
        "cost_price": 20,
        "category": "Electronics",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["selling_price"] == 59.99
    assert body["source"] == "fallback"
    assert len(body["bullets"]) >= 3


def test_generate_product_validation_error(make_client, sample_content):
    client = make_client(content=sample_content)
    payload = {
        "product_name": "Bad",
        "cost_price": -5,
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product", json=payload)
    assert r.status_code == 422


def test_variations_returned_when_requested(
    make_client, sample_content, sample_variations
):
    client = make_client(content=sample_content, variations=sample_variations)
    payload = {
        "product_name": "Portable Blender",
        "cost_price": 15,
        "category": "Kitchen",
        "features": ["a", "b", "c"],
        "target_audience": "Health-conscious",
    }
    r = client.post("/generate-product?variations=urgency,humor", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["variations_source"] == "llm"
    assert [v["axis"] for v in body["ad_variations"]] == ["urgency", "humor"]


def test_variations_empty_param_skipped(make_client, sample_content):
    client = make_client(content=sample_content)
    payload = {
        "product_name": "X",
        "cost_price": 15,
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product?variations=", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["variations_source"] == "skipped"
    assert body["ad_variations"] is None


def test_unknown_axis_returns_422(make_client, sample_content):
    client = make_client(content=sample_content)
    payload = {
        "product_name": "X",
        "cost_price": 15,
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product?variations=urgency,bogus", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["field"] == "variations"
    assert "bogus" in detail["invalid"]


def test_segments_returned_when_requested(
    make_client, sample_content, sample_segments
):
    client = make_client(content=sample_content, segments=sample_segments)
    payload = {
        "product_name": "Portable Blender",
        "cost_price": 15,
        "category": "Kitchen",
        "features": ["a", "b", "c"],
        "target_audience": "Health-conscious",
    }
    r = client.post("/generate-product?segments=true", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["segments_source"] == "llm"
    assert len(body["audience_segments"]) == 3


def test_segments_false_skipped(make_client, sample_content):
    client = make_client(content=sample_content)
    payload = {
        "product_name": "X",
        "cost_price": 15,
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product?segments=false", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["segments_source"] == "skipped"
    assert body["audience_segments"] is None


def test_partial_failure_variations_only(
    make_client, sample_content, sample_segments
):
    client = make_client(
        content=sample_content,
        variations=LLMError("variations down"),
        segments=sample_segments,
    )
    payload = {
        "product_name": "Portable Blender",
        "cost_price": 15,
        "category": "Kitchen",
        "features": ["a", "b"],
        "target_audience": "",
    }
    r = client.post(
        "/generate-product?variations=urgency,humor&segments=true",
        json=payload,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "llm"
    assert body["variations_source"] == "fallback"
    assert body["segments_source"] == "llm"
    assert len(body["ad_variations"]) == 2
    assert len(body["audience_segments"]) == 3


def test_all_three_fail_returns_200_with_all_fallback(make_client):
    client = make_client(
        content=LLMError("c"),
        variations=LLMError("v"),
        segments=LLMError("s"),
    )
    payload = {
        "product_name": "Z",
        "cost_price": 15,
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post(
        "/generate-product?variations=urgency&segments=true", json=payload
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "fallback"
    assert body["variations_source"] == "fallback"
    assert body["segments_source"] == "fallback"
