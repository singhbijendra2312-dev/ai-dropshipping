import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client
from app.config import Settings, get_settings
from app.llm.base import LLMError
from app.main import app


class _FakeClient:
    def __init__(self, content_or_error):
        self._payload = content_or_error

    async def generate_content(self, product):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def client_factory():
    def _make(payload):
        fake_settings = Settings(  # type: ignore[call-arg]
            anthropic_api_key="test-key-not-real",
        )
        app.dependency_overrides[get_settings] = lambda: fake_settings
        app.dependency_overrides[get_llm_client] = lambda: _FakeClient(payload)
        return TestClient(app)
    yield _make
    app.dependency_overrides.clear()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_generate_product_happy_path(client_factory, sample_content):
    client = client_factory(sample_content)
    payload = {
        "product_name": "Portable Blender",
        "cost_price": 15,
        "category": "Kitchen",
        "features": ["USB rechargeable", "Compact design", "Easy to clean"],
        "target_audience": "Health-conscious users",
    }
    r = client.post("/generate-product", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["selling_price"] == 37.99
    assert body["product_title"] == sample_content.product_title
    assert body["source"] == "llm"


def test_generate_product_fallback_path(client_factory):
    client = client_factory(LLMError("provider down"))
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


def test_generate_product_validation_error(client_factory, sample_content):
    client = client_factory(sample_content)
    payload = {
        "product_name": "Bad",
        "cost_price": -5,  # invalid
        "category": "Kitchen",
        "features": [],
        "target_audience": "",
    }
    r = client.post("/generate-product", json=payload)
    assert r.status_code == 422
