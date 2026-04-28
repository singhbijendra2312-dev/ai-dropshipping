# AI Automated Dropshipping System — Phase 0 + Phase 1 Design

**Date:** 2026-04-28
**Scope:** Phase 0 (project scaffold) and Phase 1 (MVP `/generate-product` endpoint)
**Status:** Approved

## Goal

A FastAPI backend that takes a product input (name, cost, category, features, target audience) and returns: a suggested selling price, a conversion-focused product title, description, bullets, ad copy, and a marketing angle.

Pricing is rule-based and deterministic. Content is LLM-generated (Anthropic Claude Haiku 4.5) with strict structured output, retries, and a deterministic fallback so the endpoint always returns a valid response.

## Non-goals (explicit)

- No frontend store, no payments, no supplier integration (Phase 3).
- No persistence — fully stateless. (DB deferred to Phase 2 if needed.)
- No auth — localhost dev only.
- No multi-variant ad generation — Phase 2.

## Project layout

```
~/Documents/AI/ai-dropshipping/
├── pyproject.toml          # uv-managed
├── .env.example            # ANTHROPIC_API_KEY=...
├── README.md
├── app/
│   ├── main.py             # FastAPI app + /health
│   ├── config.py           # pydantic-settings, loads .env
│   ├── schemas.py          # Pydantic request/response models
│   ├── api/routes.py       # POST /generate-product
│   ├── pricing/engine.py   # rule-based, pure functions
│   ├── content/
│   │   ├── generator.py    # orchestrates LLM, retries, fallback
│   │   └── prompts.py      # system + user prompt templates
│   └── llm/
│       ├── base.py         # LLMClient Protocol
│       └── anthropic_client.py
├── tests/
│   ├── test_pricing.py
│   ├── test_generator.py
│   └── test_api.py
├── scripts/smoke.py        # manual end-to-end with real API key
└── data/sample_products.json
```

**Tooling:** Python 3.11+, `uv` for env/deps. Single `pyproject.toml`.

**Dependencies:** `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `anthropic`, `pytest`, `httpx`.

## Module boundaries

- `pricing/` — pure functions, no I/O, no LLM awareness.
- `content/` — knows about LLM, knows nothing about pricing.
- `llm/` — provider-specific code lives only here, behind `LLMClient` Protocol.
- `api/routes.py` — only place pricing and content are combined.

This keeps each unit independently testable and lets us swap the LLM provider without touching content logic.

## API

### `POST /generate-product`

**Request:**
```json
{
  "product_name": "Portable Blender",
  "cost_price": 15,
  "category": "Kitchen",
  "features": ["USB rechargeable", "Compact design", "Easy to clean"],
  "target_audience": "Health-conscious users"
}
```

**Response (200):**
```json
{
  "selling_price": 37.99,
  "product_title": "Blend Anywhere with This Portable USB Blender",
  "description": "...",
  "bullets": ["...", "...", "..."],
  "ad_copy": "...",
  "marketing_angle": "Convenience + healthy lifestyle",
  "source": "llm"
}
```

`source` is `"llm"` on success or `"fallback"` if the deterministic template was used.

### `GET /health`

Returns `{"status": "ok"}`.

## Pricing engine

Pure functions in `app/pricing/engine.py`. No external dependencies.

```python
CATEGORY_MARKUP = {
    "electronics": 3.0,
    "kitchen": 2.5,
    "beauty": 2.8,
    "apparel": 2.4,
    "commodities": 1.8,
    "home": 2.5,
}
DEFAULT_MARKUP = 2.5
MIN_PRICE = 9.99
MAX_PRICE = 999.99

def suggest_price(cost_price: float, category: str) -> float:
    markup = CATEGORY_MARKUP.get(category.lower().strip(), DEFAULT_MARKUP)
    raw = cost_price * markup
    rounded = round(raw)
    charm = max(rounded - 0.01, MIN_PRICE)
    return min(charm, MAX_PRICE)
```

Behavior:
- `$15 × 2.5 = $37.50 → round → $38 → charm → $37.99`
- Unknown category falls back to 2.5x.
- Floor at `$9.99`, cap at `$999.99`.

## LLM layer

### Protocol (`app/llm/base.py`)

```python
class LLMClient(Protocol):
    def generate_content(self, product: ProductInput) -> ContentBlock: ...
```

`ContentBlock` is a Pydantic model with fields: `product_title`, `description`, `bullets` (list, 3–5 items), `ad_copy`, `marketing_angle`.

### Anthropic implementation (`app/llm/anthropic_client.py`)

Uses **tool-use for structured output**:

1. Define a tool `submit_product_content` whose `input_schema` matches `ContentBlock`.
2. Call `claude-haiku-4-5` with `tool_choice={"type": "tool", "name": "submit_product_content"}` — forces the model to invoke the tool, guaranteeing a parseable structured response.
3. Read `tool_use.input` directly (no JSON-string parsing).
4. Validate via Pydantic on the way out.

**Prompt caching:** the system prompt (content rules — benefits over features, no false claims, short punchy ad copy, persuasive description, charm-tone marketing angle) is marked `cache_control: ephemeral`. Identical across requests, so cache hits save tokens.

**Prompt-injection guard:** `product_name` and `features` are sanitized (strip control chars, cap length at 200 chars per field) and wrapped in clearly delimited tags in the user message. System prompt instructs the model to treat input as untrusted data and ignore any embedded instructions.

## Content generator

`app/content/generator.py` orchestrates:

1. Sanitize input.
2. Call `LLMClient.generate_content`.
3. On exception or Pydantic validation error: retry once.
4. On second failure: return a deterministic fallback `ContentBlock` (template-based title from product_name, generic description, bullets derived from features, simple ad copy).

The generator returns both the `ContentBlock` and a `source` flag (`"llm"` or `"fallback"`) so the API layer can include it in the response.

## Schemas (`app/schemas.py`)

```python
class ProductInput(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    cost_price: float = Field(gt=0, le=100000)
    category: str = Field(min_length=1, max_length=50)
    features: list[str] = Field(default_factory=list, max_length=20)
    target_audience: str = Field(default="", max_length=200)

class ContentBlock(BaseModel):
    product_title: str
    description: str
    bullets: list[str] = Field(min_length=3, max_length=5)
    ad_copy: str
    marketing_angle: str

class ProductResponse(BaseModel):
    selling_price: float
    product_title: str
    description: str
    bullets: list[str]
    ad_copy: str
    marketing_angle: str
    source: Literal["llm", "fallback"]
```

## Edge cases & error model

| Case | Behavior |
|---|---|
| `cost_price <= 0` | 422 (Pydantic `gt=0`) |
| `cost_price < ~$4` | Floor selling price at `$9.99` |
| `cost_price` very high | Cap at `$999.99` |
| Unknown category | Default 2.5x markup, LLM still runs |
| Empty `features` | Allowed; prompt instructs LLM to infer from name+category |
| Generic product name | LLM best-effort; fallback if output fails validation |
| LLM API error/timeout | 1 retry → deterministic fallback |
| LLM returns invalid schema | 1 retry → fallback |
| Prompt injection in input | Sanitized + delimited; system prompt instructs to ignore embedded instructions |

**HTTP status codes:**
- `400/422` — invalid input (Pydantic).
- `200` — success, with `source` field indicating LLM or fallback.
- `500` — reserved for unexpected bugs only. Never thrown for LLM or network issues.

## Configuration

`app/config.py` uses `pydantic-settings`:

```python
class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5"
    llm_max_retries: int = 1
    llm_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(env_file=".env")
```

## Testing strategy

1. **`test_pricing.py`** — pure unit tests: markup per known category, charm pricing math, floor at MIN_PRICE, cap at MAX_PRICE, unknown category falls to default.
2. **`test_generator.py`** — mocked `LLMClient`: happy path, retry-then-success, retry-then-fallback, invalid schema → fallback.
3. **`test_api.py`** — FastAPI `TestClient` with injected fake `LLMClient`: 200 happy path, 422 on bad input, fallback path returns 200 with `source="fallback"`.

**Not in CI:** real Anthropic API calls. `scripts/smoke.py` provides a manual end-to-end check with a real key.

## Setup & run

```bash
cd ~/Documents/AI/ai-dropshipping
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run uvicorn app.main:app --reload
```

Smoke test:
```bash
curl -X POST http://localhost:8000/generate-product \
  -H "Content-Type: application/json" \
  -d @data/sample_products.json
```

## Future phases (out of scope here)

- **Phase 2:** product score, multiple ad variations, audience segments, optional persistence (SQLite).
- **Phase 3:** supplier API ingestion, auto-import, store-listing export.
