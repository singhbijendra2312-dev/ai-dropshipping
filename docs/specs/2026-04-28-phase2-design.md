# AI Dropshipping Phase 2 — Product Intelligence

**Date:** 2026-04-28
**Scope:** Extend `POST /generate-product` with deterministic product score, axis-tagged ad variations, and structured audience segments. Move LLM calls to async with per-section failure isolation.
**Status:** Approved

## Goal

Phase 1 returns one block of content + a price. Phase 2 adds three intelligence enhancements:

1. A deterministic 0–100 **product score** signalling dropshipping fit.
2. Up to 5 **axis-tagged ad variations** (urgency, aspirational, social_proof, problem_solution, humor) opt-in via query param.
3. Three structured **audience segments** with name, description, pain point, and recommended channel — opt-in via query param.

Each LLM-backed section retries once and falls back deterministically. Sections are independent — when variations fail, base content + segments still ship.

## Non-goals

- No persistence (still stateless; deferred to Phase 3 with supplier integration).
- No score tuning from real conversion data — rubric is hand-crafted.
- No multi-segment-count knob — fixed at 3.
- No frontend, no auth, no payments.

## API

### `POST /generate-product`

**Body** unchanged from Phase 1.

**Query params (all optional):**

| Param | Type | Default | Behavior |
|---|---|---|---|
| `variations` | csv of axis names | `""` (none) | Subset of `{urgency, aspirational, social_proof, problem_solution, humor}`. Unknown axis → 422. Duplicates deduped. Empty/missing → no variations. |
| `segments` | bool (`true`/`false`) | `false` | Include 3 audience segments when `true`. |

**Response (extended):**

```jsonc
{
  "selling_price": 37.99,
  "product_score": 72,                     // always present, 0–100 integer
  "product_title": "...",
  "description": "...",
  "bullets": [...],
  "ad_copy": "...",
  "marketing_angle": "...",
  "source": "llm",                         // base content source: "llm" | "fallback"

  "ad_variations": [                       // present iff variations requested
    {"axis": "urgency", "ad_copy": "..."},
    {"axis": "humor",   "ad_copy": "..."}
  ],
  "variations_source": "skipped",          // "llm" | "fallback" | "skipped"

  "audience_segments": [                   // present iff segments=true
    {"name": "...", "description": "...", "pain_point": "...", "recommended_channel": "tiktok"},
    {"name": "...", "description": "...", "pain_point": "...", "recommended_channel": "instagram"},
    {"name": "...", "description": "...", "pain_point": "...", "recommended_channel": "facebook"}
  ],
  "segments_source": "skipped"             // "llm" | "fallback" | "skipped"
}
```

When `variations` is empty/missing, `ad_variations` is omitted (or `null`) and `variations_source` is `"skipped"`. Same pattern for segments.

### `GET /health`

Unchanged.

## Module layout

```
app/
├── scoring/
│   ├── __init__.py            NEW
│   └── engine.py              NEW — pure functions, no I/O
├── content/
│   ├── prompts.py             EDIT — add VARIATIONS_TOOL, SEGMENTS_TOOL, system prompts
│   ├── variations.py          NEW — run(llm, product, axes), fallback templates
│   └── segments.py            NEW — run(llm, product), fallback segments
├── llm/
│   ├── base.py                EDIT — async Protocol with 3 methods
│   └── anthropic_client.py    EDIT — AsyncAnthropic, 3 methods, each its own tool
├── api/
│   ├── deps.py                EDIT — return AsyncLLMClient
│   └── routes.py              EDIT — async, query params, asyncio.gather, partial sources
└── schemas.py                 EDIT — AdAxis, AdVariation, AudienceSegment, extended ProductResponse

tests/
├── test_scoring.py            NEW
├── test_variations.py         NEW (mocked async client)
├── test_segments.py           NEW (mocked async client)
├── test_generator.py          EDIT — async migration
└── test_api.py                EDIT — query param paths, partial-failure paths
```

`pricing/engine.py`, `app/main.py`, `app/config.py` untouched.

## Async LLM transition

`LLMClient` Protocol moves to async. `AnthropicLLMClient` switches to `anthropic.AsyncAnthropic`. Routes become `async def`; FastAPI handles them natively.

```python
class LLMClient(Protocol):
    async def generate_content(self, product: ProductInput) -> ContentBlock: ...
    async def generate_variations(self, product: ProductInput, axes: list[AdAxis]) -> list[AdVariation]: ...
    async def generate_segments(self, product: ProductInput) -> list[AudienceSegment]: ...
```

Each method has its own tool definition and its own retry-once-then-fallback. The route uses `asyncio.gather` to fan out:

```python
tasks = [generator.generate_with_fallback(llm, product)]
if axes: tasks.append(variations.run(llm, product, axes))
if want_segments: tasks.append(segments.run(llm, product))
results = await asyncio.gather(*tasks)
```

Failure isolation: each section returns its own `(value, source)` tuple. One section's fallback does not affect the others.

## Product score (deterministic)

Pure functions in `app/scoring/engine.py`. No imports from `llm/` or `content/`.

```python
def compute_score(product: ProductInput) -> int:
    score = 0
    score += _feature_score(product.features)         # 0–35
    score += _category_score(product.category)        # 5–25
    score += _audience_score(product.target_audience) # 0–15
    score += _price_score(product.cost_price)         # 10–25
    return min(score, 100)
```

**Rubric:**

| Component | Inputs | Points |
|---|---|---|
| Feature richness | 0 features → 0; 1–2 → 15; 3–4 → 25; 5+ → 35 |
| Category margin | High (`electronics`, `beauty`) → 25; mid (`kitchen`, `home`, `apparel`) → 15; low (`commodities`) → 5; unknown → 15 |
| Audience specificity | empty/whitespace → 0; ≤30 chars → 8; >30 chars → 15 |
| Price sweet spot | $20 ≤ `cost_price` ≤ $60 → 25; otherwise → 10 |

Max realistic score: 35 + 25 + 15 + 25 = 100. Final value clamped at 100.

**Rationale:** these are signals of a winning dropshipping product — rich features make stronger copy, high-margin categories leave room for ads, specific audiences enable better targeting, and the $20–$60 cost sweet spot lands at ~$60–$180 retail (impulse-buy zone for ~3× markup).

## Ad variations

**Fixed axes:**

```python
AdAxis = Literal["urgency", "aspirational", "social_proof", "problem_solution", "humor"]
```

**Single LLM call** returns all requested axes in one shot — cheaper than per-axis calls and lets the model differentiate angles in one pass.

**Tool schema (`submit_ad_variations`):**

```json
{
  "type": "object",
  "properties": {
    "variations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "axis": {"enum": ["urgency", "aspirational", "social_proof", "problem_solution", "humor"]},
          "ad_copy": {"type": "string"}
        },
        "required": ["axis", "ad_copy"]
      }
    }
  },
  "required": ["variations"]
}
```

**Post-validation:** the model may include or omit axes; the variations module filters to exactly the requested set, in the order requested. Missing axes after retry trigger fallback.

**System prompt** explains each axis tersely (e.g., `urgency: time pressure, scarcity` / `humor: light, playful, never demeaning`) so output matches the angle.

**Fallback templates** (one per axis) used when the LLM fails twice:

| Axis | Template |
|---|---|
| `urgency` | `"Limited stock — order today before they're gone."` |
| `aspirational` | `"Become the version of you that already has it."` |
| `social_proof` | `"Thousands of customers can't be wrong. See why everyone is talking about it."` |
| `problem_solution` | `"Tired of {pain}? Meet {name} — the simple fix."` |
| `humor` | `"It's not magic. It just feels like it."` |

Templates are short and intentionally generic; the variations module substitutes `{name}` and `{pain}` where placeholders exist (pain falls back to "the daily hassle").

## Audience segments

**Fixed count: 3.** Three is enough for actionable variety without noise.

**Schema:**

```python
class AudienceSegment(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    pain_point: str = Field(min_length=1, max_length=300)
    recommended_channel: Literal[
        "tiktok", "instagram", "facebook", "youtube", "google_ads", "email"
    ]
```

**Single LLM call** with tool `submit_audience_segments`. System prompt instructs distinct personas with non-overlapping channels when possible.

**Post-validation:** must return exactly 3 segments, each with a valid channel. Anything else after retry triggers fallback.

**Fallback (deterministic, category-based)** — when LLM fails twice, return 3 segments tied to the product's category. For unknown category, use a generic set:

| Generic segment | Channel |
|---|---|
| "Practical buyers" | google_ads |
| "Trend-driven shoppers" | tiktok |
| "Gift buyers" | facebook |

Per-category overrides (kitchen, electronics, beauty, apparel) live in a small dict in `segments.py`.

## Schemas (extensions)

```python
AdAxis = Literal["urgency", "aspirational", "social_proof", "problem_solution", "humor"]
SectionSource = Literal["llm", "fallback", "skipped"]

class AdVariation(BaseModel):
    axis: AdAxis
    ad_copy: str = Field(min_length=1, max_length=500)

class AudienceSegment(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    pain_point: str = Field(min_length=1, max_length=300)
    recommended_channel: Literal[
        "tiktok", "instagram", "facebook", "youtube", "google_ads", "email"
    ]

class ProductResponse(BaseModel):
    selling_price: float
    product_score: int = Field(ge=0, le=100)
    product_title: str
    description: str
    bullets: list[str]
    ad_copy: str
    marketing_angle: str
    source: Literal["llm", "fallback"]              # base content
    ad_variations: list[AdVariation] | None = None
    variations_source: SectionSource = "skipped"
    audience_segments: list[AudienceSegment] | None = None
    segments_source: SectionSource = "skipped"
```

`ProductInput` and `ContentBlock` unchanged.

## Configuration

`app/config.py` adds nothing. Existing `LLM_MAX_RETRIES` and `LLM_TIMEOUT_SECONDS` apply to all three LLM call sites.

## Edge cases

| Case | Behavior |
|---|---|
| `?variations=` (empty value) | No variations; `variations_source: "skipped"` |
| `?variations=urgency,bogus` | 422 with `detail` listing valid axes |
| `?variations=urgency,urgency` | Dedupe → one variation |
| `?variations=URGENCY` | Lowercased before validation; accepted |
| All 5 axes requested | Allowed; one LLM call returns all 5 |
| `?segments=true` LLM fails twice | Category-based fallback; `segments_source: "fallback"` |
| Variations LLM returns extra axes | Filtered to requested set |
| Variations LLM returns subset of requested axes | Retry once; if still incomplete → fallback for the entire variations response |
| Score with `cost_price=0.5` | Price component → 10 (not in sweet spot) |
| Score with empty features, unknown category, empty audience, $5 cost | 0 + 15 + 0 + 10 = 25. Always 0–100 integer. |
| Base content + variations + segments all fail | All three return their fallbacks; HTTP 200 with all three `_source: "fallback"`; never 500 for LLM/network |

## Error model

Unchanged from Phase 1:
- `200` — success (any combination of `llm`/`fallback` per section)
- `422` — request validation (unknown axis, malformed query)
- `500` — only for unexpected bugs

## Testing strategy

1. **`test_scoring.py`** — pure unit tests, ~10 cases:
   - Each rubric component independently (feature buckets, category tiers, audience buckets, price buckets)
   - Boundary values ($20.00 / $60.00 / 30-char audience)
   - Clamping at 100 (high-margin + 5+ features + long audience + sweet-spot price scenario)
   - Lowest plausible score (empty/unknown everything)

2. **`test_variations.py`** — async, mocked `LLMClient` via `pytest-asyncio`:
   - Happy path: requested 2 axes returns exactly 2 variations in requested order
   - Dedupe: requested `[urgency, urgency]` → one variation
   - LLM returns extra axes → filtered down
   - LLM returns missing axes → retry → fallback
   - LLM raises → retry → fallback templates with `{name}` substitution

3. **`test_segments.py`** — async, mocked client:
   - Happy path: returns 3 segments with valid channels
   - Returns 2 or 4 segments → schema invalid → retry → fallback
   - Invalid channel → retry → fallback
   - Category-based fallback for known categories; generic for unknown

4. **`test_generator.py`** (migrated to async) — same 5 Phase 1 cases, now `async def` and using `pytest-asyncio`.

5. **`test_api.py`** (extended):
   - `product_score` always present in response
   - `?variations=urgency,humor` returns 2 variations, source `llm`
   - `?variations=` (empty) → no variations, source `skipped`
   - `?variations=bogus` → 422
   - `?segments=true` returns 3 segments
   - `?segments=false` → no segments, source `skipped`
   - Partial failure: variations LLM raises → 200, base content still `llm`, variations `fallback`
   - All three LLM calls fail → 200, all three `_source: "fallback"`

**New dev dep:** `pytest-asyncio>=0.23` (configured `asyncio_mode = "auto"` in `pyproject.toml`).
**No new prod deps:** Anthropic SDK already exposes `AsyncAnthropic`.

## Future phases (out of scope)

- **Phase 3:** SQLite persistence for generation history, supplier API ingestion, batch generation, store-listing export.
- **Score tuning:** once real conversion data exists, fit rubric weights to actual outcomes.
- **Per-segment ad copy:** combine variations × segments matrix on demand.
- **Caller-controlled segment count.**
