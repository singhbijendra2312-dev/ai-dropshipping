# AI Dropshipping

Stateless FastAPI service: rule-based pricing + LLM-generated product content.

## Setup

```bash
uv sync
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

## Test

```bash
uv run pytest
```

## Smoke test (real API)

```bash
uv run python scripts/smoke.py
```

## Example request

```bash
curl -X POST http://localhost:8000/generate-product \
  -H "Content-Type: application/json" \
  -d @data/sample_products.json
```
