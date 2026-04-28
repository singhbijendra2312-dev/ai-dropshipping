"""Manual end-to-end check: hits real Anthropic API. Requires .env."""
import json
from pathlib import Path

from app.config import get_settings
from app.content.generator import generate_with_fallback
from app.llm.anthropic_client import AnthropicLLMClient
from app.pricing.engine import suggest_price
from app.schemas import ProductInput


def main() -> None:
    settings = get_settings()
    sample = json.loads(Path("data/sample_products.json").read_text())
    product = ProductInput(**sample)
    client = AnthropicLLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    price = suggest_price(product.cost_price, product.category)
    content, source = generate_with_fallback(
        client, product, max_retries=settings.llm_max_retries
    )
    print(f"selling_price: {price}")
    print(f"source: {source}")
    print(content.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
