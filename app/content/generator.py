from typing import Literal

from app.llm.base import LLMClient, LLMError
from app.schemas import ContentBlock, ProductInput

Source = Literal["llm", "fallback"]


def _fallback_content(product: ProductInput) -> ContentBlock:
    name = product.product_name.strip() or "This Product"
    audience = (product.target_audience or "everyday shoppers").strip()
    features = [f for f in product.features if f.strip()]

    if len(features) >= 3:
        bullets = [f"{f}" for f in features[:5]]
    else:
        base = features + [
            "Designed for everyday use",
            "Quality you can rely on",
            "Easy to use right out of the box",
        ]
        bullets = base[:3]

    return ContentBlock(
        product_title=f"{name} — Built for {audience}",
        description=(
            f"Discover the {name}, made with {audience} in mind. "
            "Practical, dependable, and ready when you are."
        ),
        bullets=bullets,
        ad_copy=f"Meet the {name}. Simple. Useful. Yours today.",
        marketing_angle="Practical value for everyday life",
    )


def generate_with_fallback(
    client: LLMClient,
    product: ProductInput,
    max_retries: int = 1,
) -> tuple[ContentBlock, Source]:
    attempts = max_retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return client.generate_content(product), "llm"
        except LLMError as exc:
            last_error = exc
            continue
    # Exhausted retries — return deterministic fallback.
    _ = last_error  # available for logging if added later
    return _fallback_content(product), "fallback"
