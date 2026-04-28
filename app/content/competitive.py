import sys
from typing import Literal

from app.llm.base import LLMClient, LLMError
from app.schemas import CompetitiveIntel, ProductInput

Source = Literal["llm", "fallback"]


DIFFERENTIATION_FALLBACKS: dict[str, list[str]] = {
    "kitchen": [
        "Faster setup than competitors",
        "Easier to clean",
        "More compact storage",
    ],
    "electronics": [
        "Longer battery life",
        "Better build quality",
        "Simpler setup",
    ],
    "beauty": [
        "Cleaner ingredient list",
        "Better value per use",
        "Travel-friendly format",
    ],
    "apparel": [
        "Better fabric quality",
        "More inclusive sizing",
        "More versatile styling",
    ],
}


WEAKNESSES_FALLBACKS: dict[str, list[str]] = {
    "kitchen": [
        "Hard to clean",
        "Underpowered motor",
        "Loud during operation",
    ],
    "electronics": [
        "Short battery life",
        "Confusing setup",
        "Flimsy build",
    ],
    "beauty": [
        "Greasy texture",
        "Harsh fragrance",
        "Small product size",
    ],
    "apparel": [
        "Inconsistent sizing",
        "Fabric pills quickly",
        "Poor color retention",
    ],
}


GENERIC_DIFFERENTIATION_FALLBACK: list[str] = [
    "Better value at a similar price point",
    "Designed for everyday practicality",
    "Easier to use right out of the box",
]

GENERIC_WEAKNESSES_FALLBACK: list[str] = [
    "Quality inconsistent across batches",
    "Setup or unboxing more confusing than expected",
    "Hidden costs after purchase (shipping, accessories)",
]


def _fallback(product: ProductInput) -> CompetitiveIntel:
    cat = product.category.lower().strip()
    return CompetitiveIntel(
        price_benchmarks=None,
        competitors=[],
        differentiation_suggestions=DIFFERENTIATION_FALLBACKS.get(
            cat, GENERIC_DIFFERENTIATION_FALLBACK
        ),
        common_weaknesses=WEAKNESSES_FALLBACKS.get(cat, GENERIC_WEAKNESSES_FALLBACK),
    )


async def run(
    client: LLMClient,
    product: ProductInput,
    max_retries: int = 1,
) -> tuple[CompetitiveIntel, Source]:
    attempts = max_retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            intel = await client.generate_competitive_intel(product)
        except LLMError as exc:
            last_error = exc
            continue
        return intel, "llm"
    print(
        f"[WARN] Competitive intel LLM failed after {attempts} attempt(s): "
        f"{last_error}",
        file=sys.stderr,
    )
    return _fallback(product), "fallback"
