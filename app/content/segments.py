import sys
from typing import Literal

from app.llm.base import LLMClient, LLMError
from app.schemas import AudienceSegment, ProductInput

Source = Literal["llm", "fallback"]


GENERIC_FALLBACK: list[AudienceSegment] = [
    AudienceSegment(
        name="Practical buyers",
        description="Shoppers who research before they buy and value reliability.",
        pain_point="Tired of overpriced products that underdeliver on basics.",
        recommended_channel="google_ads",
    ),
    AudienceSegment(
        name="Trend-driven shoppers",
        description="Early adopters who follow social trends and want what's new.",
        pain_point="Fear of missing out on what everyone else is using.",
        recommended_channel="tiktok",
    ),
    AudienceSegment(
        name="Gift buyers",
        description="People shopping for friends and family on birthdays and holidays.",
        pain_point="Difficulty finding meaningful, practical gifts that get used.",
        recommended_channel="facebook",
    ),
]


CATEGORY_FALLBACKS: dict[str, list[AudienceSegment]] = {
    "kitchen": [
        AudienceSegment(
            name="Time-pressed home cooks",
            description="Busy professionals who want fast meal prep without giving up quality.",
            pain_point="Limited time after work to cook from scratch.",
            recommended_channel="instagram",
        ),
        AudienceSegment(
            name="Health-conscious shoppers",
            description="People prioritizing wellness in everyday routines.",
            pain_point="Few convenient kitchen tools that fit a healthy lifestyle.",
            recommended_channel="tiktok",
        ),
        AudienceSegment(
            name="Gift buyers",
            description="Shoppers looking for thoughtful, practical kitchen gifts.",
            pain_point="Hard to find practical gifts that feel personal.",
            recommended_channel="facebook",
        ),
    ],
    "electronics": [
        AudienceSegment(
            name="Tech enthusiasts",
            description="Early adopters who follow gadget reviews and product launches.",
            pain_point="Wanting the latest, most capable gear before the mainstream.",
            recommended_channel="youtube",
        ),
        AudienceSegment(
            name="Productivity seekers",
            description="Professionals optimizing their work-from-home setup.",
            pain_point="Generic gear that doesn't keep up with their workflow.",
            recommended_channel="google_ads",
        ),
        AudienceSegment(
            name="Gift buyers",
            description="People shopping electronics gifts for tech-savvy friends.",
            pain_point="Hard to pick gear someone will actually use.",
            recommended_channel="facebook",
        ),
    ],
    "beauty": [
        AudienceSegment(
            name="Skincare devotees",
            description="People with daily routines who research ingredients before buying.",
            pain_point="Most products overpromise and underdeliver on real skin concerns.",
            recommended_channel="instagram",
        ),
        AudienceSegment(
            name="Trend-following shoppers",
            description="Beauty fans who follow viral product reviews on social.",
            pain_point="Fear of missing out on the next breakthrough product.",
            recommended_channel="tiktok",
        ),
        AudienceSegment(
            name="Gift buyers",
            description="Shoppers buying beauty gifts for partners or friends.",
            pain_point="Picking the right shade, scent, or formula for someone else.",
            recommended_channel="facebook",
        ),
    ],
    "apparel": [
        AudienceSegment(
            name="Style-conscious shoppers",
            description="People keeping up with current looks and seasonal trends.",
            pain_point="Mass-market clothing that fits poorly or feels generic.",
            recommended_channel="instagram",
        ),
        AudienceSegment(
            name="Comfort-first buyers",
            description="People who prioritize fabric quality and fit over brand.",
            pain_point="Most fast-fashion items don't last past a few washes.",
            recommended_channel="google_ads",
        ),
        AudienceSegment(
            name="Gift buyers",
            description="People buying clothing gifts for partners or family.",
            pain_point="Sizing is hard to guess, and tastes are personal.",
            recommended_channel="facebook",
        ),
    ],
}


def _fallback(product: ProductInput) -> list[AudienceSegment]:
    return CATEGORY_FALLBACKS.get(
        product.category.lower().strip(), GENERIC_FALLBACK
    )


async def run(
    client: LLMClient,
    product: ProductInput,
    max_retries: int = 1,
) -> tuple[list[AudienceSegment], Source]:
    attempts = max_retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            segments = await client.generate_segments(product)
        except LLMError as exc:
            last_error = exc
            continue
        if len(segments) == 3:
            return segments, "llm"
        last_error = LLMError(f"expected 3 segments, got {len(segments)}")
    print(
        f"[WARN] Segments LLM failed after {attempts} attempt(s): {last_error}",
        file=sys.stderr,
    )
    return _fallback(product), "fallback"
