import sys
from typing import Literal

from app.llm.base import LLMClient, LLMError
from app.schemas import AdAxis, AdVariation, ProductInput

Source = Literal["llm", "fallback"]

FALLBACK_TEMPLATES: dict[str, str] = {
    "urgency": "Limited stock — order today before they're gone.",
    "aspirational": "Become the version of you that already has it.",
    "social_proof": (
        "Thousands of customers can't be wrong. See why everyone is talking about it."
    ),
    "problem_solution": "Tired of {pain}? Meet {name} — the simple fix.",
    "humor": "It's not magic. It just feels like it.",
}


def _normalize_axes(axes: list[AdAxis]) -> list[AdAxis]:
    seen: set[str] = set()
    out: list[AdAxis] = []
    for a in axes:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def _filter_to_requested(
    raw: list[AdVariation], axes: list[AdAxis]
) -> list[AdVariation] | None:
    by_axis: dict[str, AdVariation] = {v.axis: v for v in raw}
    if not all(a in by_axis for a in axes):
        return None
    return [by_axis[a] for a in axes]


def _fallback(product: ProductInput, axes: list[AdAxis]) -> list[AdVariation]:
    name = product.product_name.strip() or "this product"
    pain = "the daily hassle"
    out: list[AdVariation] = []
    for a in axes:
        template = FALLBACK_TEMPLATES[a]
        ad_copy = template.format(name=name, pain=pain)[:500]
        out.append(AdVariation(axis=a, ad_copy=ad_copy))
    return out


async def run(
    client: LLMClient,
    product: ProductInput,
    axes: list[AdAxis],
    max_retries: int = 1,
) -> tuple[list[AdVariation], Source]:
    normalized = _normalize_axes(axes)
    attempts = max_retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            raw = await client.generate_variations(product, normalized)
        except LLMError as exc:
            last_error = exc
            continue
        filtered = _filter_to_requested(raw, normalized)
        if filtered is not None:
            return filtered, "llm"
        last_error = LLMError("variations missing some requested axes")
    print(
        f"[WARN] Variations LLM failed after {attempts} attempt(s): {last_error}",
        file=sys.stderr,
    )
    return _fallback(product, normalized), "fallback"
