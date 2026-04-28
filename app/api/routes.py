import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_llm_client
from app.config import Settings, get_settings
from app.content import competitive as competitive_mod
from app.content import segments as segments_mod
from app.content import variations as variations_mod
from app.content.generator import generate_with_fallback
from app.llm.base import LLMClient
from app.pricing.engine import suggest_price
from app.scoring.engine import compute_score
from app.schemas import AdAxis, ProductInput, ProductResponse, SectionSource

router = APIRouter()

VALID_AXES: tuple[str, ...] = (
    "urgency",
    "aspirational",
    "social_proof",
    "problem_solution",
    "humor",
)


def _parse_variations(raw: str | None) -> list[AdAxis]:
    if not raw:
        return []
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    invalid = [p for p in parts if p not in VALID_AXES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "variations",
                "invalid": invalid,
                "valid_axes": list(VALID_AXES),
            },
        )
    return parts  # type: ignore[return-value]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate-product", response_model=ProductResponse)
async def generate_product(
    product: ProductInput,
    variations: Annotated[str | None, Query()] = None,
    segments: Annotated[bool, Query()] = False,
    compete: Annotated[bool, Query()] = False,
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> ProductResponse:
    axes = _parse_variations(variations)
    price = suggest_price(product.cost_price, product.category)
    score = compute_score(product)

    base_task = generate_with_fallback(
        llm, product, max_retries=settings.llm_max_retries
    )
    coros: dict[str, Any] = {"base": base_task}
    if axes:
        coros["variations"] = variations_mod.run(
            llm, product, axes, max_retries=settings.llm_max_retries
        )
    if segments:
        coros["segments"] = segments_mod.run(
            llm, product, max_retries=settings.llm_max_retries
        )
    if compete:
        coros["compete"] = competitive_mod.run(
            llm, product, max_retries=settings.llm_max_retries
        )

    keys = list(coros.keys())
    results = await asyncio.gather(*coros.values())
    out = dict(zip(keys, results))

    content, content_source = out["base"]
    var_list = None
    var_source: SectionSource = "skipped"
    if "variations" in out:
        var_list, var_source = out["variations"]
    seg_list = None
    seg_source: SectionSource = "skipped"
    if "segments" in out:
        seg_list, seg_source = out["segments"]
    comp_intel = None
    comp_source: SectionSource = "skipped"
    if "compete" in out:
        comp_intel, comp_source = out["compete"]

    return ProductResponse(
        selling_price=price,
        product_score=score,
        product_title=content.product_title,
        description=content.description,
        bullets=content.bullets,
        ad_copy=content.ad_copy,
        marketing_angle=content.marketing_angle,
        source=content_source,
        ad_variations=var_list,
        variations_source=var_source,
        audience_segments=seg_list,
        segments_source=seg_source,
        competitive_intel=comp_intel,
        compete_source=comp_source,
    )
