from fastapi import APIRouter, Depends

from app.api.deps import get_llm_client
from app.config import Settings, get_settings
from app.content.generator import generate_with_fallback
from app.llm.base import LLMClient
from app.pricing.engine import suggest_price
from app.schemas import ProductInput, ProductResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate-product", response_model=ProductResponse)
def generate_product(
    product: ProductInput,
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> ProductResponse:
    price = suggest_price(product.cost_price, product.category)
    content, source = generate_with_fallback(
        llm, product, max_retries=settings.llm_max_retries
    )
    return ProductResponse(
        selling_price=price,
        product_title=content.product_title,
        description=content.description,
        bullets=content.bullets,
        ad_copy=content.ad_copy,
        marketing_angle=content.marketing_angle,
        source=source,
    )
