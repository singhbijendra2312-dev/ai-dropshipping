from app.schemas import ProductInput

CATEGORY_TIER: dict[str, str] = {
    "electronics": "high",
    "beauty": "high",
    "kitchen": "mid",
    "home": "mid",
    "apparel": "mid",
    "commodities": "low",
}
TIER_POINTS: dict[str, int] = {"high": 25, "mid": 15, "low": 5}
UNKNOWN_CATEGORY_POINTS = 15
PRICE_SWEET_SPOT_MIN = 20.0
PRICE_SWEET_SPOT_MAX = 60.0


def _feature_score(features: list[str]) -> int:
    n = len([f for f in features if f.strip()])
    if n == 0:
        return 0
    if n <= 2:
        return 15
    if n <= 4:
        return 25
    return 35


def _category_score(category: str) -> int:
    tier = CATEGORY_TIER.get(category.lower().strip())
    if tier is None:
        return UNKNOWN_CATEGORY_POINTS
    return TIER_POINTS[tier]


def _audience_score(audience: str) -> int:
    cleaned = audience.strip()
    if not cleaned:
        return 0
    if len(cleaned) <= 30:
        return 8
    return 15


def _price_score(cost_price: float) -> int:
    if PRICE_SWEET_SPOT_MIN <= cost_price <= PRICE_SWEET_SPOT_MAX:
        return 25
    return 10


def compute_score(product: ProductInput) -> int:
    total = (
        _feature_score(product.features)
        + _category_score(product.category)
        + _audience_score(product.target_audience)
        + _price_score(product.cost_price)
    )
    return min(total, 100)
