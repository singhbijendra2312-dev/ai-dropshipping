CATEGORY_MARKUP: dict[str, float] = {
    "electronics": 3.0,
    "kitchen": 2.5,
    "beauty": 2.8,
    "apparel": 2.4,
    "commodities": 1.8,
    "home": 2.5,
}
DEFAULT_MARKUP = 2.5
MIN_PRICE = 9.99
MAX_PRICE = 999.99


def suggest_price(cost_price: float, category: str) -> float:
    markup = CATEGORY_MARKUP.get(category.lower().strip(), DEFAULT_MARKUP)
    raw = cost_price * markup
    rounded = round(raw)
    charm = max(rounded - 0.01, MIN_PRICE)
    return min(charm, MAX_PRICE)
