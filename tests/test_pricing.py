import pytest
from app.pricing.engine import suggest_price, MIN_PRICE, MAX_PRICE


def test_kitchen_markup_charm_pricing():
    # 15 * 2.5 = 37.50 -> round to 38 -> charm 37.99
    assert suggest_price(15.0, "Kitchen") == 37.99


def test_electronics_higher_markup():
    # 20 * 3.0 = 60 -> 59.99
    assert suggest_price(20.0, "Electronics") == 59.99


def test_commodities_lower_markup():
    # 10 * 1.8 = 18 -> 17.99
    assert suggest_price(10.0, "Commodities") == 17.99


def test_unknown_category_uses_default():
    # 12 * 2.5 = 30 -> 29.99
    assert suggest_price(12.0, "Widgets") == 29.99


def test_category_case_insensitive():
    assert suggest_price(15.0, "kitchen") == suggest_price(15.0, "KITCHEN")


def test_floor_at_min_price():
    # 1 * 2.5 = 2.5 -> 3 -> 2.99 -> floor at 9.99
    assert suggest_price(1.0, "Kitchen") == MIN_PRICE


def test_cap_at_max_price():
    # 50000 * 2.5 = 125000 -> capped at 999.99
    assert suggest_price(50000.0, "Kitchen") == MAX_PRICE
