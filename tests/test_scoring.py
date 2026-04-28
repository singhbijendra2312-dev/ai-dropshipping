import pytest
from app.schemas import ProductInput
from app.scoring.engine import compute_score


def _input(**overrides) -> ProductInput:
    base = dict(
        product_name="Test",
        cost_price=30.0,
        category="Kitchen",
        features=[],
        target_audience="",
    )
    base.update(overrides)
    return ProductInput(**base)


def test_no_features_zero_feature_component():
    # 0 (features) + 15 (kitchen=mid) + 0 (audience) + 25 (price sweet spot) = 40
    assert compute_score(_input(features=[])) == 40


def test_one_to_two_features_15():
    # 15 + 15 + 0 + 25 = 55
    assert compute_score(_input(features=["a", "b"])) == 55


def test_three_to_four_features_25():
    # 25 + 15 + 0 + 25 = 65
    assert compute_score(_input(features=["a", "b", "c"])) == 65


def test_five_plus_features_35():
    # 35 + 15 + 0 + 25 = 75
    assert compute_score(_input(features=["a", "b", "c", "d", "e", "f"])) == 75


def test_high_margin_category():
    # 0 + 25 (electronics=high) + 0 + 25 = 50
    assert compute_score(_input(category="Electronics")) == 50


def test_low_margin_category():
    # 0 + 5 (commodities=low) + 0 + 25 = 30
    assert compute_score(_input(category="Commodities")) == 30


def test_unknown_category_default():
    # 0 + 15 (unknown=default) + 0 + 25 = 40
    assert compute_score(_input(category="Toys")) == 40


def test_short_audience_8():
    # 0 + 15 + 8 (audience<=30) + 25 = 48
    assert compute_score(_input(target_audience="busy parents")) == 48


def test_long_audience_15():
    # >30 chars audience
    audience = "Health-conscious millennial parents in urban areas"
    assert len(audience) > 30
    # 0 + 15 + 15 + 25 = 55
    assert compute_score(_input(target_audience=audience)) == 55


def test_price_outside_sweet_spot():
    # 0 + 15 + 0 + 10 (price not in sweet spot) = 25
    assert compute_score(_input(cost_price=10.0)) == 25
    assert compute_score(_input(cost_price=80.0)) == 25


def test_price_at_sweet_spot_boundaries():
    # $20 and $60 inclusive
    assert compute_score(_input(cost_price=20.0)) == 40
    assert compute_score(_input(cost_price=60.0)) == 40


def test_max_score_clamped_to_100():
    # 35 + 25 + 15 + 25 = 100
    score = compute_score(_input(
        features=["a", "b", "c", "d", "e"],
        category="Electronics",
        target_audience="Health-conscious millennials in urban environments",
        cost_price=40.0,
    ))
    assert score == 100


def test_minimum_realistic_score():
    # 0 + 15 (unknown) + 0 + 10 = 25
    score = compute_score(_input(
        features=[],
        category="Toys",
        target_audience="",
        cost_price=5.0,
    ))
    assert score == 25
