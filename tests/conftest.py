import pytest
from app.schemas import ContentBlock, ProductInput


@pytest.fixture
def sample_input() -> ProductInput:
    return ProductInput(
        product_name="Portable Blender",
        cost_price=15.0,
        category="Kitchen",
        features=["USB rechargeable", "Compact design", "Easy to clean"],
        target_audience="Health-conscious users",
    )


@pytest.fixture
def sample_content() -> ContentBlock:
    return ContentBlock(
        product_title="Blend Anywhere Portable USB Blender",
        description="Make smoothies wherever life takes you.",
        bullets=[
            "Charges via USB anywhere",
            "Fits in any bag",
            "Rinses clean in seconds",
        ],
        ad_copy="Smoothies on the go. No outlet needed.",
        marketing_angle="Convenience + healthy lifestyle",
    )


@pytest.fixture
def sample_variations():
    from app.schemas import AdVariation
    return [
        AdVariation(axis="urgency", ad_copy="Grab one now."),
        AdVariation(axis="humor", ad_copy="It's not magic. It just feels like it."),
    ]


@pytest.fixture
def sample_segments():
    from app.schemas import AudienceSegment
    return [
        AudienceSegment(
            name="A",
            description="d",
            pain_point="p",
            recommended_channel="tiktok",
        ),
        AudienceSegment(
            name="B",
            description="d",
            pain_point="p",
            recommended_channel="instagram",
        ),
        AudienceSegment(
            name="C",
            description="d",
            pain_point="p",
            recommended_channel="facebook",
        ),
    ]
