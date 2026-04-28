from typing import Literal
from pydantic import BaseModel, Field, field_validator

AdAxis = Literal["urgency", "aspirational", "social_proof", "problem_solution", "humor"]
SectionSource = Literal["llm", "fallback", "skipped"]
RecommendedChannel = Literal[
    "tiktok", "instagram", "facebook", "youtube", "google_ads", "email"
]


class ProductInput(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    cost_price: float = Field(gt=0, le=100000)
    category: str = Field(min_length=1, max_length=50)
    features: list[str] = Field(default_factory=list, max_length=20)
    target_audience: str = Field(default="", max_length=200)

    @field_validator("product_name", "category", mode="before")
    @classmethod
    def _require_non_blank(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class ContentBlock(BaseModel):
    product_title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    bullets: list[str] = Field(min_length=3, max_length=5)
    ad_copy: str = Field(min_length=1, max_length=500)
    marketing_angle: str = Field(min_length=1, max_length=200)


class AdVariation(BaseModel):
    axis: AdAxis
    ad_copy: str = Field(min_length=1, max_length=500)


class AudienceSegment(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    pain_point: str = Field(min_length=1, max_length=300)
    recommended_channel: RecommendedChannel


class ProductResponse(BaseModel):
    selling_price: float
    product_score: int = Field(ge=0, le=100)
    product_title: str
    description: str
    bullets: list[str]
    ad_copy: str
    marketing_angle: str
    source: Literal["llm", "fallback"]
    ad_variations: list[AdVariation] | None = None
    variations_source: SectionSource = "skipped"
    audience_segments: list[AudienceSegment] | None = None
    segments_source: SectionSource = "skipped"
