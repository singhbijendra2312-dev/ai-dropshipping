from typing import Literal
from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    cost_price: float = Field(gt=0, le=100000)
    category: str = Field(min_length=1, max_length=50)
    features: list[str] = Field(default_factory=list, max_length=20)
    target_audience: str = Field(default="", max_length=200)


class ContentBlock(BaseModel):
    product_title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    bullets: list[str] = Field(min_length=3, max_length=5)
    ad_copy: str = Field(min_length=1, max_length=500)
    marketing_angle: str = Field(min_length=1, max_length=200)


class ProductResponse(BaseModel):
    selling_price: float
    product_title: str
    description: str
    bullets: list[str]
    ad_copy: str
    marketing_angle: str
    source: Literal["llm", "fallback"]
