from typing import Protocol
from app.schemas import ProductInput, ContentBlock


class LLMError(Exception):
    """Raised when the LLM provider fails or returns invalid output."""


class LLMClient(Protocol):
    def generate_content(self, product: ProductInput) -> ContentBlock:
        """Generate structured product content. Raises LLMError on failure."""
        ...
