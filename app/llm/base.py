from typing import Protocol
from app.schemas import ContentBlock, ProductInput


class LLMError(Exception):
    """Raised when the LLM provider fails or returns invalid output."""


class LLMClient(Protocol):
    async def generate_content(self, product: ProductInput) -> ContentBlock:
        """Generate structured product content. Raises LLMError on failure."""
        ...
