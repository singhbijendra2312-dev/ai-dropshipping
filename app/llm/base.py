from typing import Protocol
from app.schemas import AdAxis, AdVariation, ContentBlock, ProductInput


class LLMError(Exception):
    """Raised when the LLM provider fails or returns invalid output."""


class LLMClient(Protocol):
    async def generate_content(self, product: ProductInput) -> ContentBlock:
        """Generate structured product content. Raises LLMError on failure."""
        ...

    async def generate_variations(
        self, product: ProductInput, axes: list[AdAxis]
    ) -> list[AdVariation]:
        """Generate ad variations for the given axes. Raises LLMError on failure."""
        ...
