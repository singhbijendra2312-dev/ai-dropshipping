from anthropic import AsyncAnthropic, APIError
from pydantic import ValidationError

from app.content.prompts import CONTENT_TOOL, SYSTEM_PROMPT, build_user_message
from app.llm.base import LLMError
from app.schemas import ContentBlock, ProductInput


class AnthropicLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5",
        timeout_seconds: float = 20.0,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    async def generate_content(self, product: ProductInput) -> ContentBlock:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[CONTENT_TOOL],
                tool_choice={"type": "tool", "name": "submit_product_content"},
                messages=[
                    {"role": "user", "content": build_user_message(product)}
                ],
            )
        except APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_product_content":
                try:
                    return ContentBlock.model_validate(block.input)
                except ValidationError as exc:
                    raise LLMError(f"LLM output failed schema: {exc}") from exc

        raise LLMError("LLM did not invoke the submit_product_content tool")

    async def generate_variations(
        self, product, axes
    ):
        from app.content.prompts import (
            VARIATIONS_SYSTEM_PROMPT,
            VARIATIONS_TOOL,
            build_variations_user_message,
        )
        from app.schemas import AdVariation

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": VARIATIONS_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[VARIATIONS_TOOL],
                tool_choice={"type": "tool", "name": "submit_ad_variations"},
                messages=[
                    {
                        "role": "user",
                        "content": build_variations_user_message(product, list(axes)),
                    }
                ],
            )
        except APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_ad_variations":
                try:
                    raw_list = block.input.get("variations", [])
                    return [AdVariation.model_validate(v) for v in raw_list]
                except ValidationError as exc:
                    raise LLMError(f"Variations output failed schema: {exc}") from exc

        raise LLMError("LLM did not invoke the submit_ad_variations tool")
