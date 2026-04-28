from fastapi import Depends

from app.config import Settings, get_settings
from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.base import LLMClient


def get_llm_client(
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    return AnthropicLLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
