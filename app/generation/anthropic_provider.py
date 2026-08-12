from app.config import get_settings
from app.generation.anthropic_provider import AnthropicProvider

_llm_provider: AnthropicProvider | None = None


def get_llm_provider() -> AnthropicProvider:
    global _llm_provider
    if _llm_provider is None:
        settings = get_settings()
        _llm_provider = AnthropicProvider(
            api_key=settings.anthropic_api_key, model=settings.generation_model
        )
    return _llm_provider


async def close_llm_provider() -> None:
    global _llm_provider
    if _llm_provider is not None:
        await _llm_provider.close()
        _llm_provider = None
