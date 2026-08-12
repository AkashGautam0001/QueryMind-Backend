from collections.abc import AsyncIterator


class LLMProvider:
    async def stream(
        self, system_prompt: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def complete(
        self, system_prompt: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        """Single-shot non-streaming completion for LLM-judge calls."""
        raise NotImplementedError
