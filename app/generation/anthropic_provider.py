from collections.abc import AsyncIterator
from anthropic import AsyncAnthropic

from app.generation.llm_provider import LLMProvider

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model : str):
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def stream(self, system_prompt: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[str]:
        async with self._client.message.stream(
            model=self.model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    
    async def complete(
        self, system_prompt: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def close(self) -> None:
        await self._client.close()