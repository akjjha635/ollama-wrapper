from __future__ import annotations

from typing import Any, AsyncIterator, Iterable

from ollama import AsyncClient, Client

from .contracts import LLMChatResult, LLMMessage, LLMProvider, LLMUsage


class OllamaProvider(LLMProvider):
    """LLM provider adapter for local Ollama deployments."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self._sync_client = Client(host=host)
        self._async_client = AsyncClient(host=host)

    def provider_name(self) -> str:
        return "ollama"

    @staticmethod
    def _extract_usage(response: Any) -> LLMUsage:
        input_tokens = getattr(response, "prompt_eval_count", None)
        output_tokens = getattr(response, "eval_count", None)
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw={
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            },
        )

    def chat(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": model, "messages": payload_messages}
        if response_schema:
            kwargs["format"] = response_schema
        if options:
            kwargs["options"] = options

        response = self._sync_client.chat(**kwargs)
        return LLMChatResult(
            content=response.message.content,
            model=model,
            usage=self._extract_usage(response),
            raw={"provider": "ollama"},
        )

    async def chat_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": model, "messages": payload_messages}
        if response_schema:
            kwargs["format"] = response_schema
        if options:
            kwargs["options"] = options

        response = await self._async_client.chat(**kwargs)
        return LLMChatResult(
            content=response.message.content,
            model=model,
            usage=self._extract_usage(response),
            raw={"provider": "ollama"},
        )

    def embed(self, model: str, input_text: str, options: dict[str, Any] | None = None) -> list[float]:
        kwargs: dict[str, Any] = {"model": model, "input": input_text}
        if options:
            kwargs["options"] = options
        response = self._sync_client.embed(**kwargs)
        return list(response.embeddings[0])

    async def embed_async(
        self,
        model: str,
        input_text: str,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        kwargs: dict[str, Any] = {"model": model, "input": input_text}
        if options:
            kwargs["options"] = options
        response = await self._async_client.embed(**kwargs)
        return list(response.embeddings[0])

    def health_check(self) -> bool:
        try:
            self._sync_client.list()
            return True
        except Exception:
            return False

    def chat_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterable[str]:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": model, "messages": payload_messages, "stream": True}
        if response_schema:
            kwargs["format"] = response_schema
        if options:
            kwargs["options"] = options

        for chunk in self._sync_client.chat(**kwargs):
            content = getattr(getattr(chunk, "message", None), "content", "")
            if content:
                yield content

    async def chat_stream_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": model, "messages": payload_messages, "stream": True}
        if response_schema:
            kwargs["format"] = response_schema
        if options:
            kwargs["options"] = options

        stream = await self._async_client.chat(**kwargs)
        async for chunk in stream:
            content = getattr(getattr(chunk, "message", None), "content", "")
            if content:
                yield content
