from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

from .contracts import LLMChatResult, LLMMessage, LLMProvider, LLMUsage


class OpenAIProvider(LLMProvider):
    """Cloud-provider adapter scaffold for OpenAI-compatible APIs.

    The constructor accepts injected callables so this module does not force an
    `openai` dependency until you decide to enable it in production.
    """

    def __init__(
        self,
        chat_fn: Callable[..., Any] | None = None,
        chat_async_fn: Callable[..., Awaitable[Any]] | None = None,
        chat_stream_fn: Callable[..., Iterable[str]] | None = None,
        chat_stream_async_fn: Callable[..., AsyncIterator[str]] | None = None,
        embed_fn: Callable[..., Any] | None = None,
        embed_async_fn: Callable[..., Awaitable[Any]] | None = None,
        health_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._chat_fn = chat_fn
        self._chat_async_fn = chat_async_fn
        self._chat_stream_fn = chat_stream_fn
        self._chat_stream_async_fn = chat_stream_async_fn
        self._embed_fn = embed_fn
        self._embed_async_fn = embed_async_fn
        self._health_fn = health_fn

    def provider_name(self) -> str:
        return "openai"

    def chat(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        if self._chat_fn is None:
            raise NotImplementedError("OpenAI sync chat adapter is not configured yet.")

        response = self._chat_fn(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_schema=response_schema,
            options=options or {},
        )
        return LLMChatResult(content=str(response), model=model, usage=LLMUsage(raw={}), raw={"provider": "openai"})

    async def chat_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        if self._chat_async_fn is None:
            raise NotImplementedError("OpenAI async chat adapter is not configured yet.")

        response = await self._chat_async_fn(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_schema=response_schema,
            options=options or {},
        )
        return LLMChatResult(content=str(response), model=model, usage=LLMUsage(raw={}), raw={"provider": "openai"})

    def embed(self, model: str, input_text: str, options: dict[str, Any] | None = None) -> list[float]:
        if self._embed_fn is None:
            raise NotImplementedError("OpenAI sync embeddings adapter is not configured yet.")
        return list(self._embed_fn(model=model, input_text=input_text, options=options or {}))

    async def embed_async(
        self,
        model: str,
        input_text: str,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        if self._embed_async_fn is None:
            raise NotImplementedError("OpenAI async embeddings adapter is not configured yet.")
        return list(await self._embed_async_fn(model=model, input_text=input_text, options=options or {}))

    def health_check(self) -> bool:
        if self._health_fn is None:
            return False
        return bool(self._health_fn())

    def chat_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterable[str]:
        if self._chat_stream_fn is None:
            raise NotImplementedError("OpenAI stream chat adapter is not configured yet.")
        return self._chat_stream_fn(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_schema=response_schema,
            options=options or {},
        )

    async def chat_stream_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if self._chat_stream_async_fn is None:
            raise NotImplementedError("OpenAI async stream chat adapter is not configured yet.")
        async for chunk in self._chat_stream_async_fn(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_schema=response_schema,
            options=options or {},
        ):
            yield chunk
