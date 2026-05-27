from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str


@dataclass(slots=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMChatResult:
    content: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Provider-agnostic LLM contract used by orchestration/server layers."""

    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        raise NotImplementedError

    @abstractmethod
    async def chat_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMChatResult:
        raise NotImplementedError

    @abstractmethod
    def embed(self, model: str, input_text: str, options: dict[str, Any] | None = None) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_async(
        self,
        model: str,
        input_text: str,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError

    def chat_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterable[str]:
        raise NotImplementedError("Streaming chat is not implemented for this provider.")

    async def chat_stream_async(
        self,
        model: str,
        messages: list[LLMMessage],
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Async streaming chat is not implemented for this provider.")
