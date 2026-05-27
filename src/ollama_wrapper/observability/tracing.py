from __future__ import annotations

from collections import deque

from .contracts import RequestTrace


class RequestTraceStore:
    """In-memory bounded trace store for debug visibility."""

    def __init__(self, max_size: int = 500) -> None:
        self._items: deque[RequestTrace] = deque(maxlen=max(10, max_size))

    def add(self, trace: RequestTrace) -> None:
        self._items.append(trace)

    def latest(self, limit: int = 20) -> list[RequestTrace]:
        n = max(1, min(limit, len(self._items)))
        return list(self._items)[-n:][::-1]
