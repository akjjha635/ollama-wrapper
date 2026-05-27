from __future__ import annotations

from typing import Callable

from .backends import FaissDenseBackend, LinearDenseBackend
from .contracts import DenseBackend


class DenseBackendRegistry:
    """Factory registry for retrieval dense backends."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], DenseBackend]] = {
            "linear": LinearDenseBackend,
            "faiss": FaissDenseBackend,
        }

    def register(self, name: str, factory: Callable[[], DenseBackend]) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Backend name cannot be empty.")
        self._factories[key] = factory

    def create(self, name: str) -> DenseBackend:
        key = str(name).strip().lower()
        if key not in self._factories:
            raise KeyError(key)
        return self._factories[key]()

    def has(self, name: str) -> bool:
        return str(name).strip().lower() in self._factories


def create_default_dense_backend_registry() -> DenseBackendRegistry:
    return DenseBackendRegistry()
