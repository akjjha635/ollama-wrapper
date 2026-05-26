# src/ollama_wrapper/__init__.py

from .core import OllamaWrapper

# Define the explicit public API exports for wildcard imports
__all__ = ["OllamaWrapper"]