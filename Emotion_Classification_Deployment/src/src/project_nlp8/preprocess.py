"""Proxy module that re-exports cleaning helpers from pipeline.py
   so we can measure coverage only for these three functions."""
from .textclean import (
    optimized_preprocessor,
    remove_placeholders,
    split_into_sentences,
)

__all__ = [
    "optimized_preprocessor",
    "remove_placeholders",
    "split_into_sentences",
]
