"""src.agents.chunker

Compatibility shim
------------------

The project originally implemented the chunking engine in `chuncker.py` (typo).
Unit tests and downstream callers expect to import from `src.agents.chunker`.

This module keeps the stable import path by re-exporting the public chunking API
from :mod:`src.agents.chuncker`.
"""

from __future__ import annotations

from .chuncker import ChunkingEngine, ChunkValidator, ChunkValidationError

__all__ = [
    "ChunkingEngine",
    "ChunkValidator",
    "ChunkValidationError",
]
