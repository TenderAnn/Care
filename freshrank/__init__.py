"""Convenience package wrapper for the freshrank service modules."""
from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["service", "scoring", "config"]


def __getattr__(name: str) -> ModuleType | Any:
    if name in __all__:
        return _import_module(f"freshrank.{name}")
    raise AttributeError(name)


# Ensure sub-packages under `freshrank/freshrank` remain importable when this shim is used.
__path__ = [str(Path(__file__).resolve().parent / "freshrank"), *__path__]
