"""Helpers for grouped namespace wrappers over the legacy flat package layout."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def reexport_module(module_name: str) -> tuple[dict[str, Any], list[str]]:
    """Return public attributes from a legacy module for namespace wrappers."""
    module = import_module(module_name)
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [name for name in dir(module) if not name.startswith("_")]
    namespace = {name: getattr(module, name) for name in exported}
    return namespace, list(exported)
