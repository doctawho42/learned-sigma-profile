"""Grouped namespace wrapper around `tgnn_solv.model`."""

from __future__ import annotations

from .._namespace import reexport_module as _reexport_module

_globals, __all__ = _reexport_module("tgnn_solv.model")
globals().update(_globals)
