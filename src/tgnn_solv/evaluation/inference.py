"""Grouped namespace wrapper around `tgnn_solv.inference`."""

from __future__ import annotations

from .._namespace import reexport_module as _reexport_module

_globals, __all__ = _reexport_module("tgnn_solv.inference")
globals().update(_globals)
