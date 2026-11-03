"""Lightweight progress-bar helpers with graceful fallback."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm not installed
    _tqdm = None


def progress(iterable: Iterable[T], **kwargs: object) -> Iterable[T]:
    """Return a tqdm-wrapped iterable if available, otherwise the iterable."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


def trange(n: int, **kwargs: object) -> Iterable[int]:
    """Return a tqdm-wrapped range if available, otherwise range."""
    if _tqdm is None:
        return range(n)
    return _tqdm(range(n), **kwargs)
