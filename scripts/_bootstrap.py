"""Shared bootstrap for scripts/ directory.

Usage at the top of any script:
    import _bootstrap  # noqa: F401
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path so `import tgnn_solv` works even without
# `pip install -e .`
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if _src_dir.is_dir() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Also make repo root available for configs/ and data/ resolution
REPO_ROOT = _repo_root


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to repo root if not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
