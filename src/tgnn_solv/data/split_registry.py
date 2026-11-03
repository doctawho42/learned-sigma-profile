"""Canonical helpers for named data split variants."""

from __future__ import annotations

from pathlib import Path


SPLIT_MODE_ALIASES = {
    "solute_scaffold": "solute_scaffold",
    "scaffold": "solute_scaffold",
    "scaffold_split": "solute_scaffold",
    "solute": "solute",
    "random_solute": "solute",
    "random_by_solute": "solute",
    "solvent": "solvent",
    "solvent_holdout": "solvent",
}


SPLIT_MODES = ("solute_scaffold", "solute", "solvent")
SPLIT_FILE_SUFFIXES = {
    "solute_scaffold": "",
    "solute": "_solute",
    "solvent": "_solvent",
}
SPLIT_DISPLAY_NAMES = {
    "solute_scaffold": "Solute scaffold split",
    "solute": "Random solute split",
    "solvent": "Solvent holdout split",
}


def normalize_split_mode(
    split_mode: str | None,
    *,
    default: str = "solute_scaffold",
) -> str:
    """Normalize a split-mode string and validate it."""
    mode = (split_mode or default).strip()
    mode = SPLIT_MODE_ALIASES.get(mode, mode)
    if mode not in SPLIT_FILE_SUFFIXES:
        raise ValueError(
            f"Unknown split mode: {split_mode!r}. Expected one of {SPLIT_MODES}."
        )
    return mode


def resolve_split_modes(split_spec: str | None) -> list[str]:
    """Resolve a comma-separated split spec into canonical split modes."""
    if split_spec is None:
        return ["solute_scaffold"]

    normalized_items = [item.strip() for item in split_spec.split(",") if item.strip()]
    if not normalized_items:
        return ["solute_scaffold"]
    if len(normalized_items) == 1 and normalized_items[0] == "all":
        return list(SPLIT_MODES)

    resolved: list[str] = []
    for item in normalized_items:
        mode = normalize_split_mode(item)
        if mode not in resolved:
            resolved.append(mode)
    return resolved


def get_split_suffix(split_mode: str) -> str:
    """Return the filename suffix for a split mode."""
    return SPLIT_FILE_SUFFIXES[normalize_split_mode(split_mode)]


def get_split_display_name(split_mode: str) -> str:
    """Return the human-readable label for a split mode."""
    return SPLIT_DISPLAY_NAMES[normalize_split_mode(split_mode)]


def split_filename(stem: str, split_mode: str, suffix: str = ".csv") -> str:
    """Build a canonical split filename for one stem and split mode."""
    return f"{stem}{get_split_suffix(split_mode)}{suffix}"


def split_paths(processed_dir: Path, split_mode: str) -> dict[str, Path]:
    """Return train/val/test CSV paths for one split mode."""
    mode = normalize_split_mode(split_mode)
    return {
        split: processed_dir / split_filename(split, mode)
        for split in ("train", "val", "test")
    }


def infer_split_mode_from_path(path: str | Path) -> str | None:
    """Infer the split mode from a canonical split CSV path."""
    stem = Path(path).stem
    for mode, suffix in (
        ("solute", "_solute"),
        ("solvent", "_solvent"),
    ):
        if stem.endswith(suffix):
            return mode
    if stem in {"train", "val", "test"}:
        return "solute_scaffold"
    return None


def build_split_metadata(
    *,
    split_mode: str | None,
    train_data: str | Path | None = None,
    val_data: str | Path | None = None,
    test_data: str | Path | None = None,
) -> dict[str, str | None]:
    """Build a JSON-safe metadata block for split-aware experiment outputs."""
    inferred = (
        split_mode
        or infer_split_mode_from_path(train_data or "")
        or infer_split_mode_from_path(test_data or "")
        or "unknown"
    )
    metadata = {
        "mode": inferred,
        "display_name": (
            get_split_display_name(inferred) if inferred in SPLIT_DISPLAY_NAMES else inferred
        ),
        "train_data": str(train_data) if train_data is not None else None,
        "val_data": str(val_data) if val_data is not None else None,
        "test_data": str(test_data) if test_data is not None else None,
    }
    return metadata
