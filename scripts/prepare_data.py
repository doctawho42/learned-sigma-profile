#!/usr/bin/env python3
"""CLI data preparation pipeline matching `notebooks/01_prepare_data.ipynb`."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import _bootstrap  # noqa: F401
import pandas as pd

SRC_ROOT = _bootstrap.REPO_ROOT / "src"
DATA_MODULE_ROOT = SRC_ROOT / "tgnn_solv" / "data"


def _ensure_namespace_packages() -> None:
    """Create lightweight namespace packages without importing package `__init__`."""
    if "tgnn_solv" not in sys.modules:
        package = types.ModuleType("tgnn_solv")
        package.__path__ = [str(SRC_ROOT / "tgnn_solv")]
        sys.modules["tgnn_solv"] = package

    if "tgnn_solv.data" not in sys.modules:
        package = types.ModuleType("tgnn_solv.data")
        package.__path__ = [str(DATA_MODULE_ROOT)]
        sys.modules["tgnn_solv.data"] = package


def _load_module(module_name: str, module_path: Path) -> types.ModuleType:
    """Load a module from disk without importing package-level side effects."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace_packages()
data_utils = _load_module("tgnn_solv.data.utils", DATA_MODULE_ROOT / "utils.py")
data_sources = _load_module("tgnn_solv.data.sources", DATA_MODULE_ROOT / "sources.py")
builder_module = _load_module("tgnn_solv.data.builder", DATA_MODULE_ROOT / "builder.py")
split_module = _load_module("tgnn_solv.data.split", DATA_MODULE_ROOT / "split.py")
split_registry_module = _load_module(
    "tgnn_solv.data.split_registry",
    DATA_MODULE_ROOT / "split_registry.py",
)
config_module = _load_module("tgnn_solv.config", SRC_ROOT / "tgnn_solv" / "config.py")

DataBuilder = builder_module.DataBuilder
filter_for_sle = builder_module.filter_for_sle
load_bigsoldb = data_sources.load_bigsoldb
load_fusion_enthalpies = data_sources.load_fusion_enthalpies
load_hansen = data_sources.load_hansen
load_idac = data_sources.load_idac
load_melting_points = data_sources.load_melting_points
scaffold_split = split_module.scaffold_split
SPLIT_MODES = split_registry_module.SPLIT_MODES
build_split_metadata = split_registry_module.build_split_metadata
get_split_display_name = split_registry_module.get_split_display_name
normalize_split_mode = split_registry_module.normalize_split_mode
split_paths = split_registry_module.split_paths
TGNNSolvConfig = config_module.TGNNSolvConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare TGNN-Solv processed splits from raw data sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML config. When provided, include_water_solubility is "
            "read from TGNNSolvConfig unless overridden by the CLI flag below."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks/data/processed",
        help="Directory where train/val/test CSV files will be written.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="solute_scaffold",
        choices=["solute_scaffold", "solute", "solvent"],
        help="Grouping mode used by scaffold_split().",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of rows assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to the test split.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Disable network downloads and use only files already present on disk.",
    )
    parser.add_argument(
        "--include-water-solubility",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep water-solvent rows in the supervised subset even though "
            "water has only one atom. Defaults to the config value or True."
        ),
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios.

    Args:
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Raises:
        ValueError: If ratios are invalid.
    """
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be strictly positive.")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f}."
        )


def configure_data_paths(output_dir: Path) -> tuple[Path, Path]:
    """Point the data modules to the requested raw and processed directories.

    Args:
        output_dir: Processed-data output directory.

    Returns:
        Tuple of `(data_root, raw_dir)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = output_dir.parent
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    data_utils.DATA_DIR = data_root
    data_utils.RAW_DIR = raw_dir
    data_utils.PROCESSED_DIR = output_dir

    data_sources.RAW_DIR = raw_dir
    data_sources.BIGSOLDB_PATH = raw_dir / "BigSolDBv2.1.csv"

    return data_root, raw_dir


@contextmanager
def maybe_skip_downloads(skip_download: bool) -> Iterator[None]:
    """Optionally disable download attempts inside source loaders.

    Args:
        skip_download: Whether download attempts should be disabled.
    """
    original_download_file = data_sources.download_file

    if skip_download:
        def local_only_download(url: str, path: Path, desc: str = "") -> bool:
            """Report only local file availability without downloading."""
            if path.exists():
                size_mb = path.stat().st_size / 1e6
                print(f"  Already exists: {path.name} ({size_mb:.1f} MB)")
                return True
            print(
                f"  Skipping download for {desc or path.name}: "
                f"file not found at {path}"
            )
            return False

        data_sources.download_file = local_only_download

    try:
        yield
    finally:
        data_sources.download_file = original_download_file


def print_split_statistics(name: str, df: pd.DataFrame) -> None:
    """Print summary statistics for a split.

    Args:
        name: Human-readable split name.
        df: Split dataframe.
    """
    print(
        f"  {name:5s}: {len(df):7,d} rows | "
        f"{df['solute_smiles'].nunique():6,d} solutes | "
        f"{df['solvent_smiles'].nunique():5,d} solvents | "
        f"T_m: {int(df.get('has_T_m', pd.Series(False, index=df.index)).sum()):6,d} | "
        f"dH_fus: {int(df.get('has_dH_fus', pd.Series(False, index=df.index)).sum()):6,d} | "
        f"Hansen: {int(df.get('has_hansen', pd.Series(False, index=df.index)).sum()):6,d} | "
        f"gamma_inf: {int(df.get('has_gamma_inf', pd.Series(False, index=df.index)).sum()):6,d}"
    )


def save_split(df: pd.DataFrame, path: Path) -> None:
    """Save one split to CSV.

    Args:
        df: Split dataframe.
        path: Output CSV path.
    """
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_split_bundle(
    output_dir: Path,
    split_mode: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, str]:
    """Save train/val/test CSVs for one named split mode."""
    paths = split_paths(output_dir, split_mode)
    save_split(train_df, paths["train"])
    save_split(val_df, paths["val"])
    save_split(test_df, paths["test"])
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    """Run the notebook-equivalent data preparation pipeline."""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    primary_split_mode = normalize_split_mode(args.split_mode)
    resolved_include_water_solubility = True
    if args.config:
        cfg = TGNNSolvConfig.from_yaml(args.config)
        resolved_include_water_solubility = bool(cfg.include_water_solubility)
    if args.include_water_solubility is not None:
        resolved_include_water_solubility = bool(args.include_water_solubility)

    output_dir = Path(args.output_dir)
    data_root, raw_dir = configure_data_paths(output_dir)

    print("=" * 70)
    print("TGNN-Solv Data Preparation")
    print("=" * 70)
    print(f"Processed output: {output_dir}")
    print(f"Raw data root:    {raw_dir}")
    print(f"Primary split:    {primary_split_mode}")
    print(f"Seed:             {args.seed}")
    print(f"Water supervision:{resolved_include_water_solubility}")
    print(
        f"Ratios:           train={args.train_ratio:.2f}, "
        f"val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
    )

    if args.skip_download and not data_sources.BIGSOLDB_PATH.exists():
        raise FileNotFoundError(
            "BigSolDB raw file not found while --skip-download is enabled: "
            f"{data_sources.BIGSOLDB_PATH}"
        )

    with maybe_skip_downloads(args.skip_download):
        print("\n1. Loading primary source...")
        bigsoldb = load_bigsoldb()

        print("\n2. Filtering for SLE compatibility...")
        bigsoldb = filter_for_sle(
            bigsoldb,
            x2_max=0.98,
            include_water_solubility=resolved_include_water_solubility,
        )
        print(f"  After SLE filter: {len(bigsoldb):,}")

        print("\n3. Loading auxiliary sources...")
        mp_data = load_melting_points()
        dh_data = load_fusion_enthalpies()
        hansen_data = load_hansen()
        idac_data = load_idac()

    print("\n4. Building unified dataset...")
    builder = DataBuilder()
    builder.add_mp(mp_data)
    builder.add_dh(dh_data)
    builder.add_hansen(hansen_data)
    builder.add_gamma(idac_data)
    unified = builder.build(bigsoldb)
    print(f"  Unified shape: {unified.shape}")

    print("\n5. Creating all split variants...")
    split_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for split_mode in SPLIT_MODES:
        print(f"\n  -> {get_split_display_name(split_mode)}")
        split_results[split_mode] = scaffold_split(
            unified,
            train_frac=args.train_ratio,
            val_frac=args.val_ratio,
            test_frac=args.test_ratio,
            seed=args.seed,
            mode=split_mode,
        )

    print("\n6. Saving split CSV files...")
    saved_paths: dict[str, dict[str, str]] = {}
    split_manifest = {
        "primary_split_mode": primary_split_mode,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "splits": {},
    }
    for split_mode, (train_df, val_df, test_df) in split_results.items():
        print(f"\n  Saving {get_split_display_name(split_mode)}:")
        saved_paths[split_mode] = save_split_bundle(
            output_dir,
            split_mode,
            train_df,
            val_df,
            test_df,
        )
        split_manifest["splits"][split_mode] = build_split_metadata(
            split_mode=split_mode,
            train_data=saved_paths[split_mode]["train"],
            val_data=saved_paths[split_mode]["val"],
            test_data=saved_paths[split_mode]["test"],
        )

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved: {manifest_path}")

    print("\nFinal statistics")
    print("-" * 70)
    primary_train_df, primary_val_df, primary_test_df = split_results[primary_split_mode]
    print(f"Primary split: {get_split_display_name(primary_split_mode)}")
    print_split_statistics("Train", primary_train_df)
    print_split_statistics("Val", primary_val_df)
    print_split_statistics("Test", primary_test_df)
    print("-" * 70)
    print("Additional split files:")
    for split_mode in SPLIT_MODES:
        if split_mode == primary_split_mode:
            continue
        bundle = saved_paths[split_mode]
        print(
            f"  {split_mode:16s}: "
            f"{Path(bundle['train']).name}, {Path(bundle['val']).name}, {Path(bundle['test']).name}"
        )
    print("-" * 70)
    print(
        f"  Total: {len(primary_train_df) + len(primary_val_df) + len(primary_test_df):7,d} rows | "
        f"{unified['solute_smiles'].nunique():6,d} solutes | "
        f"{unified['solvent_smiles'].nunique():5,d} solvents | "
        f"T_m: {int(unified['has_T_m'].sum()):6,d} | "
        f"dH_fus: {int(unified['has_dH_fus'].sum()):6,d} | "
        f"Hansen: {int(unified['has_hansen'].sum()):6,d} | "
        f"gamma_inf: {int(unified['has_gamma_inf'].sum()):6,d}"
    )
    print("=" * 70)
    print(f"Completed successfully. Split files saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
