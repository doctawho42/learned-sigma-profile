"""Shared article-reproduction planning helpers.

This module centralizes the maintained paper-reproduction workflow so that:

- `reproduce.sh`
- the structured CLI runner
- the Streamlit Experiment Lab
- documentation

can all reference the same step graph and defaults instead of drifting apart.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "notebooks" / "data" / "processed"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_TABLES_DIR = REPO_ROOT / "tables"

PROFILE_CHOICES = ("core", "article", "full")


@dataclass(slots=True)
class ReproductionSettings:
    """Runtime settings for the maintained article-reproduction workflow."""

    profile: str = "article"
    python_command: str = "python"
    device: str = "auto"
    processed_dir: str = str(DEFAULT_PROCESSED_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    checkpoints_dir: str = str(DEFAULT_CHECKPOINTS_DIR)
    figures_dir: str = str(DEFAULT_FIGURES_DIR)
    tables_dir: str = str(DEFAULT_TABLES_DIR)
    seed: int = 42
    n_seeds: int = 5
    split_mode: str = "solute_scaffold"
    fastsolv_mode: str = "both"
    solprop_mode: str = "native"
    fastsolv_python: str | None = None
    solprop_python: str | None = None
    solprop_runtime_dir: str | None = None
    solprop_native_device: str = "auto"
    external_continue_on_error: bool = True


@dataclass(slots=True)
class ReproductionStep:
    """One maintained reproduction step."""

    step_id: str
    name: str
    description: str
    category: str
    command_preview: list[str]
    expected_outputs: list[str]
    optional: bool = False


def _python_prefix(settings: ReproductionSettings) -> list[str]:
    return shlex.split(settings.python_command)


def _pycmd(settings: ReproductionSettings, script_path: str, *args: str) -> list[str]:
    return [*_python_prefix(settings), script_path, *args]


def _device_value(settings: ReproductionSettings) -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _processed_paths(settings: ReproductionSettings) -> tuple[str, str, str]:
    processed_dir = Path(settings.processed_dir)
    return (
        str(processed_dir / "train.csv"),
        str(processed_dir / "val.csv"),
        str(processed_dir / "test.csv"),
    )


def reproduction_profiles() -> dict[str, str]:
    """Return maintained reproduction profiles and their intent."""

    return {
        "core": (
            "Minimal maintained paper path: canonical data, tuned TGNN multi-seed, "
            "best-checkpoint evaluation, split comparison, supplementary tables, and figures."
        ),
        "article": (
            "Current article-comparison path: core TGNN reproduction plus medium-budget "
            "matched baselines and external FastSolv/SolProp benchmarking."
        ),
        "full": (
            "Expanded diagnostics: article profile plus split-late, DirectGNN multi-seed, "
            "ablations, learning curves, temperature extrapolation, physics validation, "
            "statistical tests, and the full-budget diagnostic export."
        ),
    }


def _step_prepare_data(settings: ReproductionSettings) -> ReproductionStep:
    return ReproductionStep(
        step_id="prepare_data",
        name="Prepare scaffold-aware data",
        description="Build canonical processed train/val/test splits with auxiliary labels.",
        category="data",
        command_preview=_pycmd(
            settings,
            "scripts/data/prepare_data.py",
            "--output-dir",
            settings.processed_dir,
            "--split-mode",
            settings.split_mode,
            "--seed",
            str(int(settings.seed)),
            "--skip-download",
        ),
        expected_outputs=[
            str(Path(settings.processed_dir) / "train.csv"),
            str(Path(settings.processed_dir) / "val.csv"),
            str(Path(settings.processed_dir) / "test.csv"),
        ],
    )


def _step_tgnn_multi_seed(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="tgnn_multi_seed",
        name="Run tuned TGNN multi-seed sweep",
        description="Train the maintained physics-informed baseline with the tuned paper configuration.",
        category="experiments",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_seeds.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--n-seeds",
            str(int(settings.n_seeds)),
            "--base-seed",
            str(int(settings.seed)),
            "--output",
            str(Path(settings.results_dir) / "multi_seed_results.json"),
            "--checkpoint-dir",
            str(Path(settings.checkpoints_dir) / "seeds"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[
            str(Path(settings.results_dir) / "multi_seed_results.json"),
            str(Path(settings.checkpoints_dir) / "seeds"),
        ],
    )


def _step_medium_budget(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="medium_budget",
        name="Run medium-budget architecture comparison",
        description="Train tuned TGNN, GC-prior variants, DirectGNN baselines, and RF on the full scaffold split.",
        category="experiments",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_medium_budget_comparison.py",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--output-dir",
            str(Path(settings.results_dir) / "medium_budget"),
            "--seed",
            str(int(settings.seed)),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "medium_budget")],
    )


def _step_external_benchmarks(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    command = _pycmd(
        settings,
        "scripts/experiments/run_external_baseline_benchmark.py",
        "--train-data",
        train_csv,
        "--val-data",
        val_csv,
        "--test-data",
        test_csv,
        "--out-dir",
        str(Path(settings.results_dir) / "external_baselines" / "article_benchmark"),
        "--split-mode",
        settings.split_mode,
        "--fastsolv-mode",
        settings.fastsolv_mode,
        "--solprop-mode",
        settings.solprop_mode,
        "--solprop-native-device",
        settings.solprop_native_device if settings.solprop_native_device != "auto" else _device_value(settings),
    )
    if settings.fastsolv_python:
        command += ["--fastsolv-python", settings.fastsolv_python]
    if settings.solprop_python:
        command += ["--solprop-python", settings.solprop_python]
    if settings.solprop_runtime_dir:
        command += ["--solprop-runtime-dir", settings.solprop_runtime_dir]
    if settings.external_continue_on_error:
        command.append("--continue-on-error")
    return ReproductionStep(
        step_id="external_benchmarks",
        name="Run external baseline benchmark",
        description="Benchmark FastSolv and native-retrained SolProp on the repo's own scaffold-aware split.",
        category="baseline",
        command_preview=command,
        expected_outputs=[
            str(Path(settings.results_dir) / "external_baselines" / "article_benchmark" / "summary.csv"),
            str(Path(settings.results_dir) / "external_baselines" / "article_benchmark" / "comparison.json"),
        ],
        optional=True,
    )


def _step_evaluate_best(settings: ReproductionSettings) -> ReproductionStep:
    _, _, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="evaluate_best",
        name="Evaluate best TGNN checkpoint",
        description="Resolve the best seed from the tuned multi-seed run and export the canonical evaluation report.",
        category="evaluation",
        command_preview=_pycmd(
            settings,
            "scripts/evaluation/evaluate_complete.py",
            "--test-data",
            test_csv,
            "--tgnn-checkpoint",
            "<best-seed-checkpoint>",
            "--output",
            str(Path(settings.results_dir) / "full_evaluation.json"),
            "--verbose",
        ),
        expected_outputs=[str(Path(settings.results_dir) / "full_evaluation.json")],
    )


def _step_split_comparisons(settings: ReproductionSettings) -> ReproductionStep:
    return ReproductionStep(
        step_id="split_comparisons",
        name="Run split-wise comparison",
        description="Compare TGNN-Solv, DirectGNN, and RF families across scaffold, solute, and solvent holdout protocols.",
        category="experiments",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_split_comparisons.py",
            "--processed-dir",
            settings.processed_dir,
            "--splits",
            "solute_scaffold,solute,solvent",
            "--models",
            "tgnn_solv,direct_gnn,rf_baseline,rf_morgan,rf_hybrid",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--n-seeds",
            "3",
            "--base-seed",
            str(int(settings.seed)),
            "--results-dir",
            str(Path(settings.results_dir) / "split_comparisons"),
            "--output",
            str(Path(settings.results_dir) / "split_comparisons.json"),
            "--checkpoint-root",
            str(Path(settings.checkpoints_dir) / "split_comparisons"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "split_comparisons.json")],
    )


def _step_generate_supplementary(settings: ReproductionSettings) -> ReproductionStep:
    return ReproductionStep(
        step_id="generate_supplementary",
        name="Generate supplementary tables",
        description="Collect paper-facing CSV/LaTeX tables from all currently available experiment artifacts.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/generate_supplementary.py",
            "--results-dir",
            settings.results_dir,
            "--output-dir",
            settings.tables_dir,
        ),
        expected_outputs=[settings.tables_dir],
    )


def _step_generate_figures(settings: ReproductionSettings) -> ReproductionStep:
    return ReproductionStep(
        step_id="generate_figures",
        name="Generate paper figures",
        description="Build publication figures from available evaluation and experiment outputs.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/generate_paper_figures.py",
            "--results-dir",
            settings.results_dir,
            "--output-dir",
            settings.figures_dir,
        ),
        expected_outputs=[settings.figures_dir],
    )


def _step_split_late(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="split_late_multi_seed",
        name="Run split-late backbone comparison",
        description="Run the matched split-late encoder ablation under the same seed budget.",
        category="experiments",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_seeds.py",
            "--config",
            "configs/paper_config_split_late.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--n-seeds",
            str(int(settings.n_seeds)),
            "--base-seed",
            str(int(settings.seed)),
            "--output",
            str(Path(settings.results_dir) / "split_late_multi_seed_results.json"),
            "--checkpoint-dir",
            str(Path(settings.checkpoints_dir) / "split_late_seeds"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[
            str(Path(settings.results_dir) / "split_late_multi_seed_results.json"),
            str(Path(settings.checkpoints_dir) / "split_late_seeds"),
        ],
    )


def _step_direct_multi_seed(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="directgnn_multi_seed",
        name="Run DirectGNN multi-seed baseline",
        description="Train the matched no-physics backbone across the same seed budget used for TGNN-Solv.",
        category="baseline",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_seeds.py",
            "--train-script",
            "scripts/training/train_directgnn.py",
            "--config",
            "configs/paper_config_directgnn_tuned.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--n-seeds",
            str(int(settings.n_seeds)),
            "--base-seed",
            str(int(settings.seed)),
            "--output",
            str(Path(settings.results_dir) / "directgnn_multi_seed_results.json"),
            "--checkpoint-dir",
            str(Path(settings.checkpoints_dir) / "directgnn_seeds"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[
            str(Path(settings.results_dir) / "directgnn_multi_seed_results.json"),
            str(Path(settings.checkpoints_dir) / "directgnn_seeds"),
        ],
    )


def _step_error_analysis(settings: ReproductionSettings) -> ReproductionStep:
    _, _, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="error_analysis",
        name="Run error analysis",
        description="Derive chemistry- and regime-level error slices from the canonical evaluation output.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/evaluation/error_analysis.py",
            "--predictions",
            str(Path(settings.results_dir) / "full_evaluation.json"),
            "--test-data",
            test_csv,
            "--output",
            str(Path(settings.results_dir) / "error_analysis.json"),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "error_analysis.json")],
    )


def _step_ablation(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="ablation",
        name="Run ablation suite",
        description="Run bridge/oracle/architecture ablations on the canonical split.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_ablation.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--n-seeds",
            "3",
            "--output",
            str(Path(settings.results_dir) / "ablation.json"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "ablation.json")],
    )


def _step_learning_curves(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="learning_curves",
        name="Run learning-curve study",
        description="Measure how TGNN-Solv and descriptor baselines scale with available training data.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/learning_curves.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--fractions",
            "0.01,0.05,0.1,0.2,0.5,1.0",
            "--n-seeds",
            "3",
            "--output",
            str(Path(settings.results_dir) / "learning_curves.json"),
            "--device",
            _device_value(settings),
            "--models",
            "tgnn_solv,rf_baseline",
        ),
        expected_outputs=[str(Path(settings.results_dir) / "learning_curves.json")],
    )


def _step_temperature_extrapolation(settings: ReproductionSettings) -> ReproductionStep:
    combined_path = Path(settings.results_dir) / "reproduction" / "temperature_extrapolation_input.csv"
    return ReproductionStep(
        step_id="temperature_extrapolation",
        name="Run temperature extrapolation study",
        description="Concatenate train/val/test into one CSV and evaluate extrapolation across multiple temperature cutoffs.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/temperature_extrapolation.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--data",
            str(combined_path),
            "--t-cuts",
            "298.15,323.15,348.15,373.15",
            "--n-seeds",
            "3",
            "--output",
            str(Path(settings.results_dir) / "temperature_extrapolation.json"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "temperature_extrapolation.json")],
    )


def _step_physics_validation(settings: ReproductionSettings) -> ReproductionStep:
    _, _, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="physics_validation",
        name="Validate solver-facing physics",
        description="Export thermodynamic sanity checks and physical-parameter validation on the best tuned TGNN checkpoint.",
        category="evaluation",
        command_preview=_pycmd(
            settings,
            "scripts/evaluation/validate_physics.py",
            "--checkpoint",
            "<best-seed-checkpoint>",
            "--test-data",
            test_csv,
            "--output",
            str(Path(settings.results_dir) / "physics_validation.json"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "physics_validation.json")],
    )


def _step_statistical_tests(settings: ReproductionSettings) -> ReproductionStep:
    return ReproductionStep(
        step_id="statistical_tests",
        name="Run significance tests",
        description="Compare TGNN-Solv against DirectGNN and split-late using the multi-seed result bundles.",
        category="analysis",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/statistical_tests.py",
            "--results",
            str(Path(settings.results_dir) / "multi_seed_results.json"),
            str(Path(settings.results_dir) / "directgnn_multi_seed_results.json"),
            str(Path(settings.results_dir) / "split_late_multi_seed_results.json"),
            "--labels",
            "TGNN-Solv",
            "DirectGNN",
            "SplitLate",
            "--output",
            str(Path(settings.results_dir) / "significance.json"),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "significance.json")],
    )


def _step_full_budget(settings: ReproductionSettings) -> ReproductionStep:
    train_csv, val_csv, test_csv = _processed_paths(settings)
    return ReproductionStep(
        step_id="full_budget_diagnostics",
        name="Run full-budget diagnostic study",
        description="Budget-matched TGNN-vs-DirectGNN diagnostic export with solver intermediates and oracle metrics.",
        category="experiments",
        command_preview=_pycmd(
            settings,
            "scripts/experiments/run_full_budget_experiment.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--train-data",
            train_csv,
            "--val-data",
            val_csv,
            "--test-data",
            test_csv,
            "--seeds",
            str(int(settings.seed)),
            "--output-dir",
            str(Path(settings.results_dir) / "full_budget_experiment"),
            "--device",
            _device_value(settings),
        ),
        expected_outputs=[str(Path(settings.results_dir) / "full_budget_experiment")],
        optional=True,
    )


def build_reproduction_steps(settings: ReproductionSettings) -> list[ReproductionStep]:
    """Build the maintained reproduction plan for one profile."""

    if settings.profile not in PROFILE_CHOICES:
        raise ValueError(f"Unknown profile: {settings.profile}")

    base_steps = [
        _step_prepare_data(settings),
        _step_tgnn_multi_seed(settings),
    ]
    core_tail = [
        _step_evaluate_best(settings),
        _step_split_comparisons(settings),
        _step_generate_supplementary(settings),
        _step_generate_figures(settings),
    ]
    article_extra = [
        _step_medium_budget(settings),
        _step_external_benchmarks(settings),
    ]
    full_extra = [
        _step_split_late(settings),
        _step_direct_multi_seed(settings),
        _step_error_analysis(settings),
        _step_ablation(settings),
        _step_learning_curves(settings),
        _step_temperature_extrapolation(settings),
        _step_physics_validation(settings),
        _step_statistical_tests(settings),
        _step_full_budget(settings),
    ]

    if settings.profile == "core":
        return [*base_steps, *core_tail]
    if settings.profile == "article":
        return [*base_steps, *article_extra, *core_tail]
    return [
        *base_steps,
        _step_split_late(settings),
        _step_direct_multi_seed(settings),
        *article_extra,
        _step_evaluate_best(settings),
        _step_error_analysis(settings),
        _step_ablation(settings),
        _step_learning_curves(settings),
        _step_temperature_extrapolation(settings),
        _step_physics_validation(settings),
        _step_split_comparisons(settings),
        _step_statistical_tests(settings),
        _step_full_budget(settings),
        _step_generate_supplementary(settings),
        _step_generate_figures(settings),
    ]


def resolve_best_checkpoint(results_json: str | Path) -> str:
    """Resolve the best TGNN checkpoint path from a multi-seed results JSON."""

    path = Path(results_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best_seed")
    if not isinstance(best, dict):
        raise KeyError("`best_seed` is missing or malformed.")
    checkpoint = best.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint.strip():
        return checkpoint
    best_seed = best.get("seed")
    for row in payload.get("per_seed", []):
        if isinstance(row, dict) and row.get("seed") == best_seed and row.get("checkpoint"):
            return str(row["checkpoint"])
    raise KeyError("Could not resolve best-seed checkpoint from multi_seed_results.json")


def ensure_temperature_extrapolation_dataset(settings: ReproductionSettings) -> str:
    """Materialize a concatenated train/val/test CSV for temperature extrapolation."""

    train_csv, val_csv, test_csv = _processed_paths(settings)
    output_path = Path(settings.results_dir) / "reproduction" / "temperature_extrapolation_input.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    return str(output_path)


def actual_command_for_step(
    step: ReproductionStep,
    settings: ReproductionSettings,
    state: dict[str, Any],
) -> list[str]:
    """Return the concrete subprocess command for one step."""

    if step.step_id == "evaluate_best":
        _, _, test_csv = _processed_paths(settings)
        best_checkpoint = state.get("best_checkpoint") or resolve_best_checkpoint(Path(settings.results_dir) / "multi_seed_results.json")
        state["best_checkpoint"] = best_checkpoint
        return _pycmd(
            settings,
            "scripts/evaluation/evaluate_complete.py",
            "--test-data",
            test_csv,
            "--tgnn-checkpoint",
            best_checkpoint,
            "--output",
            str(Path(settings.results_dir) / "full_evaluation.json"),
            "--verbose",
        )

    if step.step_id == "error_analysis":
        _, _, test_csv = _processed_paths(settings)
        return _pycmd(
            settings,
            "scripts/evaluation/error_analysis.py",
            "--predictions",
            str(Path(settings.results_dir) / "full_evaluation.json"),
            "--test-data",
            test_csv,
            "--output",
            str(Path(settings.results_dir) / "error_analysis.json"),
        )

    if step.step_id == "physics_validation":
        _, _, test_csv = _processed_paths(settings)
        best_checkpoint = state.get("best_checkpoint") or resolve_best_checkpoint(Path(settings.results_dir) / "multi_seed_results.json")
        state["best_checkpoint"] = best_checkpoint
        return _pycmd(
            settings,
            "scripts/evaluation/validate_physics.py",
            "--checkpoint",
            best_checkpoint,
            "--test-data",
            test_csv,
            "--output",
            str(Path(settings.results_dir) / "physics_validation.json"),
            "--device",
            _device_value(settings),
        )

    if step.step_id == "temperature_extrapolation":
        combined_path = state.get("temperature_dataset_path") or ensure_temperature_extrapolation_dataset(settings)
        state["temperature_dataset_path"] = combined_path
        return _pycmd(
            settings,
            "scripts/experiments/temperature_extrapolation.py",
            "--config",
            "configs/paper_config_tuned.yaml",
            "--data",
            combined_path,
            "--t-cuts",
            "298.15,323.15,348.15,373.15",
            "--n-seeds",
            "3",
            "--output",
            str(Path(settings.results_dir) / "temperature_extrapolation.json"),
            "--device",
            _device_value(settings),
        )

    return list(step.command_preview)


def quote_command(command: list[str]) -> str:
    """Shell-quote a command for human-readable previews."""

    return shlex.join([str(item) for item in command])

