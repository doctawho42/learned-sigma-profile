"""Publication-quality plotting utilities for TGNN-Solv."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("pip install matplotlib") from exc

ArrayLike: TypeAlias = npt.ArrayLike
SavePath: TypeAlias = str | Path | None


MODEL_COLORS = {
    "tgnn_solv": "#2196F3",
    "direct_gnn": "#FF9800",
    "fastsolv": "#4CAF50",
    "ideal_sle": "#9E9E9E",
    "rf_baseline": "#E91E63",
    "unifac": "#9C27B0",
}


MODEL_DISPLAY_NAMES = {
    "tgnn_solv": "TGNN-Solv",
    "direct_gnn": "DirectGNN",
    "rf_baseline": "RF Baseline",
    "split_late": "SplitLate",
}


def setup_plot_style() -> None:
    """Apply a consistent publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def _finalize_figure(target: Axes, save_path: SavePath) -> Axes:
    """Tighten layout, optionally save, and close when requested."""
    target.figure.tight_layout()
    if save_path is not None:
        target.figure.savefig(save_path)
        plt.close(target.figure)
    return target


def parity_plot(
    true: ArrayLike,
    pred: ArrayLike,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = "#2196F3",
    show_metrics: bool = True,
    density: bool = True,
    save_path: SavePath = None,
) -> Axes:
    """Create a parity plot for experimental vs predicted `ln x2`."""
    setup_plot_style()

    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if density and len(true) > 1000:
        ax.hist2d(true, pred, bins=100, cmap="Blues", cmin=1)
    else:
        ax.scatter(true, pred, alpha=0.3, s=5, color=color)

    if len(true) > 0:
        lower = float(min(true.min(), pred.min()))
        upper = float(max(true.max(), pred.max()))
    else:
        lower, upper = -1.0, 1.0

    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental ln x₂")
    ax.set_ylabel("Predicted ln x₂")

    if title is not None:
        ax.set_title(title)

    if show_metrics and len(true) > 0:
        mae = mean_absolute_error(true, pred)
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        r2 = r2_score(true, pred)
        metrics_text = (
            f"n = {len(true)}\n"
            f"MAE = {mae:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"R² = {r2:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    return _finalize_figure(ax, save_path)


def residual_plot(
    true: ArrayLike,
    pred: ArrayLike,
    ax: Axes | None = None,
    save_path: SavePath = None,
) -> Axes:
    """Plot residuals as a function of the experimental target."""
    setup_plot_style()

    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    residuals = pred - true

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(true, residuals, alpha=0.3, s=5)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Experimental ln x₂")
    ax.set_ylabel("Residual")

    if len(residuals) > 0:
        bias = float(np.mean(residuals))
        std = float(np.std(residuals))
        ax.text(
            0.05,
            0.95,
            f"Bias = {bias:.3f}\nStd = {std:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    return _finalize_figure(ax, save_path)


def error_distribution(
    true: ArrayLike,
    pred: ArrayLike,
    ax: Axes | None = None,
    save_path: SavePath = None,
) -> Axes:
    """Plot the absolute-error distribution."""
    setup_plot_style()

    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(true) & np.isfinite(pred)
    errors = np.abs(pred[mask] - true[mask])

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.hist(errors, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    if len(errors) > 0:
        ax.axvline(np.mean(errors), color="red", linestyle="--", linewidth=1.2, label="Mean")
        ax.axvline(np.median(errors), color="black", linestyle=":", linewidth=1.2, label="Median")
        ax.legend()
    ax.set_xlabel("Absolute Error (ln x₂)")
    ax.set_ylabel("Count")

    return _finalize_figure(ax, save_path)


def ablation_bar_chart(
    results: dict[str, dict[str, float]], save_path: SavePath = None
) -> Figure:
    """Create a horizontal MAE bar chart for ablation results."""
    setup_plot_style()

    variants = list(results.keys())
    mae_means = [results[name]["mae_mean"] for name in variants]
    mae_stds = [results[name]["mae_std"] for name in variants]
    colors = [
        MODEL_COLORS["tgnn_solv"] if name == "full" else "#BDBDBD"
        for name in variants
    ]

    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.45 * len(variants) + 1.0)))
    positions = np.arange(len(variants))
    ax.barh(positions, mae_means, xerr=mae_stds, color=colors, alpha=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(variants)
    ax.set_xlabel("MAE")
    ax.set_ylabel("Variant")

    if "full" in results:
        ax.axvline(results["full"]["mae_mean"], color=MODEL_COLORS["tgnn_solv"], linestyle="--")

    return _finalize_figure(ax, save_path).figure


def learning_curve_plot(results: dict[str, object], save_path: SavePath = None) -> Figure:
    """Plot MAE vs data fraction with uncertainty bands for multiple models."""
    setup_plot_style()

    if "results" in results and isinstance(results["results"], dict):
        results = results["results"]

    fractions = sorted(
        [key for key in results.keys() if isinstance(key, str)],
        key=float,
    )
    models = sorted(
        {
            model_name
            for fraction in fractions
            for model_name in results[fraction].keys()
            if isinstance(results[fraction].get(model_name), dict)
            and "mae_mean" in results[fraction][model_name]
            and "mae_std" in results[fraction][model_name]
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(fractions, dtype=float)

    for model_name in models:
        means = []
        stds = []
        for fraction in fractions:
            metrics = results[fraction].get(model_name)
            if not isinstance(metrics, dict):
                means.append(np.nan)
                stds.append(np.nan)
                continue
            means.append(metrics.get("mae_mean", np.nan))
            stds.append(metrics.get("mae_std", np.nan))

        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        color = MODEL_COLORS.get(model_name, "#607D8B")

        ax.plot(x, means, marker="o", label=model_name, color=color)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xlabel("Training Fraction")
    ax.set_ylabel("MAE")
    ax.legend()

    return _finalize_figure(ax, save_path).figure


def split_comparison_plot(results: dict[str, object], save_path: SavePath = None) -> Figure:
    """Plot split-wise model MAE as a grouped bar chart."""
    setup_plot_style()

    split_order = results.get("split_order", [])
    splits = results.get("splits", {})
    if not isinstance(split_order, list) or not isinstance(splits, dict):
        raise ValueError("Split comparison payload must contain 'split_order' and 'splits'.")

    model_order = results.get("model_order", [])
    if not isinstance(model_order, list) or not model_order:
        model_order = sorted(
            {
                model_name
                for split_mode in split_order
                for model_name in splits.get(split_mode, {}).get("models", {}).keys()
            }
        )

    labels = []
    for split_mode in split_order:
        split_payload = splits.get(split_mode, {})
        split_meta = split_payload.get("split", {})
        display_name = split_meta.get("display_name") if isinstance(split_meta, dict) else None
        labels.append(display_name or split_mode)

    x = np.arange(len(split_order), dtype=float)
    width = 0.8 / max(len(model_order), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for index, model_name in enumerate(model_order):
        means = []
        stds = []
        for split_mode in split_order:
            model_data = splits.get(split_mode, {}).get("models", {}).get(model_name, {})
            aggregated = model_data.get("aggregated", {}) if isinstance(model_data, dict) else {}
            mae_stats = aggregated.get("mae", {}) if isinstance(aggregated, dict) else {}
            means.append(mae_stats.get("mean", np.nan))
            stds.append(mae_stats.get("std", np.nan))

        means_array = np.asarray(means, dtype=float)
        stds_array = np.asarray(stds, dtype=float)
        offset = (index - (len(model_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            means_array,
            width=width,
            yerr=stds_array,
            capsize=3,
            alpha=0.9,
            color=MODEL_COLORS.get(model_name, "#607D8B"),
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Split protocol")
    ax.legend()

    return _finalize_figure(ax, save_path).figure
