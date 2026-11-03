"""Heuristics and dataframe helpers for source-level uncertainty priors."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METHOD_SIGMA_PRIORS: dict[str, float] = {
    "gravimetric_equilibrium": 0.20,
    "spectrophotometric_uv": 0.35,
    "hplc": 0.20,
    "polythermal_visual": 1.00,
    "dsc": 0.75,
    "single_temperature_primary": 0.30,
    "multi_temperature_primary": 0.60,
    "compilation_or_secondary": 1.50,
    "computed_or_modeled": 2.00,
    "unknown_primary": 0.50,
    "unknown": 0.75,
}

SOURCE_UNCERTAINTY_MERGE_KEYS: tuple[str, ...] = (
    "solute_smiles",
    "solvent_smiles",
    "temperature",
    "ln_x2",
)


_TEXT_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "hplc",
        (
            "hplc",
            "high-performance liquid chromatography",
            "high performance liquid chromatography",
            "chromatograph",
            "chromatographic",
        ),
        "HPLC keywords",
    ),
    (
        "spectrophotometric_uv",
        (
            "uv",
            "uv-vis",
            "uv vis",
            "spectrophotometric",
            "spectrophotometer",
            "ultraviolet",
        ),
        "UV/spectrophotometry keywords",
    ),
    (
        "gravimetric_equilibrium",
        (
            "gravimetric",
            "shake flask",
            "shake-flask",
            "saturation shake-flask",
            "static gravimetric method",
            "gravimetric method",
            "isothermal saturation",
            "equilibrium method",
            "analytical balance",
            "equilibrium solubility was determined",
        ),
        "gravimetric/equilibrium keywords",
    ),
    (
        "polythermal_visual",
        (
            "polythermal",
            "cloud point",
            "synthetic method",
            "visual method",
            "laser monitoring",
            "disappearance of the solid phase",
            "clear point",
            "turbid",
            "turbidity",
        ),
        "polythermal/visual keywords",
    ),
    (
        "dsc",
        (
            "differential scanning calorimetry",
            "dsc",
        ),
        "DSC keywords",
    ),
    (
        "compilation_or_secondary",
        (
            "solubility data series",
            "compiled",
            "compilation",
            "critically evaluated",
            "review of",
            "review article",
            "handbook",
            "database",
            "iupac",
        ),
        "compilation keywords",
    ),
    (
        "computed_or_modeled",
        (
            "cosmo-rs",
            "cosmo rs",
            "predicted",
            "prediction of",
            "predictions of",
            "modeled",
            "simulation",
            "simulated",
            "molecular dynamics",
            "quantum chemical",
            "abraham model",
        ),
        "modeling keywords",
    ),
)


_DOI_RE = re.compile(r"^\s*10\.\S+/\S+\s*$", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)
    return text


def looks_like_doi(source: Any) -> bool:
    """Return True if the string looks like a DOI."""
    return bool(_DOI_RE.match(str(source or "").strip()))


def build_source_text(
    *,
    source: Any,
    title: Any = None,
    journal: Any = None,
    abstract: Any = None,
) -> str:
    """Concatenate and normalize source-side textual fields."""
    chunks = [
        _clean_text(source),
        _clean_text(title),
        _clean_text(journal),
        _clean_text(abstract),
    ]
    return " ".join(chunk for chunk in chunks if chunk).strip()


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _pattern_guess(stats: Mapping[str, Any] | None) -> tuple[str, float, str, str]:
    if not stats:
        return "unknown", METHOD_SIGMA_PRIORS["unknown"], "none", "no source-level pattern stats"

    median_temps_per_pair = _safe_float(stats.get("median_temps_per_pair"))
    fraction_pairs_multi_temp = _safe_float(stats.get("fraction_pairs_multi_temp"))
    fraction_pairs_ge5_temps = _safe_float(stats.get("fraction_pairs_ge5_temps"))
    fraction_uniform_step_pairs = _safe_float(stats.get("fraction_uniform_step_pairs"))
    fraction_rows_at_298_15 = _safe_float(stats.get("fraction_rows_at_298_15"))
    unique_solutes = _safe_float(stats.get("unique_solutes"))
    unique_solvents = _safe_float(stats.get("unique_solvents"))
    unique_pairs = _safe_float(stats.get("unique_pairs"))

    if (
        unique_solutes is not None
        and unique_solvents is not None
        and unique_pairs is not None
        and unique_solutes >= 40
        and unique_solvents >= 15
        and unique_pairs >= 100
        and median_temps_per_pair is not None
        and median_temps_per_pair <= 2.0
    ):
        return (
            "compilation_or_secondary",
            1.25,
            "pattern",
            "broad source coverage with mostly sparse per-pair temperature grids",
        )

    if (
        median_temps_per_pair is not None
        and fraction_pairs_ge5_temps is not None
        and fraction_pairs_multi_temp is not None
        and fraction_uniform_step_pairs is not None
        and median_temps_per_pair >= 5.0
        and fraction_pairs_ge5_temps >= 0.45
        and fraction_pairs_multi_temp >= 0.65
        and fraction_uniform_step_pairs >= 0.30
    ):
        return (
            "multi_temperature_primary",
            METHOD_SIGMA_PRIORS["multi_temperature_primary"],
            "pattern",
            "dense, nearly uniform temperature sweeps suggest a multi-temperature primary source",
        )

    if (
        median_temps_per_pair is not None
        and fraction_rows_at_298_15 is not None
        and median_temps_per_pair <= 1.5
        and fraction_rows_at_298_15 >= 0.50
    ):
        return (
            "single_temperature_primary",
            METHOD_SIGMA_PRIORS["single_temperature_primary"],
            "pattern",
            "mostly single-temperature measurements near 298.15 K suggest a single-temperature primary source",
        )

    if (
        median_temps_per_pair is not None
        and fraction_pairs_multi_temp is not None
        and median_temps_per_pair <= 2.5
        and fraction_pairs_multi_temp <= 0.35
    ):
        return (
            "unknown_primary",
            METHOD_SIGMA_PRIORS["unknown_primary"],
            "pattern",
            "primary-source-like footprint without strong method cues",
        )

    return (
        "unknown",
        METHOD_SIGMA_PRIORS["unknown"],
        "none",
        "no reliable text or pattern signal",
    )


def classify_source_method(
    *,
    source: Any,
    title: Any = None,
    journal: Any = None,
    abstract: Any = None,
    stats: Mapping[str, Any] | None = None,
    override_method: Any = None,
    override_sigma_ln_x2: Any = None,
    override_confidence: Any = None,
    override_rationale: Any = None,
) -> dict[str, Any]:
    """
    Return a pragmatic source-level method prior.

    This is not a literal extraction of experimental pointwise uncertainty.
    It provides source-level method classes and sigma priors that can later be
    audited or manually overridden.
    """
    override_method_str = _safe_text(override_method)
    if override_method_str:
        sigma_override = _safe_float(override_sigma_ln_x2)
        sigma_value = (
            sigma_override
            if sigma_override is not None
            else METHOD_SIGMA_PRIORS.get(override_method_str, METHOD_SIGMA_PRIORS["unknown"])
        )
        return {
            "method_guess": override_method_str,
            "sigma_ln_x2_guess": float(sigma_value),
            "confidence": _safe_text(override_confidence) or "manual",
            "heuristic_level": "manual_override",
            "rationale": _safe_text(override_rationale) or "manual override",
        }

    text = build_source_text(
        source=source,
        title=title,
        journal=journal,
        abstract=abstract,
    )
    for method, keywords, rationale in _TEXT_RULES:
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            confidence = "medium"
            if abstract and matched:
                confidence = "high"
            elif title and matched:
                confidence = "medium"
            return {
                "method_guess": method,
                "sigma_ln_x2_guess": float(METHOD_SIGMA_PRIORS[method]),
                "confidence": confidence,
                "heuristic_level": "text",
                "rationale": f"{rationale}: {', '.join(matched[:4])}",
            }

    method_guess, sigma_ln_x2_guess, heuristic_level, rationale = _pattern_guess(stats)
    confidence = "low" if heuristic_level == "pattern" else "none"
    return {
        "method_guess": method_guess,
        "sigma_ln_x2_guess": float(sigma_ln_x2_guess),
        "confidence": confidence,
        "heuristic_level": heuristic_level,
        "rationale": rationale,
    }


def compute_source_uncertainty_weights(
    sigma_ln_x2: np.ndarray | pd.Series | list[float],
    *,
    mode: str = "inverse_variance",
    min_sigma_ln_x2: float = 0.20,
    min_weight: float | None = 0.25,
    max_weight: float | None = 4.0,
) -> np.ndarray:
    """Convert source-level sigma priors into bounded per-row training weights."""
    sigma = np.asarray(sigma_ln_x2, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)
    sigma = np.clip(sigma, min_sigma_ln_x2, None)

    if mode == "inverse_variance":
        weight = 1.0 / np.square(sigma)
    elif mode == "inverse_sigma":
        weight = 1.0 / sigma
    elif mode == "uniform":
        weight = np.ones_like(sigma, dtype=float)
    else:
        raise ValueError(
            "Unsupported source uncertainty weight mode: "
            f"{mode!r}. Expected one of "
            "'inverse_variance', 'inverse_sigma', 'uniform'."
        )

    finite = np.isfinite(weight)
    if finite.any():
        weight[finite] = weight[finite] / max(float(np.mean(weight[finite])), 1.0e-12)
    else:
        weight = np.ones_like(sigma, dtype=float)
        finite = np.isfinite(weight)

    if min_weight is not None:
        weight[finite] = np.maximum(weight[finite], float(min_weight))
    if max_weight is not None:
        weight[finite] = np.minimum(weight[finite], float(max_weight))

    weight = np.where(np.isfinite(weight), weight, 1.0)
    return weight.astype(float, copy=False)


def attach_source_uncertainty(
    df: pd.DataFrame,
    *,
    uncertainty_csv: str,
    weight_mode: str = "inverse_variance",
    default_sigma_ln_x2: float = 0.75,
    min_sigma_ln_x2: float = 0.20,
    min_weight: float | None = 0.25,
    max_weight: float | None = 4.0,
    strict_for_supervised: bool = True,
) -> pd.DataFrame:
    """
    Attach source-level method/sigma priors and derived row weights to a split dataframe.

    The merge is keyed by exact maintained row identity:
    `(solute_smiles, solvent_smiles, temperature, ln_x2)`.
    """
    path = Path(str(uncertainty_csv).strip())
    if not path.is_file():
        raise FileNotFoundError(
            "Source uncertainty CSV not found: "
            f"{path}. Generate it via "
            "`scripts/analysis/run_source_uncertainty_audit.py` first."
        )

    work = df.copy()
    source_df = pd.read_csv(path, low_memory=False)

    missing_keys = [key for key in SOURCE_UNCERTAINTY_MERGE_KEYS if key not in work.columns]
    if missing_keys:
        raise ValueError(
            "Dataframe is missing required source-uncertainty merge keys: "
            f"{missing_keys}"
        )
    missing_source_keys = [
        key for key in SOURCE_UNCERTAINTY_MERGE_KEYS if key not in source_df.columns
    ]
    if missing_source_keys:
        raise ValueError(
            "Source uncertainty CSV is missing required merge keys: "
            f"{missing_source_keys}"
        )

    duplicate_counts = (
        source_df.groupby(list(SOURCE_UNCERTAINTY_MERGE_KEYS), dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    if bool((duplicate_counts["n_rows"] > 1).any()):
        raise ValueError(
            "Source uncertainty CSV contains duplicate merge keys; "
            "refusing ambiguous merge."
        )

    source_subset = source_df[
        list(SOURCE_UNCERTAINTY_MERGE_KEYS)
        + [
            "source",
            "method_guess",
            "sigma_ln_x2_guess",
            "confidence",
            "heuristic_level",
            "rationale",
        ]
    ].rename(
        columns={
            "source": "source_detail",
            "method_guess": "source_method_guess",
            "sigma_ln_x2_guess": "source_sigma_ln_x2",
            "confidence": "source_uncertainty_confidence",
            "heuristic_level": "source_uncertainty_heuristic_level",
            "rationale": "source_uncertainty_rationale",
        }
    )

    merged = work.merge(
        source_subset,
        on=list(SOURCE_UNCERTAINTY_MERGE_KEYS),
        how="left",
    )
    supervised_mask = (
        merged["has_solubility"].fillna(False).astype(bool).to_numpy()
        if "has_solubility" in merged.columns
        else np.ones(len(merged), dtype=bool)
    )
    matched_mask = np.isfinite(pd.to_numeric(merged["source_sigma_ln_x2"], errors="coerce"))

    if strict_for_supervised:
        missing_supervised = supervised_mask & ~matched_mask
        if bool(missing_supervised.any()):
            raise ValueError(
                "Source uncertainty merge missed supervised rows: "
                f"{int(missing_supervised.sum())} unmatched rows."
            )

    merged["source_sigma_ln_x2"] = pd.to_numeric(
        merged["source_sigma_ln_x2"],
        errors="coerce",
    )
    merged.loc[supervised_mask & ~matched_mask, "source_sigma_ln_x2"] = float(default_sigma_ln_x2)
    merged.loc[~supervised_mask, "source_sigma_ln_x2"] = float(default_sigma_ln_x2)

    weights = np.ones(len(merged), dtype=float)
    if bool(supervised_mask.any()):
        weights[supervised_mask] = compute_source_uncertainty_weights(
            merged.loc[supervised_mask, "source_sigma_ln_x2"].to_numpy(dtype=float),
            mode=weight_mode,
            min_sigma_ln_x2=min_sigma_ln_x2,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    merged["source_solubility_weight"] = weights
    return merged
