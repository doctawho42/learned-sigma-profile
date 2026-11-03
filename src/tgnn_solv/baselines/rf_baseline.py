"""Random Forest baseline on RDKit descriptors for TGNN-Solv comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "scikit-learn is required for rf_baseline.py. "
        "Install it with `pip install scikit-learn`."
    ) from exc

try:
    from ..features import compute_molecular_descriptors, compute_pair_morgan_features
except ImportError:  # pragma: no cover - script fallback
    try:
        from tgnn_solv.features import (
            compute_molecular_descriptors,
            compute_pair_morgan_features,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RDKit-backed feature computation is required for rf_baseline.py. "
            "Install TGNN-Solv with its RDKit dependencies."
        ) from exc


SUPPORTED_TREE_REGRESSORS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}
SUPPORTED_FEATURE_MODES = {"descriptors", "morgan", "hybrid"}

REQUIRED_COLUMNS = {"solute_smiles", "solvent_smiles", "temperature", "ln_x2"}


def compute_pair_descriptors(
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float,
) -> np.ndarray | None:
    """Compute concatenated RDKit descriptors for a solute-solvent pair.

    Args:
        solute_smiles: Solute SMILES string.
        solvent_smiles: Solvent SMILES string.
        temperature: Temperature in Kelvin.

    Returns:
        Concatenated descriptor vector, or None if either molecule is invalid.
    """
    desc_sol = compute_molecular_descriptors(solute_smiles)
    desc_slv = compute_molecular_descriptors(solvent_smiles)
    if desc_sol is None or desc_slv is None:
        return None

    inv_temperature = 1.0 / float(temperature)
    return np.concatenate(
        [
            desc_sol.astype(float, copy=False),
            desc_slv.astype(float, copy=False),
            [float(temperature), inv_temperature],
        ]
    )


class RFBaseline:
    """Random Forest baseline trained on RDKit pair descriptors."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 30,
        n_jobs: int = -1,
        random_state: int = 42,
        feature_mode: str = "descriptors",
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> None:
        """Initialize the baseline model and feature scaler.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth.
            n_jobs: Number of parallel jobs for fitting and inference.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.fitted = False
        if feature_mode not in SUPPORTED_FEATURE_MODES:
            raise ValueError(
                f"Unsupported feature_mode '{feature_mode}'. "
                f"Expected one of {sorted(SUPPORTED_FEATURE_MODES)}."
            )
        self.feature_mode = feature_mode
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self._descriptor_cache: dict[str, np.ndarray | None] = {}
        self._morgan_cache: dict[tuple[str, int, int], np.ndarray | None] = {}

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Validate that a dataframe contains the required columns.

        Args:
            df: Input dataframe.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def fit(self, train_df: pd.DataFrame) -> RFBaseline:
        """Fit the baseline model on a training dataframe.

        Args:
            train_df: Training dataframe with pair identifiers and targets.

        Returns:
            The fitted baseline instance.
        """
        self._validate_columns(train_df)

        valid_descriptors: list[np.ndarray] = []
        targets: list[float] = []

        train_view = self._supervised_view(train_df)
        for row in train_view.itertuples(index=False):
            descriptor = self._compute_pair_features(
                row.solute_smiles,
                row.solvent_smiles,
                row.temperature,
            )
            if descriptor is None:
                continue

            target = float(row.ln_x2)
            if not np.isfinite(target):
                continue

            valid_descriptors.append(descriptor)
            targets.append(target)

        if not valid_descriptors:
            raise ValueError("No valid training samples were found for RFBaseline.")

        X = np.array(valid_descriptors, dtype=float)
        y = np.array(targets, dtype=float)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

        print(
            f"RFBaseline[{self.feature_mode}] fitted on {len(X)} samples "
            f"({len(train_view) - len(X)} supervised rows skipped)"
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Predict solubility values for valid rows in a dataframe.

        Args:
            test_df: Test dataframe with pair identifiers.

        Returns:
            Tuple of predictions and valid dataframe indices.
        """
        assert self.fitted, "RFBaseline must be fitted before calling predict()."
        self._validate_columns(test_df)
        test_df = self._supervised_view(test_df).reset_index(drop=True)

        valid_descriptors: list[np.ndarray] = []
        valid_indices: list[int] = []

        for idx, row in enumerate(test_df.itertuples(index=False)):
            descriptor = self._compute_pair_features(
                row.solute_smiles,
                row.solvent_smiles,
                row.temperature,
            )
            if descriptor is None:
                continue

            valid_descriptors.append(descriptor)
            valid_indices.append(idx)

        if not valid_descriptors:
            return np.array([], dtype=float), []

        X = np.array(valid_descriptors, dtype=float)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.asarray(predictions, dtype=float), valid_indices

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, float | int]:
        """Evaluate the baseline on a test dataframe.

        Args:
            test_df: Test dataframe with targets.

        Returns:
            Dictionary with regression metrics and sample counts.
        """
        eval_df = self._supervised_view(test_df).reset_index(drop=True)
        predictions, valid_idx = self.predict(eval_df)
        if len(valid_idx) == 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "pearson_r": float("nan"),
                "n_samples": 0,
                "n_skipped": len(eval_df),
            }

        true = eval_df.iloc[valid_idx]["ln_x2"].values.astype(float)

        mae = float(mean_absolute_error(true, predictions))
        rmse = float(np.sqrt(mean_squared_error(true, predictions)))
        r2 = float(r2_score(true, predictions))

        if len(true) > 1 and np.std(true) > 0 and np.std(predictions) > 0:
            pearson_r = float(np.corrcoef(true, predictions)[0, 1])
        else:
            pearson_r = float("nan")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": pearson_r,
            "n_samples": int(len(valid_idx)),
            "n_skipped": int(len(eval_df) - len(valid_idx)),
        }

    def _compute_pair_features(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        temperature: float,
    ) -> np.ndarray | None:
        """Compute a pair feature vector according to the configured feature mode."""
        descriptor_features = None
        morgan_features = None

        if self.feature_mode in {"descriptors", "hybrid"}:
            descriptor_features = self._compute_pair_descriptors_cached(
                solute_smiles,
                solvent_smiles,
                temperature,
            )
        if self.feature_mode in {"morgan", "hybrid"}:
            morgan_features = self._compute_pair_morgan_cached(
                solute_smiles,
                solvent_smiles,
                temperature,
            )

        if self.feature_mode == "descriptors":
            return descriptor_features
        if self.feature_mode == "morgan":
            return morgan_features
        if descriptor_features is None or morgan_features is None:
            return None
        return np.concatenate([descriptor_features, morgan_features]).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _supervised_view(df: pd.DataFrame) -> pd.DataFrame:
        """Restrict fitting/evaluation to rows with experimental solubility."""
        if "has_solubility" not in df.columns:
            return df
        series = df["has_solubility"]
        if pd.api.types.is_bool_dtype(series):
            mask = series.fillna(False).to_numpy(dtype=bool)
        else:
            mask = (
                series.fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y", "t"})
                .to_numpy(dtype=bool)
            )
        return df.loc[mask].copy()

    def _molecular_descriptors_cached(self, smiles: str) -> np.ndarray | None:
        key = str(smiles)
        if key not in self._descriptor_cache:
            self._descriptor_cache[key] = compute_molecular_descriptors(key)
        return self._descriptor_cache[key]

    def _morgan_fp_cached(self, smiles: str) -> np.ndarray | None:
        from tgnn_solv.features import smiles_to_morgan_fp

        key = (str(smiles), int(self.morgan_radius), int(self.morgan_n_bits))
        if key not in self._morgan_cache:
            self._morgan_cache[key] = smiles_to_morgan_fp(
                key[0],
                radius=key[1],
                n_bits=key[2],
            )
        return self._morgan_cache[key]

    def _compute_pair_descriptors_cached(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        temperature: float,
    ) -> np.ndarray | None:
        desc_sol = self._molecular_descriptors_cached(solute_smiles)
        desc_slv = self._molecular_descriptors_cached(solvent_smiles)
        if desc_sol is None or desc_slv is None:
            return None
        temp = float(temperature)
        return np.concatenate(
            [
                desc_sol.astype(float, copy=False),
                desc_slv.astype(float, copy=False),
                [temp, 1.0 / temp],
            ]
        )

    def _compute_pair_morgan_cached(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        temperature: float,
    ) -> np.ndarray | None:
        solute_fp = self._morgan_fp_cached(solute_smiles)
        solvent_fp = self._morgan_fp_cached(solvent_smiles)
        if solute_fp is None or solvent_fp is None:
            return None
        temp = float(temperature)
        return np.concatenate([solute_fp, solvent_fp, [temp, 1.0 / temp]]).astype(
            np.float32,
            copy=False,
        )


def _json_ready(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    """Convert non-finite metric values to JSON-safe values.

    Args:
        metrics: Metrics dictionary.

    Returns:
        JSON-safe metrics dictionary.
    """
    output: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            output[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, (int, np.integer)):
            output[key] = int(value)
        else:
            output[key] = value
    return output


def main() -> None:
    """Train and evaluate the Random Forest baseline from CSV files."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest baseline on RDKit descriptors."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/rf_baseline.json",
        help="Path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="descriptors",
        choices=sorted(SUPPORTED_FEATURE_MODES),
        help="Feature family used by the tree baseline.",
    )
    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--morgan-n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    baseline = RFBaseline(
        feature_mode=args.feature_mode,
        morgan_radius=args.morgan_radius,
        morgan_n_bits=args.morgan_n_bits,
    )
    baseline.fit(train_df)
    metrics = baseline.evaluate(test_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics), handle, indent=2)

    print("Random Forest baseline results:")
    for key in ("mae", "rmse", "r2", "pearson_r", "n_samples", "n_skipped"):
        print(f"  {key}: {metrics[key]}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
