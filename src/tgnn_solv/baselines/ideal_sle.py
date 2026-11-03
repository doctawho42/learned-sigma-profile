"""Ideal-solubility baseline with gamma fixed to one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"solute_smiles"}
REQUIRED_TEST_COLUMNS = {"solute_smiles", "temperature", "ln_x2"}


class IdealSLEBaseline:
    """Ideal solubility baseline using fusion properties from the training set."""

    def __init__(self) -> None:
        """Initialize the ideal-solubility baseline."""
        self.R = 8.314
        self.property_db: dict[str, dict[str, float | None]] = {}

    @staticmethod
    def _finite_series(
        frame: pd.DataFrame,
        value_col: str,
        mask_col: str | None = None,
    ) -> pd.Series:
        """Extract finite values from a dataframe column, optionally using a mask.

        Args:
            frame: Input dataframe slice.
            value_col: Name of the numeric column.
            mask_col: Optional boolean mask column name.

        Returns:
            Series of finite numeric values.
        """
        if value_col not in frame.columns:
            return pd.Series(dtype=float)

        values = pd.to_numeric(frame[value_col], errors="coerce")
        valid = values.notna()
        if mask_col is not None and mask_col in frame.columns:
            valid &= frame[mask_col].fillna(False).astype(bool)
        return values[valid]

    @staticmethod
    def _to_json_ready(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
        """Convert non-finite values into JSON-safe values.

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

    def fit(self, train_df: pd.DataFrame) -> IdealSLEBaseline:
        """Collect melting and fusion properties from the training data.

        For each unique solute:
        - Use the median `T_m` if available.
        - Use the median `dH_fus` if available.
        - Use the median `dCp_fus` if available.
        - Skip solutes without `T_m`.
        - Use Walden's rule if `dH_fus` is missing.

        Args:
            train_df: Training dataframe.

        Returns:
            Fitted baseline instance.
        """
        missing = REQUIRED_TRAIN_COLUMNS - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing required training columns: {sorted(missing)}")

        self.property_db = {}
        n_with_both = 0
        n_walden = 0
        n_missing = 0

        for solute_smiles, group in train_df.groupby("solute_smiles", dropna=True):
            tm_values = self._finite_series(group, "T_m", "has_T_m")
            if tm_values.empty:
                n_missing += 1
                continue

            T_m = float(tm_values.median())

            dh_values = self._finite_series(group, "dH_fus", "has_dH_fus")
            if dh_values.empty:
                dH_fus = float(56.5 * T_m)
                n_walden += 1
            else:
                dH_fus = float(dh_values.median())
                n_with_both += 1

            dcp_values = self._finite_series(group, "dCp_fus", "has_dCp_fus")
            dCp_fus = float(dcp_values.median()) if not dcp_values.empty else None

            self.property_db[solute_smiles] = {
                "T_m": T_m,
                "dH_fus": dH_fus,
                "dCp_fus": dCp_fus,
            }

        print(
            "IdealSLEBaseline: "
            f"{n_with_both} solutes with T_m + dH_fus, "
            f"{n_walden} using Walden's rule, "
            f"{n_missing} without T_m (will be skipped)"
        )
        return self

    def _predict_single(self, properties: dict[str, float | None], temperature: float) -> float | None:
        """Predict log-solubility for a single sample.

        Args:
            properties: Solute property dictionary.
            temperature: Evaluation temperature in Kelvin.

        Returns:
            Predicted `ln(x2)` or None if the inputs are invalid.
        """
        T = float(temperature)
        T_m = float(properties["T_m"])
        dH_fus = float(properties["dH_fus"])
        dCp_fus = properties.get("dCp_fus")

        if not np.isfinite(T) or not np.isfinite(T_m) or not np.isfinite(dH_fus):
            return None
        if T <= 0.0 or T_m <= 0.0:
            return None

        ln_x2 = -(dH_fus / self.R) * (1.0 / T - 1.0 / T_m)
        if dCp_fus is not None and np.isfinite(dCp_fus):
            ratio = T_m / T
            ln_x2 += (float(dCp_fus) / self.R) * (ratio - 1.0 - np.log(ratio))

        if not np.isfinite(ln_x2):
            return None

        return float(np.clip(ln_x2, -30.0, 0.0))

    def predict(self, test_df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Predict `ln(x2)` for rows with available solute properties.

        Args:
            test_df: Test dataframe.

        Returns:
            Tuple of predictions and valid row indices.
        """
        if not self.property_db:
            raise AssertionError("IdealSLEBaseline must be fitted before calling predict().")

        missing = REQUIRED_TEST_COLUMNS - set(test_df.columns)
        if missing:
            raise ValueError(f"Missing required test columns: {sorted(missing)}")

        predictions: list[float] = []
        valid_indices: list[int] = []

        for idx, row in enumerate(test_df.itertuples(index=False)):
            properties = self.property_db.get(row.solute_smiles)
            if properties is None:
                continue

            prediction = self._predict_single(properties, row.temperature)
            if prediction is None:
                continue

            predictions.append(prediction)
            valid_indices.append(idx)

        return np.asarray(predictions, dtype=float), valid_indices

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, float | int]:
        """Evaluate the baseline on a labeled test dataframe.

        Args:
            test_df: Test dataframe with `ln_x2` targets.

        Returns:
            Dictionary with regression metrics and coverage statistics.
        """
        predictions, valid_idx = self.predict(test_df)
        if len(valid_idx) == 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "n_samples": 0,
                "n_skipped": int(len(test_df)),
            }

        true = pd.to_numeric(
            test_df.iloc[valid_idx]["ln_x2"],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_mask = np.isfinite(true) & np.isfinite(predictions)
        true = true[finite_mask]
        predictions = predictions[finite_mask]

        if len(true) == 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "n_samples": 0,
                "n_skipped": int(len(test_df)),
            }

        errors = predictions - true
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((true - np.mean(true)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_samples": int(len(true)),
            "n_skipped": int(len(test_df) - len(true)),
        }


def main() -> None:
    """Train and evaluate the ideal-solubility baseline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate an ideal-solubility baseline."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/ideal_sle_baseline.json",
        help="Path to save evaluation metrics as JSON.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    baseline = IdealSLEBaseline()
    baseline.fit(train_df)
    metrics = baseline.evaluate(test_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(baseline._to_json_ready(metrics), handle, indent=2)

    print("Ideal solubility baseline results:")
    for key in ("mae", "rmse", "r2", "n_samples", "n_skipped"):
        print(f"  {key}: {metrics[key]}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
