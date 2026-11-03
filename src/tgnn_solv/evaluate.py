"""
Evaluation suite for TGNN-Solv.

Computes metrics broken down by:
  - Overall
  - Solvent type (water vs organic)
  - Solubility range (high / medium / low / very low)
  - Temperature (standard vs non-standard)
  - Auxiliary target availability (with/without T_m)

Usage::

    from tgnn_solv.evaluate import Evaluator

    evaluator = Evaluator(model, cfg)
    report = evaluator.evaluate(test_loader, test_df)
    evaluator.print_report(report)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TGNNSolvConfig
from .model import TGNNSolv
from .data.utils import canonicalize


# ================================================================== #
#  Metric computations                                                #
# ================================================================== #

def _compute_metrics(
    pred: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics on numpy arrays."""
    if len(pred) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "bias": float("nan")}

    errors = pred - true
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    bias = float(errors.mean())

    ss_res = float((errors ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return {"n": len(pred), "mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


# ================================================================== #
#  Evaluator                                                          #
# ================================================================== #

class Evaluator:
    """
    Comprehensive evaluation with stratified metrics.

    Parameters
    ----------
    model : TGNNSolv
    cfg : TGNNSolvConfig
    """

    WATER_SMILES = canonicalize("O")

    # ln(x₂) boundaries for solubility categories
    SOL_RANGES = {
        "high":     (-0.5, 0.0),      # x₂ > 0.61
        "moderate": (-3.0, -0.5),      # 0.05 < x₂ < 0.61
        "low":      (-7.0, -3.0),      # 9e-4 < x₂ < 0.05
        "very_low": (-30.0, -7.0),     # x₂ < 9e-4
    }

    def __init__(self, model: TGNNSolv, cfg: TGNNSolvConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _collect_predictions(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Run model on entire loader, collect predictions and metadata.

        Returns
        -------
        pred_ln_x2 : (N,)
        true_ln_x2 : (N,)
        meta : dict of (N,) arrays — T_m_pred, dH_pred, gamma_pred, etc.
        """
        self.model.eval()

        all_pred, all_true = [], []
        meta_lists = defaultdict(list)

        for sol_b, slv_b, tgt in loader:
            sol_b = sol_b.to(self.device)
            slv_b = slv_b.to(self.device)
            T = tgt["T"].to(self.device)
            solvent_type = tgt.get("solvent_type")
            solute_morgan_fp = tgt.get("solute_morgan_fp")
            solvent_morgan_fp = tgt.get("solvent_morgan_fp")
            solute_descriptors = tgt.get("solute_descriptors")
            solvent_descriptors = tgt.get("solvent_descriptors")
            solute_descriptor_prior_features = tgt.get(
                "solute_descriptor_prior_features"
            )
            solvent_descriptor_prior_features = tgt.get(
                "solvent_descriptor_prior_features"
            )
            solute_group_prior_features = tgt.get(
                "solute_group_prior_features"
            )
            solvent_group_prior_features = tgt.get(
                "solvent_group_prior_features"
            )
            T_m_gc = tgt.get("T_m_gc")
            dH_fus_gc = tgt.get("dH_fus_gc")
            dCp_fus_gc = tgt.get("dCp_fus_gc")
            mask = tgt["has_solubility"].to(self.device)

            out = self.model(
                sol_b,
                slv_b,
                T,
                solvent_type=solvent_type,
                solute_morgan_fp=(
                    solute_morgan_fp.to(self.device)
                    if isinstance(solute_morgan_fp, torch.Tensor)
                    else None
                ),
                solvent_morgan_fp=(
                    solvent_morgan_fp.to(self.device)
                    if isinstance(solvent_morgan_fp, torch.Tensor)
                    else None
                ),
                solute_descriptors=(
                    solute_descriptors.to(self.device)
                    if isinstance(solute_descriptors, torch.Tensor)
                    else None
                ),
                solvent_descriptors=(
                    solvent_descriptors.to(self.device)
                    if isinstance(solvent_descriptors, torch.Tensor)
                    else None
                ),
                solute_descriptor_prior_features=(
                    solute_descriptor_prior_features.to(self.device)
                    if isinstance(solute_descriptor_prior_features, torch.Tensor)
                    else None
                ),
                solvent_descriptor_prior_features=(
                    solvent_descriptor_prior_features.to(self.device)
                    if isinstance(solvent_descriptor_prior_features, torch.Tensor)
                    else None
                ),
                solute_group_prior_features=(
                    solute_group_prior_features.to(self.device)
                    if isinstance(solute_group_prior_features, torch.Tensor)
                    else None
                ),
                solvent_group_prior_features=(
                    solvent_group_prior_features.to(self.device)
                    if isinstance(solvent_group_prior_features, torch.Tensor)
                    else None
                ),
                T_m_gc=(
                    T_m_gc.to(self.device)
                    if isinstance(T_m_gc, torch.Tensor)
                    else None
                ),
                dH_fus_gc=(
                    dH_fus_gc.to(self.device)
                    if isinstance(dH_fus_gc, torch.Tensor)
                    else None
                ),
                dCp_fus_gc=(
                    dCp_fus_gc.to(self.device)
                    if isinstance(dCp_fus_gc, torch.Tensor)
                    else None
                ),
            )

            if mask.any():
                mask_cpu = mask.detach().cpu()
                all_pred.append(
                    out["ln_x2"].detach().cpu()[mask_cpu].numpy()
                )
                all_true.append(tgt["ln_x2"][mask_cpu].cpu().numpy())

                # Metadata for stratification
                meta_lists["T_m_pred"].append(
                    out["fusion_params"]["T_m"].detach().cpu()[mask_cpu].numpy()
                )
                meta_lists["dH_pred"].append(
                    out["fusion_params"]["dH_fus"].detach().cpu()[mask_cpu].numpy()
                )
                meta_lists["gamma_pred"].append(
                    torch.exp(
                        out["physics"]["ln_gamma_2"].detach().cpu()[mask_cpu]
                    ).numpy()
                )
                meta_lists["correction"].append(
                    out["correction"].detach().cpu()[mask_cpu].numpy()
                )
                meta_lists["Phi"].append(
                    out["physics"]["Phi"].detach().cpu()[mask_cpu].numpy()
                )

        pred = np.concatenate(all_pred) if all_pred else np.array([])
        true = np.concatenate(all_true) if all_true else np.array([])
        meta = {k: np.concatenate(v) for k, v in meta_lists.items()}

        return pred, true, meta

    def evaluate(
        self,
        loader: DataLoader,
        df: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """
        Full evaluation with stratified metrics.

        Parameters
        ----------
        loader : test DataLoader
        df : corresponding DataFrame (for solvent/T metadata).
             If None, only overall metrics are computed.

        Returns
        -------
        dict with keys: overall, by_solvent, by_range, by_temp,
                        physics_summary
        """
        pred, true, meta = self._collect_predictions(loader)
        report = {}

        # --- Overall ---
        report["overall"] = _compute_metrics(pred, true)

        # --- By solubility range ---
        report["by_range"] = {}
        for name, (lo, hi) in self.SOL_RANGES.items():
            mask = (true >= lo) & (true < hi)
            report["by_range"][name] = _compute_metrics(pred[mask], true[mask])

        # --- By solvent type (if df provided) ---
        report["by_solvent"] = {}
        if df is not None:
            # Align df with loader's valid records
            sol_df = df[df["has_solubility"]].reset_index(drop=True)
            n_records = min(len(pred), len(sol_df))
            sol_df = sol_df.iloc[:n_records]

            is_water = (sol_df["solvent_smiles"] == self.WATER_SMILES).values
            report["by_solvent"]["water"] = _compute_metrics(
                pred[is_water[:len(pred)]], true[is_water[:len(pred)]]
            )
            report["by_solvent"]["organic"] = _compute_metrics(
                pred[~is_water[:len(pred)]], true[~is_water[:len(pred)]]
            )

            # Top 5 individual solvents
            if "solvent_smiles" in sol_df.columns:
                top_slv = sol_df["solvent_smiles"].value_counts().head(5)
                for smi, cnt in top_slv.items():
                    mask = (sol_df["solvent_smiles"] == smi).values[:len(pred)]
                    name = smi[:30]
                    if "solvent_name" in sol_df.columns:
                        names = sol_df.loc[
                            sol_df["solvent_smiles"] == smi, "solvent_name"
                        ].dropna()
                        if len(names) > 0:
                            name = str(names.iloc[0])[:30]
                    report["by_solvent"][name] = _compute_metrics(
                        pred[mask], true[mask]
                    )

            # By temperature
            report["by_temp"] = {}
            T_vals = sol_df["temperature"].values[:len(pred)]
            at_298 = np.abs(T_vals - 298.15) < 1.0
            report["by_temp"]["298.15K"] = _compute_metrics(
                pred[at_298], true[at_298]
            )
            report["by_temp"]["other_T"] = _compute_metrics(
                pred[~at_298], true[~at_298]
            )

            # With/without T_m
            report["by_aux"] = {}
            if "has_T_m" in sol_df.columns:
                has_tm = sol_df["has_T_m"].values[:len(pred)]
                report["by_aux"]["with_T_m"] = _compute_metrics(
                    pred[has_tm], true[has_tm]
                )
                report["by_aux"]["without_T_m"] = _compute_metrics(
                    pred[~has_tm], true[~has_tm]
                )

        # --- Physics summary ---
        report["physics_summary"] = {}
        if "T_m_pred" in meta:
            report["physics_summary"]["T_m_median"] = float(
                np.median(meta["T_m_pred"])
            )
            report["physics_summary"]["dH_fus_median"] = float(
                np.median(meta["dH_pred"])
            )
            report["physics_summary"]["gamma_median"] = float(
                np.median(meta["gamma_pred"])
            )
            report["physics_summary"]["correction_mean_abs"] = float(
                np.abs(meta["correction"]).mean()
            )
            report["physics_summary"]["correction_max_abs"] = float(
                np.abs(meta["correction"]).max()
            )

        return report

    # -------------------------------------------------------------- #
    #  Pretty printing                                                #
    # -------------------------------------------------------------- #

    @staticmethod
    def print_report(report: dict[str, object]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 65)
        print("  TGNN-Solv Evaluation Report")
        print("=" * 65)

        def _fmt(m: dict[str, float]) -> str:
            if m["n"] == 0:
                return "  (no data)"
            return (
                f"  n={m['n']:>6,d}  MAE={m['mae']:.3f}  "
                f"RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}  "
                f"bias={m['bias']:+.3f}"
            )

        # Overall
        print("\n  OVERALL")
        print(_fmt(report["overall"]))

        # By solubility range
        print("\n  BY SOLUBILITY RANGE")
        for name, metrics in report.get("by_range", {}).items():
            lo, hi = Evaluator.SOL_RANGES[name]
            print(f"    {name:10s} [{lo:6.1f}, {hi:5.1f})")
            print(f"  {_fmt(metrics)}")

        # By solvent
        if report.get("by_solvent"):
            print("\n  BY SOLVENT")
            for name, metrics in report["by_solvent"].items():
                print(f"    {name:30s}")
                print(f"  {_fmt(metrics)}")

        # By temperature
        if report.get("by_temp"):
            print("\n  BY TEMPERATURE")
            for name, metrics in report["by_temp"].items():
                print(f"    {name:15s}")
                print(f"  {_fmt(metrics)}")

        # By auxiliary target
        if report.get("by_aux"):
            print("\n  BY AUXILIARY DATA")
            for name, metrics in report["by_aux"].items():
                print(f"    {name:15s}")
                print(f"  {_fmt(metrics)}")

        # Physics summary
        ps = report.get("physics_summary", {})
        if ps:
            print("\n  PHYSICS SUMMARY")
            print(f"    Predicted T_m median:       "
                  f"{ps.get('T_m_median', 0):.0f} K")
            print(f"    Predicted ΔH_fus median:    "
                  f"{ps.get('dH_fus_median', 0):.0f} J/mol")
            print(f"    Predicted γ₂ median:        "
                  f"{ps.get('gamma_median', 0):.2f}")
            print(f"    |correction| mean:          "
                  f"{ps.get('correction_mean_abs', 0):.3f}")
            print(f"    |correction| max:           "
                  f"{ps.get('correction_max_abs', 0):.3f}")

        print("\n" + "=" * 65)
