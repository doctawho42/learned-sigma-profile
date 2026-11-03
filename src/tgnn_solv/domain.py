"""
Applicability Domain (AD) for TGNN-Solv.

Detects when a query system is outside the training distribution
(OOD = out-of-domain). The current implementation combines:

1. **Latent distance** — Mahalanobis distance in the pair
   representation space to the training set centroid.
2. **Fingerprint similarity** — Tanimoto similarity to the
   nearest training molecule (solute and solvent separately).

Usage::

    from tgnn_solv.domain import ApplicabilityDomain

    ad = ApplicabilityDomain(model, train_loader)
    ad.fit()  # compute training statistics

    score = ad.score("CC(=O)Nc1ccc(O)cc1", "CCO", T=298.15)
    print(score["in_domain"])         # True / False
    print(score["mahalanobis"])       # distance
    print(score["tanimoto_solute"])   # similarity to nearest train solute
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .model import TGNNSolv
from .layers import make_temperature_features
from .features import smiles_to_graph
from .data.solvent_types import (
    solvent_type_id_from_smiles,
    SOLVENT_TYPE_OTHER_ID,
)

ScoreDict: TypeAlias = dict[str, float | bool]


class ApplicabilityDomain:
    """
    Applicability domain detector.

    Call .fit(train_loader) once after training, then .score()
    for each query.

    Parameters
    ----------
    model : TGNNSolv
        Trained model (used to extract latent pair vectors).
    train_loader : DataLoader
        Training DataLoader (for computing reference statistics).
    fp_radius : int
        Morgan fingerprint radius (default 2 = ECFP4).
    fp_bits : int
        Fingerprint bit length.
    mahalanobis_threshold : float
        Percentile of training Mahalanobis distances used as
        the in-domain cutoff (default 0.95 = 95th percentile).
    tanimoto_threshold : float
        Minimum Tanimoto similarity to nearest training molecule.
    """

    def __init__(
        self,
        model: TGNNSolv,
        train_loader: DataLoader | None = None,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        mahalanobis_threshold: float = 0.95,
        tanimoto_threshold: float = 0.3,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.mahal_pct = mahalanobis_threshold
        self.tani_threshold = tanimoto_threshold

        # Computed by fit()
        self.centroid = None         # (D_pair,) mean pair vector
        self.cov_inv = None          # (D_pair, D_pair) inverse covariance
        self.mahal_cutoff = None     # scalar threshold
        self.train_solute_fps = []   # list of RDKit fingerprints
        self.train_solvent_fps = []
        self.train_solute_smiles: set[str] = set()
        self.train_solvent_smiles: set[str] = set()

        self.is_fitted = False

        if train_loader is not None:
            self.fit(train_loader)

    # -------------------------------------------------------------- #
    #  Fingerprint helpers                                            #
    # -------------------------------------------------------------- #

    def _smiles_to_fp(self, smi: str) -> object | None:
        """Compute Morgan fingerprint from SMILES."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_radius, nBits=self.fp_bits,
        )

    def _max_tanimoto(self, query_fp: object | None, ref_fps: list[object]) -> float:
        """Max Tanimoto similarity between query and reference set."""
        if query_fp is None or len(ref_fps) == 0:
            return 0.0
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
        return float(max(sims)) if sims else 0.0

    # -------------------------------------------------------------- #
    #  Extract pair vectors from model                                #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def _extract_pair_vectors(
        self, loader: DataLoader
    ) -> torch.Tensor:
        """Extract pair representation vectors for all samples."""
        self.model.eval()
        all_pairs = []

        for sol_b, slv_b, tgt in loader:
            sol_b = sol_b.to(self.device)
            slv_b = slv_b.to(self.device)
            T = tgt["T"].to(self.device)
            solvent_type = tgt.get("solvent_type")

            # Get pair vectors (stop before SLE solver)
            t_feat = make_temperature_features(T)
            _, g_sol_payload, _, _ = self.model._encode_and_readout(
                sol_b, "solute", temp_feat=t_feat
            )
            _, g_slv_payload, _, _ = self.model._encode_and_readout(
                slv_b, "solvent", temp_feat=t_feat
            )
            g_pair = self.model._build_pair_representation(
                g_sol_payload,
                g_slv_payload,
            )
            if getattr(self.model, "solvent_moe", None) is not None:
                if solvent_type is None:
                    solvent_type = torch.full(
                        (g_pair.shape[0],),
                        SOLVENT_TYPE_OTHER_ID,
                        device=g_pair.device,
                        dtype=torch.long,
                    )
                else:
                    solvent_type = solvent_type.to(g_pair.device)
                g_pair, _ = self.model.solvent_moe(g_pair, solvent_type)
            all_pairs.append(g_pair.cpu())

        return torch.cat(all_pairs, dim=0)  # (N_train, D_pair)

    # -------------------------------------------------------------- #
    #  Fit: compute training statistics                               #
    # -------------------------------------------------------------- #

    def fit(self, train_loader: DataLoader) -> None:
        """
        Compute reference statistics from training data.

        This extracts pair vectors and fingerprints for the
        entire training set.  Call once after training.
        """
        print("Fitting Applicability Domain...")

        # 1. Pair vectors → centroid + covariance
        pair_vectors = self._extract_pair_vectors(train_loader)
        N, D = pair_vectors.shape
        if N < 2:
            raise ValueError(
                "ApplicabilityDomain.fit requires at least 2 samples"
            )

        self.centroid = pair_vectors.mean(dim=0)  # (D,)
        centered = pair_vectors - self.centroid.unsqueeze(0)
        cov = (centered.T @ centered) / (N - 1)

        # Regularize covariance for numerical stability
        cov += torch.eye(D) * 1e-4
        self.cov_inv = torch.linalg.inv(cov)

        # Mahalanobis distances for training set
        train_dists = self._batch_mahalanobis(centered)
        self.mahal_cutoff = float(
            np.percentile(train_dists.numpy(), self.mahal_pct * 100)
        )
        print(f"  Pair vectors: {N} × {D}")
        print(f"  Mahalanobis cutoff ({self.mahal_pct:.0%}): "
              f"{self.mahal_cutoff:.2f}")
        print(f"  Training distances: median={float(train_dists.median()):.2f}, "
              f"max={float(train_dists.max()):.2f}")

        # 2. Fingerprints from training SMILES
        self.train_solute_smiles = set()
        self.train_solvent_smiles = set()

        for sol_b, slv_b, tgt in train_loader:
            for i in range(len(sol_b.smiles) if hasattr(sol_b, "smiles") else 0):
                self.train_solute_smiles.add(sol_b.smiles[i])
                self.train_solvent_smiles.add(slv_b.smiles[i])

        # If SMILES not stored in batch, extract from dataset
        if len(self.train_solute_smiles) == 0:
            ds = train_loader.dataset
            if hasattr(ds, "df"):
                self.train_solute_smiles = set(ds.df["solute_smiles"].unique())
                self.train_solvent_smiles = set(ds.df["solvent_smiles"].unique())

        self.train_solute_fps = []
        for smi in self.train_solute_smiles:
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                self.train_solute_fps.append(fp)

        self.train_solvent_fps = []
        for smi in self.train_solvent_smiles:
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                self.train_solvent_fps.append(fp)

        print(f"  Solute fingerprints:  {len(self.train_solute_fps)}")
        print(f"  Solvent fingerprints: {len(self.train_solvent_fps)}")

        self.is_fitted = True
        print("  AD fitting complete.")

    # -------------------------------------------------------------- #
    #  Mahalanobis distance                                           #
    # -------------------------------------------------------------- #

    def _batch_mahalanobis(self, centered: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance for centered vectors. Returns (N,)."""
        # d² = x^T Σ⁻¹ x
        left = centered @ self.cov_inv  # (N, D)
        d_sq = (left * centered).sum(dim=-1)  # (N,)
        return torch.sqrt(d_sq.clamp(min=0))

    @torch.no_grad()
    def _query_pair_vector(
        self, solute_smiles: str, solvent_smiles: str, T: float
    ) -> torch.Tensor:
        """Get pair vector for a single query."""
        self.model.eval()
        sol_g = smiles_to_graph(
            solute_smiles,
            use_gasteiger_charges=bool(getattr(self.model.cfg, "use_gasteiger_charges", False)),
            use_phys_edge_features=bool(getattr(self.model.cfg, "use_phys_edge_features", False)),
            explicit_h_small_molecules=bool(getattr(self.model.cfg, "explicit_h_small_molecules", False)),
            explicit_h_max_heavy_atoms=int(getattr(self.model.cfg, "explicit_h_max_heavy_atoms", 3)),
        )
        slv_g = smiles_to_graph(
            solvent_smiles,
            use_gasteiger_charges=bool(getattr(self.model.cfg, "use_gasteiger_charges", False)),
            use_phys_edge_features=bool(getattr(self.model.cfg, "use_phys_edge_features", False)),
            explicit_h_small_molecules=bool(getattr(self.model.cfg, "explicit_h_small_molecules", False)),
            explicit_h_max_heavy_atoms=int(getattr(self.model.cfg, "explicit_h_max_heavy_atoms", 3)),
        )
        if sol_g is None or slv_g is None:
            raise ValueError("Invalid SMILES")

        sol_b = Batch.from_data_list([sol_g]).to(self.device)
        slv_b = Batch.from_data_list([slv_g]).to(self.device)
        T_tensor = torch.tensor([T], device=self.device)
        t_feat = make_temperature_features(T_tensor)

        _, g_sol_payload, _, _ = self.model._encode_and_readout(
            sol_b, "solute", temp_feat=t_feat
        )
        _, g_slv_payload, _, _ = self.model._encode_and_readout(
            slv_b, "solvent", temp_feat=t_feat
        )
        g_pair = self.model._build_pair_representation(
            g_sol_payload,
            g_slv_payload,
        )
        if getattr(self.model, "solvent_moe", None) is not None:
            solvent_type = torch.tensor(
                [solvent_type_id_from_smiles(solvent_smiles)],
                device=g_pair.device,
                dtype=torch.long,
            )
            g_pair, _ = self.model.solvent_moe(g_pair, solvent_type)
        return g_pair.cpu().squeeze(0)  # (D,)

    # -------------------------------------------------------------- #
    #  Score a query                                                  #
    # -------------------------------------------------------------- #

    def score(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        T: float = 298.15,
    ) -> ScoreDict:
        """
        Score how "in-domain" a query system is.

        Returns
        -------
        dict with:
          mahalanobis        — distance to training centroid
          mahalanobis_cutoff — threshold for in-domain
          tanimoto_solute    — max similarity to nearest training solute
          tanimoto_solvent   — max similarity to nearest training solvent
          solute_seen        — whether exact solute is in training set
          solvent_seen       — whether exact solvent is in training set
          in_domain          — overall verdict (bool)
          confidence         — 0.0 (far OOD) to 1.0 (clearly in-domain)
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit(train_loader) first")

        # Mahalanobis distance
        pair_vec = self._query_pair_vector(solute_smiles, solvent_smiles, T)
        centered = pair_vec - self.centroid
        mahal = self._batch_mahalanobis(centered.unsqueeze(0)).item()

        # Tanimoto similarities
        sol_fp = self._smiles_to_fp(solute_smiles)
        slv_fp = self._smiles_to_fp(solvent_smiles)
        tani_sol = self._max_tanimoto(sol_fp, self.train_solute_fps)
        tani_slv = self._max_tanimoto(slv_fp, self.train_solvent_fps)

        # Exact match flags
        solute_seen = solute_smiles in self.train_solute_smiles
        solvent_seen = solvent_smiles in self.train_solvent_smiles

        # Overall in-domain verdict
        mahal_ok = mahal <= self.mahal_cutoff
        tani_ok = (tani_sol >= self.tani_threshold and
                   tani_slv >= self.tani_threshold)
        in_domain = mahal_ok and tani_ok

        # Confidence score: 0 (far OOD) to 1 (clearly in-domain)
        mahal_conf = max(0.0, 1.0 - mahal / (self.mahal_cutoff * 2.0))
        tani_conf = min(tani_sol, tani_slv)
        confidence = 0.5 * mahal_conf + 0.5 * tani_conf

        return {
            "mahalanobis": mahal,
            "mahalanobis_cutoff": self.mahal_cutoff,
            "tanimoto_solute": tani_sol,
            "tanimoto_solvent": tani_slv,
            "solute_seen": solute_seen,
            "solvent_seen": solvent_seen,
            "in_domain": in_domain,
            "confidence": confidence,
        }

    def report(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        T: float = 298.15,
    ) -> str:
        """Human-readable AD report."""
        s = self.score(solute_smiles, solvent_smiles, T)
        lines = [
            "=" * 50,
            "Applicability Domain Report",
            "=" * 50,
            f"Solute:  {solute_smiles}",
            f"Solvent: {solvent_smiles}",
            "",
            f"Mahalanobis distance:  {s['mahalanobis']:.2f}  "
            f"(cutoff: {s['mahalanobis_cutoff']:.2f})  "
            f"{'OK' if s['mahalanobis'] <= s['mahalanobis_cutoff'] else 'OOD'}",
            f"Tanimoto (solute):     {s['tanimoto_solute']:.3f}  "
            f"(min: {self.tani_threshold})  "
            f"{'OK' if s['tanimoto_solute'] >= self.tani_threshold else 'LOW'}",
            f"Tanimoto (solvent):    {s['tanimoto_solvent']:.3f}  "
            f"{'OK' if s['tanimoto_solvent'] >= self.tani_threshold else 'LOW'}",
            f"Solute in train:       {'Yes' if s['solute_seen'] else 'No'}",
            f"Solvent in train:      {'Yes' if s['solvent_seen'] else 'No'}",
            "",
            f"Verdict:    {'IN DOMAIN' if s['in_domain'] else 'OUT OF DOMAIN'}",
            f"Confidence: {s['confidence']:.2f}",
        ]
        if not s["in_domain"]:
            lines.append("")
            lines.append("⚠ Prediction may be unreliable!")
            if s["mahalanobis"] > s["mahalanobis_cutoff"]:
                lines.append(
                    "  → System is far from training distribution "
                    "in latent space"
                )
            if s["tanimoto_solute"] < self.tani_threshold:
                lines.append(
                    "  → Solute structure is dissimilar to any "
                    "training solute"
                )
            if s["tanimoto_solvent"] < self.tani_threshold:
                lines.append(
                    "  → Solvent not well-represented in training data"
                )
        return "\n".join(lines)
