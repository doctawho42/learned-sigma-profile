"""
PyTorch Dataset and DataLoader factories for TGNN-Solv.

TGNNSolvDataset wraps a unified DataFrame and produces
(solute_graph, solvent_graph, targets_dict) triples.

All auxiliary targets carry boolean masks so the loss function
can skip missing values.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from typing import TypeAlias

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch, Data

from ..features import (
    compute_molecular_descriptors,
    smiles_to_descriptor_prior_features,
    smiles_to_graph,
    smiles_to_group_prior_features,
    smiles_to_morgan_fp,
)
from ..group_contribution import GC_FALLBACK_PRIORS, compute_gc_priors
from ..hansen_contrastive import pseudo_hansen_from_smiles
from ..ionic_features import (
    IONIC_FEATURE_DIM,
    compute_ionic_features,
    flag_no_melting_point,
)
from .source_uncertainty import attach_source_uncertainty
from .solvent_types import solvent_type_id_from_smiles

TargetValue: TypeAlias = torch.Tensor | str
BatchTargetValue: TypeAlias = torch.Tensor | list[str]
TargetDict: TypeAlias = dict[str, TargetValue]
Sample: TypeAlias = tuple[Data, Data, TargetDict]
BatchTargetDict: TypeAlias = dict[str, BatchTargetValue]
BatchTriplet: TypeAlias = tuple[Batch, Batch, BatchTargetDict]


class PairTemperatureBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps multi-temperature solute/solvent pairs together."""

    def __init__(
        self,
        pair_keys: Sequence[str],
        batch_size: int,
        *,
        drop_last: bool = False,
        min_group_size: int = 2,
        group_chunk_size: int = 4,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")
        if group_chunk_size < min_group_size:
            raise ValueError("group_chunk_size must be >= min_group_size")

        self.pair_keys = [str(key) for key in pair_keys]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.min_group_size = min_group_size
        self.group_chunk_size = max(min(group_chunk_size, batch_size), min_group_size)
        self.seed = seed
        self._epoch = 0
        self._groups = self._build_groups()

    def _build_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx, key in enumerate(self.pair_keys):
            groups.setdefault(key, []).append(idx)
        return groups

    def _split_group(
        self,
        indices: list[int],
        leftovers: list[int],
    ) -> list[list[int]]:
        chunks: list[list[int]] = []
        start = 0
        remaining = len(indices)

        while remaining >= self.min_group_size:
            take = min(self.group_chunk_size, remaining)
            if remaining - take == 1:
                take -= 1
            if take < self.min_group_size:
                break
            chunks.append(indices[start:start + take])
            start += take
            remaining -= take

        if start < len(indices):
            leftovers.extend(indices[start:])

        return chunks

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.pair_keys) // self.batch_size
        return math.ceil(len(self.pair_keys) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        pair_chunks: list[list[int]] = []
        leftovers: list[int] = []
        for indices in self._groups.values():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            if len(shuffled) < self.min_group_size:
                leftovers.extend(shuffled)
                continue
            pair_chunks.extend(self._split_group(shuffled, leftovers))

        rng.shuffle(pair_chunks)
        pair_chunks.sort(key=len, reverse=True)
        rng.shuffle(leftovers)

        candidate_batches: list[list[int]] = []
        current_batch: list[int] = []
        for chunk in pair_chunks:
            if len(current_batch) + len(chunk) > self.batch_size:
                if current_batch:
                    candidate_batches.append(current_batch)
                current_batch = []
            current_batch.extend(chunk)
        if current_batch:
            candidate_batches.append(current_batch)

        leftover_cursor = 0
        finalized_batches: list[list[int]] = []
        for batch in candidate_batches:
            while len(batch) < self.batch_size and leftover_cursor < len(leftovers):
                batch.append(leftovers[leftover_cursor])
                leftover_cursor += 1
            if len(batch) == self.batch_size or not self.drop_last:
                rng.shuffle(batch)
                finalized_batches.append(batch)

        remaining = leftovers[leftover_cursor:]
        while remaining:
            batch = remaining[:self.batch_size]
            remaining = remaining[self.batch_size:]
            if len(batch) == self.batch_size or not self.drop_last:
                rng.shuffle(batch)
                finalized_batches.append(batch)

        if not finalized_batches and not self.drop_last and not self.pair_keys:
            return

        rng.shuffle(finalized_batches)
        yield from finalized_batches


class TGNNSolvDataset(Dataset):
    """
    PyTorch Dataset for TGNN-Solv.

    Each sample is a tuple:
      (solute_graph, solvent_graph, targets_dict)

    targets_dict keys:
      T              — temperature (K)
      ln_x2          — log mole fraction (0.0 if no solubility data)
      has_solubility — bool mask
      pair_key       — string key "{solute}>>{solvent}" for same-pair temperature losses
      solvent_type   — int solvent type id
      T_m, T_m_mask
      dH_fus, dH_mask
      hansen_sol (3,), hansen_mask
      hansen_*_effective, pair_Ra, pair_hansen_mask for Hansen contrastive
      ln_gamma_inf, gamma_mask
      ln_gamma_2_target, gamma2_mask, activity_x2
      dG_solv, dG_mask
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache: bool = True,
        *,
        use_morgan_features: bool = False,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        use_descriptor_augmentation: bool = False,
        use_ionic_features: bool = False,
        use_descriptor_priors: bool = False,
        use_group_priors: bool = False,
        use_gc_priors_crystal: bool = False,
        use_gasteiger_charges: bool = False,
        use_phys_edge_features: bool = False,
        explicit_h_small_molecules: bool = False,
        explicit_h_max_heavy_atoms: int = 3,
        use_pseudo_hansen: bool = False,
        pseudo_hansen_weight_discount: float = 0.3,
        source_uncertainty_csv: str = "",
        source_uncertainty_weight_mode: str = "inverse_variance",
        source_uncertainty_default_sigma_ln_x2: float = 0.75,
        source_uncertainty_min_sigma_ln_x2: float = 0.20,
        source_uncertainty_min_weight: float = 0.25,
        source_uncertainty_max_weight: float = 4.0,
        expected_sigma_bins: int | None = None,
    ) -> None:
        self.cache: dict[str, Data] | None = {} if cache else None
        self.use_morgan_features = use_morgan_features
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.use_descriptor_augmentation = use_descriptor_augmentation
        self.use_ionic_features = use_ionic_features
        self.use_descriptor_priors = use_descriptor_priors
        self.use_group_priors = use_group_priors
        self.use_gc_priors_crystal = use_gc_priors_crystal
        self.use_gasteiger_charges = use_gasteiger_charges
        self.use_phys_edge_features = use_phys_edge_features
        self.explicit_h_small_molecules = explicit_h_small_molecules
        self.explicit_h_max_heavy_atoms = int(explicit_h_max_heavy_atoms)
        self.use_pseudo_hansen = use_pseudo_hansen
        self.pseudo_hansen_weight_discount = float(pseudo_hansen_weight_discount)
        self.expected_sigma_bins = expected_sigma_bins
        if str(source_uncertainty_csv).strip():
            df = attach_source_uncertainty(
                df,
                uncertainty_csv=str(source_uncertainty_csv).strip(),
                weight_mode=source_uncertainty_weight_mode,
                default_sigma_ln_x2=float(source_uncertainty_default_sigma_ln_x2),
                min_sigma_ln_x2=float(source_uncertainty_min_sigma_ln_x2),
                min_weight=float(source_uncertainty_min_weight),
                max_weight=float(source_uncertainty_max_weight),
                strict_for_supervised=True,
            )
        self.fp_cache: dict[str, torch.Tensor] | None = {} if cache and use_morgan_features else None
        self.descriptor_aug_cache: dict[str, torch.Tensor] | None = (
            {} if cache and use_descriptor_augmentation else None
        )
        self.ionic_feature_cache: dict[tuple[str, str], torch.Tensor] | None = (
            {} if cache and use_ionic_features else None
        )
        self.descriptor_cache: dict[str, torch.Tensor] | None = (
            {} if cache and use_descriptor_priors else None
        )
        self.group_prior_cache: dict[str, torch.Tensor] | None = (
            {} if cache and use_group_priors else None
        )
        self.crystal_gc_cache: dict[str, torch.Tensor] | None = (
            {} if cache and use_gc_priors_crystal else None
        )
        self.pseudo_hansen_cache: dict[str, torch.Tensor | None] | None = (
            {} if cache and use_pseudo_hansen else None
        )

        # Validate all SMILES upfront (fast: uses cache after first pass)
        valid = []
        for row in df.itertuples():
            sol_ok = self._graph(row.solute_smiles) is not None
            slv_ok = self._graph(row.solvent_smiles) is not None
            if sol_ok and slv_ok:
                valid.append(row.Index)

        self.df = df.loc[valid].reset_index(drop=True)
        # Precompute solvent type ids (for MoE routing)
        unique_solv = self.df["solvent_smiles"].unique()
        type_map = {
            smi: solvent_type_id_from_smiles(smi) for smi in unique_solv
        }
        self.df["solvent_type"] = self.df["solvent_smiles"].map(type_map)
        self.df["pair_key"] = (
            self.df["solute_smiles"].astype(str)
            + ">>"
            + self.df["solvent_smiles"].astype(str)
        )
        self.pair_keys: list[str] = self.df["pair_key"].tolist()
        n_drop = len(df) - len(valid)
        n_cached = len(self.cache) if self.cache else 0
        print(
            f"  Dataset: {len(self.df):,} valid "
            f"({n_drop} dropped, {n_cached} cached graphs)"
        )

    def _graph(self, smi: str) -> Data | None:
        """Get graph from cache or build it."""
        if self.cache is not None and smi in self.cache:
            return self.cache[smi]
        try:
            g = smiles_to_graph(
                smi,
                use_gasteiger_charges=self.use_gasteiger_charges,
                use_phys_edge_features=self.use_phys_edge_features,
                explicit_h_small_molecules=self.explicit_h_small_molecules,
                explicit_h_max_heavy_atoms=self.explicit_h_max_heavy_atoms,
            )
        except TypeError:
            # Legacy tests and some external callers monkeypatch `smiles_to_graph`
            # with the historical single-argument signature.
            g = smiles_to_graph(smi)
        if self.cache is not None and g is not None:
            self.cache[smi] = g
        return g

    def _morgan_fp(self, smi: str) -> torch.Tensor | None:
        """Get a Morgan fingerprint tensor from cache or compute it."""
        if not self.use_morgan_features:
            return None
        if self.fp_cache is not None and smi in self.fp_cache:
            return self.fp_cache[smi]

        fp = smiles_to_morgan_fp(
            smi,
            radius=self.morgan_radius,
            n_bits=self.morgan_n_bits,
        )
        if fp is None:
            return None
        fp_tensor = torch.tensor(fp, dtype=torch.float)
        if self.fp_cache is not None:
            self.fp_cache[smi] = fp_tensor
        return fp_tensor

    def _descriptor_features(self, smi: str) -> torch.Tensor | None:
        """Get cached full RDKit descriptor features for DirectGNN augmentation."""
        if not self.use_descriptor_augmentation:
            return None
        if self.descriptor_aug_cache is not None and smi in self.descriptor_aug_cache:
            return self.descriptor_aug_cache[smi]

        features = compute_molecular_descriptors(smi)
        if features is None:
            return None
        tensor = torch.tensor(features, dtype=torch.float)
        if self.descriptor_aug_cache is not None:
            self.descriptor_aug_cache[smi] = tensor
        return tensor

    def _ionic_features(self, solute_smi: str, solvent_smi: str) -> torch.Tensor | None:
        """Get cached ionic/contact-pair context features for one pair."""
        if not self.use_ionic_features:
            return None
        key = (str(solute_smi), str(solvent_smi))
        if self.ionic_feature_cache is not None and key in self.ionic_feature_cache:
            return self.ionic_feature_cache[key]
        features = compute_ionic_features(key[0], key[1])
        tensor = torch.tensor(features, dtype=torch.float)
        if tensor.numel() != IONIC_FEATURE_DIM:
            raise ValueError("Unexpected ionic feature dimensionality.")
        if self.ionic_feature_cache is not None:
            self.ionic_feature_cache[key] = tensor
        return tensor

    def _descriptor_prior_features(self, smi: str) -> torch.Tensor | None:
        """Get cached fixed descriptor features for prior-conditioned heads."""
        if not self.use_descriptor_priors:
            return None
        if self.descriptor_cache is not None and smi in self.descriptor_cache:
            return self.descriptor_cache[smi]

        features = smiles_to_descriptor_prior_features(smi)
        if features is None:
            return None
        tensor = torch.tensor(features, dtype=torch.float)
        if self.descriptor_cache is not None:
            self.descriptor_cache[smi] = tensor
        return tensor

    def _group_prior_features(self, smi: str) -> torch.Tensor | None:
        """Get cached fixed group-count features for group priors."""
        if not self.use_group_priors:
            return None
        if self.group_prior_cache is not None and smi in self.group_prior_cache:
            return self.group_prior_cache[smi]

        features = smiles_to_group_prior_features(smi)
        if features is None:
            return None
        tensor = torch.tensor(features, dtype=torch.float)
        if self.group_prior_cache is not None:
            self.group_prior_cache[smi] = tensor
        return tensor

    def _crystal_gc_priors(self, smi: str) -> torch.Tensor | None:
        """Get cached fixed crystal GC priors for the solute."""
        if not self.use_gc_priors_crystal:
            return None
        if self.crystal_gc_cache is not None and smi in self.crystal_gc_cache:
            return self.crystal_gc_cache[smi]

        priors = compute_gc_priors(smi)
        if any(priors[key] is None for key in ("T_m_gc", "dH_fus_gc", "dCp_fus_gc")):
            priors = GC_FALLBACK_PRIORS
        tensor = torch.tensor(
            [priors["T_m_gc"], priors["dH_fus_gc"], priors["dCp_fus_gc"]],
            dtype=torch.float,
        )
        if self.crystal_gc_cache is not None:
            self.crystal_gc_cache[smi] = tensor
        return tensor

    def _pseudo_hansen(self, smi: str) -> torch.Tensor | None:
        """Get cached pseudo-Hansen labels for contrastive regularization."""
        if not self.use_pseudo_hansen:
            return None
        if self.pseudo_hansen_cache is not None and smi in self.pseudo_hansen_cache:
            return self.pseudo_hansen_cache[smi]
        values = pseudo_hansen_from_smiles(smi)
        tensor = (
            torch.tensor(values, dtype=torch.float)
            if values is not None
            else None
        )
        if self.pseudo_hansen_cache is not None:
            self.pseudo_hansen_cache[smi] = tensor
        return tensor

    @staticmethod
    def _finite_triplet(values: list[object]) -> torch.Tensor | None:
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in floats):
            return None
        return torch.tensor(floats, dtype=torch.float)

    @staticmethod
    def _row_bool(r: pd.Series, names: Sequence[str]) -> bool | None:
        for name in names:
            if name in r.index and pd.notna(r[name]):
                value = r[name]
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in {"true", "1", "yes", "y", "t"}:
                        return True
                    if normalized in {"false", "0", "no", "n", "f", ""}:
                        return False
                return bool(value)
        return None

    def _crystal_validity_flags(self, r: pd.Series) -> dict[str, object]:
        """Resolve valid-fusion vs decomposition flags from columns and curation."""
        raw_has_tm = bool(self._row_bool(r, ("has_T_m",)) or False)
        raw_has_dh = bool(self._row_bool(r, ("has_dH_fus",)) or False)
        known_handling = flag_no_melting_point(str(r["solute_smiles"]))
        known_decomposition = known_handling != "standard"

        column_decomposition = self._row_bool(
            r,
            (
                "has_decomposition_T",
                "has_decomposition_temperature",
                "T_m_is_decomposition",
                "is_decomposition_temperature",
            ),
        )
        text_decomposition = False
        for name in ("T_m_type", "melting_point_type", "crystal_status", "phase_change_type"):
            if name in r.index and pd.notna(r[name]):
                text = str(r[name]).strip().lower()
                if "decomp" in text or "degrad" in text:
                    text_decomposition = True
                    break

        has_decomposition = bool(
            known_decomposition
            or (column_decomposition is True)
            or text_decomposition
        )
        explicit_valid_tm = self._row_bool(
            r,
            ("has_valid_T_m", "has_valid_tm", "valid_T_m", "T_m_valid"),
        )
        explicit_valid_dh = self._row_bool(
            r,
            (
                "has_valid_dH_fus",
                "has_valid_dh_fus",
                "valid_dH_fus",
                "dH_fus_valid",
            ),
        )

        has_valid_tm = raw_has_tm and not has_decomposition
        has_valid_dh = raw_has_dh and not has_decomposition
        if explicit_valid_tm is not None:
            has_valid_tm = bool(explicit_valid_tm) and not has_decomposition
        if explicit_valid_dh is not None:
            has_valid_dh = bool(explicit_valid_dh) and not has_decomposition

        return {
            "has_raw_T_m": raw_has_tm,
            "has_raw_dH_fus": raw_has_dh,
            "has_valid_T_m": has_valid_tm,
            "has_valid_dH_fus": has_valid_dh,
            "has_decomposition_T": has_decomposition,
            "crystal_handling": known_handling,
        }

    def _row_hansen_triplet(
        self,
        r: pd.Series,
        *,
        prefixes: Sequence[tuple[str, str, str]],
        masks: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read optional experimental Hansen triplets from flexible column names."""
        mask_value = self._row_bool(r, masks)
        for d_col, p_col, h_col in prefixes:
            if {d_col, p_col, h_col} <= set(r.index):
                triplet = self._finite_triplet([r[d_col], r[p_col], r[h_col]])
                if triplet is not None and (mask_value is None or mask_value):
                    return triplet, torch.tensor(True, dtype=torch.bool)
        return torch.zeros(3, dtype=torch.float), torch.tensor(False, dtype=torch.bool)

    def _effective_hansen(
        self,
        real_values: torch.Tensor,
        real_mask: torch.Tensor,
        smiles: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return values, mask, and confidence weight for contrastive Hansen."""
        if bool(real_mask):
            return (
                real_values,
                torch.tensor(True, dtype=torch.bool),
                torch.tensor(1.0, dtype=torch.float),
            )
        pseudo = self._pseudo_hansen(smiles)
        if pseudo is None:
            return (
                torch.zeros(3, dtype=torch.float),
                torch.tensor(False, dtype=torch.bool),
                torch.tensor(0.0, dtype=torch.float),
            )
        return (
            pseudo,
            torch.tensor(True, dtype=torch.bool),
            torch.tensor(self.pseudo_hansen_weight_discount, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        r = self.df.iloc[idx]

        # Early preflight: validate sigma-profile columns before any other work.
        _sig_cols_early = sorted(
            (c for c in r.index if str(c).startswith("sigma_p_")),
            key=lambda c: int(str(c).rsplit("_", 1)[-1]),
        )
        if _sig_cols_early:
            if (
                self.expected_sigma_bins is not None
                and len(_sig_cols_early) != self.expected_sigma_bins
            ):
                raise ValueError(
                    f"sigma-profile row has {len(_sig_cols_early)} sigma_p_* columns but "
                    f"cfg.cosmo_sac_n_bins={self.expected_sigma_bins}; the profile "
                    f"grid is mis-registered against the COSMO-SAC layer grid."
                )
            _has_sig_early = bool(self._row_bool(r, ("has_sigma_profile",)) or False)
            if _has_sig_early and not (
                "sigma_area" in r.index and pd.notna(r["sigma_area"])
            ):
                raise ValueError(
                    "sigma-profile row has has_sigma_profile=True but is missing a "
                    "valid 'sigma_area'; refusing to default it to 0.0."
                )

        sol_g = self._graph(r["solute_smiles"]).clone()
        slv_g = self._graph(r["solvent_smiles"]).clone()
        sol_fp = self._morgan_fp(r["solute_smiles"])
        slv_fp = self._morgan_fp(r["solvent_smiles"])
        sol_aug_desc = self._descriptor_features(r["solute_smiles"])
        slv_aug_desc = self._descriptor_features(r["solvent_smiles"])
        ionic_features = self._ionic_features(r["solute_smiles"], r["solvent_smiles"])
        sol_desc = self._descriptor_prior_features(r["solute_smiles"])
        slv_desc = self._descriptor_prior_features(r["solvent_smiles"])
        sol_group = self._group_prior_features(r["solute_smiles"])
        slv_group = self._group_prior_features(r["solvent_smiles"])
        sol_gc = self._crystal_gc_priors(r["solute_smiles"])
        hansen_sol_triplet = self._finite_triplet(
            [r["hansen_d"], r["hansen_p"], r["hansen_h"]]
        )
        hansen_sol_real = (
            hansen_sol_triplet
            if hansen_sol_triplet is not None
            else torch.zeros(3, dtype=torch.float)
        )
        hansen_sol_mask = torch.tensor(
            bool(r["has_hansen"]) and hansen_sol_triplet is not None,
            dtype=torch.bool,
        )
        hansen_slv_real, hansen_slv_mask = self._row_hansen_triplet(
            r,
            prefixes=(
                ("solvent_hansen_d", "solvent_hansen_p", "solvent_hansen_h"),
                ("hansen_slv_d", "hansen_slv_p", "hansen_slv_h"),
                ("slv_hansen_d", "slv_hansen_p", "slv_hansen_h"),
            ),
            masks=("has_solvent_hansen", "has_hansen_slv", "slv_hansen_mask"),
        )
        hansen_sol_eff, hansen_sol_contrastive_mask, hansen_sol_weight = (
            self._effective_hansen(
                hansen_sol_real,
                hansen_sol_mask,
                str(r["solute_smiles"]),
            )
        )
        hansen_slv_eff, hansen_slv_contrastive_mask, hansen_slv_weight = (
            self._effective_hansen(
                hansen_slv_real,
                hansen_slv_mask,
                str(r["solvent_smiles"]),
            )
        )
        pseudo_hansen_sol = self._pseudo_hansen(str(r["solute_smiles"]))
        pseudo_hansen_slv = self._pseudo_hansen(str(r["solvent_smiles"]))
        pseudo_hansen_sol_mask = pseudo_hansen_sol is not None
        pseudo_hansen_slv_mask = pseudo_hansen_slv is not None
        pseudo_hansen_sol = (
            pseudo_hansen_sol
            if pseudo_hansen_sol is not None
            else torch.zeros(3, dtype=torch.float)
        )
        pseudo_hansen_slv = (
            pseudo_hansen_slv
            if pseudo_hansen_slv is not None
            else torch.zeros(3, dtype=torch.float)
        )
        pair_hansen_mask = bool(hansen_sol_contrastive_mask.item()) and bool(
            hansen_slv_contrastive_mask.item()
        )
        if pair_hansen_mask:
            diff = hansen_sol_eff - hansen_slv_eff
            pair_Ra = torch.sqrt(
                4.0 * diff[0].pow(2) + diff[1].pow(2) + diff[2].pow(2)
            )
            pair_weight = hansen_sol_weight * hansen_slv_weight
        else:
            pair_Ra = torch.tensor(0.0, dtype=torch.float)
            pair_weight = torch.tensor(0.0, dtype=torch.float)
        crystal_flags = self._crystal_validity_flags(r)
        has_valid_T_m = bool(crystal_flags["has_valid_T_m"])
        has_valid_dH_fus = bool(crystal_flags["has_valid_dH_fus"])

        t = {
            "T": torch.tensor(float(r["temperature"]), dtype=torch.float),
            "ln_x2": torch.tensor(
                float(r["ln_x2"]) if r["has_solubility"] else 0.0,
                dtype=torch.float,
            ),
            "has_solubility": torch.tensor(
                bool(r["has_solubility"]), dtype=torch.bool
            ),
            "solvent_type": torch.tensor(
                int(r["solvent_type"]), dtype=torch.long
            ),
            "T_m": torch.tensor(
                float(r["T_m"]) if has_valid_T_m else 0.0,
                dtype=torch.float,
            ),
            "T_m_mask": torch.tensor(
                has_valid_T_m, dtype=torch.bool
            ),
            "has_T_m": torch.tensor(
                has_valid_T_m, dtype=torch.bool
            ),
            "has_raw_T_m": torch.tensor(
                bool(crystal_flags["has_raw_T_m"]), dtype=torch.bool
            ),
            "has_valid_T_m": torch.tensor(has_valid_T_m, dtype=torch.bool),
            "has_decomposition_T": torch.tensor(
                bool(crystal_flags["has_decomposition_T"]), dtype=torch.bool
            ),
            "dH_fus": torch.tensor(float(r["dH_fus"]), dtype=torch.float),
            "dH_mask": torch.tensor(
                has_valid_dH_fus, dtype=torch.bool
            ),
            "has_dH_fus": torch.tensor(
                has_valid_dH_fus, dtype=torch.bool
            ),
            "has_raw_dH_fus": torch.tensor(
                bool(crystal_flags["has_raw_dH_fus"]), dtype=torch.bool
            ),
            "has_valid_dH_fus": torch.tensor(has_valid_dH_fus, dtype=torch.bool),
            "hansen_sol": hansen_sol_real,
            "hansen_mask": hansen_sol_mask,
            "hansen_slv": hansen_slv_real,
            "hansen_slv_mask": hansen_slv_mask,
            "hansen_sol_effective": hansen_sol_eff,
            "hansen_slv_effective": hansen_slv_eff,
            "hansen_contrastive_mask": hansen_sol_contrastive_mask,
            "hansen_slv_contrastive_mask": hansen_slv_contrastive_mask,
            "hansen_sol_contrastive_weight": hansen_sol_weight,
            "hansen_slv_contrastive_weight": hansen_slv_weight,
            "pseudo_hansen_params": pseudo_hansen_sol,
            "pseudo_hansen_mask": torch.tensor(
                pseudo_hansen_sol_mask,
                dtype=torch.bool,
            ),
            "pseudo_hansen_slv": pseudo_hansen_slv,
            "pseudo_hansen_slv_mask": torch.tensor(
                pseudo_hansen_slv_mask,
                dtype=torch.bool,
            ),
            "pair_Ra": pair_Ra,
            "pair_hansen_mask": torch.tensor(pair_hansen_mask, dtype=torch.bool),
            "pair_hansen_weight": pair_weight,
            "ln_gamma_inf": torch.tensor(
                float(r["ln_gamma_inf"]), dtype=torch.float
            ),
            "gamma_mask": torch.tensor(
                bool(r["has_gamma_inf"]), dtype=torch.bool
            ),
            "gamma_weight": torch.tensor(
                float(r["gamma_weight"])
                if "gamma_weight" in r.index
                and pd.notna(r["gamma_weight"])
                else 1.0,
                dtype=torch.float,
            ),
            "ln_gamma_2_target": torch.tensor(
                float(r["ln_gamma_2"])
                if "ln_gamma_2" in r.index
                and pd.notna(r["ln_gamma_2"])
                else (
                    float(r["ln_gamma_unifac"])
                    if "ln_gamma_unifac" in r.index
                    and pd.notna(r["ln_gamma_unifac"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "gamma2_mask": torch.tensor(
                bool(r["has_gamma_2"])
                if "has_gamma_2" in r.index
                and pd.notna(r["has_gamma_2"])
                else (
                    ("ln_gamma_2" in r.index and pd.notna(r["ln_gamma_2"]))
                    or (
                        "ln_gamma_unifac" in r.index
                        and pd.notna(r["ln_gamma_unifac"])
                    )
                ),
                dtype=torch.bool,
            ),
            "gamma2_weight": torch.tensor(
                float(r["gamma2_weight"])
                if "gamma2_weight" in r.index
                and pd.notna(r["gamma2_weight"])
                else 1.0,
                dtype=torch.float,
            ),
            "activity_x2": torch.tensor(
                float(r["activity_x2"])
                if "activity_x2" in r.index
                and pd.notna(r["activity_x2"])
                else (
                    float(r["solute_mole_fraction"])
                    if "solute_mole_fraction" in r.index
                    and pd.notna(r["solute_mole_fraction"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "unifac_ln_gamma_inf": torch.tensor(
                float(r["unifac_ln_gamma_inf"])
                if "unifac_ln_gamma_inf" in r.index
                and pd.notna(r["unifac_ln_gamma_inf"])
                else 0.0,
                dtype=torch.float,
            ),
            "unifac_gamma_mask": torch.tensor(
                bool(r["has_unifac_gamma_inf"])
                if "has_unifac_gamma_inf" in r.index
                and pd.notna(r["has_unifac_gamma_inf"])
                else False,
                dtype=torch.bool,
            ),
            "vh_anchor_ln_x2": torch.tensor(
                float(r["vh_anchor_ln_x2"])
                if "vh_anchor_ln_x2" in r.index
                and pd.notna(r["vh_anchor_ln_x2"])
                else 0.0,
                dtype=torch.float,
            ),
            "vh_anchor_mask": torch.tensor(
                bool(r["has_vh_anchor"])
                if "has_vh_anchor" in r.index
                and pd.notna(r["has_vh_anchor"])
                else False,
                dtype=torch.bool,
            ),
            "vh_anchor_weight": torch.tensor(
                float(r["vh_anchor_weight"])
                if "vh_anchor_weight" in r.index
                and pd.notna(r["vh_anchor_weight"])
                else 1.0,
                dtype=torch.float,
            ),
            "vh_fit_slope": torch.tensor(
                float(r["vh_fit_slope"])
                if "vh_fit_slope" in r.index
                and pd.notna(r["vh_fit_slope"])
                else (
                    float(r["vh_anchor_slope"])
                    if "vh_anchor_slope" in r.index
                    and pd.notna(r["vh_anchor_slope"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "vh_fit_intercept": torch.tensor(
                float(r["vh_fit_intercept"])
                if "vh_fit_intercept" in r.index
                and pd.notna(r["vh_fit_intercept"])
                else (
                    float(r["vh_anchor_intercept"])
                    if "vh_anchor_intercept" in r.index
                    and pd.notna(r["vh_anchor_intercept"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "vh_fit_r2": torch.tensor(
                float(r["vh_fit_r2"])
                if "vh_fit_r2" in r.index
                and pd.notna(r["vh_fit_r2"])
                else (
                    float(r["vh_anchor_fit_r2"])
                    if "vh_anchor_fit_r2" in r.index
                    and pd.notna(r["vh_anchor_fit_r2"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "vh_fit_rmse": torch.tensor(
                float(r["vh_fit_rmse"])
                if "vh_fit_rmse" in r.index
                and pd.notna(r["vh_fit_rmse"])
                else (
                    float(r["vh_anchor_fit_rmse"])
                    if "vh_anchor_fit_rmse" in r.index
                    and pd.notna(r["vh_anchor_fit_rmse"])
                    else 0.0
                ),
                dtype=torch.float,
            ),
            "vh_fit_mask": torch.tensor(
                bool(r["has_vh_fit"])
                if "has_vh_fit" in r.index
                and pd.notna(r["has_vh_fit"])
                else (
                    "vh_fit_slope" in r.index
                    and pd.notna(r["vh_fit_slope"])
                    or "vh_anchor_slope" in r.index
                    and pd.notna(r["vh_anchor_slope"])
                ),
                dtype=torch.bool,
            ),
            "vh_fit_weight": torch.tensor(
                float(r["vh_fit_weight"])
                if "vh_fit_weight" in r.index
                and pd.notna(r["vh_fit_weight"])
                else (
                    float(r["vh_anchor_weight"])
                    if "vh_anchor_weight" in r.index
                    and pd.notna(r["vh_anchor_weight"])
                    else 1.0
                ),
                dtype=torch.float,
            ),
            "pair_key": str(r["pair_key"]),
            "solute_smiles": str(r["solute_smiles"]),
            "solvent_smiles": str(r["solvent_smiles"]),
            "source_method_guess": (
                str(r["source_method_guess"])
                if "source_method_guess" in r.index and pd.notna(r["source_method_guess"])
                else ""
            ),
            "source_detail": (
                str(r["source_detail"])
                if "source_detail" in r.index and pd.notna(r["source_detail"])
                else ""
            ),
            "crystal_handling": str(crystal_flags["crystal_handling"]),
            "source_sigma_ln_x2": torch.tensor(
                float(r["source_sigma_ln_x2"])
                if "source_sigma_ln_x2" in r.index and pd.notna(r["source_sigma_ln_x2"])
                else 0.75,
                dtype=torch.float,
            ),
            "source_solubility_weight": torch.tensor(
                float(r["source_solubility_weight"])
                if "source_solubility_weight" in r.index and pd.notna(r["source_solubility_weight"])
                else 1.0,
                dtype=torch.float,
            ),
        }
        if self.use_gc_priors_crystal:
            if sol_gc is None:
                raise ValueError(
                    "Crystal GC prior computation failed for a supposedly valid sample."
                )
            t["T_m_gc"] = sol_gc[0]
            t["dH_fus_gc"] = sol_gc[1]
            t["dCp_fus_gc"] = sol_gc[2]
        # External single-component sigma-profile labels (present only in the
        # sigma aux stream CSV). Bin count is inferred from the columns.
        sig_cols = sorted(
            (c for c in r.index if str(c).startswith("sigma_p_")),
            key=lambda c: int(str(c).rsplit("_", 1)[-1]),
        )
        if sig_cols:
            # Contract (bin count + sigma_area presence) is validated in the
            # early preflight at the top of __getitem__; do not duplicate it here.
            has_sig = bool(self._row_bool(r, ("has_sigma_profile",)) or False)
            t["sigma_profile_target"] = torch.tensor(
                [float(r[c]) for c in sig_cols], dtype=torch.float
            )
            t["sigma_area_target"] = torch.tensor(
                float(r["sigma_area"]) if has_sig else 0.0,
                dtype=torch.float,
            )
            t["sigma_profile_mask"] = torch.tensor(has_sig, dtype=torch.bool)
        if self.use_morgan_features:
            if sol_fp is None or slv_fp is None:
                raise ValueError(
                    "Morgan fingerprint computation failed for a supposedly valid sample."
                )
            t["solute_morgan_fp"] = sol_fp
            t["solvent_morgan_fp"] = slv_fp
        if self.use_descriptor_augmentation:
            if sol_aug_desc is None or slv_aug_desc is None:
                raise ValueError(
                    "Full RDKit descriptor computation failed for a supposedly valid sample."
                )
            t["solute_descriptors"] = sol_aug_desc
            t["solvent_descriptors"] = slv_aug_desc
        if self.use_ionic_features:
            if ionic_features is None:
                raise ValueError(
                    "Ionic feature computation failed for a supposedly valid sample."
                )
            t["ionic_features"] = ionic_features
        if self.use_descriptor_priors:
            if sol_desc is None or slv_desc is None:
                raise ValueError(
                    "Descriptor prior feature computation failed for a supposedly valid sample."
                )
            t["solute_descriptor_prior_features"] = sol_desc
            t["solvent_descriptor_prior_features"] = slv_desc
        if self.use_group_priors:
            if sol_group is None or slv_group is None:
                raise ValueError(
                    "Group prior feature computation failed for a supposedly valid sample."
                )
            t["solute_group_prior_features"] = sol_group
            t["solvent_group_prior_features"] = slv_group
        return sol_g, slv_g, t


def collate_fn(batch: list[Sample]) -> BatchTriplet:
    """
    Collate function for DataLoader.

    Batches solute/solvent graphs separately via PyG Batch,
    stacks all target tensors.
    """
    sol_gs, slv_gs, tgts = zip(*batch)

    sol_batch = Batch.from_data_list(list(sol_gs))
    slv_batch = Batch.from_data_list(list(slv_gs))

    t_batch = {}
    for key in tgts[0]:
        first_value = tgts[0][key]
        if isinstance(first_value, torch.Tensor):
            t_batch[key] = torch.stack([t[key] for t in tgts])
        else:
            t_batch[key] = [t[key] for t in tgts]

    return sol_batch, slv_batch, t_batch


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    cache: bool = True,
    drop_last: bool | None = None,
    use_pair_temperature_batching: bool = False,
    pair_temperature_min_group_size: int = 2,
    pair_temperature_group_chunk_size: int = 4,
    use_morgan_features: bool = False,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    use_descriptor_augmentation: bool = False,
    use_ionic_features: bool = False,
    use_descriptor_priors: bool = False,
    use_group_priors: bool = False,
    use_gc_priors_crystal: bool = False,
    use_gasteiger_charges: bool = False,
    use_phys_edge_features: bool = False,
    explicit_h_small_molecules: bool = False,
    explicit_h_max_heavy_atoms: int = 3,
    use_pseudo_hansen: bool = False,
    pseudo_hansen_weight_discount: float = 0.3,
    source_uncertainty_csv: str = "",
    source_uncertainty_weight_mode: str = "inverse_variance",
    source_uncertainty_default_sigma_ln_x2: float = 0.75,
    source_uncertainty_min_sigma_ln_x2: float = 0.20,
    source_uncertainty_min_weight: float = 0.25,
    source_uncertainty_max_weight: float = 4.0,
    expected_sigma_bins: int | None = None,
    seed: int = 42,
) -> DataLoader:
    """Create a single DataLoader with optional same-pair temperature batching."""
    dataset = TGNNSolvDataset(
        df,
        cache=cache,
        use_morgan_features=use_morgan_features,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        use_descriptor_augmentation=use_descriptor_augmentation,
        use_ionic_features=use_ionic_features,
        use_descriptor_priors=use_descriptor_priors,
        use_group_priors=use_group_priors,
        use_gc_priors_crystal=use_gc_priors_crystal,
        use_gasteiger_charges=use_gasteiger_charges,
        use_phys_edge_features=use_phys_edge_features,
        explicit_h_small_molecules=explicit_h_small_molecules,
        explicit_h_max_heavy_atoms=explicit_h_max_heavy_atoms,
        use_pseudo_hansen=use_pseudo_hansen,
        pseudo_hansen_weight_discount=pseudo_hansen_weight_discount,
        source_uncertainty_csv=source_uncertainty_csv,
        source_uncertainty_weight_mode=source_uncertainty_weight_mode,
        source_uncertainty_default_sigma_ln_x2=source_uncertainty_default_sigma_ln_x2,
        source_uncertainty_min_sigma_ln_x2=source_uncertainty_min_sigma_ln_x2,
        source_uncertainty_min_weight=source_uncertainty_min_weight,
        source_uncertainty_max_weight=source_uncertainty_max_weight,
        expected_sigma_bins=expected_sigma_bins,
    )

    if drop_last is None:
        drop_last = shuffle and len(dataset) > batch_size

    kwargs = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # The per-SMILES featurization cache (self.cache) lives on the dataset in each
    # worker process. Without persistent workers, workers are respawned every epoch
    # and their cache is discarded -> re-featurize every epoch (slower than num_workers=0,
    # which caches after epoch 1). Keep workers alive so the cache warms once and persists.
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    if shuffle and use_pair_temperature_batching:
        batch_sampler = PairTemperatureBatchSampler(
            dataset.pair_keys,
            batch_size=batch_size,
            drop_last=drop_last,
            min_group_size=pair_temperature_min_group_size,
            group_chunk_size=pair_temperature_group_chunk_size,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs,
    )


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 0,
    use_pair_temperature_batching: bool = False,
    pair_temperature_min_group_size: int = 2,
    pair_temperature_group_chunk_size: int = 4,
    use_morgan_features: bool = False,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    use_descriptor_augmentation: bool = False,
    use_ionic_features: bool = False,
    use_descriptor_priors: bool = False,
    use_group_priors: bool = False,
    use_gc_priors_crystal: bool = False,
    use_gasteiger_charges: bool = False,
    use_phys_edge_features: bool = False,
    explicit_h_small_molecules: bool = False,
    explicit_h_max_heavy_atoms: int = 3,
    use_pseudo_hansen: bool = False,
    pseudo_hansen_weight_discount: float = 0.3,
    source_uncertainty_csv: str = "",
    source_uncertainty_weight_mode: str = "inverse_variance",
    source_uncertainty_default_sigma_ln_x2: float = 0.75,
    source_uncertainty_min_sigma_ln_x2: float = 0.20,
    source_uncertainty_min_weight: float = 0.25,
    source_uncertainty_max_weight: float = 4.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders.

    Parameters
    ----------
    train_df, val_df, test_df : split DataFrames from scaffold_split
    batch_size : samples per batch
    num_workers : DataLoader workers (0 = main process)

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    print("\nCreating DataLoaders...")

    train_ld = make_loader(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        cache=True,
        drop_last=(len(train_df) > batch_size),
        use_pair_temperature_batching=use_pair_temperature_batching,
        pair_temperature_min_group_size=pair_temperature_min_group_size,
        pair_temperature_group_chunk_size=pair_temperature_group_chunk_size,
        use_morgan_features=use_morgan_features,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        use_descriptor_augmentation=use_descriptor_augmentation,
        use_ionic_features=use_ionic_features,
        use_descriptor_priors=use_descriptor_priors,
        use_group_priors=use_group_priors,
        use_gc_priors_crystal=use_gc_priors_crystal,
        use_gasteiger_charges=use_gasteiger_charges,
        use_phys_edge_features=use_phys_edge_features,
        explicit_h_small_molecules=explicit_h_small_molecules,
        explicit_h_max_heavy_atoms=explicit_h_max_heavy_atoms,
        use_pseudo_hansen=use_pseudo_hansen,
        pseudo_hansen_weight_discount=pseudo_hansen_weight_discount,
        source_uncertainty_csv=source_uncertainty_csv,
        source_uncertainty_weight_mode=source_uncertainty_weight_mode,
        source_uncertainty_default_sigma_ln_x2=source_uncertainty_default_sigma_ln_x2,
        source_uncertainty_min_sigma_ln_x2=source_uncertainty_min_sigma_ln_x2,
        source_uncertainty_min_weight=source_uncertainty_min_weight,
        source_uncertainty_max_weight=source_uncertainty_max_weight,
        seed=seed,
    )
    val_ld = make_loader(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache=True,
        drop_last=False,
        use_morgan_features=use_morgan_features,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        use_descriptor_augmentation=use_descriptor_augmentation,
        use_ionic_features=use_ionic_features,
        use_descriptor_priors=use_descriptor_priors,
        use_group_priors=use_group_priors,
        use_gc_priors_crystal=use_gc_priors_crystal,
        use_gasteiger_charges=use_gasteiger_charges,
        use_phys_edge_features=use_phys_edge_features,
        explicit_h_small_molecules=explicit_h_small_molecules,
        explicit_h_max_heavy_atoms=explicit_h_max_heavy_atoms,
        use_pseudo_hansen=use_pseudo_hansen,
        pseudo_hansen_weight_discount=pseudo_hansen_weight_discount,
        source_uncertainty_csv=source_uncertainty_csv,
        source_uncertainty_weight_mode=source_uncertainty_weight_mode,
        source_uncertainty_default_sigma_ln_x2=source_uncertainty_default_sigma_ln_x2,
        source_uncertainty_min_sigma_ln_x2=source_uncertainty_min_sigma_ln_x2,
        source_uncertainty_min_weight=source_uncertainty_min_weight,
        source_uncertainty_max_weight=source_uncertainty_max_weight,
    )
    test_ld = make_loader(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache=True,
        drop_last=False,
        use_morgan_features=use_morgan_features,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        use_descriptor_augmentation=use_descriptor_augmentation,
        use_ionic_features=use_ionic_features,
        use_descriptor_priors=use_descriptor_priors,
        use_group_priors=use_group_priors,
        use_gc_priors_crystal=use_gc_priors_crystal,
        use_gasteiger_charges=use_gasteiger_charges,
        use_phys_edge_features=use_phys_edge_features,
        explicit_h_small_molecules=explicit_h_small_molecules,
        explicit_h_max_heavy_atoms=explicit_h_max_heavy_atoms,
        use_pseudo_hansen=use_pseudo_hansen,
        pseudo_hansen_weight_discount=pseudo_hansen_weight_discount,
        source_uncertainty_csv=source_uncertainty_csv,
        source_uncertainty_weight_mode=source_uncertainty_weight_mode,
        source_uncertainty_default_sigma_ln_x2=source_uncertainty_default_sigma_ln_x2,
        source_uncertainty_min_sigma_ln_x2=source_uncertainty_min_sigma_ln_x2,
        source_uncertainty_min_weight=source_uncertainty_min_weight,
        source_uncertainty_max_weight=source_uncertainty_max_weight,
    )

    for name, frame, loader in [
        ("Train", train_df, train_ld),
        ("Val", val_df, val_ld),
        ("Test", test_df, test_ld),
    ]:
        print(f"  {name}: {len(frame):,d} samples, {len(loader):,d} batches")

    return train_ld, val_ld, test_ld
