"""Ionic-system flags and coarse solvent dielectric descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem


IONIC_FEATURE_DIM = 10


def canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None when parsing fails."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


SOLVENT_DIELECTRIC: dict[str, float] = {
    "O": 78.4,
    "CO": 32.6,
    "CCO": 24.6,
    "CC(C)=O": 20.7,
    "CCOC(C)=O": 6.0,
    "ClCCl": 8.9,
    "c1ccccc1": 2.3,
    "CCCCC": 1.8,
    "CS(C)=O": 46.7,
    "CC#N": 37.5,
    "CCOCC": 4.3,
    "CC(C)O": 18.3,
    "CCCCO": 17.8,
    "CC(C)(C)O": 12.5,
    "CC(C)C": 1.9,
    "CCCCCC": 1.9,
    "O=C1CCCC1": 39.0,
    "CN(C)C=O": 36.7,
    "CN(C)C(C)=O": 37.8,
    "CCOCCOCC": 7.2,
    "C1CCOC1": 7.6,
    "O=C(O)C": 6.2,
    "CC(=O)O": 6.2,
}

PROTIC_SOLVENTS: set[str] = {
    "O",
    "CO",
    "CCO",
    "CC(C)O",
    "CCCCO",
    "CC(C)(C)O",
    "O=C(O)C",
    "CC(=O)O",
}

DECOMPOSE_BEFORE_MELT: dict[str, tuple[str, str]] = {
    "Oc1cc(O)c2cc(O)c(-c3cc(O)c(O)c(O)c3)[o+]c2c1.[Cl-]": (
        "reported as decomposition-prone anthocyanidin chloride; no ordinary "
        "fusion point is available in the processed corpus",
        "exclude_from_crystal_branch",
    ),
}


@dataclass(frozen=True)
class IonicFeatureSummary:
    """Human-readable summary of ionic/coarse dielectric status."""

    n_fragments: int
    total_formal_charge: int
    n_charged_atoms: int
    max_abs_formal_charge: int
    is_explicit_salt: bool
    is_zwitterion: bool
    solvent_eps_r: float
    solvent_is_protic: bool
    ion_pair_low_eps_proxy: float
    possible_dissociation_proxy: float


def solvent_dielectric(solvent_smiles: str, default: float = 25.0) -> float:
    """Return a coarse relative permittivity estimate for common solvents."""
    canonical = canonical_smiles(solvent_smiles)
    if canonical is None:
        return float(default)
    return float(SOLVENT_DIELECTRIC.get(canonical, default))


def is_protic_solvent(solvent_smiles: str) -> bool:
    """Return whether the solvent is in the curated protic-solvent set."""
    canonical = canonical_smiles(solvent_smiles)
    return bool(canonical in PROTIC_SOLVENTS)


def flag_no_melting_point(smiles: str) -> str:
    """Return handling mode for compounds known to decompose before melting."""
    canonical = canonical_smiles(smiles)
    if canonical is None:
        return "standard"
    return DECOMPOSE_BEFORE_MELT.get(canonical, ("", "standard"))[1]


def no_melting_reason(smiles: str) -> str:
    """Return a short reason for special no-melting handling, if known."""
    canonical = canonical_smiles(smiles)
    if canonical is None:
        return ""
    return DECOMPOSE_BEFORE_MELT.get(canonical, ("", "standard"))[0]


def _charge_summary(smiles: str) -> tuple[int, int, int, bool, bool, int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0, 0, 0, False, False, 1
    charges = [int(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    total_charge = int(sum(charges))
    n_charged_atoms = int(sum(1 for charge in charges if charge != 0))
    max_abs_charge = int(max((abs(charge) for charge in charges), default=0))
    has_positive = any(charge > 0 for charge in charges)
    has_negative = any(charge < 0 for charge in charges)
    n_fragments = len(Chem.GetMolFrags(mol))
    is_explicit_salt = n_fragments > 1 and n_charged_atoms > 0
    is_zwitterion = n_fragments == 1 and total_charge == 0 and has_positive and has_negative
    return (
        total_charge,
        n_charged_atoms,
        max_abs_charge,
        is_explicit_salt,
        is_zwitterion,
        n_fragments,
    )


def ionic_feature_summary(
    solute_smiles: str,
    solvent_smiles: str,
    *,
    default_eps_r: float = 25.0,
) -> IonicFeatureSummary:
    """Compute interpretable ionic/contact-pair descriptors for one pair."""
    (
        total_charge,
        n_charged_atoms,
        max_abs_charge,
        is_explicit_salt,
        is_zwitterion,
        n_fragments,
    ) = _charge_summary(solute_smiles)
    eps_r = solvent_dielectric(solvent_smiles, default=default_eps_r)
    solvent_is_protic = is_protic_solvent(solvent_smiles)
    charged = n_charged_atoms > 0
    ion_pair_low_eps_proxy = (
        (n_charged_atoms / max(eps_r, 1.0)) if is_explicit_salt else 0.0
    )
    possible_dissociation_proxy = (
        max((eps_r - 50.0) / 30.0, 0.0) if charged else 0.0
    )
    return IonicFeatureSummary(
        n_fragments=n_fragments,
        total_formal_charge=total_charge,
        n_charged_atoms=n_charged_atoms,
        max_abs_formal_charge=max_abs_charge,
        is_explicit_salt=is_explicit_salt,
        is_zwitterion=is_zwitterion,
        solvent_eps_r=eps_r,
        solvent_is_protic=solvent_is_protic,
        ion_pair_low_eps_proxy=float(ion_pair_low_eps_proxy),
        possible_dissociation_proxy=float(possible_dissociation_proxy),
    )


def compute_ionic_features(
    solute_smiles: str,
    solvent_smiles: str,
    *,
    default_eps_r: float = 25.0,
) -> np.ndarray:
    """Return a fixed numeric feature vector for ionic/contact-pair context."""
    s = ionic_feature_summary(
        solute_smiles,
        solvent_smiles,
        default_eps_r=default_eps_r,
    )
    eps_norm = (s.solvent_eps_r - 1.0) / (80.0 - 1.0)
    log_eps_norm = math.log(s.solvent_eps_r + 1.0) / math.log(81.0)
    values = np.asarray(
        [
            float(np.clip(s.total_formal_charge / 3.0, -1.0, 1.0)),
            float(min(s.n_charged_atoms, 6) / 6.0),
            float(min(s.max_abs_formal_charge, 3) / 3.0),
            float(s.is_explicit_salt),
            float(s.is_zwitterion),
            float(np.clip(eps_norm, 0.0, 1.5)),
            float(np.clip(log_eps_norm, 0.0, 1.5)),
            float(s.solvent_is_protic),
            float(np.clip(s.ion_pair_low_eps_proxy, 0.0, 1.0)),
            float(np.clip(s.possible_dissociation_proxy, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    if values.shape != (IONIC_FEATURE_DIM,):
        raise RuntimeError("Internal ionic feature dimensionality mismatch.")
    return values


def applicability_domain_flags(
    solute_smiles: str,
    solvent_smiles: str,
    *,
    has_T_m: bool | None = None,
    predicted_ln_x2: float | None = None,
) -> list[str]:
    """Return rule-based AD warnings for chemically difficult systems."""
    flags: list[str] = []
    summary = ionic_feature_summary(solute_smiles, solvent_smiles)
    no_melting = flag_no_melting_point(solute_smiles) != "standard"
    if no_melting:
        flags.append(
            "decomposes_before_melting: standard fusion-based crystal term is "
            "not physically well-defined for this solute"
        )
    elif has_T_m is False:
        flags.append(
            "no_T_m: crystal term is inferred from structure; uncertainty is "
            "high for unusual scaffolds"
        )

    charged = summary.n_charged_atoms > 0
    if charged:
        if summary.is_explicit_salt and summary.solvent_eps_r < 30.0:
            flags.append(
                "ion_pair_regime: explicit salt in low-dielectric solvent; "
                "contact ion-pair geometry is only approximate"
            )
        elif summary.solvent_eps_r > 50.0:
            flags.append(
                "possible_dissociation: ionic solute in high-dielectric solvent; "
                "electrolyte thermodynamics may be needed"
            )
        else:
            flags.append("charged_solute: activity model reliability is reduced")

    if predicted_ln_x2 is not None and math.isfinite(float(predicted_ln_x2)):
        if float(predicted_ln_x2) <= -15.0:
            flags.append(
                "extreme_low_solubility: prediction is in the corpus left tail"
            )
    return flags
