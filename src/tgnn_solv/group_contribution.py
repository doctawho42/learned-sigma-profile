"""Group-contribution priors for crystal thermodynamics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

GC_FALLBACK_PRIORS = {
    "T_m_gc": 400.0,
    "dH_fus_gc": 20000.0,
    "dCp_fus_gc": 0.0,
}

JOBACK_TM_BASE_K = 122.5
JOBACK_HFUS_OFFSET_KJMOL = -0.88


@dataclass(frozen=True)
class JobackGroup:
    """Single Joback fragment with crystal-property increments."""

    group_id: int
    name: str
    smarts: str
    tm_increment: float | None
    hfus_increment: float | None

    # There is deliberately no per-group dCp increment. dH_fus/T_m is a ratio of two molecular
    # quantities and does not decompose over fragments; a property that returned one invited the
    # sum that _estimate_dcp_from_counts used to perform, and that sum was wrong by a median factor
    # of 2.95. Aggregate the increments first, divide once.


# Joback/Reid crystal-property increments and SMARTS adapted from the public
# thermo implementation of Joback fragmentation.
JOBACK_GROUPS: tuple[JobackGroup, ...] = (
    JobackGroup(1, "-CH3", "[CX4H3]", -5.10, 0.908),
    JobackGroup(2, "-CH2-", "[!R;CX4H2]", 11.27, 2.590),
    JobackGroup(3, ">CH-", "[!R;CX4H]", 12.64, 0.749),
    JobackGroup(4, ">C<", "[!R;CX4H0]", 46.43, -1.460),
    JobackGroup(5, "=CH2", "[CX3H2]", -4.32, -0.473),
    JobackGroup(6, "=CH-", "[!R;CX3H1;!$([CX3H1](=O))]", 8.73, 2.691),
    JobackGroup(7, "=C<", "[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]", 11.14, 3.063),
    JobackGroup(8, "=C=", "[$([CX2H0](=*)=*)]", 17.78, 4.720),
    JobackGroup(9, "≡CH", "[$([CX2H1]#[!#7])]", -11.18, 2.322),
    JobackGroup(10, "≡C-", "[$([CX2H0]#[!#7])]", 64.32, 4.151),
    JobackGroup(11, "-CH2- (ring)", "[R;CX4H2]", 7.75, 0.490),
    JobackGroup(12, ">CH- (ring)", "[R;CX4H]", 19.88, 3.243),
    JobackGroup(13, ">C< (ring)", "[R;CX4H0]", 60.15, -1.373),
    JobackGroup(14, "=CH- (ring)", "[R;CX3H1,cX3H1]", 8.13, 1.101),
    JobackGroup(15, "=C< (ring)", "[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]", 37.02, 2.394),
    JobackGroup(16, "-F", "[F]", -15.78, 1.398),
    JobackGroup(17, "-Cl", "[Cl]", 13.55, 2.515),
    JobackGroup(18, "-Br", "[Br]", 43.43, 3.603),
    JobackGroup(19, "-I", "[I]", 41.69, 2.724),
    JobackGroup(20, "-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]", 44.45, 2.406),
    JobackGroup(21, "-OH (phenol)", "[$([OX2H]-a)]", 82.83, 4.490),
    JobackGroup(22, "-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]", 22.23, 1.188),
    JobackGroup(23, "-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]", 23.05, 5.879),
    JobackGroup(24, ">C=O (nonring)", "[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O", 61.20, 4.189),
    JobackGroup(25, ">C=O (ring)", "[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O", 75.97, 0.000),
    JobackGroup(26, "O=CH- (aldehyde)", "[CH;D2;$(C-!@C)](=O)", 36.90, 3.197),
    JobackGroup(27, "-COOH (acid)", "[OX2H]-[C]=O", 155.50, 11.051),
    JobackGroup(28, "-COO- (ester)", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]", 53.60, 6.959),
    JobackGroup(29, "=O (other than above)", "[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]", 2.08, 3.624),
    JobackGroup(30, "-NH2", "[NX3H2]", 66.89, 3.515),
    JobackGroup(31, ">NH (nonring)", "[NX3H1;!R]", 52.66, 5.099),
    JobackGroup(32, ">NH (ring)", "[#7X3H1;R]", 101.51, 7.490),
    JobackGroup(33, ">N- (nonring)", "[#7X3H0;!$([#7](~O)~O)]", 48.84, 4.703),
    JobackGroup(34, "-N= (nonring)", "[#7X2H0;!R]", None, None),
    JobackGroup(35, "-N= (ring)", "[#7X2H0;R]", 68.40, 3.649),
    JobackGroup(36, "=NH", "[#7X2H1]", 68.91, None),
    JobackGroup(37, "-CN", "[#6X2]#[#7X1H0]", 59.89, 2.414),
    JobackGroup(38, "-NO2", "[$([#7X3,#7X3+][!#8])](=[O])~[O-]", 127.24, 9.679),
    JobackGroup(39, "-SH", "[SX2H]", 20.09, 2.360),
    JobackGroup(40, "-S- (nonring)", "[#16X2H0;!R]", 34.40, 4.130),
    JobackGroup(41, "-S- (ring)", "[#16X2H0;R]", 79.93, 1.557),
)

JOBACK_GROUPS_BY_ID = {group.group_id: group for group in JOBACK_GROUPS}


@lru_cache(maxsize=1)
def _compiled_groups() -> tuple[tuple[JobackGroup, Chem.Mol, int, int], ...]:
    compiled: list[tuple[JobackGroup, Chem.Mol, int, int]] = []
    for group in JOBACK_GROUPS:
        query = Chem.MolFromSmarts(group.smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS for Joback group {group.group_id}: {group.smarts}")
        hetero_atoms = sum(
            1
            for atom in query.GetAtoms()
            if atom.GetAtomicNum() not in (0, 1, 6)
        )
        compiled.append((group, query, query.GetNumAtoms(), hetero_atoms))
    compiled.sort(key=lambda item: (item[2], item[3], item[0].group_id), reverse=True)
    return tuple(compiled)


def _heavy_atom_indices(mol: Chem.Mol) -> set[int]:
    return {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}


def _fragment_joback(mol: Chem.Mol) -> tuple[dict[int, int], bool]:
    """Greedy non-overlapping SMARTS fragmentation for the Joback groups."""
    assigned_atoms: set[int] = set()
    counts: dict[int, int] = {}

    for group, query, _, _ in _compiled_groups():
        matches = mol.GetSubstructMatches(query, uniquify=True)
        for match in matches:
            heavy_match = tuple(
                atom_idx
                for atom_idx in match
                if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() > 1
            )
            if not heavy_match:
                continue
            if any(atom_idx in assigned_atoms for atom_idx in heavy_match):
                continue
            assigned_atoms.update(heavy_match)
            counts[group.group_id] = counts.get(group.group_id, 0) + 1

    return counts, assigned_atoms == _heavy_atom_indices(mol)


def _estimate_tm_from_counts(counts: dict[int, int], mol: Chem.Mol) -> float | None:
    total = 0.0
    for group_id, count in counts.items():
        increment = JOBACK_GROUPS_BY_ID[group_id].tm_increment
        if increment is None:
            return None
        total += increment * count

    tm_gc = JOBACK_TM_BASE_K + total
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    if n_aromatic_rings > 1:
        tm_gc += 50.0 * float(n_aromatic_rings - 1)
    return tm_gc


def _estimate_dh_from_counts(counts: dict[int, int]) -> float | None:
    total = JOBACK_HFUS_OFFSET_KJMOL
    for group_id, count in counts.items():
        increment = JOBACK_GROUPS_BY_ID[group_id].hfus_increment
        if increment is None:
            return None
        total += increment * count
    return max(total * 1000.0, 1000.0)


def _estimate_dcp_from_counts(counts: dict[int, int]) -> float | None:
    """Heuristic dH_fus/T_m in J/mol/K, aggregated over the MOLECULE.

    This was previously summed over `JobackGroup.dcp_increment`, a per-group ratio. That is not the
    same quantity: the Joback base temperature enters T_m once per molecule, not once per fragment
    occurrence, and each group's enthalpy increment belongs over the molecule's T_m rather than over
    its own. Summing the per-group ratios added the 122.5 K base once per occurrence and divided
    each numerator by its own increment, inflating the result by a median factor of 2.95 (max 11.6)
    over 166 solutes. Found 2026-08-06 while auditing a crystal fidelity ladder whose third rung the
    inflation alone produced: repairing this line moved that rung's MSE from 12.56 to 5.49, erasing
    92% of the effect it was built to show. No published number rests on it -- `use_gc_priors_crystal`
    is False by default and none of the manuscript's arms enable it.
    """
    tm_total = 0.0
    hfus_total = 0.0
    for group_id, count in counts.items():
        group = JOBACK_GROUPS_BY_ID[group_id]
        if group.tm_increment is None or group.hfus_increment is None:
            return None
        tm_total += group.tm_increment * count
        hfus_total += group.hfus_increment * count
    denom = max(100.0, JOBACK_TM_BASE_K + tm_total)
    return 1000.0 * hfus_total / denom


def fit_tm_gc_calibration(
    raw_tm_gc: Sequence[float],
    tm_true: Sequence[float],
) -> tuple[float, float]:
    """Fit an affine calibration ``scale * T_m_gc + bias`` on training data."""
    raw = np.asarray(raw_tm_gc, dtype=float)
    target = np.asarray(tm_true, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(target)
    raw = raw[mask]
    target = target[mask]

    if raw.size == 0:
        return 1.0, 0.0
    if raw.size == 1 or float(np.std(raw)) <= 1.0e-8:
        return 1.0, float(target.mean() - raw.mean())

    design = np.column_stack([raw, np.ones_like(raw)])
    scale, bias = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(scale), float(bias)


def compute_gc_priors(smiles: str) -> dict[str, float | None]:
    """Return fixed crystal GC priors in SI units for a single solute SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"T_m_gc": None, "dH_fus_gc": None, "dCp_fus_gc": None}

    # The completeness flag is deliberately not a gate: a partial fragmentation still carries more
    # information than the hard-coded fallback, which is why test_group_contribution.py asserts that
    # an incompletely fragmented molecule keeps its estimate. It affects 24 of the 166 crystal-set
    # solutes (14.5%), so a caller comparing GC priors against measured labels should stratify on it
    # rather than assume uniform quality. Named with a leading underscore because it is unused here,
    # not because it is unimportant -- an audit read the underscore as an oversight and proposed
    # gating on it, which the test above correctly rejects.
    counts, _complete = _fragment_joback(mol)
    if not counts:
        return dict(GC_FALLBACK_PRIORS)

    tm_gc = _estimate_tm_from_counts(counts, mol)
    dh_gc = _estimate_dh_from_counts(counts)
    dcp_gc = _estimate_dcp_from_counts(counts)
    if tm_gc is None or dh_gc is None or dcp_gc is None:
        return dict(GC_FALLBACK_PRIORS)

    return {
        "T_m_gc": float(tm_gc),
        "dH_fus_gc": float(dh_gc),
        "dCp_fus_gc": float(dcp_gc),
    }
