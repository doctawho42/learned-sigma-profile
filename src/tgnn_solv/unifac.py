"""Thin wrappers around `thermo` Modified UNIFAC for cached preprocessing.

The model should not call these functions inside a batched forward pass.
Use them in data-preparation scripts to precompute scalar priors/objectives.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Optional

from rdkit import Chem


@lru_cache(maxsize=1)
def _load_modified_unifac_tables():
    try:
        import thermo.unifac as unifac
        from thermo.unifac import load_group_assignments_DDBST, load_unifac_ip
    except ImportError as exc:
        raise ImportError(
            "Modified UNIFAC support requires optional packages: "
            "`pip install thermo chemicals lxml`."
        ) from exc

    load_group_assignments_DDBST()
    load_unifac_ip()
    return unifac


@lru_cache(maxsize=100_000)
def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Return RDKit Standard InChIKey for a SMILES string, or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _smiles_without_stereo(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def _largest_fragment_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    best = max(
        fragments,
        key=lambda frag: (
            any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms()),
            frag.GetNumHeavyAtoms(),
            frag.GetNumAtoms(),
        ),
    )
    return Chem.MolToSmiles(best, canonical=True)


def _neutralized_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        neutral = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(neutral, canonical=True)


def _approximate_lookup_candidates(smiles: str) -> tuple[str, ...]:
    candidates: list[str] = []

    no_stereo = _smiles_without_stereo(smiles)
    if no_stereo and no_stereo != smiles:
        candidates.append(no_stereo)

    largest_fragment = _largest_fragment_smiles(smiles)
    if largest_fragment and largest_fragment != smiles:
        candidates.append(largest_fragment)

    neutralized = _neutralized_smiles(smiles)
    if neutralized and neutralized != smiles:
        candidates.append(neutralized)

    if largest_fragment is not None:
        largest_fragment_neutral = _neutralized_smiles(largest_fragment)
        if (
            largest_fragment_neutral
            and largest_fragment_neutral != smiles
            and largest_fragment_neutral != largest_fragment
        ):
            candidates.append(largest_fragment_neutral)

    return tuple(dict.fromkeys(candidates))


@lru_cache(maxsize=200_000)
def _modified_unifac_groups_cached(
    smiles: str,
    allow_approximate_fallback: bool,
) -> Optional[tuple[tuple[int, int], ...]]:
    key = smiles_to_inchikey(str(smiles))
    if key is not None:
        unifac = _load_modified_unifac_tables()
        groups = unifac.DDBST_MODIFIED_UNIFAC_assignments.get(key)
        if groups:
            return tuple(sorted((int(k), int(v)) for k, v in groups.items()))

    if not allow_approximate_fallback:
        return None

    unifac = _load_modified_unifac_tables()
    for candidate in _approximate_lookup_candidates(str(smiles)):
        candidate_key = smiles_to_inchikey(candidate)
        if candidate_key is None:
            continue
        groups = unifac.DDBST_MODIFIED_UNIFAC_assignments.get(candidate_key)
        if groups:
            return tuple(sorted((int(k), int(v)) for k, v in groups.items()))
    return None


@lru_cache(maxsize=100_000)
def modified_unifac_groups(
    smiles: str,
    *,
    allow_approximate_fallback: bool = False,
) -> Optional[tuple[tuple[int, int], ...]]:
    """Return DDBST Modified UNIFAC group assignment as a hashable tuple.

    When ``allow_approximate_fallback`` is enabled, several low-cost
    standardization fallbacks are tried after the exact Standard-InChIKey
    lookup:

    - remove stereochemistry
    - keep the largest fragment
    - neutralize charges
    - neutralize the largest fragment

    This is intended for synthetic pseudo-label generation, not for the default
    exact prior path.
    """
    return _modified_unifac_groups_cached(
        str(smiles),
        bool(allow_approximate_fallback),
    )


def _groups_dict(groups: tuple[tuple[int, int], ...]) -> dict[int, int]:
    return {int(k): int(v) for k, v in groups}


@lru_cache(maxsize=250_000)
def modified_unifac_lngamma_inf(
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float,
    *,
    temperature_decimals: int = 3,
    allow_approximate_fallback: bool = False,
) -> Optional[float]:
    """Compute ln(gamma_inf) for solute infinitely diluted in solvent.

    Uses Modified UNIFAC Dortmund 2016 interaction parameters as exposed by
    `thermo.unifac`. Returns None when DDBST has no group assignment or when
    UNIFAC cannot evaluate the pair.
    """
    T = round(float(temperature), int(temperature_decimals))
    if not math.isfinite(T) or T <= 0.0:
        return None

    sol_groups = modified_unifac_groups(
        str(solute_smiles),
        allow_approximate_fallback=allow_approximate_fallback,
    )
    slv_groups = modified_unifac_groups(
        str(solvent_smiles),
        allow_approximate_fallback=allow_approximate_fallback,
    )
    if sol_groups is None or slv_groups is None:
        return None

    unifac = _load_modified_unifac_tables()
    try:
        ge = unifac.UNIFAC.from_subgroups(
            T=T,
            xs=[1e-9, 1.0 - 1e-9],
            chemgroups=[_groups_dict(sol_groups), _groups_dict(slv_groups)],
            subgroups=unifac.DOUFSG,
            interaction_data=unifac.DOUFIP2016,
            version=1,
        )
        gamma = float(ge.gammas_infinite_dilution()[0])
    except Exception:
        return None

    if not math.isfinite(gamma) or gamma <= 0.0:
        return None
    return float(math.log(gamma))


@lru_cache(maxsize=500_000)
def modified_unifac_lngamma_binary(
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float,
    solute_mole_fraction: float,
    *,
    temperature_decimals: int = 3,
    composition_decimals: int = 6,
    allow_approximate_fallback: bool = False,
) -> Optional[float]:
    """Compute finite-composition ln(gamma) for solute in a binary mixture.

    The first component is always treated as the solute/target component whose
    finite-composition activity coefficient should supervise the activity
    branch. Returns None when group assignments are unavailable or UNIFAC
    fails to evaluate the state.
    """
    T = round(float(temperature), int(temperature_decimals))
    x_raw = float(solute_mole_fraction)
    if not math.isfinite(T) or T <= 0.0:
        return None
    if not math.isfinite(x_raw) or x_raw <= 0.0 or x_raw >= 1.0:
        return None
    # Validity is judged on the RAW fraction; the rounding below only stabilizes
    # the xs vector. Preserve dilute states that would round to exactly 0 (or 1):
    # x2 < 5e-7 (ln x2 < ~-14.5) is a legitimate, solubility-relevant finite
    # composition, not an invalid input, and must not be silently dropped.
    x_solute = round(x_raw, int(composition_decimals))
    if x_solute <= 0.0:
        x_solute = max(x_raw, 1e-9)
    elif x_solute >= 1.0:
        x_solute = min(x_raw, 1.0 - 1e-9)

    sol_groups = modified_unifac_groups(
        str(solute_smiles),
        allow_approximate_fallback=allow_approximate_fallback,
    )
    slv_groups = modified_unifac_groups(
        str(solvent_smiles),
        allow_approximate_fallback=allow_approximate_fallback,
    )
    if sol_groups is None or slv_groups is None:
        return None

    unifac = _load_modified_unifac_tables()
    try:
        ge = unifac.UNIFAC.from_subgroups(
            T=T,
            xs=[x_solute, 1.0 - x_solute],
            chemgroups=[_groups_dict(sol_groups), _groups_dict(slv_groups)],
            subgroups=unifac.DOUFSG,
            interaction_data=unifac.DOUFIP2016,
            version=1,
        )
        gamma = float(ge.gammas()[0])
    except Exception:
        return None

    if not math.isfinite(gamma) or gamma <= 0.0:
        return None
    return float(math.log(gamma))
