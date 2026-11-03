from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles: object) -> str:
    if smiles is None:
        raise ValueError("SMILES cannot be None")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def smiles_cache_key(smiles: object) -> str:
    canonical = canonicalize_smiles(smiles)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def infer_charge_and_multiplicity(mol: Chem.Mol) -> tuple[int, int]:
    charge = int(Chem.GetFormalCharge(mol))
    n_radical = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
    multiplicity = max(1, n_radical + 1)
    return charge, multiplicity


def embed_smiles_3d(
    smiles: object,
    *,
    random_seed: int = 0xF00D,
    max_iters: int = 200,
) -> tuple[Chem.Mol, int, int]:
    raw = Chem.MolFromSmiles(str(smiles))
    if raw is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")

    mol = Chem.AddHs(raw)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"RDKit 3D embedding failed for {smiles!r}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
    except Exception:
        pass

    charge, multiplicity = infer_charge_and_multiplicity(raw)
    return mol, charge, multiplicity


def mol_to_xyz_lines(mol: Chem.Mol) -> list[str]:
    conf = mol.GetConformer()
    lines: list[str] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2} {pos.x: .12f} {pos.y: .12f} {pos.z: .12f}"
        )
    return lines


def write_cosmorsxyz(
    mol: Chem.Mol,
    path: str | Path,
    *,
    charge: int,
    multiplicity: int,
) -> Path:
    path = Path(path)
    xyz_lines = mol_to_xyz_lines(mol)
    payload = [str(len(xyz_lines)), f"{int(charge)} {int(multiplicity)}", *xyz_lines]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    return path


def write_orca_cosmors_input(
    mol: Chem.Mol,
    path: str | Path,
    *,
    basename: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
    maxcore_mb: int,
) -> Path:
    path = Path(path)
    xyz_lines = mol_to_xyz_lines(mol)
    payload = [
        f"%pal nprocs {max(1, int(nprocs))} end",
        f"%maxcore {max(1, int(maxcore_mb))}",
        "%cosmors",
        f'  solventfilename "{basename}"',
        "end",
        f"* xyz {int(charge)} {int(multiplicity)}",
        *xyz_lines,
        "*",
        "",
    ]
    path.write_text("\n".join(payload), encoding="utf-8")
    return path


def default_orca_binary_candidates() -> list[Path]:
    return [
        Path(os.environ.get("ORCA_BIN", "")).expanduser(),
        Path.home() / "Library/orca_6_1_1/orca",
        Path.home() / "orca/orca",
    ]


def detect_orca_binary(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    resolved = shutil.which("orca")
    if resolved:
        candidates.append(Path(resolved))
    candidates.extend(default_orca_binary_candidates())
    for candidate in candidates:
        if candidate and str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "ORCA binary not found. Pass --orca-bin or expose `orca` on PATH."
    )


def detect_mpi_lib(explicit: str | Path | None = None) -> Path | None:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(os.environ.get("LIBMPI_40_PATH", "")).expanduser(),
        Path("/opt/homebrew/lib/libmpi.40.dylib"),
        Path("/usr/local/lib/libmpi.40.dylib"),
    ]
    for candidate in candidates:
        if candidate and str(candidate) and candidate.is_file():
            return candidate.resolve()
    return None


def _link_mpi_runtime(workdir: Path, mpi_lib: Path | None) -> Path | None:
    if mpi_lib is None or not mpi_lib.is_file():
        return None
    target = workdir / mpi_lib.name
    if target.exists() or target.is_symlink():
        return target
    target.symlink_to(mpi_lib)
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _expected_partial_success(stdout_path: Path) -> bool:
    if not stdout_path.is_file():
        return False
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    return "openCOSMORS" in text and "No such file or directory" in text


def cleanup_orca_self_pair_workdir(workdir: str | Path, *, basename: str) -> None:
    workdir = Path(workdir)
    keep = {
        workdir / f"{basename}.inp",
        workdir / f"{basename}.out",
        workdir / f"{basename}.cosmorsxyz",
        workdir / f"{basename}.orcacosmo",
        workdir / "metadata.json",
        workdir / "libmpi.40.dylib",
    }
    for path in workdir.iterdir():
        if path in keep:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)


@dataclass(frozen=True)
class CosmoArtifact:
    canonical_smiles: str
    cache_key: str
    workdir: Path
    cosmo_path: Path
    metadata_path: Path


def run_orca_self_cosmors(
    smiles: object,
    *,
    workdir: str | Path,
    orca_bin: str | Path | None = None,
    mpi_lib: str | Path | None = None,
    basename: str = "molecule",
    nprocs: int = 2,
    maxcore_mb: int = 1000,
    keep_workdir: bool = False,
) -> CosmoArtifact:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    canonical = canonicalize_smiles(smiles)
    cache_key = smiles_cache_key(canonical)
    mol, charge, multiplicity = embed_smiles_3d(canonical)
    orca_path = detect_orca_binary(orca_bin)
    mpi_path = detect_mpi_lib(mpi_lib)
    linked_mpi = _link_mpi_runtime(workdir, mpi_path)

    write_cosmorsxyz(
        mol,
        workdir / f"{basename}.cosmorsxyz",
        charge=charge,
        multiplicity=multiplicity,
    )
    write_orca_cosmors_input(
        mol,
        workdir / f"{basename}.inp",
        basename=basename,
        charge=charge,
        multiplicity=multiplicity,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
    )

    stdout_path = workdir / f"{basename}.out"
    with stdout_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [str(orca_path), f"{basename}.inp"],
            cwd=str(workdir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    solute_orcacosmo = workdir / f"{basename}.solute.orcacosmo"
    if not solute_orcacosmo.is_file():
        raise RuntimeError(
            f"ORCA did not produce {solute_orcacosmo.name} for {canonical!r}. "
            f"See {stdout_path}."
        )

    final_cosmo = workdir / f"{basename}.orcacosmo"
    shutil.copy2(solute_orcacosmo, final_cosmo)

    metadata = {
        "smiles": str(smiles),
        "canonical_smiles": canonical,
        "cache_key": cache_key,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "orca_bin": str(orca_path),
        "mpi_lib": str(mpi_path) if mpi_path else None,
        "mpi_lib_linked": str(linked_mpi) if linked_mpi else None,
        "orca_returncode": int(proc.returncode),
        "expected_partial_success": bool(_expected_partial_success(stdout_path)),
        "workdir": str(workdir),
        "cosmo_path": str(final_cosmo),
    }
    metadata_path = workdir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2),
        encoding="utf-8",
    )

    if not keep_workdir:
        cleanup_orca_self_pair_workdir(workdir, basename=basename)

    return CosmoArtifact(
        canonical_smiles=canonical,
        cache_key=cache_key,
        workdir=workdir,
        cosmo_path=final_cosmo,
        metadata_path=metadata_path,
    )


def ensure_orca_cosmo_file(
    smiles: object,
    *,
    cache_dir: str | Path,
    orca_bin: str | Path | None = None,
    mpi_lib: str | Path | None = None,
    nprocs: int = 2,
    maxcore_mb: int = 1000,
    keep_workdir: bool = False,
    force: bool = False,
) -> CosmoArtifact:
    canonical = canonicalize_smiles(smiles)
    cache_key = smiles_cache_key(canonical)
    workdir = Path(cache_dir) / cache_key[:2] / cache_key
    cosmo_path = workdir / "molecule.orcacosmo"
    metadata_path = workdir / "metadata.json"

    if cosmo_path.is_file() and metadata_path.is_file() and not force:
        return CosmoArtifact(
            canonical_smiles=canonical,
            cache_key=cache_key,
            workdir=workdir,
            cosmo_path=cosmo_path,
            metadata_path=metadata_path,
        )

    return run_orca_self_cosmors(
        canonical,
        workdir=workdir,
        orca_bin=orca_bin,
        mpi_lib=mpi_lib,
        basename="molecule",
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
        keep_workdir=keep_workdir,
    )


def _require_opencosmorspy():
    try:
        from opencosmorspy.cosmors import COSMORS
        from opencosmorspy.parameterization import openCOSMORS24a
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openCOSMO-RS_py is required for COSMO-RS calculations. "
            "Install `opencosmorspy` in the active environment."
        ) from exc
    return COSMORS, openCOSMORS24a


class OpenCosmoBinarySystem:
    def __init__(
        self,
        solute_orcacosmo: str | Path,
        solvent_orcacosmo: str | Path,
    ) -> None:
        COSMORS, openCOSMORS24a = _require_opencosmorspy()
        self.solute_orcacosmo = Path(solute_orcacosmo)
        self.solvent_orcacosmo = Path(solvent_orcacosmo)
        self._model = COSMORS(par=openCOSMORS24a())
        self._model.add_molecule([str(self.solute_orcacosmo)])
        self._model.add_molecule([str(self.solvent_orcacosmo)])

    def ln_gamma_2(self, x2: float, temperature_k: float) -> float:
        x2 = float(x2)
        temperature_k = float(temperature_k)
        if not (0.0 < x2 < 1.0):
            raise ValueError(f"Expected 0 < x2 < 1, got {x2!r}")
        if temperature_k <= 0.0:
            raise ValueError(
                f"Expected positive temperature, got {temperature_k!r}"
            )
        self._model.clear_jobs()
        self._model.add_job(
            np.array([x2, 1.0 - x2], dtype=float),
            temperature_k,
            refst="pure_component",
        )
        res = self._model.calculate()
        value = float(res["tot"]["lng"][0, 0])
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite COSMO-RS ln(gamma_2) for x2={x2}, T={temperature_k}"
            )
        return value


def calculate_cosmors_lngamma_binary(
    solute_orcacosmo: str | Path,
    solvent_orcacosmo: str | Path,
    *,
    x2: float,
    temperature_k: float,
) -> float:
    system = OpenCosmoBinarySystem(solute_orcacosmo, solvent_orcacosmo)
    return system.ln_gamma_2(x2, temperature_k)
