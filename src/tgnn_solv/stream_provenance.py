"""Point-of-use certification and provenance for grounding streams.

WHY THIS EXISTS
---------------
The scaffold guarantee on an auxiliary grounding stream used to be a property of
the *builder*: ``build_sigma_profile_aux_stream.py`` excludes pool molecules whose
Bemis--Murcko scaffold appears in a held-out split, and aborts on a missing split
file. That is necessary and it is not sufficient. One build on disk was generated
against split files from a different project directory; it passed the builder's
guard, leaked held-out solutes into the stream, and the defect was found only in
review, because nothing recorded *which build fed which run*.

The fix has two halves and this module is both of them:

  (a) certify at the point of use. The training run itself re-reads the stream it
      is about to consume and the val/test files it is about to be scored against,
      and asserts the intersection is empty by canonical SMILES and by Murcko
      scaffold. A builder can be pointed at the wrong files; a run cannot, because
      the files it checks are the files it uses.

  (b) record the certification. The stream's SHA-256, its row count and the
      recomputed leak counts go into the run manifest beside the SHA-256 of the
      pinned split, so the artifact says which build fed which arm rather than
      leaving it to file timestamps.

Both halves are cheap (one RDKit pass over a few thousand SMILES) and neither
needs a GPU, so nothing is gained by making them optional.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from rdkit import Chem

from .data.utils import scaffold_key


__all__ = [
    "canonical_smiles",
    "held_out_keys",
    "sha256_of",
    "certify_stream_disjoint",
    "StreamLeakError",
]


class StreamLeakError(RuntimeError):
    """Raised when a grounding stream shares a molecule with a held-out split."""


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file's bytes.

    Byte-identical to the ``build_sha256`` the stream builder writes into its
    ``summary.json``, so a manifest hash and a builder hash compare directly.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(smiles: Any) -> str:
    """Canonical SMILES, falling back to the input string when RDKit cannot parse.

    The fallback is deliberate: an unparseable SMILES must still be *comparable*,
    otherwise a molecule RDKit chokes on would silently drop out of the leak check
    on both sides and be certified clean by absence.
    """
    text = str(smiles)
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol) if mol is not None else text


def _solute_smiles(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, usecols=lambda c: c == "solute_smiles", low_memory=False)
    if "solute_smiles" not in frame.columns:
        raise ValueError(f"{path} has no solute_smiles column; cannot run the leak check")
    return [str(s) for s in frame["solute_smiles"].dropna().unique()]


def held_out_keys(paths: Sequence[str | Path]) -> tuple[set[str], set[str]]:
    """Canonical-SMILES and scaffold key sets for the union of held-out splits.

    Fails closed: a path that does not exist, or that yields no usable solute,
    raises rather than contributing an empty set. An empty held-out set would
    certify any stream as clean, which is exactly the failure this guards.
    """
    smiles_keys: set[str] = set()
    scaffold_keys: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"held-out split for the stream leak check does not exist: {path}. "
                "Refusing to certify a stream against a missing split."
            )
        solutes = _solute_smiles(path)
        if not solutes:
            raise ValueError(
                f"{path} yielded no solutes; refusing to certify a stream against an "
                "empty held-out set."
            )
        for smiles in solutes:
            smiles_keys.add(canonical_smiles(smiles))
            key = scaffold_key(smiles)
            if key:
                scaffold_keys.add(key)
    return smiles_keys, scaffold_keys


def certify_stream_disjoint(
    stream_csv: str | Path,
    held_out: Sequence[str | Path],
    *,
    role: str = "sigma_train",
    max_examples: int = 20,
) -> dict[str, Any]:
    """Certify one stream file against the held-out splits, at the point of use.

    Returns a provenance record suitable for a run manifest. Does not raise on a
    leak -- the caller decides whether a leak is fatal -- but sets
    ``certified_no_leak`` False and lists the offending molecules.
    """
    stream_path = Path(stream_csv)
    stream_solutes = _solute_smiles(stream_path)
    held_smiles, held_scaffolds = held_out_keys(held_out)

    leak_smiles: set[str] = set()
    leak_scaffolds: set[str] = set()
    for smiles in stream_solutes:
        if canonical_smiles(smiles) in held_smiles:
            leak_smiles.add(smiles)
        key = scaffold_key(smiles)
        if key and key in held_scaffolds:
            leak_scaffolds.add(smiles)

    n_rows = int(len(pd.read_csv(stream_path, usecols=[0], low_memory=False)))
    return {
        "role": role,
        "path": str(stream_path.resolve()),
        "sha256": sha256_of(stream_path),
        "n_rows": n_rows,
        "n_distinct_solutes": len(stream_solutes),
        "held_out_checked": [str(Path(p).resolve()) for p in held_out],
        "n_held_out_smiles": len(held_smiles),
        "n_held_out_scaffolds": len(held_scaffolds),
        "leak_by_canonical_smiles": len(leak_smiles),
        "leak_by_murcko_scaffold": len(leak_scaffolds),
        "leak_examples_smiles": sorted(leak_smiles)[:max_examples],
        "leak_examples_scaffold": sorted(leak_scaffolds)[:max_examples],
        "certified_no_leak": not (leak_smiles or leak_scaffolds),
    }


def certify_streams_or_raise(
    streams: Iterable[tuple[str, str | Path | None]],
    held_out: Sequence[str | Path],
    *,
    allow_overlap: bool = False,
) -> list[dict[str, Any]]:
    """Certify several streams; raise :class:`StreamLeakError` unless overridden.

    ``streams`` is an iterable of ``(role, path_or_None)``; a ``None`` path yields
    a record stating the arm carried no stream of that role. Recording the absence
    matters: for the ungrounded arm, "no stream" is the thing being asserted, and
    an empty manifest entry cannot distinguish it from a stream that was not logged.
    """
    records: list[dict[str, Any]] = []
    leaking: list[dict[str, Any]] = []
    for role, path in streams:
        if path is None:
            records.append({"role": role, "path": None, "present": False,
                            "certified_no_leak": None})
            continue
        record = certify_stream_disjoint(path, held_out, role=role)
        record["present"] = True
        record["override_allowed"] = bool(allow_overlap)
        records.append(record)
        if not record["certified_no_leak"]:
            leaking.append(record)
    if leaking and not allow_overlap:
        detail = "; ".join(
            f"{r['role']} ({Path(str(r['path'])).name}): "
            f"{r['leak_by_canonical_smiles']} by canonical SMILES, "
            f"{r['leak_by_murcko_scaffold']} by Murcko scaffold"
            for r in leaking
        )
        raise StreamLeakError(
            "point-of-use stream certification failed -- a grounding stream shares "
            f"molecules with the held-out splits: {detail}. This run would not carry "
            "the scaffold guarantee. Rebuild the stream against these split files, or "
            "pass --allow-stream-scaffold-overlap to proceed deliberately (the override "
            "is recorded in the run manifest)."
        )
    return records
