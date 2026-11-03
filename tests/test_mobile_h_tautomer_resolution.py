"""A mobile-hydrogen InChI must not leave a SMILES on a minor tautomer.

A standard InChI whose ``/h`` layer carries a mobile-hydrogen group does not fix the tautomer, and
``Chem.MolFromInchi`` resolves it to a form that is not the dominant one: formamide arrived as the
imidic acid, caprolactam as the lactim. Since every table downstream keys on canonical SMILES, such
a record silently fails to match a reference profile deposited under the amide form. That cost 27
infinite-dilution records and made the two matching rules of the Supporting Information differ by
25 pairs where they should have agreed.

These tests pin the reader: it resolves the ambiguous case and leaves the determined one alone.
The companion check, that no cached table still disagrees with its own InChI, runs in the
development tree instead, because it reads the raw infinite-dilution tables and this package
distributes only the processed split.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from tgnn_solv.data.thermoml_idac import _MOBILE_H, _mol_from_inchi

ROOT = Path(__file__).resolve().parents[1]

#: The three molecules this corpus actually lost, with the form the InChI means.
KNOWN = [
    ("InChI=1S/CH3NO/c2-1-3/h1H,(H2,2,3)", "NC=O", "formamide"),
    ("InChI=1S/C2H5NO/c1-3-2-4/h2H,1H3,(H,3,4)", "CNC=O", "N-methylformamide"),
    # read out of notebooks/data/raw/idac.csv, where the record names it 1-aza-2-cycloheptanone
    ("InChI=1S/C6H11NO/c8-6-4-2-1-3-5-7-6/h1-5H2,(H,7,8)", "O=C1CCCCCN1", "caprolactam"),
]


@pytest.mark.parametrize("inchi,expected,name", KNOWN)
def test_mobile_hydrogen_resolves_to_the_dominant_tautomer(inchi, expected, name):
    assert _MOBILE_H.search(inchi), f"{name}: this InChI has no mobile-hydrogen layer to resolve"
    mol = _mol_from_inchi(inchi)
    assert mol is not None, f"{name}: the InChI did not parse"
    assert Chem.MolToSmiles(mol, canonical=True) == expected


@pytest.mark.parametrize("inchi", [
    "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",                 # benzene
    "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",                  # ethanol
    "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",                     # acetone
])
def test_a_determined_inchi_is_left_exactly_as_rdkit_parses_it(inchi):
    """Where the /h layer fixes the hydrogens the structure is settled.

    Canonicalising those too would rewrite molecules the repair has no licence to touch, so the
    reader must be a no-op on them.
    """
    assert not _MOBILE_H.search(inchi)
    plain = Chem.MolToSmiles(Chem.MolFromInchi(inchi), canonical=True)
    assert Chem.MolToSmiles(_mol_from_inchi(inchi), canonical=True) == plain
