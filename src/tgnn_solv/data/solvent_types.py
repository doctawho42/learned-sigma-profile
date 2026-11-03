"""
Solvent type classification utilities.

Provides a small, interpretable set of solvent classes based on
RDKit substructure matches. Used to route Mixture-of-Experts.
"""

from __future__ import annotations

from typing import Dict

from rdkit import Chem


SOLVENT_TYPE_NAMES = [
    "water",
    "alcohol",
    "carboxylic_acid",
    "amide",
    "sulfoxide",
    "nitrile",
    "halogenated",
    "aromatic",
    "hydrocarbon",
    "other",
]
SOLVENT_TYPE_TO_ID: Dict[str, int] = {
    name: i for i, name in enumerate(SOLVENT_TYPE_NAMES)
}
SOLVENT_TYPE_COUNT = len(SOLVENT_TYPE_NAMES)
SOLVENT_TYPE_OTHER_ID = SOLVENT_TYPE_TO_ID["other"]


_SMARTS = {
    "carboxylic_acid": Chem.MolFromSmarts("C(=O)[OX2H1]"),
    "amide": Chem.MolFromSmarts("C(=O)N"),
    "sulfoxide": Chem.MolFromSmarts("S(=O)"),
    "nitrile": Chem.MolFromSmarts("C#N"),
    "alcohol": Chem.MolFromSmarts("[CX4][OX2H]"),
    "halogen": Chem.MolFromSmarts("[F,Cl,Br,I]"),
    "aromatic": Chem.MolFromSmarts("a"),
}


def _is_water(mol: Chem.Mol) -> bool:
    return mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetSymbol() == "O"


def _is_hydrocarbon(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ("C", "H"):
            return False
    return True


def solvent_type_from_mol(mol: Chem.Mol) -> str:
    """
    Assign a single solvent type from an RDKit Mol.

    Priority order:
      water → carboxylic_acid → amide → sulfoxide → nitrile → alcohol
      → halogenated → aromatic → hydrocarbon → other
    """
    if mol is None:
        return "other"
    if _is_water(mol):
        return "water"
    if mol.HasSubstructMatch(_SMARTS["carboxylic_acid"]):
        return "carboxylic_acid"
    if mol.HasSubstructMatch(_SMARTS["amide"]):
        return "amide"
    if mol.HasSubstructMatch(_SMARTS["sulfoxide"]):
        return "sulfoxide"
    if mol.HasSubstructMatch(_SMARTS["nitrile"]):
        return "nitrile"
    if mol.HasSubstructMatch(_SMARTS["alcohol"]):
        return "alcohol"
    if mol.HasSubstructMatch(_SMARTS["halogen"]):
        return "halogenated"
    if mol.HasSubstructMatch(_SMARTS["aromatic"]):
        return "aromatic"
    if _is_hydrocarbon(mol):
        return "hydrocarbon"
    return "other"


def solvent_type_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return solvent_type_from_mol(mol)


def solvent_type_id_from_smiles(smiles: str) -> int:
    return SOLVENT_TYPE_TO_ID.get(
        solvent_type_from_smiles(smiles),
        SOLVENT_TYPE_TO_ID["other"],
    )
