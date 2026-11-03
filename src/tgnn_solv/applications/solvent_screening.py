"""Comprehensive solvent-screening workflow built on top of TGNN-Solv."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover - project runtime normally has RDKit
    Chem = None
    Descriptors = None

from ..baselines.direct_gnn import DirectGNN
from ..config import TGNNSolvConfig
from ..domain import ApplicabilityDomain
from ..inference import (
    predict_direct_solubility,
    predict_solubility,
    temperature_scan,
    temperature_scan_direct,
)
from ..model import TGNNSolv
from .core import clamp


@dataclass(frozen=True)
class SolventEntry:
    name: str
    smiles: str
    solvent_class: str
    boiling_point_K: float | None
    density_g_mL: float | None
    ild_class: str
    green_score: int | None
    cost_relative: str
    h_bond_donor: bool
    h_bond_acceptor: bool
    protic: bool
    miscible_with_water: bool
    hansen_d: float | None = None
    hansen_p: float | None = None
    hansen_h: float | None = None


def _canonicalize_smiles(smiles: str) -> str:
    raw = str(smiles or "").strip()
    if not raw:
        return raw
    if Chem is None:
        return raw
    mol = Chem.MolFromSmiles(raw)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else raw


def _entry(
    name: str,
    smiles: str,
    solvent_class: str,
    *,
    boiling_point_K: float | None,
    density_g_mL: float | None,
    ild_class: str,
    green_score: int | None,
    cost_relative: str,
    h_bond_donor: bool,
    h_bond_acceptor: bool,
    protic: bool,
    miscible_with_water: bool,
    hansen_d: float | None = None,
    hansen_p: float | None = None,
    hansen_h: float | None = None,
) -> dict[str, Any]:
    canonical_smiles = _canonicalize_smiles(smiles)
    row = asdict(
        SolventEntry(
            name=name,
            smiles=canonical_smiles,
            solvent_class=solvent_class,
            boiling_point_K=boiling_point_K,
            density_g_mL=density_g_mL,
            ild_class=ild_class,
            green_score=green_score,
            cost_relative=cost_relative,
            h_bond_donor=h_bond_donor,
            h_bond_acceptor=h_bond_acceptor,
            protic=protic,
            miscible_with_water=miscible_with_water,
            hansen_d=hansen_d,
            hansen_p=hansen_p,
            hansen_h=hansen_h,
        )
    )
    row["class"] = solvent_class
    row["toxicity_class"] = ild_class
    row["boiling_point"] = boiling_point_K
    return row


BUILTIN_SOLVENT_LIBRARY: list[dict[str, Any]] = [
    _entry("Water", "O", "water", boiling_point_K=373.15, density_g_mL=0.997, ild_class="ICH Class 3", green_score=10, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.5, hansen_p=16.0, hansen_h=42.3),
    _entry("Methanol", "CO", "alcohol", boiling_point_K=337.85, density_g_mL=0.792, ild_class="ICH Class 2", green_score=5, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.1, hansen_p=12.3, hansen_h=22.3),
    _entry("Ethanol", "CCO", "alcohol", boiling_point_K=351.52, density_g_mL=0.789, ild_class="ICH Class 3", green_score=8, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.8, hansen_p=8.8, hansen_h=19.4),
    _entry("1-Propanol", "CCCO", "alcohol", boiling_point_K=370.35, density_g_mL=0.803, ild_class="ICH Class 3", green_score=6, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.0, hansen_p=6.8, hansen_h=17.4),
    _entry("Isopropanol", "CC(C)O", "alcohol", boiling_point_K=355.35, density_g_mL=0.786, ild_class="ICH Class 3", green_score=7, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.8, hansen_p=6.1, hansen_h=16.4),
    _entry("n-Butanol", "CCCCO", "alcohol", boiling_point_K=390.85, density_g_mL=0.810, ild_class="ICH Class 3", green_score=5, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=False, hansen_d=16.0, hansen_p=5.7, hansen_h=15.8),
    _entry("sec-Butanol", "CCC(C)O", "alcohol", boiling_point_K=372.65, density_g_mL=0.806, ild_class="not classified", green_score=5, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.7, hansen_p=5.8, hansen_h=15.2),
    _entry("tert-Butanol", "CC(C)(C)O", "alcohol", boiling_point_K=355.45, density_g_mL=0.781, ild_class="ICH Class 3", green_score=6, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.2, hansen_p=5.7, hansen_h=16.0),
    _entry("Acetone", "CC(=O)C", "ketone", boiling_point_K=329.45, density_g_mL=0.785, ild_class="ICH Class 3", green_score=8, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.5, hansen_p=10.4, hansen_h=7.0),
    _entry("MEK", "CCC(C)=O", "ketone", boiling_point_K=352.75, density_g_mL=0.805, ild_class="ICH Class 3", green_score=7, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=16.0, hansen_p=9.0, hansen_h=5.1),
    _entry("MIBK", "CC(C)CC(C)=O", "ketone", boiling_point_K=390.95, density_g_mL=0.802, ild_class="ICH Class 2", green_score=5, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.3, hansen_p=6.1, hansen_h=4.1),
    _entry("Cyclohexanone", "O=C1CCCCC1", "ketone", boiling_point_K=428.65, density_g_mL=0.947, ild_class="ICH Class 2", green_score=4, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=8.4, hansen_h=5.1),
    _entry("Methyl acetate", "CC(=O)OC", "ester", boiling_point_K=330.05, density_g_mL=0.932, ild_class="ICH Class 3", green_score=7, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.5, hansen_p=7.2, hansen_h=7.6),
    _entry("Ethyl acetate", "CCOC(=O)C", "ester", boiling_point_K=350.25, density_g_mL=0.897, ild_class="ICH Class 3", green_score=8, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.8, hansen_p=5.3, hansen_h=7.2),
    _entry("Isopropyl acetate", "CC(C)OC(=O)C", "ester", boiling_point_K=361.95, density_g_mL=0.872, ild_class="not classified", green_score=7, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.3, hansen_p=4.3, hansen_h=6.9),
    _entry("n-Butyl acetate", "CCCCOC(=O)C", "ester", boiling_point_K=399.15, density_g_mL=0.882, ild_class="ICH Class 3", green_score=6, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.8, hansen_p=3.7, hansen_h=6.3),
    _entry("Propyl acetate", "CCCOC(=O)C", "ester", boiling_point_K=374.15, density_g_mL=0.888, ild_class="not classified", green_score=6, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.8, hansen_p=4.0, hansen_h=6.6),
    _entry("Isoamyl acetate", "CC(C)CCOC(=O)C", "ester", boiling_point_K=415.85, density_g_mL=0.876, ild_class="not classified", green_score=5, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.8, hansen_p=2.6, hansen_h=5.7),
    _entry("Dimethyl carbonate", "COC(=O)OC", "carbonate", boiling_point_K=363.15, density_g_mL=1.069, ild_class="not classified", green_score=9, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.5, hansen_p=9.7, hansen_h=9.2),
    _entry("Diethyl carbonate", "CCOC(=O)OCC", "carbonate", boiling_point_K=399.45, density_g_mL=0.973, ild_class="not classified", green_score=8, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.4, hansen_p=6.1, hansen_h=7.0),
    _entry("Propylene carbonate", "CC1OC(=O)OC1", "carbonate", boiling_point_K=514.15, density_g_mL=1.205, ild_class="not classified", green_score=7, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=20.0, hansen_p=18.0, hansen_h=4.1),
    _entry("DCM", "ClCCl", "chlorinated", boiling_point_K=312.95, density_g_mL=1.326, ild_class="ICH Class 2", green_score=2, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=18.2, hansen_p=6.3, hansen_h=6.1),
    _entry("Chloroform", "ClC(Cl)Cl", "chlorinated", boiling_point_K=334.35, density_g_mL=1.489, ild_class="ICH Class 2", green_score=1, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=3.1, hansen_h=5.7),
    _entry("Carbon tetrachloride", "ClC(Cl)(Cl)Cl", "chlorinated", boiling_point_K=349.85, density_g_mL=1.594, ild_class="ICH Class 1", green_score=1, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=0.0, hansen_h=0.6),
    _entry("1,2-Dichloroethane", "ClCCCl", "chlorinated", boiling_point_K=356.65, density_g_mL=1.253, ild_class="ICH Class 1", green_score=1, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=18.2, hansen_p=7.3, hansen_h=4.1),
    _entry("Chlorobenzene", "Clc1ccccc1", "chlorinated", boiling_point_K=404.95, density_g_mL=1.106, ild_class="ICH Class 2", green_score=2, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=19.0, hansen_p=4.3, hansen_h=2.0),
    _entry("Benzene", "c1ccccc1", "aromatic", boiling_point_K=353.25, density_g_mL=0.877, ild_class="ICH Class 1", green_score=1, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=18.4, hansen_p=0.0, hansen_h=2.0),
    _entry("Toluene", "Cc1ccccc1", "aromatic", boiling_point_K=383.75, density_g_mL=0.867, ild_class="ICH Class 2", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=18.0, hansen_p=1.4, hansen_h=2.0),
    _entry("Ethylbenzene", "CCc1ccccc1", "aromatic", boiling_point_K=409.35, density_g_mL=0.867, ild_class="not classified", green_score=4, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=0.6, hansen_h=1.0),
    _entry("m-Xylene", "Cc1cccc(C)c1", "aromatic", boiling_point_K=412.25, density_g_mL=0.864, ild_class="ICH Class 2", green_score=3, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=1.0, hansen_h=3.1),
    _entry("o-Xylene", "Cc1ccccc1C", "aromatic", boiling_point_K=417.55, density_g_mL=0.879, ild_class="ICH Class 2", green_score=3, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.6, hansen_p=1.0, hansen_h=3.1),
    _entry("p-Xylene", "Cc1ccc(C)cc1", "aromatic", boiling_point_K=411.45, density_g_mL=0.861, ild_class="ICH Class 2", green_score=3, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=1.0, hansen_h=3.1),
    _entry("Anisole", "COc1ccccc1", "aromatic ether", boiling_point_K=427.15, density_g_mL=0.995, ild_class="not classified", green_score=6, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=17.8, hansen_p=4.0, hansen_h=4.1),
    _entry("Cyclohexane", "C1CCCCC1", "cycloalkane", boiling_point_K=353.85, density_g_mL=0.779, ild_class="ICH Class 2", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=16.8, hansen_p=0.0, hansen_h=0.2),
    _entry("Hexane", "CCCCCC", "alkane", boiling_point_K=341.85, density_g_mL=0.655, ild_class="ICH Class 2", green_score=2, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=14.9, hansen_p=0.0, hansen_h=0.0),
    _entry("Heptane", "CCCCCCC", "alkane", boiling_point_K=371.55, density_g_mL=0.684, ild_class="ICH Class 3", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=15.3, hansen_p=0.0, hansen_h=0.0),
    _entry("Octane", "CCCCCCCC", "alkane", boiling_point_K=398.75, density_g_mL=0.703, ild_class="not classified", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=15.5, hansen_p=0.0, hansen_h=0.0),
    _entry("Isooctane", "CC(C)CC(C)(C)C", "alkane", boiling_point_K=372.35, density_g_mL=0.692, ild_class="not classified", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=14.4, hansen_p=0.0, hansen_h=0.0),
    _entry("Petroleum ether", "CCCCCC", "alkane blend", boiling_point_K=333.15, density_g_mL=0.640, ild_class="ICH Class 3", green_score=3, cost_relative="low", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=14.9, hansen_p=0.0, hansen_h=0.0),
    _entry("Diethyl ether", "CCOCC", "ether", boiling_point_K=307.75, density_g_mL=0.713, ild_class="ICH Class 3", green_score=5, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=14.5, hansen_p=2.9, hansen_h=4.6),
    _entry("THF", "C1CCOC1", "ether", boiling_point_K=339.05, density_g_mL=0.889, ild_class="ICH Class 2", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=16.8, hansen_p=5.7, hansen_h=8.0),
    _entry("2-MeTHF", "CC1CCCO1", "ether", boiling_point_K=353.65, density_g_mL=0.854, ild_class="not classified", green_score=8, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=16.7, hansen_p=5.0, hansen_h=4.3),
    _entry("1,4-Dioxane", "O1CCOCC1", "ether", boiling_point_K=374.25, density_g_mL=1.033, ild_class="ICH Class 2", green_score=2, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=19.0, hansen_p=1.8, hansen_h=7.4),
    _entry("MTBE", "COC(C)(C)C", "ether", boiling_point_K=328.25, density_g_mL=0.740, ild_class="ICH Class 3", green_score=5, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=14.5, hansen_p=3.6, hansen_h=4.7),
    _entry("CPME", "COC1CCCC1", "ether", boiling_point_K=379.15, density_g_mL=0.860, ild_class="not classified", green_score=8, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=False, hansen_d=15.6, hansen_p=3.0, hansen_h=4.3),
    _entry("Diglyme", "COCCOCCOC", "ether", boiling_point_K=435.15, density_g_mL=0.944, ild_class="not classified", green_score=4, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=16.6, hansen_p=5.3, hansen_h=8.3),
    _entry("DMF", "CN(C)C=O", "amide", boiling_point_K=426.15, density_g_mL=0.944, ild_class="ICH Class 2", green_score=2, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=17.4, hansen_p=13.7, hansen_h=11.3),
    _entry("DMAc", "CC(=O)N(C)C", "amide", boiling_point_K=438.15, density_g_mL=0.937, ild_class="ICH Class 2", green_score=2, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=16.8, hansen_p=11.5, hansen_h=10.2),
    _entry("NMP", "CN1CCCC1=O", "amide", boiling_point_K=475.15, density_g_mL=1.028, ild_class="ICH Class 2", green_score=1, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=18.0, hansen_p=12.3, hansen_h=7.2),
    _entry("DMSO", "CS(C)=O", "sulfoxide", boiling_point_K=462.15, density_g_mL=1.095, ild_class="ICH Class 3", green_score=5, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=18.4, hansen_p=16.4, hansen_h=10.2),
    _entry("Sulfolane", "O=S1(=O)CCCC1", "sulfone", boiling_point_K=558.15, density_g_mL=1.262, ild_class="not classified", green_score=3, cost_relative="high", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=20.0, hansen_p=14.0, hansen_h=6.0),
    _entry("Acetonitrile", "CC#N", "nitrile", boiling_point_K=354.75, density_g_mL=0.786, ild_class="ICH Class 2", green_score=4, cost_relative="low", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.3, hansen_p=18.0, hansen_h=6.1),
    _entry("Propionitrile", "CCC#N", "nitrile", boiling_point_K=370.15, density_g_mL=0.777, ild_class="not classified", green_score=4, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.1, hansen_p=14.1, hansen_h=6.4),
    _entry("Nitromethane", "C[N+](=O)[O-]", "nitro", boiling_point_K=374.35, density_g_mL=1.138, ild_class="not classified", green_score=3, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=True, protic=False, miscible_with_water=True, hansen_d=15.8, hansen_p=18.8, hansen_h=3.8),
    _entry("Acetic acid", "CC(=O)O", "acid", boiling_point_K=391.05, density_g_mL=1.049, ild_class="ICH Class 3", green_score=6, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=14.5, hansen_p=8.0, hansen_h=13.5),
    _entry("Formic acid", "O=CO", "acid", boiling_point_K=373.95, density_g_mL=1.220, ild_class="not classified", green_score=6, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=14.3, hansen_p=11.9, hansen_h=16.6),
    _entry("TFA", "OC(=O)C(F)(F)F", "acid", boiling_point_K=345.05, density_g_mL=1.489, ild_class="not classified", green_score=1, cost_relative="high", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.2, hansen_p=6.2, hansen_h=11.0),
    _entry("Ethyl lactate", "CCOC(=O)C(O)C", "ester", boiling_point_K=427.15, density_g_mL=1.034, ild_class="not classified", green_score=9, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.3, hansen_p=7.6, hansen_h=12.5),
    _entry("Isobutanol", "CC(C)CO", "alcohol", boiling_point_K=381.05, density_g_mL=0.802, ild_class="not classified", green_score=5, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=15.6, hansen_p=5.7, hansen_h=15.2),
    _entry("Diethylene glycol", "OCCOCCO", "glycol", boiling_point_K=517.15, density_g_mL=1.118, ild_class="not classified", green_score=6, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.2, hansen_p=10.5, hansen_h=18.0),
    _entry("Triethylene glycol", "OCCOCCOCCO", "glycol", boiling_point_K=558.15, density_g_mL=1.125, ild_class="not classified", green_score=6, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.5, hansen_p=10.2, hansen_h=18.5),
    _entry("PEG-400", "OCCOCCOCCOCCO", "polyether", boiling_point_K=None, density_g_mL=1.125, ild_class="not classified", green_score=7, cost_relative="medium", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.0, hansen_p=8.8, hansen_h=13.5),
    _entry("Propylene glycol", "CC(O)CO", "glycol", boiling_point_K=460.75, density_g_mL=1.036, ild_class="not classified", green_score=8, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=16.8, hansen_p=8.2, hansen_h=17.4),
    _entry("Glycerol", "C(C(CO)O)O", "polyol", boiling_point_K=563.15, density_g_mL=1.261, ild_class="not classified", green_score=9, cost_relative="low", h_bond_donor=True, h_bond_acceptor=True, protic=True, miscible_with_water=True, hansen_d=17.4, hansen_p=12.1, hansen_h=29.3),
    _entry("Supercritical CO2", "O=C=O", "supercritical", boiling_point_K=304.25, density_g_mL=0.800, ild_class="not classified", green_score=8, cost_relative="medium", h_bond_donor=False, h_bond_acceptor=False, protic=False, miscible_with_water=False, hansen_d=15.7, hansen_p=6.3, hansen_h=5.7),
]


def _toxicity_severity(ild_class: str) -> int:
    label = str(ild_class or "").lower()
    if "class 1" in label:
        return 3
    if "class 2" in label:
        return 2
    return 1


def _cost_rank(cost_relative: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(cost_relative or "").lower(), 2)


class SolventScreener:
    """Screen solvents, crystallization windows, and solvent swaps."""

    MG_ML_ASSUMPTION = (
        "Approximate mg/mL assumes the solvent dominates liquid volume: "
        "n1 ≈ rho/MW_solvent per mL, n2 = x2/(1-x2)*n1, and solution density "
        "is approximated by the pure-solvent density."
    )

    def __init__(
        self,
        model: TGNNSolv | DirectGNN,
        cfg: TGNNSolvConfig,
        device: torch.device,
        solvent_library: Sequence[str] | Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.model_family = "direct_gnn" if isinstance(model, DirectGNN) else "tgnn_solv"
        self.applicability_domain: ApplicabilityDomain | None = None
        self._solvent_library = self._normalize_solvent_library(solvent_library)
        self._library_by_smiles = {
            _canonicalize_smiles(str(entry["smiles"])): entry for entry in self._solvent_library
        }

    @property
    def solvent_library(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._solvent_library]

    def set_applicability_domain(self, ad: ApplicabilityDomain | None) -> None:
        self.applicability_domain = ad

    def fit_applicability_domain(self, train_loader: Any) -> ApplicabilityDomain:
        ad = ApplicabilityDomain(self.model, train_loader=train_loader)
        self.applicability_domain = ad
        return ad

    def screen(
        self,
        solute_smiles: str,
        T: float = 298.15,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        return_details: bool = True,
    ) -> pd.DataFrame:
        """Screen the configured solvent library for a target solute."""

        canonical_solute = self._require_smiles(solute_smiles, "solute")
        solute_mw = self._molecular_weight(canonical_solute)
        rows: list[dict[str, Any]] = []
        for entry in self._solvent_library:
            prediction = self._predict_one(canonical_solute, str(entry["smiles"]), float(T))
            row = self._prediction_to_row(
                canonical_solute,
                solute_mw,
                entry,
                prediction,
                temperature=T,
            )
            if row is not None:
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = self._apply_filters(df, filters)
        if df.empty:
            return df
        df = df.sort_values(["solubility_mg_mL", "x2"], ascending=[False, False]).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        if top_k and top_k > 0:
            df = df.head(int(top_k)).reset_index(drop=True)
        if not return_details:
            keep_cols = [
                "rank",
                "solvent_name",
                "solvent_smiles",
                "solvent_class",
                "ln_x2",
                "x2",
                "solubility_mg_mL",
                "solubility_g_L",
                "green_score",
                "toxicity_class",
                "boiling_point_K",
                "confidence",
            ]
            df = df[[col for col in keep_cols if col in df.columns]]
        df.attrs["assumptions"] = {
            "solubility_conversion": self.MG_ML_ASSUMPTION,
            "hansen_red": "RED is reported as Ra / 8 using a generic R0=8 MPa^0.5 screening heuristic.",
            "miscibility": "Miscibility is approximated from solvent metadata and solvent-solvent Hansen distance.",
        }
        return df

    def crystallization_window(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        T_hot: float | None = None,
        T_cold: float | None = None,
        n_points: int = 20,
    ) -> dict[str, Any]:
        """Estimate the hot-to-cold crystallization window for one solvent."""

        canonical_solute = self._require_smiles(solute_smiles, "solute")
        canonical_solvent = self._require_smiles(solvent_smiles, "solvent")
        probe_prediction = self._predict_one(canonical_solute, canonical_solvent, 298.15)
        default_cold = 278.15 if T_cold is None else float(T_cold)
        if T_hot is None:
            predicted_tm = probe_prediction.get("T_m")
            if predicted_tm is not None and math.isfinite(float(predicted_tm)):
                T_hot = min(373.15, max(default_cold + 10.0, float(predicted_tm) - 20.0))
            else:
                T_hot = 373.15
        if T_hot <= default_cold:
            T_hot = default_cold + 10.0
        scan_df = self._temperature_scan(
            canonical_solute,
            canonical_solvent,
            T_min=float(default_cold),
            T_max=float(T_hot),
            n_points=max(4, int(n_points)),
        )
        scan_df = self._augment_scan(scan_df, canonical_solute, canonical_solvent)
        x2_hot = float(scan_df.iloc[-1]["x2"])
        x2_cold = float(scan_df.iloc[0]["x2"])
        theoretical_yield = clamp((x2_hot - x2_cold) / max(x2_hot, 1e-12), 0.0, 1.0)
        supersaturation_ratio = x2_hot / max(x2_cold, 1e-12)
        slope = float(np.gradient(scan_df["ln_x2"].to_numpy(), scan_df["T"].to_numpy()).mean())
        metastable_zone_width = clamp(1.0 / max(abs(slope), 1e-3), 3.0, 25.0)
        if metastable_zone_width < 5.0:
            cooling_rate = "slow cooling recommended"
        elif metastable_zone_width > 15.0:
            cooling_rate = "fast cooling acceptable"
        else:
            cooling_rate = "moderate cooling recommended"
        return {
            "temperature_scan": scan_df,
            "T_hot": float(T_hot),
            "T_cold": float(default_cold),
            "x2_hot": x2_hot,
            "x2_cold": x2_cold,
            "theoretical_yield": theoretical_yield,
            "supersaturation_ratio": supersaturation_ratio,
            "recommended_cooling_rate": cooling_rate,
            "metastable_zone_width_estimate": metastable_zone_width,
            "dlnx2_dT_mean": slope,
        }

    def antisolvent_screening(
        self,
        solute_smiles: str,
        good_solvent_smiles: str,
        T: float = 298.15,
        n_points: int = 10,
    ) -> pd.DataFrame:
        """Rank antisolvents for drowning-out crystallization."""

        del n_points
        canonical_solute = self._require_smiles(solute_smiles, "solute")
        good_meta = self._library_by_smiles.get(_canonicalize_smiles(good_solvent_smiles))
        if good_meta is None:
            raise ValueError("The good solvent must be present in the screening library.")
        good_pred = self._predict_one(canonical_solute, str(good_meta["smiles"]), float(T))
        rows: list[dict[str, Any]] = []
        for entry in self._solvent_library:
            if str(entry["smiles"]) == str(good_meta["smiles"]):
                continue
            anti_pred = self._predict_one(canonical_solute, str(entry["smiles"]), float(T))
            miscible = self._solvent_miscibility(good_meta, entry)
            ratio = float(good_pred["x2"]) / max(float(anti_pred["x2"]), 1e-12)
            recommended = (
                ratio >= 10.0
                and miscible
                and _toxicity_severity(str(entry["ild_class"])) <= 2
                and (entry.get("boiling_point_K") is None or float(entry["boiling_point_K"]) <= 430.0)
            )
            rows.append(
                {
                    "antisolvent_name": entry["name"],
                    "antisolvent_smiles": entry["smiles"],
                    "antisolvent_class": entry["solvent_class"],
                    "x2_in_good_solvent": float(good_pred["x2"]),
                    "x2_in_antisolvent": float(anti_pred["x2"]),
                    "solubility_ratio": ratio,
                    "miscible_with_good_solvent": miscible,
                    "recommended": recommended,
                    "green_score": entry.get("green_score"),
                    "toxicity_class": entry.get("ild_class"),
                    "boiling_point_K": entry.get("boiling_point_K"),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(["recommended", "solubility_ratio", "green_score"], ascending=[False, False, False]).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def solvent_swap_path(
        self,
        solute_smiles: str,
        from_solvent: str,
        to_solvent: str,
        T: float = 298.15,
        max_steps: int = 3,
    ) -> list[dict[str, Any]]:
        """Find an approximate solvent-swap path between two library solvents."""

        canonical_solute = self._require_smiles(solute_smiles, "solute")
        start = self._library_by_smiles.get(_canonicalize_smiles(from_solvent))
        target = self._library_by_smiles.get(_canonicalize_smiles(to_solvent))
        if start is None or target is None:
            raise ValueError("Both source and target solvents must be present in the screening library.")
        if start["smiles"] == target["smiles"]:
            return []

        candidates = list(self._solvent_library)
        adjacency: dict[str, list[str]] = {str(entry["smiles"]): [] for entry in candidates}
        for left in candidates:
            for right in candidates:
                if left["smiles"] == right["smiles"]:
                    continue
                if self._solvent_miscibility(left, right):
                    adjacency[str(left["smiles"])].append(str(right["smiles"]))

        start_key = str(start["smiles"])
        target_key = str(target["smiles"])
        queue: list[list[str]] = [[start_key]]
        seen = {start_key}
        chosen_path: list[str] | None = None
        while queue:
            path = queue.pop(0)
            if path[-1] == target_key:
                chosen_path = path
                break
            if len(path) - 1 >= max_steps:
                continue
            for nxt in adjacency.get(path[-1], []):
                if nxt in seen and nxt != target_key:
                    continue
                seen.add(nxt)
                queue.append(path + [nxt])

        if not chosen_path:
            return []

        steps: list[dict[str, Any]] = []
        for idx in range(len(chosen_path) - 1):
            left = self._library_by_smiles[chosen_path[idx]]
            right = self._library_by_smiles[chosen_path[idx + 1]]
            pred_left = self._predict_one(canonical_solute, str(left["smiles"]), float(T))
            pred_right = self._predict_one(canonical_solute, str(right["smiles"]), float(T))
            volume_ratio = clamp(float(pred_left["x2"]) / max(float(pred_right["x2"]), 1e-12), 0.1, 20.0)
            steps.append(
                {
                    "from_solvent": left["name"],
                    "to_solvent": right["name"],
                    "miscibility": self._solvent_miscibility(left, right),
                    "x2_in_from": float(pred_left["x2"]),
                    "x2_in_to": float(pred_right["x2"]),
                    "volume_ratio_estimate": volume_ratio,
                }
            )
        return steps

    def green_solvent_replacement(
        self,
        solute_smiles: str,
        current_solvent_smiles: str,
        T: float = 298.15,
        min_solubility_fraction: float = 0.5,
    ) -> pd.DataFrame:
        """Find greener solvent alternatives that preserve enough solubility."""

        canonical_solute = self._require_smiles(solute_smiles, "solute")
        current_meta = self._library_by_smiles.get(_canonicalize_smiles(current_solvent_smiles))
        if current_meta is None:
            raise ValueError("The current solvent must be present in the screening library.")
        current_pred = self._predict_one(canonical_solute, str(current_meta["smiles"]), float(T))
        current_green = current_meta.get("green_score")
        current_toxicity = _toxicity_severity(str(current_meta.get("ild_class", "")))

        rows: list[dict[str, Any]] = []
        for entry in self._solvent_library:
            if str(entry["smiles"]) == str(current_meta["smiles"]):
                continue
            pred = self._predict_one(canonical_solute, str(entry["smiles"]), float(T))
            retention = float(pred["x2"]) / max(float(current_pred["x2"]), 1e-12)
            green_improvement = (
                (int(entry["green_score"]) - int(current_green))
                if entry.get("green_score") is not None and current_green is not None
                else None
            )
            toxicity_improvement = current_toxicity - _toxicity_severity(str(entry.get("ild_class", "")))
            recommended = (
                retention >= float(min_solubility_fraction)
                and (green_improvement is None or green_improvement > 0)
                and toxicity_improvement >= 0
            )
            rows.append(
                {
                    "solvent_name": entry["name"],
                    "solvent_smiles": entry["smiles"],
                    "green_score": entry.get("green_score"),
                    "green_improvement": green_improvement,
                    "x2_current": float(current_pred["x2"]),
                    "x2_alternative": float(pred["x2"]),
                    "solubility_retention": retention,
                    "toxicity_class": entry.get("ild_class"),
                    "toxicity_improvement": toxicity_improvement,
                    "recommended": recommended,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(
            ["recommended", "green_improvement", "solubility_retention"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def _normalize_solvent_library(
        self,
        solvent_library: Sequence[str] | Sequence[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if solvent_library is None:
            return [dict(entry) for entry in BUILTIN_SOLVENT_LIBRARY]
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(solvent_library):
            if isinstance(item, str):
                normalized.append(
                    _entry(
                        f"Custom solvent {idx + 1}",
                        item,
                        "custom",
                        boiling_point_K=None,
                        density_g_mL=1.0,
                        ild_class="not classified",
                        green_score=None,
                        cost_relative="medium",
                        h_bond_donor=False,
                        h_bond_acceptor=False,
                        protic=False,
                        miscible_with_water=False,
                    )
                )
            else:
                payload = dict(item)
                normalized.append(
                    _entry(
                        str(payload.get("name", f"Custom solvent {idx + 1}")),
                        str(payload.get("smiles", "")),
                        str(payload.get("solvent_class", payload.get("class", "custom"))),
                        boiling_point_K=payload.get("boiling_point_K"),
                        density_g_mL=payload.get("density_g_mL"),
                        ild_class=str(payload.get("ild_class", payload.get("toxicity_class", "not classified"))),
                        green_score=payload.get("green_score"),
                        cost_relative=str(payload.get("cost_relative", "medium")),
                        h_bond_donor=bool(payload.get("h_bond_donor", False)),
                        h_bond_acceptor=bool(payload.get("h_bond_acceptor", False)),
                        protic=bool(payload.get("protic", False)),
                        miscible_with_water=bool(payload.get("miscible_with_water", False)),
                        hansen_d=payload.get("hansen_d"),
                        hansen_p=payload.get("hansen_p"),
                        hansen_h=payload.get("hansen_h"),
                    )
                )
        return normalized

    def _require_smiles(self, smiles: str, role: str) -> str:
        canonical = _canonicalize_smiles(smiles)
        if not canonical:
            raise ValueError(f"Invalid {role} SMILES: {smiles!r}")
        return canonical

    def _molecular_weight(self, smiles: str) -> float | None:
        if Chem is None or Descriptors is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return float(Descriptors.MolWt(mol)) if mol is not None else None

    def _predict_one(self, solute_smiles: str, solvent_smiles: str, T: float) -> dict[str, Any]:
        if self.model_family == "direct_gnn":
            return predict_direct_solubility(
                self.model,
                solute_smiles,
                solvent_smiles,
                T=T,
                device=self.device,
            )
        return predict_solubility(
            self.model,
            solute_smiles,
            solvent_smiles,
            T=T,
            device=self.device,
        )

    def _temperature_scan(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        *,
        T_min: float,
        T_max: float,
        n_points: int,
    ) -> pd.DataFrame:
        if self.model_family == "direct_gnn":
            return temperature_scan_direct(
                self.model,
                solute_smiles,
                solvent_smiles,
                T_min=T_min,
                T_max=T_max,
                n_points=n_points,
                device=self.device,
            )
        return temperature_scan(
            self.model,
            solute_smiles,
            solvent_smiles,
            T_min=T_min,
            T_max=T_max,
            n_points=n_points,
            device=self.device,
        )

    def _augment_scan(
        self,
        scan_df: pd.DataFrame,
        solute_smiles: str,
        solvent_smiles: str,
    ) -> pd.DataFrame:
        df = scan_df.copy()
        solvent_meta = self._library_by_smiles.get(_canonicalize_smiles(solvent_smiles), {})
        solute_mw = self._molecular_weight(solute_smiles)
        solvent_mw = self._molecular_weight(solvent_smiles)
        density = solvent_meta.get("density_g_mL")
        df["solubility_mg_mL"] = [
            self._x2_to_mg_ml(
                float(x2),
                solute_mw=solute_mw,
                solvent_mw=solvent_mw,
                solvent_density_g_ml=density,
            )
            for x2 in df["x2"].tolist()
        ]
        if "gamma_2" not in df.columns:
            df["gamma_2"] = np.nan
        return df

    def _prediction_to_row(
        self,
        solute_smiles: str,
        solute_mw: float | None,
        entry: dict[str, Any],
        prediction: dict[str, Any],
        *,
        temperature: float,
    ) -> dict[str, Any] | None:
        solvent_smiles = str(entry["smiles"])
        solvent_mw = self._molecular_weight(solvent_smiles)
        x2 = float(prediction["x2"])
        ln_x2 = float(prediction["ln_x2"])
        if not math.isfinite(x2) or not math.isfinite(ln_x2):
            return None
        mg_per_ml = self._x2_to_mg_ml(
            x2,
            solute_mw=solute_mw,
            solvent_mw=solvent_mw,
            solvent_density_g_ml=entry.get("density_g_mL"),
        )
        ad_score = self._ad_score(solute_smiles, solvent_smiles, float(temperature))
        confidence = ad_score.get("confidence") if ad_score else None
        hansen_ra = prediction.get("Ra")
        hansen_red = None
        if hansen_ra is not None and math.isfinite(float(hansen_ra)):
            hansen_red = float(hansen_ra) / 8.0
        hansen_sol = prediction.get("hansen_sol") or [None, None, None]
        hansen_slv = prediction.get("hansen_slv") or [
            entry.get("hansen_d"),
            entry.get("hansen_p"),
            entry.get("hansen_h"),
        ]
        row = {
            "solvent_name": entry["name"],
            "solvent_smiles": solvent_smiles,
            "solvent_class": entry["solvent_class"],
            "ln_x2": ln_x2,
            "x2": x2,
            "solubility_mg_mL": mg_per_ml,
            "solubility_g_L": mg_per_ml,
            "gamma_2": prediction.get("gamma_2"),
            "Phi": prediction.get("Phi"),
            "T_m_pred": prediction.get("T_m"),
            "dH_fus_pred": prediction.get("dH_fus"),
            "hansen_Ra": hansen_ra,
            "hansen_RED": hansen_red,
            "tau_12": prediction.get("tau_12"),
            "tau_21": prediction.get("tau_21"),
            "confidence": confidence,
            "green_score": entry.get("green_score"),
            "toxicity_class": entry.get("ild_class"),
            "toxicity_rank": _toxicity_severity(str(entry.get("ild_class", ""))),
            "boiling_point_K": entry.get("boiling_point_K"),
            "boiling_point": entry.get("boiling_point_K"),
            "density_g_mL": entry.get("density_g_mL"),
            "cost_relative": entry.get("cost_relative"),
            "cost_rank": _cost_rank(str(entry.get("cost_relative", ""))),
            "miscible_with_water": entry.get("miscible_with_water"),
            "h_bond_donor": entry.get("h_bond_donor"),
            "h_bond_acceptor": entry.get("h_bond_acceptor"),
            "protic": entry.get("protic"),
            "ild_class": entry.get("ild_class"),
            "model_family": self.model_family,
            "temperature_K": float(temperature),
            "assumption_solubility_conversion": self.MG_ML_ASSUMPTION,
            "ad_in_domain": ad_score.get("in_domain") if ad_score else None,
            "ad_mahalanobis": ad_score.get("mahalanobis") if ad_score else None,
            "ad_tanimoto_solute": ad_score.get("tanimoto_solute") if ad_score else None,
            "ad_tanimoto_solvent": ad_score.get("tanimoto_solvent") if ad_score else None,
            "hansen_sol_d": hansen_sol[0],
            "hansen_sol_p": hansen_sol[1],
            "hansen_sol_h": hansen_sol[2],
            "hansen_slv_d": hansen_slv[0],
            "hansen_slv_p": hansen_slv[1],
            "hansen_slv_h": hansen_slv[2],
        }
        return row

    def _apply_filters(self, df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
        if not filters:
            return df
        filtered = df.copy()
        min_solubility = filters.get("min_solubility_mg_mL")
        if min_solubility is not None:
            filtered = filtered[filtered["solubility_mg_mL"] >= float(min_solubility)]
        max_toxicity = filters.get("max_toxicity_class")
        if max_toxicity is not None:
            filtered = filtered[filtered["toxicity_rank"] <= int(max_toxicity)]
        min_green = filters.get("min_green_score")
        if min_green is not None:
            filtered = filtered[filtered["green_score"].fillna(-1) >= int(min_green)]
        max_bp = filters.get("max_boiling_point_K")
        if max_bp is not None:
            filtered = filtered[filtered["boiling_point_K"].fillna(float(max_bp)) <= float(max_bp)]
        exclude_classes = {
            str(item).strip().lower() for item in (filters.get("exclude_classes") or []) if str(item).strip()
        }
        if exclude_classes:
            filtered = filtered[~filtered["solvent_class"].astype(str).str.lower().isin(exclude_classes)]
        require_water_miscible = filters.get("require_water_miscible")
        if require_water_miscible:
            filtered = filtered[filtered["miscible_with_water"].fillna(False)]
        return filtered.reset_index(drop=True)

    def _x2_to_mg_ml(
        self,
        x2: float,
        *,
        solute_mw: float | None,
        solvent_mw: float | None,
        solvent_density_g_ml: float | None,
    ) -> float:
        if solute_mw is None or solvent_mw is None or solvent_density_g_ml is None:
            return float("nan")
        x2 = clamp(float(x2), 0.0, 0.999999)
        mol_ratio = x2 / max(1.0 - x2, 1e-9)
        return float(solvent_density_g_ml) * mol_ratio * (float(solute_mw) / float(solvent_mw)) * 1000.0

    def _ad_score(self, solute_smiles: str, solvent_smiles: str, T: float) -> dict[str, Any] | None:
        if self.applicability_domain is None:
            return None
        try:
            return self.applicability_domain.score(solute_smiles, solvent_smiles, T=T)
        except Exception:
            return None

    def _solvent_miscibility(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left["smiles"] == right["smiles"]:
            return True
        if left["smiles"] == _canonicalize_smiles("O"):
            return bool(right.get("miscible_with_water"))
        if right["smiles"] == _canonicalize_smiles("O"):
            return bool(left.get("miscible_with_water"))
        coords_left = (left.get("hansen_d"), left.get("hansen_p"), left.get("hansen_h"))
        coords_right = (right.get("hansen_d"), right.get("hansen_p"), right.get("hansen_h"))
        if all(value is not None for value in (*coords_left, *coords_right)):
            delta_d = float(coords_left[0]) - float(coords_right[0])
            delta_p = float(coords_left[1]) - float(coords_right[1])
            delta_h = float(coords_left[2]) - float(coords_right[2])
            ra = math.sqrt(4.0 * delta_d * delta_d + delta_p * delta_p + delta_h * delta_h)
            return ra <= 10.0
        if bool(left.get("miscible_with_water")) and bool(right.get("miscible_with_water")):
            return True
        if bool(left.get("protic")) == bool(right.get("protic")) and bool(left.get("h_bond_acceptor")) == bool(right.get("h_bond_acceptor")):
            return True
        return False


__all__ = ["BUILTIN_SOLVENT_LIBRARY", "SolventScreener"]
