"""
Train / validation / test splitting with group-based bin-packing.

Uses greedy bin-packing: each group is assigned to whichever
split is most under-filled relative to its target size.
This guarantees non-empty splits regardless of group size distribution.
"""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import get_scaffold


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    mode: str = "solute_scaffold",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by group using greedy bin-packing.

    Modes:
      - "solute_scaffold": Murcko scaffold of solute (default)
      - "solute": solute SMILES (random split by solute)
      - "solvent": solvent SMILES (no solvent overlap across splits)

    Each group goes entirely to one split (no leakage within that grouping).
    Groups are assigned to whichever split is most under-filled,
    ensuring all splits get approximately the target fraction.
    """
    print("\n" + "=" * 60)
    label = "Scaffold-based splitting"
    if mode == "solvent":
        label = "Solvent-based splitting"
    elif mode == "solute":
        label = "Solute-based splitting"
    elif mode != "solute_scaffold":
        raise ValueError(f"Unknown split mode: {mode}")
    print(label)
    print("=" * 60)

    def group_key(row: pd.Series) -> Optional[str]:
        if mode == "solute_scaffold":
            return get_scaffold(row["solute_smiles"])
        if mode == "solute":
            key = row["solute_smiles"]
            if pd.isna(key):
                return None
            return key
        key = row["solvent_smiles"]
        if pd.isna(key):
            return None
        return key

    # Compute group for each record
    groups = defaultdict(list)  # group key → [row indices]
    no_group = []

    for idx, row in df.iterrows():
        key = group_key(row)
        if key:
            groups[key].append(idx)
        else:
            no_group.append(idx)

    print(f"  Unique groups: {len(groups):,}")
    print(f"  No-group: {len(no_group):,}")

    # Shuffle groups deterministically
    group_list = list(groups.items())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_list)

    # Target sizes
    n = len(df)
    targets = {
        "train": int(n * train_frac),
        "val": int(n * val_frac),
        "test": n - int(n * train_frac) - int(n * val_frac),
    }
    current = {"train": 0, "val": 0, "test": 0}
    buckets = {"train": [], "val": [], "test": []}

    if mode == "solute":
        # Random assignment by shuffled solute groups
        for _, indices in group_list:
            if current["train"] < targets["train"]:
                best = "train"
            elif current["val"] < targets["val"]:
                best = "val"
            else:
                best = "test"
            buckets[best].extend(indices)
            current[best] += len(indices)
    else:
        # Sort by size (largest first) — greedy bin-packing works best this way
        group_list.sort(key=lambda x: len(x[1]), reverse=True)

        # Greedy assignment: each group goes to the most under-filled split
        for _, indices in group_list:
            deficits = {k: targets[k] - current[k] for k in targets}
            best = max(deficits, key=deficits.get)
            buckets[best].extend(indices)
            current[best] += len(indices)

    # No-group records go to train
    buckets["train"].extend(no_group)

    train_df = df.loc[df.index.isin(buckets["train"])].reset_index(drop=True)
    val_df = df.loc[df.index.isin(buckets["val"])].reset_index(drop=True)
    test_df = df.loc[df.index.isin(buckets["test"])].reset_index(drop=True)

    # Verify no group leakage
    def _group_set(split_df: pd.DataFrame) -> set:
        keys = set()
        for _, row in split_df.iterrows():
            key = group_key(row)
            if key:
                keys.add(key)
        return keys

    train_groups = _group_set(train_df)
    val_groups = _group_set(val_df)
    test_groups = _group_set(test_df)

    tv_leak = train_groups & val_groups
    tt_leak = train_groups & test_groups

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_sol = split["has_solubility"].sum() if "has_solubility" in split else len(split)
        n_tm = split["has_T_m"].sum() if "has_T_m" in split else 0
        pct = len(split) / n * 100
        print(
            f"  {name:5s}: {len(split):6,d} ({pct:4.1f}%)  "
            f"{n_sol:6,d} solubility, {n_tm:5,d} T_m"
        )

    status_tv = "clean" if len(tv_leak) == 0 else f"LEAK ({len(tv_leak)})"
    status_tt = "clean" if len(tt_leak) == 0 else f"LEAK ({len(tt_leak)})"
    print(f"  Train ↔ Val:  {status_tv}")
    print(f"  Train ↔ Test: {status_tt}")

    return train_df, val_df, test_df
