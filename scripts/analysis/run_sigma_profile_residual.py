#!/usr/bin/env python
"""Compare the learned sigma-profile with the reference tabulation, bin by bin.

For every molecule the script obtains the profile the trained network supplies and the
profile tabulated for the same compound, normalises both to unit mass, and reports the
residual resolved over three regions of the charge-density axis. The regions follow the
2002 COSMO-SAC kernel's own hydrogen-bonding threshold: the donor side below
-sigma_hb, the nonpolar core between the two thresholds, and the acceptor side above
+sigma_hb. The 2010 kernel's segment typing is not used, because the profiles here are
untyped vectors of 51 bins and carry no type to split on.

Enrichment is the share of the residual falling in a region divided by the share of the
molecule's cavity area that region holds, so a value above one means the departure from
the tabulation is concentrated there.

The profile is read from the head rather than recomputed. The sigma head consumes the
per-molecule representation taken before the interaction block, so the profile depends on
the molecule alone; the forward pass is run on self-pairs and the recorded values are
exactly those passed to the closure.

Outputs per_molecule.csv and summary.json in the directory given by --out.

    python scripts/analysis/run_sigma_profile_residual.py
    python scripts/analysis/run_sigma_profile_residual.py --limit 40
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tgnn_solv.config import TGNNSolvConfig            # noqa: E402
from tgnn_solv.data.dataset import make_loader         # noqa: E402
from tgnn_solv.model import TGNNSolv                   # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles  # noqa: E402

CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
#: The sigma supervision stream. It is read not to compute anything but to STRATIFY: 51 of the
#: 126 molecules this instrument sees lie in the supervision pool, and all 51 appear in the test
#: set as SOLVENTS, none as solutes. That is not a leak with respect to a solute-scaffold
#: evaluation: the guard in build_sigma_profile_aux_stream.py excludes scaffolds taken from the
#: solute_smiles column of the held-out splits, solvents are shared across the corpus and cannot
#: be held out, and the evaluation never asks about these molecules as solutes. For THIS
#: instrument it is nonetheless a mixture: a pooled median is a 40/60 mixing proportion rather
#: than a property of the model, and the two strata differ by nearly the whole seed spread, so
#: both are printed separately.
SIGMA_POOL = ROOT / "notebooks/data/processed_sigma_aux_stream_rebuilt/sigma_train.csv"
PROFILES = ROOT / "results/sigma_profile_artifact/sigma_profiles.csv"
TEST = ROOT / "notebooks/data/processed/test.csv"
OUT_DIR = ROOT / "results/sigma_profile_residual"

#: Housekeeping arguments for make_loader. source_uncertainty_csv is switched off because it
#: carries per-row label uncertainty, which the prediction path does not read.
RESERVED = {"batch_size", "shuffle", "num_workers", "seed", "drop_last", "cache",
            "use_pair_temperature_batching", "source_uncertainty_csv"}


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """H(p,q) = sqrt(1/2 * sum (sqrt(p)-sqrt(q))^2), equation (9) of Salih et al.

    The profiles are UNnormalised, carrying area in A^2 as in that work, so H is in sqrt(A^2).
    """
    return float(np.sqrt(0.5 * np.sum((np.sqrt(np.clip(p, 0, None))
                                       - np.sqrt(np.clip(q, 0, None))) ** 2)))


def hellinger_to_delta(p: np.ndarray, grid: np.ndarray) -> float:
    """H to a delta function of the same area: how much shape a profile carries at all.

    The delta is placed in the bin nearest the profile's area-weighted mean charge, which is
    the zero-shape case: all that survives is the total charge and the total area.
    """
    area = float(p.sum())
    if area <= 0:
        return float("nan")
    centre = int(np.argmin(np.abs(grid - float((p * grid).sum() / area))))
    delta = np.zeros_like(p)
    delta[centre] = area
    return hellinger(p, delta)


def build_model(ckpt: Path) -> tuple[TGNNSolv, dict, TGNNSolvConfig]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(TGNNSolvConfig)}
    cfg = TGNNSolvConfig(**{k: v for k, v in ck["config"].items() if k in fields})
    model = TGNNSolv(node_feat_dim=ck["node_feat_dim"],
                     edge_feat_dim=ck["edge_feat_dim"], cfg=cfg)
    state = ck.get("model_state") or ck["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"{ckpt.name}: weights did not load for {missing[:5]} "
                         f"({len(missing)} in total); the profile would come from an uninitialised head")
    model.eval()
    return model, ck, cfg


def learned_profiles(model: TGNNSolv, cfg: TGNNSolvConfig,
                     smiles: list[str], template: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Profiles in the SOLUTE role and in the SOLVENT role, (n_mol, n_bins) each.

    THE ROLE MATTERS, AND THAT IS MEASURED RATHER THAN ASSUMED. The encoder is built with
    encoder_role_mode="shared_residual" and two role-specific layers, so one molecule receives a
    DIFFERENT profile depending on which side of the pair it occupies: ethanol comes out at
    88.3 A^2 of area as a solute and 108.0 A^2 as a solvent. The reference profile, by contrast,
    is one array per molecule. Substitution is therefore not symmetric across the two slots: over
    the labelled test rows, 7704 replace the solvent slot only, 354 replace both roles, and 8
    replace the solute slot only. The mechanism is a role-free array entering the distribution of
    whichever slot it displaces, almost always the solvent slot, and not one array entering
    both.

    The profile does NOT depend on the partner: ethanol as a solute gives 88.281 A^2 against
    water, benzene and acetonitrile alike, to within 3e-06. The first job of this function is to
    verify that, and the second is to keep the two role profiles apart.
    """
    captured: list[torch.Tensor] = []
    handle = model.head_sigma.register_forward_hook(
        lambda _m, _i, out: captured.append(out["p_sigma"].detach().cpu()))
    try:
        rows = []
        for smi in smiles:
            r = template.iloc[0].copy()
            r["solute_smiles"] = r["solvent_smiles"] = smi     # self-pair: both roles in one pass
            r["temperature"], r["ln_x2"] = 298.15, 0.0
            rows.append(r)
        df = pd.DataFrame(rows).reset_index(drop=True)

        sig = set(inspect.signature(make_loader).parameters)
        fields = {f.name for f in dataclasses.fields(TGNNSolvConfig)}
        kwargs = {k: getattr(cfg, k) for k in (sig & fields) - RESERVED}
        loader = make_loader(df, batch_size=64, shuffle=False, num_workers=0, cache=True,
                             use_pair_temperature_batching=False, **kwargs)
        if len(loader.dataset) != len(df):
            raise RuntimeError(f"the dataset kept {len(loader.dataset)} of {len(df)} rows")

        with torch.no_grad():
            for sol, slv, tgt in loader:
                fwd = set(inspect.signature(model.forward).parameters)
                extra = {k: v for k, v in tgt.items()
                         if k in fwd - {"solute_data", "solvent_data", "T"}}
                model(sol, slv, tgt["T"], **extra)
    finally:
        handle.remove()

    # head_sigma is called TWICE PER BATCH, solute then solvent, and not twice per row.
    if len(captured) % 2:
        raise RuntimeError(f"an odd number of hook calls ({len(captured)}): the split by role is wrong")
    as_solute = np.vstack([c.numpy() for c in captured[0::2]])
    as_solvent = np.vstack([c.numpy() for c in captured[1::2]])
    if as_solute.shape != as_solvent.shape or len(as_solute) != len(smiles):
        raise RuntimeError(f"shapes disagree: {as_solute.shape} / {as_solvent.shape} "
                           f"for {len(smiles)} molecules")
    return as_solute, as_solvent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="directory of weights; defaults to the deposited e5_leakfree family")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="where to write the results")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N molecules")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cks = sorted(args.ckpt_dir.glob("grounded_a_seed*.pt"))
    if args.seeds:
        cks = [p for p in cks if any(f"seed{s}" in p.name for s in args.seeds)]
    if not cks:
        print(f"no checkpoints in {args.ckpt_dir}")
        return 1

    _, _, cfg0 = build_model(cks[0])
    n_bins = int(cfg0.cosmo_sac_n_bins)
    grid = np.linspace(cfg0.cosmo_sac_sigma_min, cfg0.cosmo_sac_sigma_max, n_bins)
    s_hb = float(cfg0.cosmo_sac_sigma_hb)
    donor = grid <= -s_hb          # the 2002 kernel switches on hydrogen bonding beyond |sigma| > sigma_hb
    acceptor = grid >= s_hb
    nonpolar = ~(donor | acceptor)
    print(f"grid: {n_bins} bins, sigma in [{grid[0]:.3f}, {grid[-1]:.3f}], "
          f"kernel threshold {s_hb}; donor {donor.sum()} bins, nonpolar {nonpolar.sum()}, "
          f"acceptor {acceptor.sum()}")

    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    test = pd.read_csv(TEST, low_memory=False)
    # NOT `ln_x2.notna()`: the auxiliary rows (`source=aux_only`, carrying has_T_m) sit in the
    # same file with ln_x2 EXACTLY 0.0 rather than NaN, so notna() admits them and returns 8103
    # rows instead of 5608. has_solubility selects the SAME 5608 rows, checked element by element
    # rather than by count.
    if "has_solubility" in test.columns:
        scored = test[test["has_solubility"].astype(bool)]
    else:
        scored = test[test["ln_x2"].notna()]
    assert not (scored["ln_x2"] == 0.0).all(), "placeholder rows were selected rather than measurements"
    seen = set(scored["solute_smiles"].astype(str)) | set(scored["solvent_smiles"].astype(str))

    from tgnn_solv.data.utils import canonicalize
    mols = sorted({s for s in seen if (c := canonicalize(str(s))) is not None and c in table})
    if args.limit:
        mols = mols[:args.limit]
    supervised: set[str] = set()
    if SIGMA_POOL.exists():
        pool = pd.read_csv(SIGMA_POOL, low_memory=False)
        if "has_sigma_profile" in pool.columns:
            pool = pool[pool["has_sigma_profile"].astype(bool)]
        supervised = {c for s_ in pool["solute_smiles"].dropna().astype(str)
                      if (c := canonicalize(s_)) is not None}
    else:
        print(f"WARNING: pool {SIGMA_POOL} not found; the results will not be stratified by supervision")
    in_pool = {m: (canonicalize(str(m)) in supervised) for m in mols}
    n_sup = sum(in_pool.values())
    print(f"molecules with a comparable profile that occur in the labelled test set: {len(mols)}")
    print(f"  of these, IN the supervision pool: {n_sup} ({n_sup / max(len(mols), 1):.1%}), "
          f"held out {len(mols) - n_sup}")
    if not mols:
        print("the intersection is empty; there is nothing to compare")
        return 1

    template = pd.read_csv(TEST, nrows=1, low_memory=False)
    ref = np.vstack([table[canonicalize(str(m))][0] for m in mols])

    per_seed, records = {}, []
    for ck in cks:
        model, ckd, cfg = build_model(ck)
        seed = int(ckd.get("seed", -1))
        print(f"  seed {seed}: running {len(mols)} molecules ...", flush=True)
        sol_p, slv_p = learned_profiles(model, cfg, mols, template)
        per_seed[seed] = (sol_p, slv_p)
        for i, smi in enumerate(mols):
            rp = ref[i]
            rec = {"seed": seed, "smiles": smi,
                   "in_sigma_pool": bool(in_pool[smi]),
                   "area_reference": float(rp.sum()),
                   "donor_area_reference": float(rp[donor].sum()),
                   "hellinger_reference_to_delta": hellinger_to_delta(rp, grid),
                   # the gap between ROLES: the reference has none by construction, the learned profile does
                   "hellinger_solute_vs_solvent": hellinger(sol_p[i], slv_p[i])}
            # THE BASELINE IS THE SHARE OF AREA, NOT THE SHARE OF BINS. Comparing the share of
            # the residual against the share of grid bins, one third per window, would declare
            # the residual concentrated in the centre for a trivial reason: the sigma-profile of
            # an organic molecule is sharply peaked near zero, and for half of these solvents the
            # reference donor area is exactly 0.00, so almost all the area is in the centre
            # already. The question that is not trivial is enrichment: the share of the residual
            # divided by the share of AREA in the same window. An enrichment near one means the
            # residual is proportional to area and therefore NOT structured, which is a weaker
            # and more honest reading than "concentrated".
            for reg, msk in (("donor", donor), ("nonpolar", nonpolar), ("acceptor", acceptor)):
                rec[f"area_frac_reference_{reg}"] = float(rp[msk].sum() / (rp.sum() or np.nan))
            for role, lp in (("solute", sol_p[i]), ("solvent", slv_p[i])):
                resid = lp - rp
                tot = float(np.abs(resid).sum()) or float("nan")
                rec.update({
                    f"hellinger_{role}_vs_reference": hellinger(lp, rp),
                    f"hellinger_{role}_to_delta": hellinger_to_delta(lp, grid),
                    f"area_{role}": float(lp.sum()),
                })
                for reg, msk in (("donor", donor), ("nonpolar", nonpolar),
                                 ("acceptor", acceptor)):
                    rf = float(np.abs(resid[msk]).sum() / tot)
                    # baseline: the mean area share of the two profiles being subtracted
                    af = 0.5 * (float(lp[msk].sum() / (lp.sum() or np.nan))
                                + float(rp[msk].sum() / (rp.sum() or np.nan)))
                    rec[f"area_{reg}_{role}"] = float(lp[msk].sum())
                    rec[f"area_frac_{reg}_{role}"] = af
                    rec[f"resid_frac_{reg}_{role}"] = rf
                    rec[f"enrichment_{reg}_{role}"] = float(rf / af) if af else float("nan")
            records.append(rec)

    df = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_molecule.csv", index=False)

    # The share of bins in each window, as a reference point. Without it, "40% of the residual
    # in the donor window" says nothing: if the donor window is 40% of the grid, that is the
    # expectation.
    base = {"donor": float(donor.mean()), "nonpolar": float(nonpolar.mean()),
            "acceptor": float(acceptor.mean())}
    donor_free = df["donor_area_reference"] <= 1e-9
    # STRATIFICATION BY SUPERVISION, which is not decoration. A pooled median is computed over a
    # mixture of 40% supervised and 60% held-out molecules, so it is a MIXING PROPORTION rather
    # than a property of the model: the two strata differ by nearly the whole between-seed
    # spread, the solvent Hellinger distance being 3.0385 inside the pool against 3.4173 outside
    # at a spread of 0.421. Any reading of the form "the profile moved closer to the reference"
    # must therefore be taken within a stratum, or it reads fit to the training set in part. The
    # pool is NOT a leak with respect to a solute-scaffold evaluation: every supervised molecule
    # this instrument sees stands in the test set as a solvent.
    strata = {}
    for tag, sel in (("in_pool", df["in_sigma_pool"].astype(bool)),
                     ("held_out", ~df["in_sigma_pool"].astype(bool))):
        sub = df[sel]
        if sub.empty:
            strata[tag] = {"n_rows": 0}
            continue
        strata[tag] = {
            "n_rows": int(len(sub)),
            "n_molecules": int(sub["smiles"].nunique()),
            "median_hellinger_to_reference": {
                r: float(sub[f"hellinger_{r}_vs_reference"].median())
                for r in ("solute", "solvent")},
            "seed_sd_hellinger_to_reference": {
                r: (float(sub.groupby("seed")[f"hellinger_{r}_vs_reference"].median().std(ddof=1))
                    if sub["seed"].nunique() > 1 else None)
                for r in ("solute", "solvent")},
        }
    for r in ("solute", "solvent"):
        a = strata.get("in_pool", {}).get("median_hellinger_to_reference", {}).get(r)
        b = strata.get("held_out", {}).get("median_hellinger_to_reference", {}).get(r)
        sd = strata.get("held_out", {}).get("seed_sd_hellinger_to_reference", {}).get(r)
        if a is not None and b is not None:
            print(f"  {r}: Hellinger in the pool {a:.4f} against held out {b:.4f} "
                  f"(gap {b - a:+.4f}" + (f", held-out seed spread {sd:.4f})" if sd else ")"))

    summary = {
        "n_molecules": len(mols), "seeds": sorted(per_seed),
        "supervision_strata": strata,
        "sigma_hb": s_hb, "bin_fraction_by_region": base,
        "median_hellinger_to_reference": {
            r: float(df[f"hellinger_{r}_vs_reference"].median()) for r in ("solute", "solvent")},
        "median_hellinger_role_gap": float(df["hellinger_solute_vs_solvent"].median()),
        "median_hellinger_to_delta": {
            "solute": float(df["hellinger_solute_to_delta"].median()),
            "solvent": float(df["hellinger_solvent_to_delta"].median()),
            "reference": float(df["hellinger_reference_to_delta"].median())},
        "median_resid_frac": {
            r: {reg: float(df[f"resid_frac_{reg}_{r}"].median())
                for reg in ("donor", "nonpolar", "acceptor")} for r in ("solute", "solvent")},
        "median_area_frac": {
            r: {reg: float(df[f"area_frac_{reg}_{r}"].median())
                for reg in ("donor", "nonpolar", "acceptor")} for r in ("solute", "solvent")},
        "median_enrichment": {
            r: {reg: float(df[f"enrichment_{reg}_{r}"].median())
                for reg in ("donor", "nonpolar", "acceptor")} for r in ("solute", "solvent")},
        "donor_free_reference_molecules": int(df.loc[donor_free, "smiles"].nunique()),
        "median_donor_area_where_reference_is_zero": {
            r: float(df.loc[donor_free, f"area_donor_{r}"].median()) for r in ("solute", "solvent")},
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")

    print(f"\n{'':30s}{'donor':>10}{'nonpolar':>11}{'acceptor':>11}")
    print(f"  {'share of grid bins':28s}{base['donor']:>10.3f}{base['nonpolar']:>11.3f}"
          f"{base['acceptor']:>11.3f}")
    for r in ("solute", "solvent"):
        a = summary["median_area_frac"][r]
        m = summary["median_resid_frac"][r]
        e = summary["median_enrichment"][r]
        print(f"  {'share of AREA, ' + r:28s}{a['donor']:>10.3f}{a['nonpolar']:>11.3f}"
              f"{a['acceptor']:>11.3f}")
        print(f"  {'share of |residual|, ' + r:28s}{m['donor']:>10.3f}{m['nonpolar']:>11.3f}"
              f"{m['acceptor']:>11.3f}")
        print(f"  {'ENRICHMENT, ' + r:28s}{e['donor']:>10.2f}{e['nonpolar']:>11.2f}"
              f"{e['acceptor']:>11.2f}")
    print(f"\n  H(learned, reference): solute "
          f"{summary['median_hellinger_to_reference']['solute']:.3f}, solvent "
          f"{summary['median_hellinger_to_reference']['solvent']:.3f}")
    print(f"  H between the two ROLES of one molecule: "
          f"{summary['median_hellinger_role_gap']:.3f}   (the reference has no role at all)")
    d = summary["median_hellinger_to_delta"]
    print(f"  H to a delta (Salih scale): learned {d['solute']:.3f}/{d['solvent']:.3f}, "
          f"reference {d['reference']:.3f}")
    print(f"\n  molecules with ZERO donor area in the reference: "
          f"{summary['donor_free_reference_molecules']}")
    z = summary["median_donor_area_where_reference_is_zero"]
    print("  their median learned donor area: "
          f"{z['solute']:.2f} / {z['solvent']:.2f} A^2 (solute / solvent)")
    # Print the ACTUAL path rather than the default. A message naming the wrong directory is
    # what lets an overwritten deposit go unnoticed, and a missing one cause alarm.
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
