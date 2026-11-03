#!/usr/bin/env python
"""Build the manuscript's summary table of diagnostics from the deposited measurements.

Every cell is read from a deposit by key rather than typed, so the table cannot drift away
from the artefacts it summarises. Each row names the control the result is read against,
because a contrast of this kind is not interpretable without one.

Outputs a LaTeX table fragment.

    python scripts/analysis/make_battery_table_tex.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/sections/tab-battery.tex"


def load(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        raise SystemExit(f"депозита нет: {rel}. Таблица не собирается по памяти.")
    return json.loads(p.read_text(encoding="utf8"))


def main() -> int:
    ident = load("results/identifiability_audit/summary.json")["base_regimes"]
    rank = load("results/closure_rank_profile_space/summary.json")
    attrib = load("results/per_bin_sensitivity_weights/closure_error_attribution.json")
    idac = load("results/idac_learned_vs_reference/summary.json")
    chan = load("results/e5_sigma_grounding/section_s1_channel_split.json")["summary"]["classes"]
    resid = load("results/sigma_profile_residual/clean_pool.json")
    floor = load("results/t2_softmax_floor/summary.json")
    costs = load("results/t2_softmax_floor/pedestal_costs.json")["per_seed"]
    refst = load("results/reference_state_violation/summary.json")["per_seed"]
    wts = load("results/per_bin_sensitivity_weights/summary.json")["by_rows"]

    # --- 1. идентифицируемость: граница Крамера-Рао в физических единицах ------------------
    cond0 = ident["solubility_only"]["cond"]
    crlb0 = ident["solubility_only"]["crlb_sd_dH_fus_Jmol"] / 1000.0
    crlb1 = ident["plus_Tm_dH"]["crlb_sd_dH_fus_Jmol"] / 1000.0

    # --- 2. обогащение остатка по окнам, чистый пул ----------------------------------------
    en = resid["median_enrichment"]["solute"]

    # --- 3. пьедестал -----------------------------------------------------------------------
    ped = [r[k]["pedestal_median"] for r in floor["per_seed"].values()
           for k in ("solute", "solvent")]
    mn = min(r[k]["min_bin_mass_anywhere"] for r in floor["per_seed"].values()
             for k in ("solute", "solvent"))
    below = [r[k]["share_of_zero_bins_below_1e_5"] for r in floor["per_seed"].values()
             for k in ("solute", "solvent")]
    ped0 = floor["untrained_null"]["pedestal_median"]
    zshare = 100 * floor["share_reference_zero_cells"]

    # --- 4. референсное состояние -----------------------------------------------------------
    med = [v["median_abs_n30"] for v in refst.values()]
    mx = max(v["max_abs_n300"] for v in refst.values())
    ctrl = max(abs(v["control_max_abs"]) for v in refst.values())

    # --- 5. цена размещения -----------------------------------------------------------------
    cc = {k: np.median(wts[k]["corner_over_core_per_seed"]) for k in ("train", "idac")}
    ccn = max(wts[k]["null"]["corner_over_core"] for k in ("train", "idac"))

    # --- 6, 7. подстановки ------------------------------------------------------------------
    fixnz = 100 * np.mean([r["recovered_fix_nonzero"] for r in costs.values()])
    zerop = 100 * np.mean([r["recovered_zero_pedestal"] for r in costs.values()])
    full_helps = sum(r["base_aad"] > r["aad_full"] for r in costs.values())
    # ВНИМАНИЕ НА ДЕПОЗИТ. Эта строка описывает подстановку ЭТАЛОНА в случайную треть бинов, и
    # читаться обязана из closure_error_attribution. Первая редакция брала долю из
    # pedestal_costs, где нуль -- ОБНУЛЕНИЕ случайных клеток, и печатала 94% там, где
    # описанная операция даёт 20%. Текст и таблица расходились на одном и том же измерении.
    ap_ = attrib["per_seed"]
    worse = 100 * np.mean([np.mean(np.asarray(r["null_aad"]) > r["base_aad"])
                           for r in ap_.values()])
    n_draw, n_seed = int(attrib["n_null"]), len(ap_)
    # --- 8: две стороны одной подстановки, с собственным плацебо --------------------------
    g_lrn = np.array([v["aad"] for v in idac["arms"]["learned"]["per_seed"].values()])
    g_ref = float(idac["arms"]["reference_vt2005"]["aad_mean"])
    x_pen = chan["solvent_only"]["mean"]
    x_sd = chan["solvent_only"]["sd_sample"]
    x_n = chan["solvent_only"]["n_per_seed"][0]
    pl_n = chan["neither"]["n_per_seed"][0]
    pl_max = chan["neither"]["max_abs_mae_difference"]

    rows = [
        (r"Does the observable determine the $\sigma$-profile?",
         rf"spectrum of the closure's Jacobian over the {rank['n_bins']} bins, "
         rf"{len(rank['solvents'])} solvents $\times$ {len(rank['seeds'])} seeds, against the "
         r"same spectrum on random profiles of equal area",
         rf"${rank['summary']['dirs_for_99pct_median']}$ of {rank['summary']['effective_dim']} "
         rf"profile directions carry $99\%$ of the information "
         rf"(range {rank['summary']['dirs_for_99pct_range'][0]}--"
         rf"{rank['summary']['dirs_for_99pct_range'][1]}); random profiles give "
         rf"{rank['summary']['null_dirs_for_99pct_median']}",
         r"No, and the deficiency is structural rather than a property of the trained head."),
        (r"Does it determine the crystal term?",
         r"Fisher information of the four-parameter fit; Cram\'er--Rao bound on the enthalpy of "
         r"fusion, against the same bound with external labels supplied",
         rf"condition number ${cond0 / 10 ** int(np.log10(cond0)):.1f}\times"
         rf"10^{{{int(np.log10(cond0))}}}$; bound $\pm{crlb0:.0f}$\,kJ\,mol$^{{-1}}$, "
         rf"falling to $\pm{crlb1:.1f}$ with the labels",
         r"No. The crystal enthalpy is collinear with the activity's temperature slope."),
        (r"Does the head use the freedom?",
         r"residual of learned minus reference profile by charge-density window, "
         rf"{resid['n_molecules_clean']} molecules, against each window's share of surface area",
         rf"donor window ${en['donor']:.1f}\times$, acceptor ${en['acceptor']:.1f}\times$, "
         rf"nonpolar core ${en['nonpolar']:.2f}\times$; {zshare:.0f}\% of reference cells are "
         rf"exactly zero",
         r"Yes, and into the windows the hydrogen-bonding term reads."),
        (r"Is the output layer responsible?",
         r"mass in bins where the reference is exactly zero, against the smallest bin the same "
         r"head reaches anywhere and against an untrained head",
         rf"head reaches $10^{{{int(np.floor(np.log10(mn)))}}}$; "
         rf"{100 * min(below):.0f}--{100 * max(below):.0f}\% of those bins already below "
         rf"$10^{{-5}}$; pedestal {min(ped):.2f}--{max(ped):.2f} against {ped0:.2f} untrained",
         r"No. The pedestal is learned, and it is the same in both roles."),
        (r"Is the reference state preserved?",
         r"activity of a molecule in itself, against the one-profile control that must return "
         r"zero identically",
         rf"control ${ctrl:.0f}$ exactly; median $|\ln\gamma|$ {min(med):.3f}--{max(med):.3f}, "
         rf"largest {mx:.2f}",
         r"No. A molecule's two role profiles do not coincide."),
        (r"What does misplaced mass cost, and where?",
         r"mean $|\Delta\ln\gamma|$ from moving a fixed mass into each bin, against random "
         r"profiles of the same area",
         rf"edge-to-core ratio ${cc['train']:.0f}\times$ and ${cc['idac']:.0f}\times$ on the two "
         rf"row sets, against ${ccn:.1f}\times$ for random profiles",
         r"Strongly non-uniform, with a step at the bonding threshold."),
        (r"Does the intermediate decompose?",
         r"reference shape substituted into part of the grid, against a randomly chosen part of "
         r"the same size",
         rf"the whole shape helps on {full_helps} of 5 seeds; a random third is worse than the "
         rf"fully learned profile in {worse:.0f}\% of {n_draw}$\times${n_seed} draws",
         r"No. The parts do not add, so per-bin attribution is undefined."),
        (r"Where does the error that reaches the closure live?",
         r"reference shape in the bins where it is nonzero, against exact zeros in the bins "
         r"where it is zero",
         rf"${fixnz:+.0f}\%$ of the gap recovered against ${zerop:+.0f}\%$",
         r"In the magnitude of the occupied bins, not in the pedestal."),
        (r"What does supplying the reference profile do?",
         rf"the same substitution scored on both quantities, against the ${pl_n}$ rows in which "
         r"no profile is replaced because neither component is tabulated",
         rf"activity coefficients ${g_lrn.mean():.2f}\pm{g_lrn.std(ddof=1):.2f}\to{g_ref:.2f}$; "
         rf"solubility worsens by ${x_pen:.2f}\pm{x_sd:.2f}$ over ${x_n}$ rows; placebo "
         rf"$<{pl_max:.0e}$",
         r"It improves the quantity the profile produces and degrades the quantity the "
         r"pipeline predicts."),
    ]

    body = "\n".join(
        f"{q} & {i} & {r} & {v} \\\\\n\\addlinespace[2pt]\n" for q, i, r, v in rows)
    tex = (
        "% СГЕНЕРИРОВАНО scripts/analysis/make_battery_table_tex.py -- НЕ ПРАВИТЬ РУКАМИ.\n"
        "% Каждое число читается из депозита по ключу; правка здесь разойдётся с артефактом молча.\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\caption{The diagnostic battery. Each test is read against the null named beside it; a "
        "result without its null is not a result. Numbers are medians over five training seeds "
        "unless a range is given.}\n"
        "\\label{tab:battery}\n"
        "\\begin{tabular}{@{}p{0.19\\linewidth}p{0.27\\linewidth}p{0.27\\linewidth}"
        "p{0.21\\linewidth}@{}}\n\\toprule\n"
        "Question & Instrument, and the null it is read against & Result & Reading \\\\\n"
        "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex, encoding="utf8")
    print(f"записано: {OUT.relative_to(ROOT)}  ({len(rows)} строк)")
    for q, _, r, _ in rows:
        print(f"  {q[:52]:54s} {r[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
