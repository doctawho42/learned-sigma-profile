# What produces what

Each item in the manuscript, the script that produces it, and the file that script reads. To
reproduce one number rather than all of them, find its row and run the script named there.

A script marked **needs checkpoints** loads the five trained models and takes minutes to hours;
run `tools/fetch_artifacts.py` first. Everything else reads a deposited file under `results/` and
runs in seconds.

## Table 1, the diagnostic battery

`scripts/analysis/make_battery_table_tex.py` writes `paper/sections/tab-battery.tex`. Every cell is
read from a deposit by key, so the table cannot drift away from the artefacts. The nine rows and
their sources:

| Row | Deposit it reads | Script that produced that deposit |
|---|---|---|
| Does the observable determine the σ-profile? | `results/closure_rank_profile_space/summary.json` | `scripts/analysis/run_closure_rank_profile_space.py` **needs checkpoints** |
| Does it determine the crystal term? | `results/identifiability_audit/summary.json` | `scripts/analysis/run_identifiability_fisher_audit.py` |
| Does the head use the freedom? | `results/sigma_profile_residual/clean_pool.json` | `scripts/analysis/read_sigma_residual_clean_pool.py`, over `scripts/analysis/run_sigma_profile_residual.py` **needs checkpoints** |
| Is the output layer responsible? | `results/t2_softmax_floor/summary.json` | `scripts/analysis/read_t2_softmax_floor_is_binding.py` **needs checkpoints** |
| Is the reference state preserved? | `results/reference_state_violation/summary.json` | `scripts/analysis/run_reference_state_violation.py` **needs checkpoints** |
| What does misplaced mass cost, and where? | `results/per_bin_sensitivity_weights/summary.json` | `scripts/analysis/run_per_bin_sensitivity_weights.py` **needs checkpoints** |
| Does the intermediate decompose? | `results/t2_softmax_floor/pedestal_costs.json` | `scripts/analysis/read_t2_pedestal_costs.py` **needs checkpoints** |
| Where does the error that reaches the closure live? | `results/per_bin_sensitivity_weights/closure_error_attribution.json` | `scripts/analysis/read_t1_closure_error_attribution.py` **needs checkpoints** |
| What does supplying the reference profile do? | `results/idac_learned_vs_reference/summary.json` and `results/e5_sigma_grounding/section_s1_channel_split.json` | `scripts/analysis/run_idac_learned_vs_reference.py` **needs checkpoints**, and `scripts/analysis/run_paradox_channel_split.py` |

## Figures

| Figure | Script | Reads |
|---|---|---|
| `fig_pipeline.pdf` | `scripts/analysis/make_pipeline_figure.py` | nothing; it is a schematic of the forward path |
| `fig_donor_window.pdf` | `scripts/analysis/make_donor_window_figure.py` | `results/closure_ladder/placebo_profile_diagnosis.csv`, `results/donor_window_rescore/per_solvent.csv` |
| `fig_pedestal_learned.pdf` | `scripts/analysis/make_intermediate_diagnostic_figures.py` | `results/t2_softmax_floor/summary.json`, `.../pedestal_costs.json`, `.../bin_mass_distributions.npz` |
| `fig_cost_map.pdf` | same | `results/per_bin_sensitivity_weights/summary.json` |
| `fig_attribution.pdf` | same | `results/per_bin_sensitivity_weights/closure_error_attribution.json` |
| `fig_toc_v2.pdf` | `scripts/analysis/make_toc_figure_v2.py` | `results/sigma_profile_artifact/sigma_profiles.csv` **needs checkpoints** |
| `fig_pka_hammett.pdf` (Supporting Information) | `scripts/analysis/run_pka_hammett_probe.py` | nothing; the series is semi-synthetic and the script writes its own `results/pka_hammett/probe.json` |

`scripts/analysis/check_figure_spec.py` checks the finished PDFs against the journal's width and
font limits.

## Every deposited file

Seventeen files, each read by at least one script above, which `tools/check_inputs_present.py`
verifies.

| File | What it holds | Where it comes from |
|---|---|---|
| `results/sigma_profile_artifact/sigma_profiles.csv` | the reference σ-profiles of 1432 molecules, resampled onto the model grid, with cavity areas | `scripts/data/ingest_vt2005_sigma_profiles.py` over the VT-2005 database, which is third party and not redistributed here |
| `results/published_idac_check/scored_records.csv` | infinite-dilution activity coefficients scored through the closure, per record | `scripts/analysis/run_published_idac_closure_check.py`, whose `--fetch` downloads the PGL6ed input files from the PGLWrapper repository |
| `results/per_bin_sensitivity_weights/summary.json` | the cost of moving a fixed mass into each of the 51 bins, per seed and per row set, with the random-profile null | `scripts/analysis/run_per_bin_sensitivity_weights.py` |
| `results/per_bin_sensitivity_weights/weights_t1.npy` | the sensitivity weight vector over the grid, on the training rows | same |
| `results/per_bin_sensitivity_weights/closure_error_attribution.json` | the share of the closure gap recovered by substituting the reference shape in the nonzero bins, against zeros in the zero bins and against a random third | `scripts/analysis/read_t1_closure_error_attribution.py` |
| `results/t2_softmax_floor/summary.json` | the profile mass in bins where the reference is exactly zero, for five seeds in both roles and for an untrained head | `scripts/analysis/read_t2_softmax_floor_is_binding.py` |
| `results/t2_softmax_floor/bin_mass_distributions.npz` | the per-bin mass distributions behind that summary | same |
| `results/t2_softmax_floor/pedestal_costs.json` | what zeroing the pedestal, and substituting parts of the grid, does to the activity coefficients | `scripts/analysis/read_t2_pedestal_costs.py` |
| `results/sigma_profile_residual/clean_pool.json` | the learned-minus-reference residual by charge-density window on the 71-molecule clean pool, against each window's share of surface area | `scripts/analysis/read_sigma_residual_clean_pool.py` over `scripts/analysis/run_sigma_profile_residual.py` |
| `results/closure_rank_profile_space/summary.json` | the spectrum of the closure's Jacobian over the 51 bins, with the equal-area random-profile null | `scripts/analysis/run_closure_rank_profile_space.py` |
| `results/identifiability_audit/summary.json` | Fisher information and the Cramér–Rao bound on the enthalpy of fusion, with and without external labels | `scripts/analysis/run_identifiability_fisher_audit.py` |
| `results/reference_state_violation/summary.json` | the activity of a molecule in itself, against the one-profile control that returns zero identically | `scripts/analysis/run_reference_state_violation.py` |
| `results/idac_learned_vs_reference/summary.json` | activity coefficients from the learned and the substituted profile on the same rows | `scripts/analysis/run_idac_learned_vs_reference.py` |
| `results/e5_sigma_grounding/section_s1_channel_split.json` | the solubility effect of substitution, split by which component was replaced, per seed | `scripts/analysis/run_paradox_channel_split.py` over the runs of record |
| `results/donor_window_rescore/per_solvent.csv` | the donor-window rescoring per solvent | `scripts/analysis/run_donor_window_rescore.py` |
| `results/pka_hammett/probe.json` | the semi-synthetic Hammett sweep behind the Supporting Information figure: the error of the fixed relation given the true constant, and the cost of supplying that constant, at each setting of the two knobs | `scripts/analysis/run_pka_hammett_probe.py`, deterministic from `--seed 0` with no external input |
| `results/closure_ladder/placebo_profile_diagnosis.csv` | donor-window and total cavity area per solvent, in the reference tabulation and in the learned profile, as medians over that solvent's rows | a deposit step of an earlier closure-damage study. That study's own claim, that the severity of closure damage orders the substitution penalty, did not hold and is not made in this manuscript. This file carries none of it: it is a direct read of the two profile arrays, window by window. |

## The runs of record

`scripts/experiments/run_e5_sigma_grounding.sh` is the driver that trained the five checkpoints,
over `configs/cosmo_sac.yaml`, seeds 42 to 46, arm `grounded_a`. Each checkpoint arrives with a
manifest recording the commit, the configuration, the seed and the environment; see
[environment.md](environment.md).

## Tests

`tests/` reads deposits only, so the suite runs without the checkpoints.

| Test | What it pins |
|---|---|
| `test_manuscript_numbers_bind.py` | every number in the article text, and the Supporting Information's sweep figures, against the deposit each is read from. Skips until the manuscript sources are added; see `paper/README.md` |
| `test_t2_gates_bind.py` | the pre-declared reading of the pedestal and substitution measurements |
| `test_sigma_bin_weights_hook.py` | the per-bin weighting of the σ-profile loss, and the attribution gate |
| `test_sigma_arm_switches.py` | that the two configuration switches change what they claim to change, and nothing when off |
| `test_package_is_self_contained.py` | that every module-level import resolves and that this file names every deposited file |
