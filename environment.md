# Environment

## Where the checkpoints were trained

All five checkpoints of record come from one commit on one machine, which the manifests shipped
beside them record:

| Field | Value |
|---|---|
| Seeds | 42, 43, 44, 45, 46, arm `grounded_a` |
| Configuration | `configs/cosmo_sac.yaml` |
| Commit | `621bf71f08d72e99e25f16e5e2826147d97aa8f3`, branch `sigma-grounded-cosmosac` |
| Working tree at training time | modified relative to that commit |
| Python | 3.10.12 |
| Platform | Linux 6.8.0 on x86-64, glibc 2.35, Google Compute Engine |
| Device | CUDA |

The working tree was modified relative to the recorded commit when these runs started, identically
for all five seeds. The commit therefore identifies the code to within those modifications rather
than exactly. The five seeds are mutually comparable, which is what the per-seed readings require,
and a reader rebuilding from this repository should expect agreement to within seed spread rather
than bit equality.

## Where the analysis was run

The deposited results under `results/` were computed on Apple silicon under macOS, Python 3.10.8,
CPU only, with the library versions below. The analysis scripts are deterministic given a
checkpoint and a seed, and the quantities they report are medians or ranges over the five seeds.

| Package | Version |
|---|---|
| torch | 2.6.0 |
| torch-geometric | 2.4.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scipy | 1.10.1 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.9.2 |
| rdkit | 2022.09.5 |
| pyyaml | 6.0.3 |
| pytest | 9.0.3 |

## Hardware is a measurable axis

Retraining the same arm on different GPU architectures does not reproduce a checkpoint to within
seed spread. On this pipeline the difference between two cluster nodes, one with Turing cards and
one with Ampere, reached several times the per-seed standard deviation of the solubility error.
Read any comparison of two checkpoints against the hardware each was trained on, which the
manifests record, before attributing a difference to the thing under study.

## One import that aborts the process

RDKit, PyTorch and scikit-learn each ship their own copy of the OpenMP runtime, and importing more
than one in the same process aborts with `OMP: Error #15`. The test suite sets
`KMP_DUPLICATE_LIB_OK=TRUE` in `tests/conftest.py`, so `pytest` is unaffected. A script run
directly needs the variable in its environment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/analysis/read_t1_closure_error_attribution.py
```

## The segment-activity iteration count

`cosmo_sac_gamma_iter_train` and `cosmo_sac_gamma_iter_eval` are separate numbers, 16 and 30 in the
shipped configuration. The checkpoints of record were trained at 16 and are scored at 30, and that
pair is close enough that the effect on the aggregate error is small: sweeping the scoring count
over 8, 16, 30, 100 and 300 with weights and rows held fixed moves the seed-mean error across an
interval of 0.021 in natural-log mole fraction.

The insensitivity of the aggregate is cancellation, not smallness. Per record at 30 iterations the
shift against a converged solve has a median of 6e-4 but a 95th percentile of 0.14 and a maximum
above 2 natural-log units, with water as the stiff case. A checkpoint trained at a count far from
its scoring count is a different matter: on one trained at 8 and scored at 30, the mismatch alone
accounted for most of a reported penalty. Read both counts from the manifest before comparing two
runs.

## Rebuilding the manuscript

The manuscript is set in STIX Two Text through `fontspec` and builds with XeLaTeX, run twice so
that references and the table of contents settle:

```bash
cd paper && xelatex learned_sigma_profile && bibtex learned_sigma_profile && \
  xelatex learned_sigma_profile && xelatex learned_sigma_profile
```
