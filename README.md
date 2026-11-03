# Reproduction package

Code, inputs and deposited results for the manuscript *When a Learned σ-Profile Is Not One:
Diagnosing the Intermediate of a Physics-Embedded Solubility Model*.

The system is a graph neural network that supplies the σ-profile consumed by a differentiable
COSMO-SAC implementation, which predicts the solubility of molecular solids through a
solid-liquid equilibrium closure. The manuscript asks whether the values the network places in
the slot of a physical quantity are that quantity. Every figure, table and number it reports is
produced by a script in this repository from a file in it.

This repository is not the development tree. It holds what the manuscript uses and nothing else,
and three checks in `tools/` make that statement testable rather than editorial.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python tools/fetch_artifacts.py            # trained checkpoints and the data split
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q
```

The test suite reads only the deposited results under `results/`, so it runs before the
checkpoints are fetched. Rebuilding a figure or a deposit needs the checkpoints.

```bash
# the table and the three diagnostic figures, from deposited results alone
python scripts/analysis/make_battery_table_tex.py
python scripts/analysis/make_intermediate_diagnostic_figures.py

# a deposit recomputed from the checkpoints
python scripts/analysis/read_t1_closure_error_attribution.py
```

[MANIFEST.md](MANIFEST.md) maps each item in the manuscript to the script that produces it and the
file that script reads. Start there to reproduce one number rather than all of them.

The LaTeX sources of the article and its Supporting Information are added when the article is
submitted. One test and one check read them and skip with their reason printed until then;
[paper/README.md](paper/README.md) says which, and everything else runs.

## Layout

```
src/tgnn_solv/        the model: encoder, COSMO-SAC layer, SLE solver, losses, trainer
scripts/train.py      the training entry point
scripts/data/         provenance of the reference σ-profile table and the supervision stream
scripts/analysis/     one script per reported quantity, plus the figure builders
scripts/experiments/  the driver that produced the runs of record
configs/              the configurations those runs used
results/              the deposited numerical results the manuscript cites
paper/                the manuscript sources, so the reported numbers can be located in the text
tests/                assertions that pin the reported quantities
tools/                the three checks described below
```

`notebooks/data/` and `checkpoints/` are created by `tools/fetch_artifacts.py`. The directory name
`notebooks/data/` is historical and is the path the configurations and scripts carry.

## What is not here

Three raw sources are third party and are named rather than redistributed: the BigSolDB solubility
compilation, the Bradley melting-point table, and the VT-2005 σ-profile database. The reference
σ-profiles resampled onto the model grid are deposited as
`results/sigma_profile_artifact/sigma_profiles.csv`, and `scripts/data/ingest_vt2005_sigma_profiles.py`
documents how that table was built from the database.

The trained checkpoints and the data split are too large for version control and are archived with
this package instead. `tools/fetch_artifacts.py` downloads and verifies them.

## The four checks

```bash
python tools/check_clean.py                               # documentation, not a work diary
python tools/check_inputs_present.py                      # complete, and no larger than needed
python tools/check_si_contract.py                         # the SI holds what the article promises
python tools/check_code_unchanged.py --source /path/to/development/tree
```

`check_clean.py` defines in checkable terms what it means for the source to read as documentation:
no non-English prose, no dated notes recording when something happened to the authors, no
first-person account of the work, no tool names, no machine-specific paths. The rules describe the
form of the text, not its content, so a comment explaining why a threshold has its value passes
while a comment recording that the value was once wrong does not.

`check_inputs_present.py` reads every artefact path the code refers to and requires each one to be
present here, to arrive from the archive, to be created at run time by a named script, or to be a
configuration default this manuscript does not exercise. It then requires every deposited file to
be read by some script here. The first half is completeness, the second is minimality, and a
declaration that stops matching the code is reported as stale.

`check_si_contract.py` reads the article's Supporting Information section as a contract, one
promise per semicolon, and requires the Supporting Information to have exactly those sections in
exactly that order. A section with no promise and a promise with no section both fail, so neither
document can drift away from the other.

`check_code_unchanged.py` compares this repository's Python against the development tree and
requires that nothing differ in executable structure except where a reason is recorded in the
script. It needs the development tree, so it is for the authors and for anyone holding both. Its
guarantee has one boundary worth stating: string constants are normalised away, and a dictionary
key is a string constant, so a renamed key passes this check. Three keys were renamed when this
package was prepared; the protection for that is the test suite, which reads the renamed keys.

## Two properties of the pipeline a reader needs

**The scaffold split is not stable across versions of the data-preparation code.** Regenerating it
reshuffles which scaffolds land in the test set, which orphans every checkpoint and every number
measured on the previous split. The split is therefore distributed as data. Treat any regeneration
as a different split and recompute everything downstream of it.

**The COSMO-SAC segment-activity fixed point is iterated a fixed number of times, and that count is
a free parameter between fitting and scoring.** Each checkpoint manifest records the count it was
trained at. A checkpoint scored at a different count is evaluated through an operator it was never
fit against, so read the count from the manifest before comparing anything.

## Citation

Polomoshnov, N. L.; Rudik, A. V. *Reproduction package for "When a Learned σ-Profile Is Not One:
Diagnosing the Intermediate of a Physics-Embedded Solubility Model"*, Zenodo,
[10.5281/zenodo.23907894](https://doi.org/10.5281/zenodo.23907894).

The concept DOI [10.5281/zenodo.23907893](https://doi.org/10.5281/zenodo.23907893) always resolves
to the most recent version. Cite the version DOI above when reproducing a specific number.

## License

MIT, see [LICENSE](LICENSE). The deposited results and the manuscript sources are released under
the same terms. The three third-party sources named above carry their own.
