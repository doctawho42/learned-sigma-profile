# Internal Package Layout

The package now exposes two parallel internal layouts:

- legacy flat modules at `tgnn_solv.<module>`
- grouped namespaces for human-friendly navigation:
  - `tgnn_solv.core`
  - `tgnn_solv.chemistry`
  - `tgnn_solv.data`
  - `tgnn_solv.models`
  - `tgnn_solv.physics`
  - `tgnn_solv.training`
  - `tgnn_solv.evaluation`
  - `tgnn_solv.baselines`
  - `tgnn_solv.research`
  - `tgnn_solv.applications`

## Why Both Exist

The flat modules remain the compatibility layer for:

- existing imports across the repo
- tests
- scripts and notebooks
- downstream users already importing `tgnn_solv.model`, `tgnn_solv.trainer`,
  and similar legacy paths

The grouped namespaces are the preferred navigation surface for contributors.
They re-export the legacy modules without changing behavior.

## Suggested Import Style

Preferred grouped imports for new code:

```python
from tgnn_solv.core.config import TGNNSolvConfig
from tgnn_solv.models.tgnn import TGNNSolv
from tgnn_solv.training.trainer import TGNNSolvTrainer
from tgnn_solv.evaluation.inference import load_model
```

Legacy imports remain supported:

```python
from tgnn_solv.config import TGNNSolvConfig
from tgnn_solv.model import TGNNSolv
from tgnn_solv.trainer import TGNNSolvTrainer
from tgnn_solv.inference import load_model
```

## Namespace Map

### `tgnn_solv.core`

- configuration and runtime helpers

### `tgnn_solv.chemistry`

- molecular featurization and fixed chemical priors

### `tgnn_solv.models`

- TGNN-Solv model, DirectGNN, heads, layers, and optional GPS / TIMP encoder paths

### `tgnn_solv.physics`

- solver and thermodynamic layers

### `tgnn_solv.training`

- trainer, losses, Stage 0 pretraining, pretrain-pipeline helpers, and tuning

### `tgnn_solv.evaluation`

- inference, evaluator/reporting, plotting, and analysis helpers
- uncertainty calibration and applicability-domain helpers consumed by the
  interactive `Experiment Lab`
- benchmark adapters, artifact sidecars, and stress-suite helpers live nearby
  and feed the same `Experiment Lab` and benchmark-bundle surfaces

### `tgnn_solv.research`

- ablation helpers that remain more experimental than the canonical paths

### `tgnn_solv.applications`

- solvent screening, process optimization, drug developability, and
  PK-relevant solubility workflows

## Additional Maintained Modules

Several maintained surfaces still live at the flat package level because they
cut across namespaces:

- `tgnn_solv.artifacts`
  - run manifests, benchmark cards, and checkpoint model cards
- `tgnn_solv.benchmark_adapters`
  - formal adapter API for arbitrary custom models
- `tgnn_solv.reproduction`
  - shared `core / article / full` reproduction planning
- `tgnn_solv.stress`
  - thermodynamic stress-suite slices for benchmark bundles
- `tgnn_solv.applications`
  - solvent screening, process optimization, drug developability, and
    PK-relevant solubility workflows

These are intended as infrastructure layers rather than end-user model APIs,
which is why they are documented here and in the script/docs guides rather than
on their own package subpages.
