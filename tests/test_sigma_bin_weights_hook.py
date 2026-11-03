r"""Bind the per-bin weight hook on the sigma-supervision shape term, and its measured vector.

WHY THIS EXISTS
---------------
T1 reweights the sigma-supervision shape term by the closure's own cost per bin. Three things in
that sentence can silently go wrong, and all three have precedents in this repo:

  * **The hook could change the unweighted path.** A treated arm is only a contrast of one
    difference if the control arm's functional is untouched. Multiplying by exactly 1.0 is exact in
    IEEE754 and the summation order is unchanged, so ``w = None`` and ``w = ones`` must give the
    BITWISE same number as before the hook existed -- not "the same to six decimals".

  * **The weight could land on the wrong surface.** In ``mode="emd"`` the term at index k prices the
    mass CROSSING the boundary between bins k and k+1; in ``mode="mse"`` it prices the mass RESIDING
    in bin k. The deposited vector is derived as the discrete gradient of a measured residence
    potential precisely so that the weighted EMD is a genuine 1-D Wasserstein under the metric where
    adjacent grid points sit at distance ``w_k``. That identity is the whole justification for the
    form, so it is checked against an INDEPENDENT optimal-transport implementation
    (``scipy.stats.wasserstein_distance``), not against a restatement of the same formula.

  * **A wrong-length vector could broadcast.** A length-1 or length-n_rows weight would pass
    silently through ``w * per_bin`` and train a different functional. It must raise.

WHAT THE DEPOSIT-BINDING TESTS PIN, AND WHY THOSE NUMBERS
---------------------------------------------------------
``results/per_bin_sensitivity_weights/summary.json``. Two readings there are counter-intuitive
enough that they will be mis-restated if they are not bound:

  * the weight peaks at the TRANSITION into the hydrogen-bonding region, not deep inside it -- the
    residence potential saturates in the corner, so it is expensive to ENTER the HB region and
    comparatively cheap to go deeper;
  * the vector's sign asymmetry about sigma = 0 is OPPOSITE on the two row sets and it is a property
    of the row set, not of the seed (train 2.26..2.97 across five seeds, idac 0.55..0.75, ranges
    disjoint). That caveat must travel with any use of the vector, because it predicts that a
    train-derived weight is NOT the weight the IDAC surface would ask for.

WHAT IS NOT CHECKED
-------------------
The corner/core ratio of the potential itself (20.6..29.6 on train, 13.3..24.8 on idac) is NOT bound
as an interval here: it is seed-unstable by 1.4x and 1.9x respectively, and the quantity T1 consumes
is the mean-1-normalised SHAPE of the vector, whose rank is stable (rho 0.75..0.95). Binding a
ratio that the arm never reads would be theatre of the kind this suite exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tgnn_solv.loss import sigma_profile_emd_loss

ROOT = Path(__file__).resolve().parents[1]
DEPOSIT = ROOT / "results/per_bin_sensitivity_weights/summary.json"
VECTOR = ROOT / "results/per_bin_sensitivity_weights/weights_t1.npy"


def _stimulus(n_rows: int = 7, n_bins: int = 51, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ps = torch.rand(n_rows, n_bins, generator=g)
    ps = ps / ps.sum(dim=-1, keepdim=True)
    ts = torch.rand(n_rows, n_bins, generator=g)
    ts = ts / ts.sum(dim=-1, keepdim=True)
    pa = torch.rand(n_rows, generator=g) * 50 + 50
    ta = torch.rand(n_rows, generator=g) * 50 + 50
    mask = torch.ones(n_rows, dtype=torch.bool)
    mask[0] = False                      # at least one row outside the mask
    return ps, ts, pa, ta, mask


@pytest.mark.parametrize("mode", ["emd", "mse"])
def test_unit_weights_are_bitwise_identical_to_no_weights(mode):
    ps, ts, pa, ta, mask = _stimulus()
    base = sigma_profile_emd_loss(ps, ts, pa, ta, mask, mode=mode)
    ones = sigma_profile_emd_loss(ps, ts, pa, ta, mask, mode=mode,
                                  shape_bin_weights=torch.ones(ps.shape[-1]))
    assert base.item() == ones.item(), (
        f"mode={mode}: a weight of ones moved the term from {base.item()!r} to {ones.item()!r}; "
        "multiplication by 1.0 is exact in IEEE754, so the order of operations changed"
    )


@pytest.mark.parametrize("mode", ["emd", "mse"])
def test_none_weights_reproduce_the_pre_hook_expression(mode):
    """The unweighted branch must agree with the expression written before the hook existed."""
    ps, ts, pa, ta, mask = _stimulus()
    m = mask.bool()
    if mode == "mse":
        want = ((ps[m] - ts[m]) ** 2).sum(dim=-1).mean()
    else:
        want = (torch.cumsum(ps[m], dim=-1)
                - torch.cumsum(ts[m], dim=-1)).abs().sum(dim=-1).mean()
    area = (((pa[m] - ta[m]) / 75.0) ** 2).mean()
    got = sigma_profile_emd_loss(ps, ts, pa, ta, mask, mode=mode)
    assert got.item() == (want + area).item()


@pytest.mark.parametrize("bad", [
    torch.ones(1), torch.ones(50), torch.ones(52), torch.ones(7, 51), torch.ones(51, 1),
])
def test_wrong_shaped_weights_raise_instead_of_broadcasting(bad):
    ps, ts, pa, ta, mask = _stimulus()
    with pytest.raises(ValueError, match="shape_bin_weights"):
        sigma_profile_emd_loss(ps, ts, pa, ta, mask, shape_bin_weights=bad)


def test_weighted_emd_is_wasserstein_on_the_stretched_grid():
    """An independent optimal-transport implementation, not a restatement of the same formula.

    If adjacent grid points sit at distance w_k, the points are at x_0 = 0 and
    x_{k+1} = x_k + w_k, and the one-dimensional Wasserstein distance between two discrete
    measures on those points must equal sum_k w_k |cumsum(p)_k - cumsum(q)_k|. That identity is
    the justification for the shape of the weighted term, so it is checked by separate code.
    """
    wd = pytest.importorskip("scipy.stats").wasserstein_distance
    rng = np.random.default_rng(7)
    n = 51
    for _ in range(12):
        w = rng.uniform(0.2, 4.0, size=n)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        x = np.concatenate([[0.0], np.cumsum(w[:-1])])
        want = float(wd(x, x, p, q))
        got = sigma_profile_emd_loss(
            torch.tensor(p, dtype=torch.double).unsqueeze(0),
            torch.tensor(q, dtype=torch.double).unsqueeze(0),
            torch.zeros(1, dtype=torch.double), torch.zeros(1, dtype=torch.double),
            torch.ones(1, dtype=torch.bool),
            shape_bin_weights=torch.tensor(w, dtype=torch.double),
        ).item()
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_weight_reorders_two_discrepancies_that_tie_unweighted():
    """Two departures with an EQUAL unweighted term must be separated by cost when weighted.

    Otherwise the hook does not do what it exists for, and the treated arm reads as a control.
    """
    n = 51
    p = np.full(n, 1.0 / n)
    cheap, costly = 24, 41              # mid-core, and the transition into the hydrogen-bonded corner
    eps = 0.01
    qa, qb = p.copy(), p.copy()
    qa[cheap] += eps
    qa[cheap + 1] -= eps
    qb[costly] += eps
    qb[costly + 1] -= eps

    def loss(q, w=None):
        return sigma_profile_emd_loss(
            torch.tensor(p, dtype=torch.double).unsqueeze(0),
            torch.tensor(q, dtype=torch.double).unsqueeze(0),
            torch.zeros(1, dtype=torch.double), torch.zeros(1, dtype=torch.double),
            torch.ones(1, dtype=torch.bool),
            shape_bin_weights=None if w is None else torch.tensor(w, dtype=torch.double),
        ).item()

    assert loss(qa) == pytest.approx(loss(qb), rel=1e-12), (
        "the stimuli are built so the unweighted term does NOT separate them; if it does, the "
        "test has lost its meaning"
    )
    w = np.load(VECTOR)
    assert loss(qb, w) > loss(qa, w), (
        f"the weighted term did not separate equal departures: cheap {loss(qa, w):.6f}, "
        f"expensive {loss(qb, w):.6f}"
    )


def test_deposited_vector_is_a_usable_metric():
    w = np.load(VECTOR)
    assert w.shape == (51,)
    assert np.all(np.isfinite(w)) and float(w.min()) > 0.0, (
        "the weight must be finite and strictly positive, or the term stops being a metric"
    )
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-9), (
        "normalising to mean(w) = 1 is the arm's controlled variable: the total amount of "
        "supervision is held fixed and only its distribution over the grid changes"
    )
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    assert w == pytest.approx(np.asarray(dep["by_rows"]["train"]["w_seed_mean"], float)), (
        "weights_t1.npy must be the vector derived on the TRAINING rows: deriving the weight on "
        "the surface the arm will be read on would make the reading circular"
    )


def test_deposited_weight_peaks_at_the_transition_not_deep_in_the_corner():
    """A counter-intuitive reading, which will be misquoted unless it is pinned."""
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    g = np.asarray(dep["sigma_grid"], float)
    b = dep["by_rows"]["train"]
    w = np.asarray(b["w_seed_mean"], float)
    R = np.asarray(b["R_seed_mean"], float)
    k = int(w.argmax())
    assert g[k] == pytest.approx(0.016, abs=1e-9), (
        f"the weight is expected to peak at the inner edge of the corner (sigma = +0.016), got "
        f"sigma = {g[k]:+.4f}"
    )
    deep = g >= 0.021
    assert w[deep].mean() < 0.5 * w[k], (
        f"the weight deep in the corner {w[deep].mean():.3f} must be far below the peak {w[k]:.3f}: "
        "the potential saturates there, so it is expensive to ENTER the hydrogen-bonded region "
        "rather than to go deeper into it"
    )
    assert R[deep].mean() > R[k], "the potential itself is higher in the corner, which is not a contradiction"


def test_sign_asymmetry_of_the_weight_flips_between_row_sets_at_every_seed():
    """A difference of surfaces rather than noise: the per-seed ranges do not overlap."""
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    g = np.asarray(dep["sigma_grid"], float)
    neg, pos = g < 0, g > 0
    ratios = {}
    for which in ("train", "idac"):
        per_seed = dep["by_rows"][which]["per_seed"]
        assert len(per_seed) == 5
        ratios[which] = [float(np.asarray(v["w"], float)[pos].mean()
                               / np.asarray(v["w"], float)[neg].mean())
                         for v in per_seed.values()]
    assert min(ratios["train"]) > max(ratios["idac"]), (
        f"the ranges must not overlap: train {min(ratios['train']):.2f}.."
        f"{max(ratios['train']):.2f}, idac {min(ratios['idac']):.2f}..{max(ratios['idac']):.2f}"
    )
    assert min(ratios["train"]) > 1.0 > max(ratios["idac"]), (
        "and must fall on opposite sides of one, or this is a difference in magnitude rather than "
        "in the SIGN of the asymmetry"
    )


def test_weight_rank_is_seed_stable_even_though_its_contrast_is_not():
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    for which in ("train", "idac"):
        b = dep["by_rows"][which]
        pairs = b["spearman_between_seeds"]
        assert len(pairs) == 10 and min(pairs) > 0.5, (
            f"{which}: rank agreement between seeds is {min(pairs):+.3f}; the vector is seed-dependent "
            "and its mean over seeds cannot be taken"
        )
        cs = np.asarray(b["corner_over_core_per_seed"], float)
        assert cs.max() / cs.min() > 1.4, (
            f"{which}: the potential contrast is stable across seeds ({cs.max() / cs.min():.2f}x), so "
            "it could be pinned as well and this test should be rewritten"
        )


def test_weight_vector_is_not_the_grid_geometry():
    """The null arm: the same map computed on random profiles of the same area."""
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    for which in ("train", "idac"):
        b = dep["by_rows"][which]
        null_cc = float(b["null"]["corner_over_core"])
        real = np.asarray(b["corner_over_core_per_seed"], float)
        assert real.min() > 5.0 * null_cc, (
            f"{which}: the worst seed gives {real.min():.1f}x against a null of {null_cc:.2f}x, so the "
            "map is about the geometry of the grid rather than the chemistry of the corpus"
        )
        assert abs(float(b["spearman_weights_vs_null"])) < 0.5


# --------------------------------------------------------------------------------------------
# The pre-declared reading of the attribution gate, pinned because a negative result gets the
# same audit as a positive one here: a null arm, a per-seed replicate, and the effect converted
# into the units of the claim before it is compared with anything.
# --------------------------------------------------------------------------------------------

GATE2 = ROOT / "results/per_bin_sensitivity_weights/closure_error_attribution.json"


def test_gate2_expensive_third_does_not_beat_a_random_third():
    g = json.loads(GATE2.read_text(encoding="utf8"))
    ps = g["per_seed"]
    assert len(ps) == 5 and g["n_null"] >= 20
    for s, r in ps.items():
        assert abs(r["base_aad"] - r["deposited_aad"]) < 5e-3, (
            f"seed {s}: the unperturbed arm does not reproduce the deposited AAD, so the "
            "substitutions cannot be read"
        )
    exp = np.array([r["recovered_expensive"] for r in ps.values()])
    null = np.array([np.mean(r["null_recovered"]) for r in ps.values()])
    assert exp.mean() < null.mean(), (
        f"the expensive third recovers {100 * exp.mean():.1f}% of the gap against "
        f"{100 * null.mean():.1f}% for a random third; if this has reversed, the premise has "
        "changed and must be re-examined"
    )
    assert exp.min() < 0.0 < exp.max(), (
        f"the sign of the expensive third's effect must change across seeds "
        f"({100 * exp.min():+.1f}..{100 * exp.max():+.1f}%): that is what makes the attribution "
        "unestablished rather than refuted"
    )
    assert min(r["p_expensive_vs_null"] for r in ps.values()) >= 0.1


def test_gate2_measures_non_additivity_whole_shape_helps_a_third_often_hurts():
    """A by-product larger than the gate itself, and therefore pinned."""
    g = json.loads(GATE2.read_text(encoding="utf8"))
    ps = g["per_seed"]
    base = np.array([r["base_aad"] for r in ps.values()])
    full = np.array([r["full_shape_substituted_aad"] for r in ps.values()])
    assert np.all(base > full), (
        "substituting the WHOLE shape must help on all five seeds: it is against that contrast "
        "that substituting a third is read as often harmful"
    )
    worse = np.array([np.mean(np.asarray(r["null_aad"]) > r["base_aad"]) for r in ps.values()])
    assert 0.05 <= worse.mean() <= 0.45 and worse.max() >= 0.2, (
        f"share of random-third draws that are WORSE than the fully learned profile: "
        f"{worse.round(2)}; this is the measurement of non-additivity in the map from profile "
        "to ln gamma"
    )
    # A global split of the gap into shape and area. This one is well posed, unlike a per-bin
    # split, because both parts are substituted whole.
    shape_part = float((base - full).mean())
    gap = float(base.mean() - 0.7649746944274517)
    assert 0.6 < shape_part / gap < 0.75, (
        f"shape accounts for {100 * shape_part / gap:.0f}% of the gap to the reference profile; "
        "area accounts for the rest"
    )
