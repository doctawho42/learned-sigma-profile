r"""Bind the two free gates that refute link 3 of frame J, and the positive that replaces them.

WHY THIS EXISTS
---------------
These gates are NEGATIVES, and a negative in this project gets the same audit as a positive: a null
arm, a seed replicate on both arms, and the effect converted into the units of the claim before it
is compared with anything. All three are present here, and this test pins them so the refutation
cannot quietly drift back into the paper as a live claim.

Link 3 said the architecture FORCES part of the σ-head's freedom, because softmax cannot emit exact
zeros while 58.2% of reference cells are exactly zero. The measurement says softmax does not bind:
the same head reaches 1e-11 somewhere on the corpus while leaving 1e-5 in reference-zero bins, a
gap of 4.3 to 6.3 orders on every seed and both roles, and 42-47% of reference-zero cells are
ALREADY below 1e-5. The pedestal is a learned quantity, not an architectural one -- the untrained
head of the same architecture sits at 0.571 against the trained 0.02.

The second gate asks what the pedestal COSTS, because a refuted premise does not make it harmless.
Zeroing it recovers -9.1% of the IDAC gap with the sign flipping across seeds; fixing the
reference-NONZERO bins recovers +89.0%, positive 5/5. That +89% is the positive finding, and it is
what the test suite should defend: the closure-relevant error lives in the MAGNITUDE of the profile
where segment density actually is.

WHAT IS NOT CHECKED
-------------------
The pedestal's cost in AAD units (+0.0772 mean) is NOT bound as a value. Its per-seed spread is
0.52 against the reference arm's seed floor of 0.0632 -- eight times the floor -- so it is not
measurable at the available seed count even in sign, and binding a number that wide would assert
precision the measurement does not have. Only its sign instability is bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "results/t2_softmax_floor/summary.json"
COSTS = ROOT / "results/t2_softmax_floor/pedestal_costs.json"
REFSTATE = ROOT / "results/reference_state_violation/summary.json"


def test_reference_profile_is_mostly_exact_zeros():
    """The part of the premise that is TRUE: the reference really does carry many zeros."""
    d = json.loads(FLOOR.read_text(encoding="utf8"))
    assert d["share_reference_zero_cells"] > 0.5
    assert d["median_zero_bins_per_molecule"] >= 25
    assert d["n_molecules"] >= 300


def test_softmax_does_not_bind_on_any_seed_or_role():
    d = json.loads(FLOOR.read_text(encoding="utf8"))
    ps = d["per_seed"]
    assert len(ps) == 5
    for s, roles in ps.items():
        for role, st in roles.items():
            assert st["orders_between_min_and_zero_bin_median"] > 2.0, (
                f"seed {s} {role}: the gap between the smallest bin mass and the median in "
                f"reference-zero bins is only "
                f"{st['orders_between_min_and_zero_bin_median']:.1f} orders; below two the "
                "softmax would be binding and this reading must be re-examined"
            )
            assert st["min_bin_mass_anywhere"] < 1e-7, (
                f"seed {s} {role}: the head never goes below "
                f"{st['min_bin_mass_anywhere']:.2e}, so it does have a floor after all"
            )
            assert st["share_of_zero_bins_below_1e_5"] > 0.3


def test_pedestal_is_learned_not_architectural():
    """The null arm: an untrained head of the same architecture."""
    d = json.loads(FLOOR.read_text(encoding="utf8"))
    null = d["untrained_null"]["pedestal_median"]
    assert null > 0.4, (
        f"the untrained pedestal is {null:.3f}; if it were low, this arm could not separate what is "
        "learned from what is architectural"
    )
    trained = [r[role]["pedestal_median"] for r in d["per_seed"].values()
               for role in ("solute", "solvent")]
    assert max(trained) < 0.1 * null, (
        f"training must move the pedestal by an order of magnitude: worst trained {max(trained):.4f} "
        f"against untrained {null:.4f}"
    )


def test_zeroing_the_pedestal_buys_nothing_and_flips_sign():
    d = json.loads(COSTS.read_text(encoding="utf8"))
    ps = d["per_seed"]
    assert len(ps) == 5 and d["n_null"] >= 20
    rec = np.array([r["recovered_zero_pedestal"] for r in ps.values()])
    assert rec.min() < 0.0 < rec.max(), (
        f"the sign must change across seeds ({100 * rec.min():+.1f}..{100 * rec.max():+.1f}%): that "
        "is what makes the quantity unmeasured rather than small"
    )
    assert rec.mean() < 0.2, (
        f"removing the pedestal recovers {100 * rec.mean():+.1f}% of the gap; if that has grown, the "
        "gate has changed and a zero-capable head should be reconsidered"
    )


def test_the_positive_closure_error_lives_in_the_reference_nonzero_bins():
    """The positive result that replaces both planned arms."""
    d = json.loads(COSTS.read_text(encoding="utf8"))
    ps = d["per_seed"]
    rec = np.array([r["recovered_fix_nonzero"] for r in ps.values()])
    assert np.all(rec > 0.6), (
        f"repairing the reference-NONZERO bins must recover most of the gap on all five seeds, "
        f"got {np.round(100 * rec, 1)}%"
    )
    assert rec.mean() > 0.8
    # the null arm: zeroing the same NUMBER of randomly chosen cells
    nulls = np.array([np.mean(r["null_recovered"]) for r in ps.values()])
    assert nulls.max() < -1.0, (
        f"the null must be catastrophic ({np.round(100 * nulls, 0)}%), or the claim that the "
        "pedestal sits in the right place is not measured"
    )
    # the non-additivity is printed rather than passed over
    na = np.array([r["non_additivity"] for r in ps.values()])
    assert np.abs(na).max() > 0.1, (
        "if the decomposition has become additive, the earlier measurement of non-additivity is "
        "refuted and must be re-examined"
    )


def test_role_invariance_would_make_link_4_a_tautology():
    """One profile on both sides gives ln gamma = 0 EXACTLY, so the arm measures nothing here."""
    d = json.loads(REFSTATE.read_text(encoding="utf8"))
    assert d["gate1_control_is_zero"] is True
    for s, r in d["per_seed"].items():
        assert r["control_max_abs"] == 0.0, (
            f"seed {s}: the one-profile control gives {r['control_max_abs']!r} rather than zero, so "
            "role invariance is NOT an identity and the arm has a structural reading again"
        )
