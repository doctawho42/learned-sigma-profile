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
    """Премиса связки 3, которая как раз ВЕРНА: нулей в эталоне действительно много."""
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
                f"сид {s} {role}: разница между минимальной массой бина и медианой в "
                f"эталонно-нулевых бинах всего "
                f"{st['orders_between_min_and_zero_bin_median']:.1f} порядка -- если это упало "
                "ниже двух, софтмакс связывает и связку 3 надо перечитать"
            )
            assert st["min_bin_mass_anywhere"] < 1e-7, (
                f"сид {s} {role}: голова нигде не доходит ниже "
                f"{st['min_bin_mass_anywhere']:.2e}; тогда у неё действительно пол"
            )
            assert st["share_of_zero_bins_below_1e_5"] > 0.3


def test_pedestal_is_learned_not_architectural():
    """Нулевое плечо: необученная голова той же архитектуры."""
    d = json.loads(FLOOR.read_text(encoding="utf8"))
    null = d["untrained_null"]["pedestal_median"]
    assert null > 0.4, (
        f"необученный пьедестал {null:.3f} -- если он низкий, то различать выученное от "
        "архитектурного этим плечом нельзя"
    )
    trained = [r[role]["pedestal_median"] for r in d["per_seed"].values()
               for role in ("solute", "solvent")]
    assert max(trained) < 0.1 * null, (
        f"обучение обязано двигать пьедестал на порядок: худший обученный {max(trained):.4f} "
        f"против необученного {null:.4f}"
    )


def test_zeroing_the_pedestal_buys_nothing_and_flips_sign():
    d = json.loads(COSTS.read_text(encoding="utf8"))
    ps = d["per_seed"]
    assert len(ps) == 5 and d["n_null"] >= 20
    rec = np.array([r["recovered_zero_pedestal"] for r in ps.values()])
    assert rec.min() < 0.0 < rec.max(), (
        f"знак обязан меняться по сидам ({100 * rec.min():+.1f}..{100 * rec.max():+.1f}%): "
        "именно это делает величину неизмеримой, а не малой"
    )
    assert rec.mean() < 0.2, (
        f"снятие пьедестала возвращает {100 * rec.mean():+.1f}% разрыва; если это выросло, "
        "ворота 2 изменились и нуль-способную голову надо пересмотреть"
    )


def test_the_positive_closure_error_lives_in_the_reference_nonzero_bins():
    """Тот самый позитив, который заменяет оба плеча T2."""
    d = json.loads(COSTS.read_text(encoding="utf8"))
    ps = d["per_seed"]
    rec = np.array([r["recovered_fix_nonzero"] for r in ps.values()])
    assert np.all(rec > 0.6), (
        f"починка эталонно-НЕнулевых бинов обязана возвращать основную часть разрыва на всех "
        f"пяти сидах, получено {np.round(100 * rec, 1)}%"
    )
    assert rec.mean() > 0.8
    # нулевое плечо: обнуление того же ЧИСЛА случайных клеток
    nulls = np.array([np.mean(r["null_recovered"]) for r in ps.values()])
    assert nulls.max() < -1.0, (
        f"нуль обязан быть катастрофическим ({np.round(100 * nulls, 0)}%): иначе 'место "
        "пьедестала правильное' не измерено"
    )
    # неаддитивность печатается, а не замалчивается
    na = np.array([r["non_additivity"] for r in ps.values()])
    assert np.abs(na).max() > 0.1, (
        "если разложение стало аддитивным, предыдущее измерение неаддитивности опровергнуто "
        "и его надо перечитать"
    )


def test_role_invariance_would_make_link_4_a_tautology():
    """Один профиль с обеих сторон даёт ln gamma = 0 РОВНО -- значит плечо не измеряет связку."""
    d = json.loads(REFSTATE.read_text(encoding="utf8"))
    assert d["gate1_control_is_zero"] is True
    for s, r in d["per_seed"].items():
        assert r["control_max_abs"] == 0.0, (
            f"сид {s}: контроль с одним профилем даёт {r['control_max_abs']!r}, не нуль -- "
            "тогда роль-инвариантность НЕ тождество и плечо снова имеет структурное чтение"
        )
