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
    mask[0] = False                      # хотя бы одна строка вне маски
    return ps, ts, pa, ta, mask


@pytest.mark.parametrize("mode", ["emd", "mse"])
def test_unit_weights_are_bitwise_identical_to_no_weights(mode):
    ps, ts, pa, ta, mask = _stimulus()
    base = sigma_profile_emd_loss(ps, ts, pa, ta, mask, mode=mode)
    ones = sigma_profile_emd_loss(ps, ts, pa, ta, mask, mode=mode,
                                  shape_bin_weights=torch.ones(ps.shape[-1]))
    assert base.item() == ones.item(), (
        f"mode={mode}: вес из единиц сдвинул член с {base.item()!r} на {ones.item()!r}; "
        "умножение на 1.0 в IEEE754 точное, значит изменился порядок операций"
    )


@pytest.mark.parametrize("mode", ["emd", "mse"])
def test_none_weights_reproduce_the_pre_hook_expression(mode):
    """Невзвешенная ветвь обязана совпасть с выражением, написанным до появления крюка."""
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
    """Независимая реализация оптимального переноса, а не пересказ той же формулы.

    Если расстояние между соседними узлами сетки равно w_k, то узлы стоят в x_0 = 0,
    x_{k+1} = x_k + w_k, и 1-D Вассерштейн между двумя дискретными мерами на этих узлах обязан
    совпасть с sum_k w_k |cumsum(p)_k - cumsum(q)_k|. Это и есть обоснование формы взвешенного
    члена, поэтому оно проверяется сторонним кодом.
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
    """Два расхождения с РАВНЫМ невзвешенным членом взвешенный обязан разделить по цене.

    Иначе крюк не делает того, ради чего существует, и арм T1 читался бы как контроль.
    """
    n = 51
    p = np.full(n, 1.0 / n)
    cheap, costly = 24, 41              # середина ядра и переход в водородносвязанный угол
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
        "стимулы подобраны так, чтобы невзвешенный член их НЕ различал; если различает, "
        "тест потерял смысл"
    )
    w = np.load(VECTOR)
    assert loss(qb, w) > loss(qa, w), (
        f"взвешенный член не развёл равные расхождения: дешёвое {loss(qa, w):.6f}, "
        f"дорогое {loss(qb, w):.6f}"
    )


def test_deposited_vector_is_a_usable_metric():
    w = np.load(VECTOR)
    assert w.shape == (51,)
    assert np.all(np.isfinite(w)) and float(w.min()) > 0.0, (
        "вес обязан быть конечным и строго положительным, иначе член перестаёт быть метрикой"
    )
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-9), (
        "нормировка на mean(w) = 1 -- это контролируемая переменная арма: суммарная величина "
        "супервизии держится фиксированной, меняется только её распределение по сетке"
    )
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    assert w == pytest.approx(np.asarray(dep["by_rows"]["train"]["w_seed_mean"], float)), (
        "weights_t1.npy обязан быть вектором с TRAIN-строк: вывод веса на поверхности, на "
        "которой арм будут читать, сделал бы чтение круговым"
    )


def test_deposited_weight_peaks_at_the_transition_not_deep_in_the_corner():
    """Контринтуитивное чтение, которое будет переврано, если не привязать."""
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    g = np.asarray(dep["sigma_grid"], float)
    b = dep["by_rows"]["train"]
    w = np.asarray(b["w_seed_mean"], float)
    R = np.asarray(b["R_seed_mean"], float)
    k = int(w.argmax())
    assert g[k] == pytest.approx(0.016, abs=1e-9), (
        f"максимум веса ожидается на внутренней кромке угла (sigma = +0.016), получено "
        f"sigma = {g[k]:+.4f}"
    )
    deep = g >= 0.021
    assert w[deep].mean() < 0.5 * w[k], (
        f"вес глубоко в углу {w[deep].mean():.3f} должен быть много меньше пикового {w[k]:.3f}: "
        "потенциал там насыщается, дорого ВОЙТИ в водородносвязанную область, а не уйти глубже"
    )
    assert R[deep].mean() > R[k], "сам потенциал при этом в углу выше -- это не противоречие"


def test_sign_asymmetry_of_the_weight_flips_between_row_sets_at_every_seed():
    """Расхождение поверхностей, а не шум: диапазоны по сидам не пересекаются."""
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
        f"диапазоны обязаны быть непересекающимися: train {min(ratios['train']):.2f}.."
        f"{max(ratios['train']):.2f}, idac {min(ratios['idac']):.2f}..{max(ratios['idac']):.2f}"
    )
    assert min(ratios["train"]) > 1.0 > max(ratios["idac"]), (
        "и лежать по разные стороны единицы: иначе это разная величина, а не разный ЗНАК "
        "асимметрии"
    )


def test_weight_rank_is_seed_stable_even_though_its_contrast_is_not():
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    for which in ("train", "idac"):
        b = dep["by_rows"][which]
        pairs = b["spearman_between_seeds"]
        assert len(pairs) == 10 and min(pairs) > 0.5, (
            f"{which}: ранговое согласие между сидами {min(pairs):+.3f} -- вектор сидозависим, "
            "брать его среднее по сидам нельзя"
        )
        cs = np.asarray(b["corner_over_core_per_seed"], float)
        assert cs.max() / cs.min() > 1.4, (
            f"{which}: контраст потенциала по сидам стабилен ({cs.max() / cs.min():.2f}x) -- "
            "тогда его тоже можно было бы привязать, и этот тест надо переписать"
        )


def test_weight_vector_is_not_the_grid_geometry():
    """Нулевое плечо: та же карта на случайных профилях той же площади."""
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))
    for which in ("train", "idac"):
        b = dep["by_rows"][which]
        null_cc = float(b["null"]["corner_over_core"])
        real = np.asarray(b["corner_over_core_per_seed"], float)
        assert real.min() > 5.0 * null_cc, (
            f"{which}: худший сид {real.min():.1f}x против нуля {null_cc:.2f}x -- карта про "
            "геометрию сетки, а не про химию корпуса"
        )
        assert abs(float(b["spearman_weights_vs_null"])) < 0.5


# --------------------------------------------------------------------------------------------
# Ворота перед покупкой T1. Привязаны потому, что оба читаются против арма, а негатив в этом
# проекте получает тот же аудит, что позитив: у обоих есть нулевое плечо и посидовый повтор, и
# оба переведены в единицы притязания до сравнения с чем-либо.
# --------------------------------------------------------------------------------------------

GATE1 = ROOT / "results/per_bin_sensitivity_weights/weight_points_at_error.json"
GATE2 = ROOT / "results/per_bin_sensitivity_weights/closure_error_attribution.json"


def test_gate1_weight_points_against_the_profile_error_on_every_seed():
    g = json.loads(GATE1.read_text(encoding="utf8"))
    assert g["n_shuffle"] >= 1000
    for which in ("train", "idac"):
        b = g["by_weight_vector"][which]
        assert b["n_expensive_bins"] == 17            # ровно треть сетки из 51 бина
        shares = [x["share_in_expensive_third"] for x in b["per_seed"]]
        rhos = [x["spearman_w_vs_error"] for x in b["per_seed"]]
        assert len(shares) == 5
        assert max(shares) < b["null_share_mean"], (
            f"{which}: доля ошибки в дорогой трети {max(shares):.3f} обязана быть НИЖЕ "
            f"перемешанного нуля {b['null_share_mean']:.3f} на всех пяти сидах"
        )
        assert max(rhos) < -0.4, (
            f"{which}: Спирмен(вес, ошибка) = {max(rhos):+.3f}; знак и порядок -- это и есть "
            "содержание ворот 1"
        )


def test_gate2_expensive_third_does_not_beat_a_random_third():
    g = json.loads(GATE2.read_text(encoding="utf8"))
    ps = g["per_seed"]
    assert len(ps) == 5 and g["n_null"] >= 20
    for s, r in ps.items():
        assert abs(r["base_aad"] - r["deposited_aad"]) < 5e-3, (
            f"сид {s}: невозмущённое плечо не воспроизводит депонированный AAD, читать "
            "подстановки нельзя"
        )
    exp = np.array([r["recovered_дорогая"] for r in ps.values()])
    null = np.array([np.mean(r["null_recovered"]) for r in ps.values()])
    assert exp.mean() < null.mean(), (
        f"дорогая треть возвращает {100 * exp.mean():.1f}% разрыва против {100 * null.mean():.1f}% "
        "у случайной трети -- если это перевернулось, премиса T1 изменилась и её надо перечитать"
    )
    assert exp.min() < 0.0 < exp.max(), (
        f"знак эффекта дорогой трети обязан меняться по сидам ({100 * exp.min():+.1f}.."
        f"{100 * exp.max():+.1f}%): именно это делает атрибуцию неустановленной, а не опровергнутой"
    )
    assert min(r["p_expensive_vs_null"] for r in ps.values()) >= 0.1


def test_gate2_measures_non_additivity_whole_shape_helps_a_third_often_hurts():
    """Побочный результат, который крупнее самих ворот, и потому обязан быть привязан."""
    g = json.loads(GATE2.read_text(encoding="utf8"))
    ps = g["per_seed"]
    base = np.array([r["base_aad"] for r in ps.values()])
    full = np.array([r["full_shape_substituted_aad"] for r in ps.values()])
    assert np.all(base > full), (
        "подстановка ВСЕЙ формы обязана помогать на всех пяти сидах: именно на контрасте с этим "
        "читается, что подстановка трети часто вредит"
    )
    worse = np.array([np.mean(np.asarray(r["null_aad"]) > r["base_aad"]) for r in ps.values()])
    assert 0.05 <= worse.mean() <= 0.45 and worse.max() >= 0.2, (
        f"доля розыгрышей случайной трети, которые ХУЖЕ полностью выученного профиля: "
        f"{worse.round(2)} -- это и есть измерение неаддитивности отображения профиль -> ln gamma"
    )
    # глобальное разложение разрыва: форма против площади. Поставлено корректно, в отличие от
    # по-бинового, потому что обе части подставляются целиком.
    shape_part = float((base - full).mean())
    gap = float(base.mean() - 0.7649746944274517)
    assert 0.6 < shape_part / gap < 0.75, (
        f"форма отвечает за {100 * shape_part / gap:.0f}% разрыва до эталонного профиля; "
        "площадь -- за остальное"
    )
