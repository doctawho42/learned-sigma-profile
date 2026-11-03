r"""Bind the manuscript's printed numbers to the deposits, and the table to the prose.

WHY THIS EXISTS
---------------
A co-author read the compiled draft and found that Table 1 and Section 3.6 reported the SAME
measurement as 94% and as 20%. Both numbers were real. The table cell had been generated from
``pedestal_costs.json``, where the control zeroes randomly chosen cells, while the sentence
described ``closure_error_attribution.json``, where the control substitutes reference values into a
randomly chosen third of bins. Two operations, one label, two numbers, and nothing in the build
compared them.

That is the same defect this project has recorded eight times before, and every previous instance
was in running prose where no gate could see it. Generating the table from deposits removed the
hand-typed value but not the possibility of reading the WRONG deposit, so the generator needs a
consumer-side check as well. This file is that check: it re-derives each printed quantity from the
artifact it claims to come from, and it compares the table against the prose wherever both report
the same thing.

WHAT IS CHECKED, AND WHAT IS NOT
--------------------------------
Only headline quantities are bound, meaning the ones a reader would quote. Ranges are checked as
ranges, because a range is the one statistic that cannot be averaged and still mean what it says.
Quantities the manuscript deliberately reports without precision ("well over half", "more than
fourteen times") are not bound, since binding a word to a float would assert a precision the
sentence does not claim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = (PAPER / "sections/results-v2.tex").read_text(encoding="utf8")
TABLE = (PAPER / "sections/tab-battery.tex").read_text(encoding="utf8")


def dep(rel: str) -> dict:
    p = ROOT / rel
    assert p.exists(), f"депозита нет: {rel}"
    return json.loads(p.read_text(encoding="utf8"))


def find(pattern: str, text: str = RESULTS) -> re.Match:
    """Поиск по тексту с любым пробелом между словами -- переносы строк не должны мешать."""
    rx = re.compile(r"\s+".join(pattern.split()))
    m = rx.search(text)
    assert m is not None, f"в тексте не найдено: {pattern[:70]}"
    return m


def test_rank_of_the_closure_in_profile_space():
    d = dep("results/closure_rank_profile_space/summary.json")["summary"]
    m = find(r"the number of directions carrying \$99\\%\$ of the information is "
             r"\$(\d+)\$, with a range of \$?(\d+)\$? to \$?(\d+)\$?")
    assert int(m.group(1)) == d["dirs_for_99pct_median"]
    assert [int(m.group(2)), int(m.group(3))] == d["dirs_for_99pct_range"]
    assert find(r"a profile can move in \$(\d+)\$").group(1) == str(d["effective_dim"])


def test_reference_sparsity_and_window_enrichment():
    f = dep("results/t2_softmax_floor/summary.json")
    c = dep("results/sigma_profile_residual/clean_pool.json")["median_enrichment"]
    share = find(r"\$(\d+)\\%\$ of all reference cells").group(1)
    assert int(share) == round(100 * f["share_reference_zero_cells"])
    assert int(find(r"a median of \$(\d+)\$ bins of \$51\$").group(1)) == \
        f["median_zero_bins_per_molecule"]
    m = find(r"over-represented by a factor of \$([\d.]+)\$ and the acceptor window by "
             r"\$([\d.]+)\$, while the nonpolar core is under-represented at \$([\d.]+)\$")
    for got, key in zip(m.groups(), ("donor", "acceptor", "nonpolar")):
        assert float(got) == pytest.approx(c["solute"][key], abs=0.05), key
    m2 = find(r"as solvent give \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$")
    for got, key in zip(m2.groups(), ("donor", "acceptor", "nonpolar")):
        assert float(got) == pytest.approx(c["solvent"][key], abs=0.05), key


def test_pedestal_numbers_and_the_two_roles():
    f = dep("results/t2_softmax_floor/summary.json")
    per = f["per_seed"]
    both = np.array([s[r]["pedestal_median"] for s in per.values()
                     for r in ("solute", "solvent")])
    m = find(r"after training the figure is \$([\d.]+)\\pm([\d.]+)\$")
    assert float(m.group(1)) == pytest.approx(both.mean(), abs=0.001)
    assert float(m.group(2)) == pytest.approx(both.std(ddof=1), abs=0.001)
    assert float(find(r"places a median of \$([\d.]+)\$").group(1)) == \
        pytest.approx(f["untrained_null"]["pedestal_median"], abs=0.001)
    low = min(s[r]["share_of_zero_bins_below_1e_5"] for s in per.values()
              for r in ("solute", "solvent"))
    high = max(s[r]["share_of_zero_bins_below_1e_5"] for s in per.values()
               for r in ("solute", "solvent"))
    m = find(r"between \$(\d+)\\%\$ and \$(\d+)\\%\$ of those bins already lie below")
    assert int(m.group(1)) == round(100 * low) and int(m.group(2)) == round(100 * high)
    # по ролям: именно это снимает возражение "вы сами сломали объект"
    m = find(r"The pedestal is \$([\d.]+)\\pm([\d.]+)\$ on the profiles a molecule receives as "
             r"solute and \$([\d.]+)\\pm([\d.]+)\$ on those it receives as solvent")
    for off, role in ((0, "solute"), (2, "solvent")):
        v = np.array([s[role]["pedestal_median"] for s in per.values()])
        assert float(m.group(off + 1)) == pytest.approx(v.mean(), abs=0.001), role
        assert float(m.group(off + 2)) == pytest.approx(v.std(ddof=1), abs=0.001), role


def test_reference_state_violation():
    per = dep("results/reference_state_violation/summary.json")["per_seed"]
    med = [v["median_abs_n30"] for v in per.values()]
    m = find(r"between \$([\d.]+)\$ and \$([\d.]+)\$ in \$\\ln\\gamma\$")
    assert float(m.group(1)) == pytest.approx(min(med), abs=0.001)
    assert float(m.group(2)) == pytest.approx(max(med), abs=0.001)
    assert float(find(r"largest single value is \$([\d.]+)\$").group(1)) == \
        pytest.approx(max(v["max_abs_n300"] for v in per.values()), abs=0.01)


def test_cost_map_ratios_against_the_random_profile_control():
    w = dep("results/per_bin_sensitivity_weights/summary.json")["by_rows"]
    m = find(r"costs \$(\d+)\$ times as much as a median over seeds, with a range of \$(\d+)\$ "
             r"to \$(\d+)\$")
    cc = np.array(w["train"]["corner_over_core_per_seed"])
    assert int(m.group(1)) == round(np.median(cc))
    assert (int(m.group(2)), int(m.group(3))) == (round(cc.min()), round(cc.max()))
    m = find(r"the ratio is \$(\d+)\$, with a range of \$(\d+)\$ to \$(\d+)\$")
    ci = np.array(w["idac"]["corner_over_core_per_seed"])
    assert int(m.group(1)) == round(np.median(ci))
    m = find(r"Random profiles of the same total area give \$([\d.]+)\$ and \$([\d.]+)\$")
    assert float(m.group(1)) == pytest.approx(w["train"]["null"]["corner_over_core"], abs=0.05)
    assert float(m.group(2)) == pytest.approx(w["idac"]["null"]["corner_over_core"], abs=0.05)


def test_nonadditivity_prose_and_table_report_the_same_operation():
    """Ровно тот дефект, который нашёл соавтор: таблица и текст про ОДНО измерение."""
    a = dep("results/per_bin_sensitivity_weights/closure_error_attribution.json")
    per = a["per_seed"]
    share = [np.mean(np.asarray(v["null_aad"]) > v["base_aad"]) for v in per.values()]
    prose = int(find(r"profile entirely alone in \$(\d+)\\%\$ of draws").group(1))
    assert prose == round(100 * float(np.mean(share)))
    tab = int(re.search(r"worse than the fully learned profile in (\d+)\\%", TABLE).group(1))
    assert tab == prose, (
        f"таблица печатает {tab}\\%, текст {prose}\\% -- это одно измерение, и расхождение "
        "здесь означает, что одна из сторон читает не тот депозит"
    )
    m = re.search(r"in \d+\\% of (\d+)\$\\times\$(\d+) draws", TABLE)
    assert (int(m.group(1)), int(m.group(2))) == (a["n_null"], len(per)), \
        "план розыгрышей в таблице обязан совпадать с депозитом"
    m = find(r"at each of five seeds and ranging from \$(\d+)\\%\$ to \$(\d+)\\%\$ by seed")
    assert (int(m.group(1)), int(m.group(2))) == (round(100 * min(share)), round(100 * max(share)))


def test_attribution_and_the_shape_area_split():
    c = dep("results/t2_softmax_floor/pedestal_costs.json")["per_seed"]
    fz = np.array([v["recovered_fix_nonzero"] for v in c.values()])
    zp = np.array([v["recovered_zero_pedestal"] for v in c.values()])
    m = find(r"recovers \$(\d+)\\%\$ of the gap")
    assert int(m.group(1)) == round(100 * fz.mean())
    m = find(r"positive on all five seeds and ranging from \$(\d+)\\%\$ to \$(\d+)\\%\$")
    assert (int(m.group(1)), int(m.group(2))) == (round(100 * fz.min()), round(100 * fz.max()))
    assert int(find(r"recovers \$-(\d+)\\%\$ on average").group(1)) == round(-100 * zp.mean())
    base = np.array([v["base_aad"] for v in c.values()])
    full = np.array([v["aad_full"] for v in c.values()])
    ref = dep("results/idac_learned_vs_reference/summary.json")["arms"]["reference_vt2005"][
        "aad_mean"]
    m = find(r"from \$([\d.]+)\$ to \$([\d.]+)\$, against \$([\d.]+)\$ when")
    assert float(m.group(1)) == pytest.approx(base.mean(), abs=0.01)
    assert float(m.group(2)) == pytest.approx(full.mean(), abs=0.01)
    assert float(m.group(3)) == pytest.approx(ref, abs=0.01)
    assert float(find(r"accounts for \$([\d.]+)\$ of the gap").group(1)) == pytest.approx(
        (base.mean() - full.mean()) / (base.mean() - ref), abs=0.01)


def test_the_two_sided_substitution_and_its_placebo():
    idac = dep("results/idac_learned_vs_reference/summary.json")
    g = np.array([v["aad"] for v in idac["arms"]["learned"]["per_seed"].values()])
    m = find(r"from \$([\d.]+)\\pm([\d.]+)\$ to \$([\d.]+)\$")
    assert float(m.group(1)) == pytest.approx(g.mean(), abs=0.002)
    assert float(m.group(2)) == pytest.approx(g.std(ddof=1), abs=0.002)
    assert float(m.group(3)) == pytest.approx(
        idac["arms"]["reference_vt2005"]["aad_mean"], abs=0.002)
    ch = dep("results/e5_sigma_grounding/section_s1_channel_split.json")["summary"]["classes"]
    so, ne = ch["solvent_only"], ch["neither"]
    m = find(r"by \$([\d.]+)\\pm([\d.]+)\$ in mean absolute error over the \$(\d+)\$ test rows")
    assert float(m.group(1)) == pytest.approx(so["mean"], abs=0.002)
    assert float(m.group(2)) == pytest.approx(so["sd_sample"], abs=0.002)
    assert int(m.group(3)) == so["n_per_seed"][0]
    per = [round(v, 3) for v in so["per_seed"].values()]
    m = find(r"per-seed values of \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$")
    assert sorted(float(x) for x in m.groups()) == sorted(per)
    assert int(find(r"for either component in \$(\d+)\$ test rows").group(1)) == \
        ne["n_per_seed"][0]
    assert ne["max_abs_mae_difference"] < 4e-6, "плацебо обязано оставаться плацебо"


def test_no_number_is_printed_without_the_control_beside_it():
    """Каждая строка таблицы обязана называть свой нуль: результат без него не читается."""
    body = TABLE.split(r"\midrule")[1].split(r"\bottomrule")[0]
    rows = [r for r in body.split(r"\\") if r.strip() and "&" in r]
    assert len(rows) == 9, f"строк в таблице {len(rows)}, ожидалось 9"
    for r in rows:
        instrument = r.split("&")[1]
        assert re.search(r"against|control|placebo|random|untrained|no profile", instrument), (
            f"строка без названного нуля: {instrument.strip()[:80]}"
        )
