r"""Bind the two arm switches that answer the reviewer's two open objections.

WHY THESE EXIST
---------------
The manuscript recommends a design it has not measured, and reports one defect that follows from a
choice made in this pipeline. Both are arms, and both need a switch.

``sigma_detached_from_closure`` cuts the PATH rather than the parameters: the profile and the area
enter the closure without history, so no gradient from the solubility loss reaches the
$\sigma$-head, while external $\sigma$-supervision still trains it. That is the construction the
nearest prior work uses, where the profile predictor is supervised against quantum chemistry and
then frozen. It is not the same as ``freeze_sigma_head_during_sle``, which freezes the head's
weights while the encoder beneath it keeps learning from the solubility loss, so the profile moves
anyway. The distinction is the whole point of the arm and is therefore tested, not assumed.

``sigma_role_invariant`` sends both molecules of a pair through the encoder under one role, so a
compound receives one profile whichever side it stands on. Under it the reference state holds
identically, because COSMO-SAC's residual term is a difference that vanishes when one profile is
used on both sides.

WHAT IS CHECKED
---------------
First that each switch is a no-op when off, bitwise, because an arm is only a contrast of one
difference if the control arm's functional is untouched. Then that each switch does what it claims,
by the mechanism it claims: gradient reachability for the first, profile identity for the second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_fixtures import first_batch, make_tiny_cosmo_trainer_and_loader  # noqa: E402


def _forward(cfg_updates: dict):
    """Один и тот же батч и те же веса, меняется только переключатель."""
    torch.manual_seed(0)
    trainer, loader = make_tiny_cosmo_trainer_and_loader()
    batch = first_batch(loader)
    model = trainer.model
    for k, v in cfg_updates.items():
        setattr(model.cfg, k, v)
    model.train()
    sol, slv, targets = trainer._move_batch_to_device(batch)
    out = model(sol, slv, targets["T"], targets=targets)
    return model, out


@pytest.mark.parametrize("switch", ["sigma_detached_from_closure", "sigma_role_invariant"])
def test_switch_off_is_a_bitwise_no_op(switch):
    _, out_b = _forward({})
    _, out_s = _forward({switch: False})
    a, b = out_b["ln_x2"].detach(), out_s["ln_x2"].detach()
    assert torch.equal(a, b), (
        f"{switch}=False сдвинул выход: плечо перестаёт быть контрастом одного отличия"
    )


def test_detach_cuts_the_solubility_gradient_but_not_the_supervision():
    model, out = _forward({"sigma_detached_from_closure": True})
    head = model.head_sigma
    model.zero_grad(set_to_none=True)
    out["ln_x2"].sum().backward(retain_graph=True)
    reached = [p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in head.parameters()]
    assert not any(reached), (
        "градиент потери по растворимости дошёл до σ-головы, хотя путь детачен"
    )
    # внешняя супервизия обязана продолжать её обучать: иначе плечо просто ломает голову
    model.zero_grad(set_to_none=True)
    out["nrtl_params"]["sigma_shape_solute"].sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in head.parameters()), (
        "σ-супервизия тоже не доходит: детач срезал обе тяги вместо одной"
    )


def test_detach_off_does_let_the_solubility_gradient_through():
    """Контроль к предыдущему: без переключателя градиент ОБЯЗАН доходить."""
    model, out = _forward({"sigma_detached_from_closure": False})
    model.zero_grad(set_to_none=True)
    out["ln_x2"].sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.head_sigma.parameters()), (
        "градиент не доходит даже без детача -- тогда первый тест ничего не проверяет"
    )


def test_role_invariance_gives_a_molecule_one_profile():
    """Та же молекула с обеих сторон обязана получить ОДИН профиль."""
    torch.manual_seed(0)
    trainer, loader = make_tiny_cosmo_trainer_and_loader()
    batch = first_batch(loader)
    model = trainer.model
    model.cfg.sigma_role_invariant = True
    model.eval()
    sol, _slv, targets = trainer._move_batch_to_device(batch)
    with torch.no_grad():
        out = model(sol, sol, targets["T"], targets=targets)   # пара молекулы с собой
    a = out["nrtl_params"]["sigma_shape_solute"]
    b = out["nrtl_params"]["sigma_shape_solvent"]
    assert torch.allclose(a, b, atol=1e-6), (
        f"профили двух ролей расходятся на {float((a - b).abs().max()):.2e}; "
        "роль-инвариантность не достигнута"
    )


def test_without_role_invariance_the_two_profiles_differ():
    """Контроль: иначе предыдущий тест прошёл бы и на сломанном переключателе."""
    torch.manual_seed(0)
    trainer, loader = make_tiny_cosmo_trainer_and_loader()
    batch = first_batch(loader)
    model = trainer.model
    model.cfg.sigma_role_invariant = False
    model.eval()
    sol, _slv, targets = trainer._move_batch_to_device(batch)
    with torch.no_grad():
        out = model(sol, sol, targets["T"], targets=targets)
    np_ = out["nrtl_params"]
    gap = float((np_["sigma_shape_solute"] - np_["sigma_shape_solvent"])
                .abs().max())
    assert gap > 1e-6, (
        "две роли дают один профиль даже без переключателя -- значит энкодер роль не читает, "
        "и раздел о референсном состоянии надо перечитать"
    )
