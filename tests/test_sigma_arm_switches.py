r"""Bind the two arm switches, each of which answers one standing objection to the pipeline.

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
    """The same batch and the same weights; only the switch changes."""
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
        f"{switch}=False moved the output: the arm stops being a contrast of a single difference"
    )


def test_detach_cuts_the_solubility_gradient_but_not_the_supervision():
    model, out = _forward({"sigma_detached_from_closure": True})
    head = model.head_sigma
    model.zero_grad(set_to_none=True)
    out["ln_x2"].sum().backward(retain_graph=True)
    reached = [p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in head.parameters()]
    assert not any(reached), (
        "the gradient of the solubility loss reached the sigma head although the path is detached"
    )
    # the external supervision must keep training it, or the arm merely breaks the head
    model.zero_grad(set_to_none=True)
    out["nrtl_params"]["sigma_shape_solute"].sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in head.parameters()), (
        "the sigma supervision does not reach it either: the detach cut both pulls, not one"
    )


def test_detach_off_does_let_the_solubility_gradient_through():
    """Control for the previous test: without the switch the gradient MUST arrive."""
    model, out = _forward({"sigma_detached_from_closure": False})
    model.zero_grad(set_to_none=True)
    out["ln_x2"].sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.head_sigma.parameters()), (
        "the gradient does not arrive even without the detach, so the first test checks nothing"
    )


def test_role_invariance_gives_a_molecule_one_profile():
    """The same molecule on both sides must receive ONE profile."""
    torch.manual_seed(0)
    trainer, loader = make_tiny_cosmo_trainer_and_loader()
    batch = first_batch(loader)
    model = trainer.model
    model.cfg.sigma_role_invariant = True
    model.eval()
    sol, _slv, targets = trainer._move_batch_to_device(batch)
    with torch.no_grad():
        out = model(sol, sol, targets["T"], targets=targets)   # the molecule paired with itself
    a = out["nrtl_params"]["sigma_shape_solute"]
    b = out["nrtl_params"]["sigma_shape_solvent"]
    assert torch.allclose(a, b, atol=1e-6), (
        f"the two role profiles differ by {float((a - b).abs().max()):.2e}; "
        "role invariance is not achieved"
    )


def test_without_role_invariance_the_two_profiles_differ():
    """Control: otherwise the previous test would pass with a broken switch too."""
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
        "the two roles give one profile even without the switch, so the encoder does not read the "
        "role at all and the reference-state section must be re-examined"
    )
