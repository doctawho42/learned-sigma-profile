"""
Solid-Liquid Equilibrium (SLE) solver.

Solves:  x₂ · γ₂(x₂, T) = exp(-Φ(T))

where Φ is the ideal (crystal) contribution and γ₂ comes from NRTL.
The solver is agnostic to where ``T_m``, ``dH_fus``, and ``dCp_fus`` come
from; the crystal branch can therefore swap from learned ``dCp_fus`` or a
fixed constant to per-sample GC priors without changing the math here.

Two modes:
  - Successive substitution (always used at eval)
  - Implicit differentiation (optional at training for stable gradients)
"""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .config import TGNNSolvConfig
from .layers import CosmoSacLayer, HansenDistanceLayer, IdealSolubilityLayer, NRTLLayer

TensorDict: TypeAlias = dict[str, Tensor]
AutogradGradients: TypeAlias = tuple[Tensor | None, ...]

# Bound the SLE exponent BEFORE exp() in the unrolled fixed-point iterations.
# These loops are fully back-propagated (the gamma_inf and cosmo_sac paths do not
# go through the no-grad implicit-diff SLESolverFunction), so a large positive
# exponent -Φ-lnγ₂ (e.g. a strongly mismatched COSMO-SAC residual reaching
# magnitudes of hundreds) overflows float32 exp() to +inf. Clamping the *value*
# after exp() masks the forward but leaves inf in the graph, poisoning the whole
# batch's backward with NaN gradients (the FASTSOLV-class failure). max=0.0 keeps
# x₂ ≤ 1; the finite min keeps gradients alive instead of an inf/NaN.
_SLE_EXP_ARG_MIN = -50.0
_SLE_EXP_ARG_MAX = 0.0


def _iterate_fixed_point(
    Phi: Tensor,
    tau_12: Tensor,
    tau_21: Tensor,
    G_12: Tensor,
    G_21: Tensor,
    *,
    n_iter: int,
    damping: float,
    min_damping: float,
    tol: float,
    adaptive_damping: bool,
    nrtl_layer: NRTLLayer,
) -> tuple[Tensor, Tensor]:
    """Solve the SLE fixed point with residual-based stopping."""
    x2 = torch.exp((-Phi).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)).clamp(
        1e-10, 1.0 - 1e-10
    )
    lng2 = torch.zeros_like(x2)
    damping_tensor = torch.full_like(x2, damping)
    prev_residual = torch.full_like(x2, float("inf"))

    for _ in range(n_iter):
        x1 = 1.0 - x2
        lng2 = nrtl_layer.ln_gamma_2(x1, x2, tau_12, tau_21, G_12, G_21)
        x2_candidate = torch.exp(
            (-Phi - lng2).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)
        ).clamp(1e-10, 1.0 - 1e-10)
        residual = (torch.log(x2_candidate) - torch.log(x2)).abs()

        if adaptive_damping:
            damping_tensor = torch.where(
                residual > prev_residual,
                torch.clamp(damping_tensor * 0.5, min=min_damping),
                torch.clamp(damping_tensor * 1.05, max=1.0),
            )

        x2 = damping_tensor * x2_candidate + (1.0 - damping_tensor) * x2
        prev_residual = residual

        if residual.max().item() < tol:
            break

    x1 = 1.0 - x2
    lng2 = nrtl_layer.ln_gamma_2(x1, x2, tau_12, tau_21, G_12, G_21)
    return x2, lng2


def _effective_lngamma_inf_from_params(
    nrtl_params: TensorDict,
    T: Tensor,
    *,
    T_ref: float,
    clamp: float,
) -> Tensor | None:
    """Return direct ln(gamma_inf)(T) when the simplified activity mode is used."""
    if "ln_gamma_inf" in nrtl_params:
        return nrtl_params["ln_gamma_inf"].to(T).clamp(-clamp, clamp)
    if "ln_gamma_inf_ref" in nrtl_params:
        inv_ratio = (T_ref / T) - 1.0
        return (
            nrtl_params["ln_gamma_inf_ref"].to(T)
            + nrtl_params["ln_gamma_inf_inv"].to(T) * inv_ratio
        ).clamp(-clamp, clamp)
    return None


def _iterate_gamma_inf_fixed_point(
    Phi: Tensor,
    lngamma_inf: Tensor,
    *,
    n_iter: int,
    damping: float,
    min_damping: float,
    tol: float,
    adaptive_damping: bool,
) -> tuple[Tensor, Tensor]:
    """Solve SLE with ln(gamma_2)=ln(gamma_inf)*(1-x_2)^2."""
    x2 = torch.exp((-Phi).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)).clamp(
        1e-10, 1.0 - 1e-10
    )
    lng2 = torch.zeros_like(x2)
    damping_tensor = torch.full_like(x2, damping)
    prev_residual = torch.full_like(x2, float("inf"))

    for _ in range(n_iter):
        x1 = 1.0 - x2
        lng2 = lngamma_inf * x1.pow(2)
        x2_candidate = torch.exp(
            (-Phi - lng2).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)
        ).clamp(1e-10, 1.0 - 1e-10)
        residual = (torch.log(x2_candidate) - torch.log(x2)).abs()

        if adaptive_damping:
            damping_tensor = torch.where(
                residual > prev_residual,
                torch.clamp(damping_tensor * 0.5, min=min_damping),
                torch.clamp(damping_tensor * 1.05, max=1.0),
            )

        x2 = damping_tensor * x2_candidate + (1.0 - damping_tensor) * x2
        prev_residual = residual

        if residual.max().item() < tol:
            break

    lng2 = lngamma_inf * (1.0 - x2).pow(2)
    return x2, lng2


def _iterate_cosmo_sac_fixed_point(
    Phi: Tensor,
    cosmo_layer: CosmoSacLayer,
    p2: Tensor,
    p1: Tensor,
    A2: Tensor,
    A1: Tensor,
    V2: Tensor | None,
    V1: Tensor | None,
    T: Tensor,
    *,
    n_iter: int,
    damping: float,
    min_damping: float,
    tol: float,
    adaptive_damping: bool,
    break_on_tol: bool,
) -> tuple[Tensor, Tensor]:
    """Solve the SLE fixed point with a COSMO-SAC activity coefficient.

    ``break_on_tol`` gates the early exit, and ``config.solver_cosmo_break_on_tol`` turns it
    OFF by default FOR THIS PATH ONLY. On CUDA the ``residual.max().item()`` it needs is a
    BLOCKING host synchronisation: the CPU stalls until the queued GPU work drains, n_iter
    times per forward.

    HOW MUCH THAT COSTS, MEASURED RATHER THAN REASONED: 1.03x, and the first version of this
    comment claimed much more. The argument was that the flush destroys the CPU run-ahead
    which hides this path's Python dispatch cost -- but the A/B on a Kaggle T4
    (``scripts/kaggle/make_speedup_notebook.py``, 2026-10-02) put the sync-only arm at 372.9
    against 384.9 ms/step. The reasoning was self-defeating: if the binding constraint IS the
    CPU issuing kernels, then the CPU never gets ahead of the GPU in the first place, so there
    is no run-ahead for a sync to destroy and the flush only drains a short queue. The real
    win in that A/B came from the stacked segment solve (1.35x) and from compiling the segment
    loop (1.75x together); this flag contributes the remaining few percent.

    It is still worth having, because the exit buys nothing to pay even that. ``scripts/analysis/run_solver_break_audit.py``
    traced the batch-max residual on 8103 real learned profile pairs and it never once reached
    the tolerance in 144 non-degenerate batch-cells, finishing 3 to 6 orders of magnitude
    above it (train Phi=2: 4.9e+04x tol; eval Phi=8, the closest call: 2.8e+03x). The only
    cell where it fires is Phi=0 -- T exactly at the melting point, where pure solute is an
    exact fixed point with residual 0.0, so further iterations move nothing. Turning it off is
    therefore bit-identical, not an approximation, and that is pinned by
    ``tests/test_solver_launch_cost_rewrites.py``.

    ``_iterate_fixed_point`` (NRTL) keeps its exit unconditionally, and the audit says why
    rather than assuming: at the shipped counts the NRTL exit does not fire either -- the
    nearest cell, a near-ideal pair at eval, lands 4.8x above tol -- but at the non-default
    n_iter_eval=30 / tol=1e-8 of ``test_physics_verification.py`` it does, at iteration 16 of
    30, and disabling it there moves x2 by 2.3e-09 instead of zero. Negligible as physics, yet
    it is the difference between an audited no-op and an undeclared operator change. A NRTL
    outer iteration carries no segment solve, so its sync is cheap next to the loop, and the
    whole 331 ms/step problem being fixed here lives on the COSMO-SAC side anyway.
    """
    x2 = torch.exp((-Phi).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)).clamp(
        1e-10, 1.0 - 1e-10
    )
    lng2 = torch.zeros_like(x2)
    damping_tensor = torch.full_like(x2, damping)
    prev_residual = torch.full_like(x2, float("inf"))

    for _ in range(n_iter):
        x1 = 1.0 - x2
        lng2 = cosmo_layer.ln_gamma_2(x1, x2, p2, p1, A2, A1, V2, V1, T)
        x2_candidate = torch.exp(
            (-Phi - lng2).clamp(_SLE_EXP_ARG_MIN, _SLE_EXP_ARG_MAX)
        ).clamp(1e-10, 1.0 - 1e-10)
        residual = (torch.log(x2_candidate) - torch.log(x2)).abs()

        if adaptive_damping:
            damping_tensor = torch.where(
                residual > prev_residual,
                torch.clamp(damping_tensor * 0.5, min=min_damping),
                torch.clamp(damping_tensor * 1.05, max=1.0),
            )

        x2 = damping_tensor * x2_candidate + (1.0 - damping_tensor) * x2
        prev_residual = residual

        if break_on_tol and residual.max().item() < tol:
            break

    x1 = 1.0 - x2
    lng2 = cosmo_layer.ln_gamma_2(x1, x2, p2, p1, A2, A1, V2, V1, T)
    return x2, lng2


# ================================================================== #
#  Implicit-differentiation autograd Function                         #
# ================================================================== #

class SLESolverFunction(torch.autograd.Function):
    """
    Custom autograd for the fixed-point SLE solution.

    Forward: run successive substitution (no gradient tracking).
    Backward: use the implicit function theorem to compute exact
    gradients through the converged fixed point.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Phi: Tensor,
        a_12: Tensor,
        b_12: Tensor,
        c_12: Tensor,
        a_21: Tensor,
        b_21: Tensor,
        c_21: Tensor,
        alpha_12: Tensor,
        T: Tensor,
        n_iter: int,
        damping: float,
        min_damping: float,
        tol: float,
        adaptive_damping: bool,
        nrtl_layer: NRTLLayer,
    ) -> tuple[Tensor, Tensor]:
        tau_12, tau_21, G_12, G_21 = nrtl_layer.compute_tau_G(
            a_12, b_12, c_12, a_21, b_21, c_21, alpha_12, T
        )

        # Fixed-point iteration (no grad)
        with torch.no_grad():
            x2, lng2_final = _iterate_fixed_point(
                Phi,
                tau_12,
                tau_21,
                G_12,
                G_21,
                n_iter=n_iter,
                damping=damping,
                min_damping=min_damping,
                tol=tol,
                adaptive_damping=adaptive_damping,
                nrtl_layer=nrtl_layer,
            )

        ctx.save_for_backward(
            x2, Phi, a_12, b_12, c_12, a_21, b_21, c_21, alpha_12,
            T, tau_12, tau_21, G_12, G_21, lng2_final,
        )
        ctx.nrtl_layer = nrtl_layer
        return x2, lng2_final

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_x2: Tensor, grad_lng2: Tensor | None
    ) -> AutogradGradients:
        (
            x2, Phi, a_12, b_12, c_12, a_21, b_21, c_21, alpha_12,
            T, tau_12, tau_21, G_12, G_21, lng2,
        ) = ctx.saved_tensors
        nrtl = ctx.nrtl_layer

        # Custom backward runs under no_grad by default.
        # We need enable_grad for internal autograd.grad calls.
        with torch.enable_grad():
            x2_var = x2.detach().requires_grad_(True)
            x1_var = 1.0 - x2_var
            a12_var = a_12.detach().requires_grad_(True)
            b12_var = b_12.detach().requires_grad_(True)
            c12_var = c_12.detach().requires_grad_(True)
            a21_var = a_21.detach().requires_grad_(True)
            b21_var = b_21.detach().requires_grad_(True)
            c21_var = c_21.detach().requires_grad_(True)
            alpha_var = alpha_12.detach().requires_grad_(True)

            T_var = T.detach().requires_grad_(True)

            tau12_v, tau21_v, G12_v, G21_v = nrtl.compute_tau_G(
                a12_var, b12_var, c12_var, a21_var, b21_var, c21_var,
                alpha_var, T_var
            )
            lng2_v = nrtl.ln_gamma_2(
                x1_var, x2_var, tau12_v, tau21_v, G12_v, G21_v
            )

            # d ln(γ₂) / d x₂
            eta = torch.autograd.grad(
                lng2_v.sum(), x2_var,
                retain_graph=True, allow_unused=True,
            )[0]
            if eta is None:
                eta = torch.zeros_like(x2)

            # dF/dx₂ = 1 + x₂ · η
            dF_dx2 = 1.0 + x2 * eta
            sign = torch.sign(dF_dx2)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            dF_dx2 = torch.where(dF_dx2.abs() < 0.01, sign * 0.01, dF_dx2)

            # Implicit multiplier
            implicit_mult = -grad_x2 / dF_dx2

            # ∂F/∂Φ = +x₂
            grad_Phi = implicit_mult * x2

            # Gradients w.r.t. NRTL parameters
            param_vars = [
                a12_var, b12_var, c12_var,
                a21_var, b21_var, c21_var,
                alpha_var,
                T_var,
            ]
            grad_terms = torch.autograd.grad(
                lng2_v,
                param_vars,
                grad_outputs=implicit_mult * x2,
                retain_graph=False,
                allow_unused=True,
            )
            grad_params = grad_terms[:-1]
            grad_T = grad_terms[-1]

        # Direct gradient from ln_gamma_2 loss (if present)
        if grad_lng2 is not None:
            with torch.enable_grad():
                x2_v2 = x2.detach().requires_grad_(True)
                x1_v2 = 1.0 - x2_v2
                a12_v2 = a_12.detach().requires_grad_(True)
                b12_v2 = b_12.detach().requires_grad_(True)
                c12_v2 = c_12.detach().requires_grad_(True)
                a21_v2 = a_21.detach().requires_grad_(True)
                b21_v2 = b_21.detach().requires_grad_(True)
                c21_v2 = c_21.detach().requires_grad_(True)
                alpha_v2 = alpha_12.detach().requires_grad_(True)
                T_v2 = T.detach().requires_grad_(True)

                t12, t21, g12, g21 = nrtl.compute_tau_G(
                    a12_v2, b12_v2, c12_v2, a21_v2, b21_v2, c21_v2,
                    alpha_v2, T_v2
                )
                lng2_v2 = nrtl.ln_gamma_2(x1_v2, x2_v2, t12, t21, g12, g21)

                direct_grads = torch.autograd.grad(
                    lng2_v2,
                    [
                        a12_v2, b12_v2, c12_v2,
                        a21_v2, b21_v2, c21_v2,
                        alpha_v2,
                        T_v2,
                    ],
                    grad_outputs=grad_lng2,
                    allow_unused=True,
                )

            grad_params = tuple(
                (g1 if g1 is not None else 0) + (g2 if g2 is not None else 0)
                for g1, g2 in zip(grad_params, direct_grads[:-1])
            )
            grad_T = (
                (grad_T if grad_T is not None else torch.zeros_like(T))
                + (
                    direct_grads[-1]
                    if direct_grads[-1] is not None
                    else torch.zeros_like(T)
                )
            )
        elif grad_T is None:
            grad_T = torch.zeros_like(T)

        return (
            grad_Phi,
            grad_params[0], grad_params[1], grad_params[2],
            grad_params[3], grad_params[4], grad_params[5],
            grad_params[6],
            grad_T,
            None, None, None, None, None, None,
        )


# ================================================================== #
#  SLE Solver Module                                                  #
# ================================================================== #

class SLESolver(nn.Module):
    """
    Full SLE solver: ideal solubility + NRTL activity coefficient.

    Given predicted crystal params and NRTL params, solves for x₂
    at temperature T via successive substitution.

    Returns a dict with all intermediate quantities for loss and
    interpretability.
    """

    def __init__(self, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.ideal_layer = IdealSolubilityLayer(R=cfg.R)
        self.nrtl_layer = NRTLLayer(
            R=cfg.R, T_ref=cfg.T_ref,
            tau_clamp=cfg.tau_clamp, eps=cfg.eps,
        )
        self.hansen_layer = HansenDistanceLayer()
        self.cosmo_sac_layer = (
            CosmoSacLayer(cfg) if getattr(cfg, "activity_model", "nrtl") == "cosmo_sac"
            else None
        )
        self.cfg = cfg

    def forward(
        self,
        T: Tensor,
        fusion_params: TensorDict,
        nrtl_params: TensorDict,
        use_implicit: bool | None = None,
    ) -> TensorDict:
        """
        Solve SLE for mole-fraction solubility x₂.

        Parameters
        ----------
        T : (B,) temperature [K]
        fusion_params : dict with T_m, dH_fus, dCp_fus
        nrtl_params : dict with a supported NRTL parameter layout
        use_implicit : override config setting

        Returns
        -------
        dict with x2, ln_x2, ln_gamma_2, ln_gamma_inf, Phi, x_ideal,
             tau_12, tau_21, G_12, G_21
        """
        # Ideal contribution. Some decomposition-prone / no-melting systems
        # can supply an effective Van't Hoff Phi(T) directly.
        if "Phi_override" in fusion_params:
            Phi = fusion_params["Phi_override"].to(T)
        else:
            Phi = self.ideal_layer(
                T, fusion_params["T_m"],
                fusion_params["dH_fus"], fusion_params["dCp_fus"],
            )

        n_iter = self.cfg.n_iter_train if self.training else self.cfg.n_iter_eval
        damping = self.cfg.damping
        min_damping = self.cfg.solver_min_damping
        tol = self.cfg.solver_tol_train if self.training else self.cfg.solver_tol_eval
        adaptive_damping = self.cfg.solver_adaptive_damping
        # Only the COSMO-SAC path drops the early exit; see _iterate_cosmo_sac_fixed_point.
        cosmo_break = getattr(self.cfg, "solver_cosmo_break_on_tol", False)
        use_impl = use_implicit if use_implicit is not None else self.cfg.use_implicit_diff

        # COSMO-SAC activity path (sigma-profiles instead of NRTL params).
        if self.cosmo_sac_layer is not None and "p_solute" in nrtl_params:
            p2 = nrtl_params["p_solute"]
            p1 = nrtl_params["p_solvent"]
            A2 = nrtl_params["A_solute"]
            A1 = nrtl_params["A_solvent"]
            V2 = nrtl_params.get("V_solute")
            V1 = nrtl_params.get("V_solvent")
            x2, lng2 = _iterate_cosmo_sac_fixed_point(
                Phi, self.cosmo_sac_layer, p2, p1, A2, A1, V2, V1, T,
                n_iter=n_iter, damping=damping, min_damping=min_damping,
                tol=tol, adaptive_damping=adaptive_damping, break_on_tol=cosmo_break,
            )
            ln_x2 = torch.log(x2 + self.cfg.eps)
            x_ideal = torch.exp(-Phi)
            lng2_inf = self.cosmo_sac_layer.ln_gamma_inf(p2, p1, A2, A1, V2, V1, T)
            zeros = torch.zeros_like(x2)
            ones = torch.ones_like(x2)
            return {
                "x2": x2,
                "ln_x2": ln_x2,
                "ln_gamma_2": lng2,
                "ln_gamma_inf": lng2_inf,
                "Phi": Phi,
                "x_ideal": x_ideal.clamp(0, 1),
                "tau_12": lng2_inf,
                "tau_21": zeros,
                "G_12": ones,
                "G_21": ones,
            }

        lngamma_inf_direct = _effective_lngamma_inf_from_params(
            nrtl_params,
            T,
            T_ref=self.cfg.T_ref,
            clamp=self.cfg.tau_clamp,
        )
        if lngamma_inf_direct is not None:
            x2, lng2 = _iterate_gamma_inf_fixed_point(
                Phi,
                lngamma_inf_direct,
                n_iter=n_iter,
                damping=damping,
                min_damping=min_damping,
                tol=tol,
                adaptive_damping=adaptive_damping,
            )
            ln_x2 = torch.log(x2 + self.cfg.eps)
            x_ideal = torch.exp(-Phi)
            zeros = torch.zeros_like(lngamma_inf_direct)
            ones = torch.ones_like(lngamma_inf_direct)
            return {
                "x2": x2,
                "ln_x2": ln_x2,
                "ln_gamma_2": lng2,
                "ln_gamma_inf": lngamma_inf_direct,
                "Phi": Phi,
                "x_ideal": x_ideal.clamp(0, 1),
                "tau_12": lngamma_inf_direct,
                "tau_21": zeros,
                "G_12": ones,
                "G_21": ones,
            }

        if "tau_12" in nrtl_params and "tau_21" in nrtl_params:
            tau_12, tau_21, G_12, G_21 = self.nrtl_layer.compute_tau_G_from_params(
                nrtl_params,
                T,
            )
            use_impl = False
        else:
            a12, b12, c12, a21, b21, c21, alpha_12 = (
                self.nrtl_layer.params_to_abc(nrtl_params)
            )

        # Solve
        if use_impl and self.training:
            x2, lng2 = SLESolverFunction.apply(
                Phi,
                a12, b12, c12, a21, b21, c21,
                alpha_12,
                T, n_iter, damping, min_damping, tol, adaptive_damping, self.nrtl_layer,
            )
            tau_12, tau_21, G_12, G_21 = self.nrtl_layer.compute_tau_G(
                a12, b12, c12, a21, b21, c21, alpha_12, T
            )
        else:
            if "tau_12" not in nrtl_params or "tau_21" not in nrtl_params:
                tau_12, tau_21, G_12, G_21 = self.nrtl_layer.compute_tau_G(
                    a12, b12, c12, a21, b21, c21, alpha_12, T
                )
            x2, lng2 = _iterate_fixed_point(
                Phi,
                tau_12,
                tau_21,
                G_12,
                G_21,
                n_iter=n_iter,
                damping=damping,
                min_damping=min_damping,
                tol=tol,
                adaptive_damping=adaptive_damping,
                nrtl_layer=self.nrtl_layer,
            )

        ln_x2 = torch.log(x2 + self.cfg.eps)
        x_ideal = torch.exp(-Phi)
        lng2_inf = self.nrtl_layer.ln_gamma_inf(tau_12, tau_21, G_21)

        return {
            "x2": x2,
            "ln_x2": ln_x2,
            "ln_gamma_2": lng2,
            "ln_gamma_inf": lng2_inf,
            "Phi": Phi,
            "x_ideal": x_ideal.clamp(0, 1),
            "tau_12": tau_12,
            "tau_21": tau_21,
            "G_12": G_12,
            "G_21": G_21,
        }
