"""
Prediction heads for TGNN-Solv.

Each head maps a learned representation to physically meaningful
parameters. Constrained activations keep outputs in valid ranges.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TGNNSolvConfig
from .features import DESCRIPTOR_PRIOR_DIM, GROUP_PRIOR_DIM


def _resolve_prior_mode(cfg: TGNNSolvConfig) -> str:
    """Return which optional molecular prior path is active."""
    if cfg.use_descriptor_priors and cfg.use_group_priors:
        raise ValueError(
            "Descriptor priors and fixed group priors are mutually exclusive."
        )
    if cfg.use_group_priors:
        return "group"
    if cfg.use_descriptor_priors:
        return "descriptor"
    return "none"


# ================================================================== #
#  Pair Representation                                                #
# ================================================================== #

class PairRepresentation(nn.Module):
    """
    Build a pair vector from solute and solvent graph-level vectors.

    Input:  g_sol (B, D_r), g_slv (B, D_r)
    Output: g_pair (B, pair_dim)

    Concatenates: [g_sol, g_slv, g_sol * g_slv, |g_sol - g_slv|]
    """

    def __init__(self, readout_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim * 4, pair_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(pair_dim * 2, pair_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

    def forward(self, g_sol: Tensor, g_slv: Tensor) -> Tensor:
        pair_input = torch.cat(
            [g_sol, g_slv, g_sol * g_slv, (g_sol - g_slv).abs()],
            dim=-1,
        )
        return self.mlp(pair_input)


class MorganFeatureAdapter(nn.Module):
    """Project Morgan fingerprints into the shared molecular representation space."""

    def __init__(
        self,
        n_bits: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, fingerprints: Tensor) -> Tensor:
        """Project fingerprint bits into a dense embedding."""
        return self.net(fingerprints)


class DescriptorPriorAdapter(nn.Module):
    """Map fixed molecular descriptors to a coarse physics prior."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DESCRIPTOR_PRIOR_DIM, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, descriptor_features: Tensor) -> Tensor:
        """Project descriptor features into the target prior space."""
        return self.net(descriptor_features)


class FixedGroupContributionPrior(nn.Module):
    """Deterministic fragment-count prior used by the optional group path."""

    def __init__(self) -> None:
        super().__init__()
        if GROUP_PRIOR_DIM != 20:
            raise ValueError(
                f"Unexpected GROUP_PRIOR_DIM={GROUP_PRIOR_DIM}; update prior indices."
            )

        self.register_buffer(
            "vm_weights",
            torch.tensor(
                [
                    0.0,
                    15.5,
                    13.0,
                    12.5,
                    3.5,
                    5.0,
                    9.5,
                    20.0,
                    3.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.5,
                    2.0,
                    1.5,
                    1.0,
                    1.5,
                    2.0,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "delta_d_weights",
            torch.tensor(
                [
                    0.0,
                    0.35,
                    0.55,
                    0.95,
                    -0.10,
                    -0.20,
                    0.10,
                    1.20,
                    0.25,
                    0.05,
                    0.15,
                    0.20,
                    0.10,
                    0.15,
                    0.10,
                    0.10,
                    0.05,
                    0.15,
                    0.10,
                    0.25,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "delta_p_weights",
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.20,
                    0.20,
                    0.90,
                    1.00,
                    0.70,
                    0.20,
                    0.10,
                    0.0,
                    8.0,
                    6.0,
                    9.0,
                    7.0,
                    4.0,
                    9.0,
                    7.0,
                    8.0,
                    10.0,
                    8.0,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "delta_h_weights",
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.10,
                    0.10,
                    0.80,
                    1.00,
                    0.40,
                    0.0,
                    0.05,
                    0.0,
                    20.0,
                    16.0,
                    18.0,
                    4.0,
                    2.5,
                    16.0,
                    7.0,
                    1.0,
                    1.0,
                    6.0,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _norm(self, group_features: Tensor) -> Tensor:
        heavy_atoms = group_features[:, 0].clamp(min=1.0)
        return heavy_atoms.pow(0.35)

    def hansen(self, group_features: Tensor) -> Tensor:
        """Return fixed Hansen priors from group-count features."""
        norm = self._norm(group_features)
        delta_d = 14.8 + (group_features @ self.delta_d_weights) / norm
        delta_p = 1.8 + (group_features @ self.delta_p_weights) / norm
        delta_h = 0.8 + (group_features @ self.delta_h_weights) / norm
        return torch.stack([delta_d, delta_p, delta_h], dim=-1).clamp(
            min=0.0,
            max=40.0,
        )

    def vm(self, group_features: Tensor) -> Tensor:
        """Return a fixed molar-volume prior from group-count features."""
        volume = 18.0 + group_features @ self.vm_weights
        return volume.clamp(min=30.0)


class SolventTypeMoE(nn.Module):
    """
    Mixture-of-Experts conditioned on solvent type.

    Uses a type embedding + gating MLP to mix expert transformations
    of the pair representation. Residual-scaled for safety.
    """

    def __init__(
        self,
        input_dim: int,
        num_types: int,
        num_experts: int = 4,
        type_emb_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.type_embed = nn.Embedding(num_types, type_emb_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + type_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim),
            )
            for _ in range(num_experts)
        ])
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, g_pair: Tensor, solvent_type: Tensor
    ) -> tuple[Tensor, Tensor]:
        type_ids = solvent_type.long().view(-1)
        type_emb = self.type_embed(type_ids)
        gate_logits = self.gate(torch.cat([g_pair, type_emb], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        expert_out = torch.stack(
            [expert(g_pair) for expert in self.experts], dim=1
        )  # (B, E, D)
        mixed = torch.sum(gate.unsqueeze(-1) * expert_out, dim=1)
        scale = torch.tanh(self.res_scale)
        return g_pair + scale * mixed, gate


# ================================================================== #
#  Fusion Head — crystal properties of the solute                     #
# ================================================================== #

class FusionHead(nn.Module):
    """
    Predict melting point and fusion thermodynamics from solute vector.

    Outputs (all per-sample):
      T_m     : melting temperature [K]
      dH_fus  : enthalpy of fusion [J/mol]
      dCp_fus : heat capacity change [J/(mol·K)]

    Default mode predicts absolute values from scratch. With
    ``fusion_output_mode="entropy_coupled"``, the absolute head predicts
    ``dH_fus`` and ``dS_fus`` and derives ``T_m = dH_fus / dS_fus`` before
    applying the configured solver-safe melting-point bounds.

    When ``use_gc_priors_crystal=True``, the head predicts only a bounded
    residual around fixed GC priors for ``T_m`` and ``dH_fus`` and passes
    through a fixed per-molecule ``dCp_fus_gc``.
    """

    def __init__(self, input_dim: int, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.fusion_output_mode not in {"direct", "entropy_coupled"}:
            raise ValueError(
                "fusion_output_mode must be either 'direct' or 'entropy_coupled'."
            )
        if cfg.fusion_entropy_min <= 0.0:
            raise ValueError("fusion_entropy_min must be positive.")
        if cfg.fusion_entropy_max <= cfg.fusion_entropy_min:
            raise ValueError("fusion_entropy_max must exceed fusion_entropy_min.")
        if cfg.use_gc_priors_crystal and cfg.predict_dCp_fus:
            raise ValueError(
                "use_gc_priors_crystal=True requires predict_dCp_fus=False because "
                "dCp_fus is fixed from the GC prior path."
            )
        self.mlp: nn.Sequential | None = None
        self.residual_mlp_Tm: nn.Sequential | None = None
        self.residual_mlp_dH: nn.Sequential | None = None
        self.direct_phi_mlp: nn.Sequential | None = None
        if cfg.use_direct_phi_branch:
            self.direct_phi_mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 2),
            )
            self._init_direct_phi_head(self.direct_phi_mlp)
        if cfg.use_gc_priors_crystal:
            self.residual_mlp_Tm = self._make_residual_mlp(input_dim)
            self.residual_mlp_dH = self._make_residual_mlp(input_dim)
            self._zero_residual_head(self.residual_mlp_Tm)
            self._zero_residual_head(self.residual_mlp_dH)
        else:
            output_dim = 3 if cfg.predict_dCp_fus else 2
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, output_dim),
            )
            if cfg.fusion_output_mode == "entropy_coupled":
                self._init_entropy_coupled_head(self.mlp)

    @staticmethod
    def _make_residual_mlp(input_dim: int) -> nn.Sequential:
        """Build a residual head for one crystal-property prior branch."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _zero_residual_head(head: nn.Sequential) -> None:
        """Initialize the last layer to zero so the prior passes through at init."""
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected residual head to end with nn.Linear.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """Numerically stable inverse of softplus for positive scalar values."""
        if value <= 0.0:
            raise ValueError("inverse softplus requires a positive value.")
        if value > 20.0:
            return value
        return math.log(math.expm1(value))

    @staticmethod
    def _logit(probability: float) -> float:
        """Return logit for a probability clipped away from 0 and 1."""
        p = min(max(probability, 1.0e-6), 1.0 - 1.0e-6)
        return math.log(p / (1.0 - p))

    def _init_entropy_coupled_head(self, head: nn.Sequential) -> None:
        """Initialize the entropy-coupled head near a Walden-like crystal prior."""
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected fusion head to end with nn.Linear.")
        with torch.no_grad():
            final.bias[0] = self._inverse_softplus(
                float(self.cfg.fusion_enthalpy_init) / float(self.cfg.S_H)
            )
            entropy_span = (
                float(self.cfg.fusion_entropy_max)
                - float(self.cfg.fusion_entropy_min)
            )
            entropy_frac = (
                float(self.cfg.fusion_entropy_init)
                - float(self.cfg.fusion_entropy_min)
            ) / entropy_span
            final.bias[1] = self._logit(entropy_frac)

    def _init_direct_phi_head(self, head: nn.Sequential) -> None:
        """Initialize direct Phi near a mild Van't Hoff slope."""
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected direct-Phi head to end with nn.Linear.")
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
            init_slope = max(
                float(self.cfg.direct_phi_slope_init),
                self.cfg.eps,
            )
            final.bias[1] = self._inverse_softplus(
                init_slope / max(float(self.cfg.direct_phi_slope_scale), self.cfg.eps)
            )

    def _add_entropy_diagnostics(
        self,
        params: dict[str, Tensor],
        *,
        T_m_unclamped: Tensor | None = None,
        dH_fus_raw: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Attach entropy-of-fusion diagnostics to a fusion parameter dict."""
        T_m = params["T_m"]
        dH_fus = params["dH_fus"]
        dS_fus = dH_fus / T_m.clamp_min(self.cfg.eps)
        params["dS_fus"] = dS_fus
        params["T_m_unclamped"] = T_m if T_m_unclamped is None else T_m_unclamped
        params["dH_fus_raw"] = dH_fus if dH_fus_raw is None else dH_fus_raw
        return params

    def set_residual_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the GC residual branches."""
        if not self.cfg.use_gc_priors_crystal:
            return
        if self.residual_mlp_Tm is None or self.residual_mlp_dH is None:
            raise ValueError("GC residual heads are not initialized.")
        for param in self.residual_mlp_Tm.parameters():
            param.requires_grad = not frozen
        for param in self.residual_mlp_dH.parameters():
            param.requires_grad = not frozen

    def _gc_dh_scale(self, residual_state: Tensor) -> Tensor:
        """Map a zero-centered residual state to the configured multiplicative range."""
        r_min, r_max = self.cfg.gc_prior_dH_residual_factor
        if r_min <= 0.0 or r_min > 1.0 or r_max < 1.0:
            raise ValueError(
                "gc_prior_dH_residual_factor must satisfy 0 < min <= 1 <= max "
                "for zero-centered residual initialization."
            )
        return torch.where(
            residual_state >= 0.0,
            1.0 + residual_state * (r_max - 1.0),
            1.0 + residual_state * (1.0 - r_min),
        )

    def forward(
        self,
        g_solute: Tensor,
        *,
        T_m_gc: Tensor | None = None,
        dH_fus_gc: Tensor | None = None,
        dCp_fus_gc: Tensor | None = None,
    ) -> dict[str, Tensor]:
        T_m_unclamped: Tensor | None = None
        dH_fus_raw: Tensor | None = None
        if self.cfg.use_gc_priors_crystal:
            if T_m_gc is None or dH_fus_gc is None or dCp_fus_gc is None:
                raise ValueError(
                    "GC crystal priors are enabled, but T_m_gc/dH_fus_gc/dCp_fus_gc "
                    "were not provided to FusionHead."
                )
            if self.residual_mlp_Tm is None or self.residual_mlp_dH is None:
                raise ValueError("GC residual heads are not initialized.")
            T_m_prior = T_m_gc.to(g_solute)
            dH_prior = dH_fus_gc.to(g_solute).clamp_min(self.cfg.eps)
            dCp_prior = dCp_fus_gc.to(g_solute)
            T_m_state = self.residual_mlp_Tm(g_solute).squeeze(-1)
            T_m_raw = (
                T_m_prior
                + self.cfg.gc_prior_Tm_residual_max * torch.tanh(T_m_state)
            )
            T_m_upper = torch.maximum(
                torch.full_like(T_m_raw, self.cfg.T_m_max),
                T_m_prior + self.cfg.gc_prior_Tm_residual_max,
            )
            T_m = torch.minimum(
                T_m_raw.clamp_min(self.cfg.T_m_min),
                T_m_upper,
            )
            dH_state = torch.tanh(self.residual_mlp_dH(g_solute).squeeze(-1))
            dH_scale = self._gc_dh_scale(dH_state)
            dH_fus = (dH_prior * dH_scale).clamp_min(self.cfg.eps)
            dCp_fus = dCp_prior
        else:
            if self.mlp is None:
                raise ValueError("Absolute fusion MLP is not initialized.")
            z = self.mlp(g_solute)
            if self.cfg.fusion_output_mode == "entropy_coupled":
                dH_fus_raw = F.softplus(z[:, 0]) * self.cfg.S_H
                dS_fus = self.cfg.fusion_entropy_min + (
                    (self.cfg.fusion_entropy_max - self.cfg.fusion_entropy_min)
                    * torch.sigmoid(z[:, 1])
                )
                T_m_unclamped = dH_fus_raw / dS_fus.clamp_min(self.cfg.eps)
                T_m = T_m_unclamped.clamp(self.cfg.T_m_min, self.cfg.T_m_max)
                dH_fus = T_m * dS_fus
            else:
                T_m = self.cfg.T_m_min + (
                    (self.cfg.T_m_max - self.cfg.T_m_min) * torch.sigmoid(z[:, 0])
                )
                dH_fus = F.softplus(z[:, 1]) * self.cfg.S_H
                dH_fus_raw = dH_fus
                T_m_unclamped = T_m
            dCp_fus = torch.full_like(T_m, self.cfg.fixed_dCp_fus)
        if self.cfg.predict_dCp_fus and not self.cfg.use_gc_priors_crystal:
            dCp_fus = z[:, 2] * self.cfg.S_Cp
        params = self._add_entropy_diagnostics(
            {"T_m": T_m, "dH_fus": dH_fus, "dCp_fus": dCp_fus},
            T_m_unclamped=T_m_unclamped,
            dH_fus_raw=dH_fus_raw,
        )
        if self.direct_phi_mlp is not None:
            phi_raw = self.direct_phi_mlp(g_solute)
            params["Phi_intercept"] = (
                torch.tanh(phi_raw[:, 0]) * self.cfg.direct_phi_intercept_scale
            )
            params["Phi_slope"] = (
                F.softplus(phi_raw[:, 1]) * self.cfg.direct_phi_slope_scale
            )
        return params


# ================================================================== #
#  NRTL Head — binary interaction parameters                          #
# ================================================================== #

class SigmaProfileHead(nn.Module):
    """
    Predict a COSMO-SAC sigma-profile from a single-molecule representation.

    Like FusionHead, this consumes the per-molecule readout (NOT the pair vector),
    so the *same* head produces the profile for solute and solvent and external
    single-component sigma-profile labels can supervise it directly.

    Outputs (per sample):
      p_shape : (B, n_bins) non-negative, sums to 1 (a proper sigma-profile shape)
      area    : (B,) cavity surface area [Å²], strictly positive
      p_sigma : (B, n_bins) = p_shape * area (area-weighted segment counts)
    """

    def __init__(self, input_dim: int, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_bins = int(cfg.cosmo_sac_n_bins)
        self.area_scale = float(cfg.sigma_area_scale)
        self.area_min = float(cfg.sigma_area_min)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.shape_head = nn.Linear(128, self.n_bins)
        self.area_head = nn.Linear(128, 1)
        # --- Arm I (Tier-3 closure-fix control): nonlinear rank-r adapter reading the
        # molecule readout g_mol, injected into the 128-d backbone feature. r>0 enriches
        # the intermediate (can reduce B_insuff); zero-init ⇒ starts at the frozen baseline.
        self.adapter_rank = int(getattr(cfg, "sigma_head_adapter_rank", 0))
        if self.adapter_rank > 0:
            self.adapter_down = nn.Linear(input_dim, self.adapter_rank, bias=False)  # (r, D_r)
            self.adapter_up = nn.Linear(self.adapter_rank, 128, bias=False)           # (128, r)
            nn.init.zeros_(self.adapter_up.weight)

    def forward(self, g_mol: Tensor) -> dict[str, Tensor]:
        h = self.backbone(g_mol)
        if getattr(self, "adapter_rank", 0) > 0:
            h = h + self.adapter_up(nn.functional.silu(self.adapter_down(g_mol)))
        p_shape = torch.softmax(self.shape_head(h), dim=-1)
        area = nn.functional.softplus(self.area_head(h)).squeeze(-1)
        area = area * self.area_scale + self.area_min
        return {
            "p_shape": p_shape,
            "area": area,
            "p_sigma": p_shape * area.unsqueeze(-1),
        }


class NRTLHead(nn.Module):
    """
    Predict NRTL parameters from the pair representation and temperature state.

    Outputs:
      If cfg.nrtl_tau_mode == "ref_invT":
        tau_ref_12, tau_ref_21 : tau values at T_ref
        tau_inv_12, tau_inv_21 : inverse-temperature slopes in
                                 tau(T) = tau_ref + tau_inv * (T_ref / T - 1)
        alpha_12               : non-randomness, sigmoid-bounded [α_min, α_max]
      If cfg.nrtl_tau_mode == "abc":
        tau_a12, tau_b12, tau_c12 : tau(T) coefficients for 1->2
        tau_a21, tau_b21, tau_c21 : tau(T) coefficients for 2->1
        alpha_12                  : non-randomness, sigmoid-bounded [α_min, α_max]
      If cfg.nrtl_tau_mode == "legacy":
        dg_12, dg_21 : energy parameters [J/mol]
        alpha_12     : non-randomness, sigmoid-bounded [α_min, α_max]
        a_T12, a_T21 : temperature-dependence coefficients
      If cfg.nrtl_tau_mode == "gamma_inf":
        ln_gamma_inf_ref : effective ln(gamma_2^inf) at T_ref
        ln_gamma_inf_inv : inverse-temperature slope for ln(gamma_2^inf)(T)
        alpha_12         : fixed compatibility tensor, not a learned parameter

    Initialization:
      - tau outputs zero-initialized → near-ideal starting point
      - alpha bias set to give α ≈ 0.3 at init
    """

    def __init__(self, input_dim: int, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.use_nrtl_group_prior and cfg.nrtl_tau_mode != "ref_invT":
            raise ValueError(
                "use_nrtl_group_prior currently supports only "
                "nrtl_tau_mode='ref_invT'."
            )
        self.use_temperature_features = cfg.use_temperature_in_nrtl_head
        backbone_input_dim = input_dim + (3 if self.use_temperature_features else 0)

        self.backbone = nn.Sequential(
            nn.Linear(backbone_input_dim, 512),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )
        output_dim = {
            "ref_invT": 5,
            "abc": 7,
            "legacy": 7,
            "gamma_inf": 2,
        }.get(cfg.nrtl_tau_mode)
        if output_dim is None:
            raise ValueError(f"Unsupported nrtl_tau_mode: {cfg.nrtl_tau_mode}")
        self.output = nn.Linear(128, output_dim)
        self.activity_global_bias = (
            nn.Parameter(
                torch.tensor(float(cfg.activity_global_bias_init), dtype=torch.float32)
            )
            if cfg.nrtl_tau_mode == "gamma_inf" and cfg.use_activity_global_bias
            else None
        )
        self.group_prior = (
            FixedGroupContributionPrior() if cfg.use_nrtl_group_prior else None
        )

        # Careful initialization: start near ideal (γ ≈ 1)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            if cfg.nrtl_tau_mode == "ref_invT":
                self.output.bias[4] = -0.405
            elif cfg.nrtl_tau_mode == "legacy":
                self.output.bias[2] = -0.405   # alpha index in legacy mode
            elif cfg.nrtl_tau_mode == "abc":
                self.output.bias[6] = -0.405   # alpha index in abc mode

    def _group_tau_prior(
        self,
        solute_group_prior_features: Tensor | None,
        solvent_group_prior_features: Tensor | None,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """Map fixed group-count features to a weak deterministic tau prior.

        This is deliberately not marketed as UNIFAC. It is a cheap
        Hansen/group-contribution surrogate that gives the NRTL head a
        physically sensible starting offset on structurally novel molecules.
        """
        if self.group_prior is None:
            return None
        if solute_group_prior_features is None or solvent_group_prior_features is None:
            raise ValueError(
                "NRTL group prior is enabled but group prior features were not "
                "provided. Build loaders with use_group_priors=True."
            )

        sol = solute_group_prior_features.to(reference)
        slv = solvent_group_prior_features.to(reference)
        hansen_sol = self.group_prior.hansen(sol)
        hansen_slv = self.group_prior.hansen(slv)
        vm_sol = self.group_prior.vm(sol).clamp_min(1e-6)
        vm_slv = self.group_prior.vm(slv).clamp_min(1e-6)

        diff = hansen_sol - hansen_slv
        ra = torch.sqrt(
            4.0 * diff[:, 0].pow(2)
            + diff[:, 1].pow(2)
            + diff[:, 2].pow(2)
            + 1e-8
        )
        mismatch = torch.tanh(ra / max(float(self.cfg.nrtl_group_prior_ra_scale), 1e-6))
        polarity = torch.tanh(
            (
                hansen_sol[:, 1]
                + hansen_sol[:, 2]
                - hansen_slv[:, 1]
                - hansen_slv[:, 2]
            )
            / 20.0
        )
        volume = torch.tanh(torch.log(vm_sol / vm_slv))

        scale = float(self.cfg.nrtl_group_prior_tau_scale)
        tau_12 = scale * (0.65 * mismatch + 0.20 * polarity + 0.15 * volume)
        tau_21 = scale * (0.65 * mismatch - 0.20 * polarity - 0.15 * volume)
        clamp = float(self.cfg.nrtl_group_prior_tau_clamp)
        return tau_12.clamp(-clamp, clamp), tau_21.clamp(-clamp, clamp)

    def _unifac_gamma_tau_prior(
        self,
        unifac_ln_gamma_inf: Tensor | None,
        unifac_gamma_mask: Tensor | None,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """Convert precomputed UNIFAC ln(gamma_inf) to a weak tau offset."""
        if not self.cfg.use_unifac_gamma_prior:
            return None
        if unifac_ln_gamma_inf is None:
            return None

        ln_gamma = unifac_ln_gamma_inf.to(reference)
        if unifac_gamma_mask is None:
            mask = torch.isfinite(ln_gamma)
        else:
            mask = unifac_gamma_mask.to(reference.device).bool() & torch.isfinite(ln_gamma)

        # Symmetric NRTL small-tau approximation:
        # ln(gamma_inf) = tau21 + tau12*exp(-alpha*tau12) ≈ 2*tau.
        scale = float(self.cfg.unifac_gamma_prior_tau_scale)
        clamp = float(self.cfg.unifac_gamma_prior_tau_clamp)
        tau = (0.5 * scale * ln_gamma).clamp(-clamp, clamp)
        tau = torch.where(mask, tau, torch.zeros_like(tau))
        return tau, tau

    def forward(
        self,
        g_pair: Tensor,
        temp_feat: Tensor | None = None,
        solute_group_prior_features: Tensor | None = None,
        solvent_group_prior_features: Tensor | None = None,
        unifac_ln_gamma_inf: Tensor | None = None,
        unifac_gamma_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.use_temperature_features:
            if temp_feat is None:
                raise ValueError(
                    "NRTLHead requires temperature features when "
                    "cfg.use_temperature_in_nrtl_head is enabled."
                )
            h = self.backbone(torch.cat([g_pair, temp_feat], dim=-1))
        else:
            h = self.backbone(g_pair)
        z = self.output(h)

        if self.cfg.nrtl_tau_mode == "gamma_inf":
            ln_gamma_inf_ref = z[:, 0] * self.cfg.S_gamma_inf_ref
            ln_gamma_inf_inv = z[:, 1] * self.cfg.S_gamma_inf_inv
            if self.activity_global_bias is not None:
                ln_gamma_inf_ref = ln_gamma_inf_ref + self.activity_global_bias.to(
                    ln_gamma_inf_ref
                )
            alpha_12 = torch.full_like(ln_gamma_inf_ref, 0.3)
            if self.cfg.use_unifac_gamma_prior and unifac_ln_gamma_inf is not None:
                ln_gamma = unifac_ln_gamma_inf.to(ln_gamma_inf_ref)
                if unifac_gamma_mask is None:
                    mask = torch.isfinite(ln_gamma)
                else:
                    mask = (
                        unifac_gamma_mask.to(ln_gamma_inf_ref.device).bool()
                        & torch.isfinite(ln_gamma)
                    )
                prior = torch.where(mask, ln_gamma, torch.zeros_like(ln_gamma))
                ln_gamma_inf_ref = ln_gamma_inf_ref + prior
            return {
                "ln_gamma_inf_ref": ln_gamma_inf_ref,
                "ln_gamma_inf_inv": ln_gamma_inf_inv,
                "alpha_12": alpha_12,
            }

        if self.cfg.nrtl_tau_mode == "ref_invT":
            tau_ref_12 = z[:, 0] * self.cfg.S_tau_ref
            tau_ref_21 = z[:, 1] * self.cfg.S_tau_ref
            tau_inv_12 = z[:, 2] * self.cfg.S_tau_inv
            tau_inv_21 = z[:, 3] * self.cfg.S_tau_inv
            alpha_12 = self.cfg.alpha_min + (
                (self.cfg.alpha_max - self.cfg.alpha_min)
                * torch.sigmoid(z[:, 4])
            )
            params = {
                "tau_ref_12": tau_ref_12,
                "tau_ref_21": tau_ref_21,
                "tau_inv_12": tau_inv_12,
                "tau_inv_21": tau_inv_21,
                "alpha_12": alpha_12,
            }
            prior = self._group_tau_prior(
                solute_group_prior_features,
                solvent_group_prior_features,
                reference=tau_ref_12,
            )
            if prior is not None:
                prior_12, prior_21 = prior
                params["tau_ref_12"] = params["tau_ref_12"] + prior_12
                params["tau_ref_21"] = params["tau_ref_21"] + prior_21
                params["nrtl_group_tau_prior_12"] = prior_12
                params["nrtl_group_tau_prior_21"] = prior_21
            unifac_prior = self._unifac_gamma_tau_prior(
                unifac_ln_gamma_inf,
                unifac_gamma_mask,
                reference=tau_ref_12,
            )
            if unifac_prior is not None:
                prior_12, prior_21 = unifac_prior
                params["tau_ref_12"] = params["tau_ref_12"] + prior_12
                params["tau_ref_21"] = params["tau_ref_21"] + prior_21
                params["unifac_gamma_tau_prior_12"] = prior_12
                params["unifac_gamma_tau_prior_21"] = prior_21
            return params

        if self.cfg.nrtl_tau_mode == "legacy":
            dg_12 = z[:, 0] * self.cfg.S_g
            dg_21 = z[:, 1] * self.cfg.S_g
            alpha_12 = self.cfg.alpha_min + (
                (self.cfg.alpha_max - self.cfg.alpha_min)
                * torch.sigmoid(z[:, 2])
            )
            a_T12 = z[:, 3] * self.cfg.S_aT
            a_T21 = z[:, 4] * self.cfg.S_aT
            return {
                "dg_12": dg_12,
                "dg_21": dg_21,
                "alpha_12": alpha_12,
                "a_T12": a_T12,
                "a_T21": a_T21,
            }

        tau_a12 = z[:, 0] * self.cfg.S_tau_a
        tau_b12 = z[:, 1] * self.cfg.S_tau_b
        tau_c12 = z[:, 2] * self.cfg.S_tau_c
        tau_a21 = z[:, 3] * self.cfg.S_tau_a
        tau_b21 = z[:, 4] * self.cfg.S_tau_b
        tau_c21 = z[:, 5] * self.cfg.S_tau_c
        alpha_12 = self.cfg.alpha_min + (
            (self.cfg.alpha_max - self.cfg.alpha_min)
            * torch.sigmoid(z[:, 6])
        )

        return {
            "tau_a12": tau_a12,
            "tau_b12": tau_b12,
            "tau_c12": tau_c12,
            "tau_a21": tau_a21,
            "tau_b21": tau_b21,
            "tau_c21": tau_c21,
            "alpha_12": alpha_12,
        }


# ================================================================== #
#  Hansen Head — solubility parameters                                #
# ================================================================== #

class HansenHead(nn.Module):
    """
    Predict Hansen solubility parameters (δd, δp, δh) in MPa^0.5.

    All three are non-negative → softplus activation.
    """

    def __init__(self, input_dim: int, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prior_mode = _resolve_prior_mode(cfg)
        self.residual_limit = (
            cfg.group_prior_hansen_residual_max
            if self.prior_mode == "group"
            else cfg.descriptor_prior_hansen_residual_max
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        self.prior_net = None
        self.group_prior = None
        if self.prior_mode == "descriptor":
            self.prior_net = DescriptorPriorAdapter(
                3,
                cfg.descriptor_prior_hidden_dim,
                cfg.dropout,
            )
        elif self.prior_mode == "group":
            self.group_prior = FixedGroupContributionPrior()
        if self.prior_mode != "none":
            with torch.no_grad():
                self.mlp[-1].weight.zero_()
                self.mlp[-1].bias.zero_()

    def forward(
        self,
        g_mol: Tensor,
        prior_features: Tensor | None = None,
        *,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        residual_raw = self.mlp(g_mol)
        if self.prior_mode == "none":
            value = F.softplus(residual_raw) * self.cfg.S_delta
            if not return_parts:
                return value
            return {
                "value": value,
                "prior": torch.zeros_like(value),
                "residual": value,
            }

        if prior_features is None:
            raise ValueError(
                "A molecular prior path is enabled, but prior_features were not "
                "provided to HansenHead."
            )

        if self.prior_mode == "descriptor":
            prior_raw = self.prior_net(prior_features)
            prior = F.softplus(prior_raw) * self.cfg.S_delta
        else:
            prior = self.group_prior.hansen(prior_features).to(g_mol)
        residual = self.residual_limit * torch.tanh(residual_raw)
        value = (prior + residual).clamp(min=0.0)
        if not return_parts:
            return value
        return {
            "value": value,
            "prior": prior,
            "residual": residual,
        }


# ================================================================== #
#  Auxiliary Properties Head                                          #
# ================================================================== #

class AuxPropsHead(nn.Module):
    """
    Predict the auxiliary molecular property used by the current objective.

    Only molar volume is retained in the maintained architecture. Earlier
    versions also emitted `eps_r`, `mu`, and `n_D`, but those outputs were not
    supervised anywhere in the training objective and only added latent noise.

    V_m : molar volume [cm³/mol], softplus → (30, +∞)
    """

    def __init__(self, input_dim: int, cfg: TGNNSolvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prior_mode = _resolve_prior_mode(cfg)
        self.residual_limit = (
            cfg.group_prior_vm_residual_max
            if self.prior_mode == "group"
            else cfg.descriptor_prior_vm_residual_max
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.prior_net = None
        self.group_prior = None
        if self.prior_mode == "descriptor":
            self.prior_net = DescriptorPriorAdapter(
                1,
                cfg.descriptor_prior_hidden_dim,
                cfg.dropout,
            )
        elif self.prior_mode == "group":
            self.group_prior = FixedGroupContributionPrior()
        if self.prior_mode != "none":
            with torch.no_grad():
                self.mlp[-1].weight.zero_()
                self.mlp[-1].bias.zero_()

    def forward(
        self,
        g_mol: Tensor,
        prior_features: Tensor | None = None,
        *,
        return_parts: bool = False,
    ) -> dict[str, Tensor]:
        residual_raw = self.mlp(g_mol)[:, 0]
        if self.prior_mode == "none":
            V_m = F.softplus(residual_raw) * 50.0 + 30.0
            if not return_parts:
                return {"V_m": V_m}
            return {
                "V_m": V_m,
                "V_m_prior": torch.zeros_like(V_m),
                "V_m_residual": V_m,
            }

        if prior_features is None:
            raise ValueError(
                "A molecular prior path is enabled, but prior_features were not "
                "provided to AuxPropsHead."
            )

        if self.prior_mode == "descriptor":
            prior_raw = self.prior_net(prior_features)[:, 0]
            V_m_prior = F.softplus(prior_raw) * 50.0 + 30.0
        else:
            V_m_prior = self.group_prior.vm(prior_features).to(g_mol)
        V_m_residual = self.residual_limit * torch.tanh(residual_raw)
        V_m = (V_m_prior + V_m_residual).clamp(min=30.0)
        if not return_parts:
            return {"V_m": V_m}
        return {
            "V_m": V_m,
            "V_m_prior": V_m_prior,
            "V_m_residual": V_m_residual,
        }


class AdaptivePhysicsCorrection(nn.Module):
    """
    Adaptive residual around the physics prediction.

    The correction path operates in parameter space.

    It predicts bounded deltas for:
      - T_m
      - dH_fus
      - tau_12(T)
      - tau_21(T)

    The corrected parameter proposal is then passed back through the SLE
    solver, and the resulting residual around the base physics solution is
    blended via a confidence gate.
    """

    def __init__(
        self,
        pair_dim: int,
        n_param_features: int = 6,
        Tm_limit: float = 60.0,
        dH_fraction_limit: float = 0.25,
        tau_limit: float = 2.0,
    ) -> None:
        super().__init__()
        self.Tm_limit = Tm_limit
        self.dH_fraction_limit = dH_fraction_limit
        self.tau_limit = tau_limit

        # Confidence network: estimates how reliable physics is
        self.confidence_net = nn.Sequential(
            nn.Linear(pair_dim + n_param_features, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Parameter-space correction: delta_Tm, delta_dH_frac, delta_tau12,
        # delta_tau21, log_sigma
        self.param_delta_net = nn.Sequential(
            nn.Linear(pair_dim + n_param_features, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 5),
        )

        # Initialize confidence bias high → σ(2.2) ≈ 0.9
        with torch.no_grad():
            self.confidence_net[-1].bias.fill_(2.2)
            self.param_delta_net[-1].weight.zero_()
            self.param_delta_net[-1].bias.zero_()

    def forward(
        self,
        g_pair: Tensor,
        param_summary: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """
        Returns
        -------
        confidence  : (B,) physics confidence σ(w)
        param_deltas : bounded parameter deltas used to build the proposal
        proposal_log_sigma : (B,) log sigma for the corrected-parameter proposal
        """
        inp = torch.cat([g_pair, param_summary], dim=-1)

        # Physics confidence: how much to trust SLE output
        confidence = torch.sigmoid(
            self.confidence_net(inp).squeeze(-1)
        )  # (B,) in [0, 1]

        delta_out = self.param_delta_net(inp)
        param_deltas = {
            "delta_T_m": self.Tm_limit * torch.tanh(delta_out[:, 0]),
            "delta_dH_fraction": self.dH_fraction_limit * torch.tanh(delta_out[:, 1]),
            "delta_tau_12": self.tau_limit * torch.tanh(delta_out[:, 2]),
            "delta_tau_21": self.tau_limit * torch.tanh(delta_out[:, 3]),
        }
        proposal_log_sigma = delta_out[:, 4].clamp(-6.0, 2.0)
        return confidence, param_deltas, proposal_log_sigma
