"""
Multi-component physics-constrained loss for TGNN-Solv.

Components:
  sol        : Huber loss on ln(x₂) — only for records WITH solubility
  T_m        : MSE on melting point (masked)
  dH         : MSE on fusion enthalpy (masked)
  hansen     : MSE on Hansen parameters (masked)
  hansen_delta : MSE on pairwise Hansen parameter deltas
  gamma_inf  : MSE on ln(γ∞) (masked)
  gamma_2    : MSE on finite-composition ln(γ₂) at supplied activity states
  mono       : Monotonicity penalty (dx₂/dT ≥ 0)
  res        : L2 on effective correction (ln_x2 - ln_x2_physics)
  bridge     : Consistency between Hansen and NRTL γ∞
  tau_reg    : L2 on NRTL τ parameters
  phys_pref  : Encourage high physics confidence
  direct_reg : Keep residual proposal close to physics
  direct_nll : Heteroskedastic NLL on residual proposal
  decorr     : Penalize coupled crystal/activity decomposition errors
  pair_temp_rank : Same-pair temperature ranking consistency
  vant_hoff_local : Local linearity in ln(x₂) vs 1/T for same-pair batches
  pair_temp_delta : Supervised same-pair adjacent Δln(x₂) consistency
  vant_hoff_slope : Supervised same-pair Van't Hoff slope consistency
  vant_hoff_intercept : Supervised same-pair Van't Hoff intercept consistency
  vh_anchor : Pseudo high-T Van't Hoff anchor distillation
  sol_variance : Batch-level penalty against prediction-range collapse
  moe_balance: Encourage balanced MoE expert usage
  walden    : Walden-rule entropy-of-fusion consistency for unsupervised samples
  hansen_contrastive_* : Align molecular / TIMP-channel / pair latent distances
                         with Hansen distances when enabled
  aux_direct_sol : training-only direct solubility head on pair representation
                   to feed gradients into interaction layers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TGNNSolvConfig
from .hansen_contrastive import (
    ChannelHansenContrastiveLoss,
    HansenContrastiveLoss,
    PairHansenContrastiveLoss,
    channel_orthogonality_penalty,
)

if TYPE_CHECKING:
    from .model import TGNNSolv

MIN_VANT_HOFF_INV_T_DIFF = 1.0e-4
MAX_VANT_HOFF_PAIR_LOSS = 100.0


def sigma_profile_emd_loss(
    pred_shape: Tensor,
    target_shape: Tensor,
    pred_area: Tensor,
    target_area: Tensor,
    mask: Tensor,
    *,
    mode: str = "emd",
    area_scale: float = 75.0,
    shape_weight: float = 1.0,
    shape_floor: float = 0.0,
    shape_bin_weights: Tensor | None = None,
    eps: float = 1e-8,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float]]:
    """Masked sigma-profile supervision loss (shape + cavity area).

    The shape term is the 1-D Wasserstein/EMD distance on the ordered sigma grid,
    SUMMED over bins (``sum|cumsum(pred) - cumsum(target)|``) so the 51-dim shape
    that defines the activity manifold is not crushed by a per-bin mean and cannot
    be dominated by the single area scalar; ``mode="mse"`` falls back to per-bin
    MSE. The area term is a scaled MSE on the cavity surface area; ``area_scale``
    should track the pool's sigma_area std (~75 Å²). Both are averaged over the
    masked single-component rows. With ``return_components=True`` also returns the
    detached scalar shape/area terms for logging.

    ``shape_floor`` > 0 replaces the shape term with ``relu(shape - floor)``: the gradient is
    bit-identical above the floor and exactly zero at or below it, i.e. the supervision
    descends to a set depth and stops. See ``config.sigma_shape_floor`` for where the number
    comes from and which nulls established that the direction, not the displacement, is what
    the deposited profile contributes. ``sigma_shape_raw`` is always reported alongside
    ``sigma_shape`` so the floored and unfloored values are both visible in a run's log; the
    key is present in BOTH branches on purpose, because ``trainer`` symmetrises solute and
    solvent component dicts key-by-key and a missing key would raise there on mixed batches.

    ``shape_bin_weights`` -- вектор длины n_bins, которым взвешиваются ПО-БИНОВЫЕ слагаемые
    форменного члена. Это не косметика, а смена метрики: взвешенная сумма
    ``sum_k w_k |C_k(pred) - C_k(target)|`` есть ровно 1-D Вассерштейн по сетке, в которой
    расстояние между соседними узлами равно ``w_k``, поэтому вес с положительными компонентами
    оставляет член метрикой, а невзвешенный EMD оказывается частным случаем ``w = const``, то есть
    предположением, что цена замыкания растёт по сетке равномерно. Измерение этого предположения
    не подтверждает (``results/per_bin_sensitivity_weights``, ступень на ``cosmo_sac_sigma_hb``),
    и вектор оттуда берётся как дискретный градиент измеренного потенциала размещения.

    ВНИМАНИЕ НА ПОВЕРХНОСТЬ. В режиме ``"emd"`` слагаемое с индексом k оценивает массу,
    ПЕРЕСЕКАЮЩУЮ границу между бинами k и k+1, а в режиме ``"mse"`` -- массу, ЛЕЖАЩУЮ в бине k.
    Это две разные поверхности, и один и тот же вектор весов означает в них разное: в ``"emd"``
    это цена переноса через границу, в ``"mse"`` -- цена размещения. Вектор, выведенный под один
    режим, в другом стоит не на своей поверхности.

    ``w = None`` и ``w`` из одних единиц дают БИТОВО тот же член, что и до появления крюка:
    умножение на ровно 1.0 в IEEE754 точное, а порядок суммирования не меняется. Это
    зафиксировано тестом, а не предположено.
    """
    m = mask.bool()
    if not bool(m.any()):
        zero = pred_shape.sum() * 0.0
        if return_components:
            return zero, {"sigma_shape": 0.0, "sigma_shape_raw": 0.0, "sigma_area": 0.0}
        return zero
    ps = pred_shape[m]
    ts = target_shape[m]
    if mode == "mse":
        per_bin = (ps - ts) ** 2
    else:
        per_bin = (torch.cumsum(ps, dim=-1) - torch.cumsum(ts, dim=-1)).abs()
    if shape_bin_weights is None:
        shape_loss = per_bin.sum(dim=-1).mean()
    else:
        w = shape_bin_weights
        if w.ndim != 1 or w.shape[0] != per_bin.shape[-1]:
            # Громко, а не широковещательно: вектор длины 1 или n_rows молча прошёл бы
            # бродкастом и обучал бы не тот функционал.
            raise ValueError(
                f"shape_bin_weights must be 1-D of length {per_bin.shape[-1]}, "
                f"got shape {tuple(w.shape)}"
            )
        shape_loss = (w.to(per_bin) * per_bin).sum(dim=-1).mean()
    shape_raw = shape_loss
    if shape_floor > 0.0:
        # torch.relu, а НЕ clamp_min(0.0): на самом полу их производные различаются, и это
        # проверено, а не предположено. clamp_min пропускает градиент при x == 0 (его backward
        # сравнивает >=), relu даёт ровно нуль (сравнивает > 0). Для "дошёл до глубины и
        # встал" нужна вторая семантика, иначе ровно на полу супервизия продолжает давить.
        shape_loss = torch.relu(shape_loss - shape_floor)
    area_loss = (((pred_area[m] - target_area[m]) / area_scale) ** 2).mean()
    total = shape_weight * shape_loss + area_loss
    if return_components:
        return total, {
            "sigma_shape": shape_loss.item(),
            "sigma_shape_raw": shape_raw.item(),
            "sigma_area": area_loss.item(),
        }
    return total


class TGNNSolvLoss(nn.Module):
    """Multi-component loss with adaptive physics correction support."""

    def __init__(
        self,
        cfg: TGNNSolvConfig,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.default_weights = {
            "sol": 1.0,
            "T_m": 0.3,
            "dH": 0.3,
            "hansen": 0.2,
            "timp_disp_hansen": 0.0,
            "timp_polar_hansen": 0.0,
            "gamma_inf": 0.5,
            "gamma_2": 0.5,
            "mono": 0.1,
            "res": 0.01,
            "bridge": cfg.bridge_loss_weight,
            "tau_reg": 0.01,
            "phys_pref": 0.1,
            "direct_reg": 0.05,
            "direct_nll": 0.2,
            "decorr": 0.0,
            "pair_temp_rank": 0.0,
            "vant_hoff_local": 0.0,
            "pair_temp_delta": 0.0,
            "vant_hoff_slope": 0.0,
            "vant_hoff_intercept": 0.0,
            "vh_anchor": 0.0,
            "sol_variance": cfg.sol_variance_loss_weight,
            "moe_balance": 0.02,
            "descriptor_prior": 0.0,
            "group_prior": 0.0,
            "hansen_contrastive_mol": 0.0,
            "hansen_contrastive_channel": 0.0,
            "hansen_contrastive_pair": 0.0,
            "hansen_channel_orth": 0.0,
            "hansen_delta": 0.0,
            "aux_direct_sol": 0.0,
        }
        if weights is not None:
            self.default_weights.update(weights)

        self.S_Tm = 50.0
        self.S_dH = 5000.0
        self.S_hansen = 5.0
        self.S_bridge = 3.0
        self.huber_delta = 1.0
        self.hansen_contrastive = HansenContrastiveLoss(
            temperature=cfg.hansen_contrastive_temperature
        )
        self.channel_hansen_contrastive = ChannelHansenContrastiveLoss(
            temperature=cfg.hansen_contrastive_temperature
        )
        self.pair_hansen_contrastive = PairHansenContrastiveLoss(
            temperature=cfg.hansen_contrastive_temperature
        )

    def huber_loss(
        self, pred: Tensor, target: Tensor, delta: float | None = None
    ) -> Tensor:
        if delta is None:
            delta = self.huber_delta
        r = pred - target
        ar = r.abs()
        return torch.where(
            ar <= delta, 0.5 * r ** 2, delta * (ar - 0.5 * delta)
        ).mean()

    def weighted_huber_loss(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        delta: float | None = None,
    ) -> Tensor:
        if weight is None:
            return self.huber_loss(pred, target, delta=delta)
        if delta is None:
            delta = self.huber_delta
        r = pred - target
        ar = r.abs()
        huber = torch.where(
            ar <= delta,
            0.5 * r ** 2,
            delta * (ar - 0.5 * delta),
        )
        weight = weight.to(pred.device, dtype=pred.dtype).clamp_min(self.cfg.eps)
        return (weight * huber).sum() / weight.sum().clamp_min(self.cfg.eps)

    def weighted_mae_loss(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        abs_error = (pred - target).abs()
        return self.weighted_mean(abs_error, weight=weight)

    def _solubility_bin_weights(self, target: Tensor) -> Tensor | None:
        """Build per-row weights over true ln(x2) bins."""
        mode = str(self.cfg.sol_bin_weight_mode).lower()
        if mode in {"", "none", "off", "false"}:
            return None
        if mode not in {"inverse_frequency", "fixed"}:
            raise ValueError(
                "sol_bin_weight_mode must be 'none', 'inverse_frequency', or 'fixed', "
                f"got {self.cfg.sol_bin_weight_mode!r}."
            )
        edges = torch.as_tensor(
            tuple(float(x) for x in self.cfg.sol_bin_edges),
            device=target.device,
            dtype=target.dtype,
        )
        if edges.numel() < 2:
            return None
        # bucketize against inner edges; out-of-range values go to end buckets.
        bucket = torch.bucketize(target.detach(), edges[1:-1], right=False)
        if mode == "fixed":
            configured = tuple(float(x) for x in self.cfg.sol_bin_weights)
            expected = int(edges.numel() - 1)
            if len(configured) != expected:
                raise ValueError(
                    "sol_bin_weights length must equal len(sol_bin_edges) - 1 "
                    f"for fixed bin weights; got {len(configured)} and expected {expected}."
                )
            bin_weights = torch.as_tensor(
                configured,
                device=target.device,
                dtype=target.dtype,
            )
            raw_weight = bin_weights[bucket]
        else:
            counts = torch.bincount(bucket, minlength=int(edges.numel() - 1)).to(
                device=target.device,
                dtype=target.dtype,
            )
            valid_counts = counts.clamp_min(1.0)
            raw_weight = 1.0 / valid_counts[bucket]
        raw_weight = raw_weight / raw_weight.mean().clamp_min(self.cfg.eps)
        return raw_weight.clamp(
            min=float(self.cfg.sol_bin_weight_min),
            max=float(self.cfg.sol_bin_weight_max),
        )

    def solubility_loss(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        loss_type = str(self.cfg.sol_loss_type).lower()
        bin_weight = self._solubility_bin_weights(target)
        if weight is None:
            effective_weight = bin_weight
        elif bin_weight is None:
            effective_weight = weight
        else:
            effective_weight = weight.to(pred.device, dtype=pred.dtype) * bin_weight
            effective_weight = effective_weight / effective_weight.mean().clamp_min(self.cfg.eps)

        if loss_type == "huber":
            return self.weighted_huber_loss(pred, target, weight=effective_weight)
        if loss_type == "weighted_huber":
            if effective_weight is None:
                effective_weight = torch.ones_like(target)
            return self.weighted_huber_loss(pred, target, weight=effective_weight)
        if loss_type == "mae":
            return self.weighted_mae_loss(pred, target, weight=effective_weight)
        if loss_type == "weighted_mae":
            if effective_weight is None:
                effective_weight = torch.ones_like(target)
            return self.weighted_mae_loss(pred, target, weight=effective_weight)
        raise ValueError(
            "sol_loss_type must be one of 'huber', 'mae', 'weighted_huber', "
            f"'weighted_mae'; got {self.cfg.sol_loss_type!r}."
        )

    def solubility_variance_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.numel() < 2:
            return pred.new_zeros(())
        pred_std = pred.std(unbiased=False)
        target_std = target.std(unbiased=False)
        denom = target_std.clamp_min(float(self.cfg.sol_variance_eps))
        return ((pred_std - target_std) / denom).pow(2)

    def weighted_mean(
        self,
        value: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        if weight is None:
            return value.mean()
        weight = weight.to(value.device, dtype=value.dtype).clamp_min(self.cfg.eps)
        return (weight * value).sum() / weight.sum().clamp_min(self.cfg.eps)

    def masked_mse(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        scale: float = 1.0,
    ) -> Tensor:
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred[mask] - target[mask]) / scale).pow(2).mean()

    def _crystal_activity_decorrelation_loss(
        self,
        output: dict[str, object],
        targets: dict[str, object],
        T: Tensor,
        sol_mask: Tensor,
        dev: torch.device,
    ) -> dict[str, Tensor]:
        """Penalize correlation between crystal and activity decomposition errors."""
        zero = torch.zeros((), device=dev)
        stats = {
            "decorr": zero,
            "decorr_corr": zero,
            "decorr_joint_rows": zero,
        }

        physics = output.get("physics")
        if not isinstance(physics, dict):
            return stats

        phi_pred = physics.get("Phi")
        gamma_pred = physics.get("ln_gamma_2")
        T_m_target = targets.get("T_m")
        dH_target = targets.get("dH_fus")
        ln_x2_target = targets.get("ln_x2")
        T_m_mask = targets.get("T_m_mask")
        dH_mask = targets.get("dH_mask")
        required = (
            phi_pred,
            gamma_pred,
            T_m_target,
            dH_target,
            ln_x2_target,
            T_m_mask,
            dH_mask,
        )
        if not all(isinstance(value, Tensor) for value in required):
            return stats

        joint_mask = sol_mask & T_m_mask.to(dev).bool() & dH_mask.to(dev).bool()
        stats["decorr_joint_rows"] = joint_mask.sum().to(
            device=dev,
            dtype=phi_pred.dtype,
        )
        min_samples = max(int(self.cfg.decorr_min_samples), 2)
        if int(joint_mask.sum().item()) < min_samples:
            return stats

        dtype = phi_pred.dtype
        T_joint = T.to(dev, dtype=dtype)[joint_mask]
        T_m_joint = T_m_target.to(dev, dtype=dtype)[joint_mask]
        dH_joint = dH_target.to(dev, dtype=dtype)[joint_mask]
        phi_pred_joint = phi_pred[joint_mask]
        gamma_pred_joint = gamma_pred.to(dev, dtype=dtype)[joint_mask]
        ln_x2_joint = ln_x2_target.to(dev, dtype=dtype)[joint_mask]

        phi_true = (dH_joint / float(self.cfg.R)) * (
            1.0 / T_joint - 1.0 / T_m_joint
        )
        delta_phi = phi_pred_joint - phi_true
        gamma_required = -ln_x2_joint - phi_true
        delta_gamma = gamma_pred_joint - gamma_required

        finite_mask = torch.isfinite(delta_phi) & torch.isfinite(delta_gamma)
        if int(finite_mask.sum().item()) < min_samples:
            return stats

        delta_phi = delta_phi[finite_mask]
        delta_gamma = delta_gamma[finite_mask]
        centered_phi = delta_phi - delta_phi.mean()
        centered_gamma = delta_gamma - delta_gamma.mean()
        phi_var = centered_phi.pow(2).mean()
        gamma_var = centered_gamma.pow(2).mean()
        if (
            phi_var <= float(self.cfg.decorr_eps)
            or gamma_var <= float(self.cfg.decorr_eps)
        ):
            return stats

        corr = (centered_phi * centered_gamma).mean() / torch.sqrt(
            phi_var * gamma_var
        ).clamp_min(float(self.cfg.decorr_eps))
        stats["decorr_corr"] = corr.clamp(-1.0, 1.0)
        stats["decorr"] = stats["decorr_corr"].pow(2)
        return stats

    def _find_grad_anchor(self, value: object) -> Tensor | None:
        """Find a tensor with autograd history inside a nested output structure."""
        if isinstance(value, Tensor):
            return value if value.requires_grad else None
        if isinstance(value, dict):
            for child in value.values():
                anchor = self._find_grad_anchor(child)
                if anchor is not None:
                    return anchor
            return None
        if isinstance(value, (list, tuple)):
            for child in value:
                anchor = self._find_grad_anchor(child)
                if anchor is not None:
                    return anchor
        return None

    def forward(
        self,
        output: dict[str, object],
        targets: dict[str, object],
        weights: dict[str, float] | None = None,
        compute_mono: bool = False,
        T: Tensor | None = None,
        model: TGNNSolv | None = None,
        solute_data: object | None = None,
        solvent_data: object | None = None,
        solvent_type: Tensor | None = None,
        solute_morgan_fp: Tensor | None = None,
        solvent_morgan_fp: Tensor | None = None,
        solute_descriptors: Tensor | None = None,
        solvent_descriptors: Tensor | None = None,
        ionic_features: Tensor | None = None,
        solute_descriptor_prior_features: Tensor | None = None,
        solvent_descriptor_prior_features: Tensor | None = None,
        solute_group_prior_features: Tensor | None = None,
        solvent_group_prior_features: Tensor | None = None,
        T_m_gc: Tensor | None = None,
        dH_fus_gc: Tensor | None = None,
        dCp_fus_gc: Tensor | None = None,
        detach_crystal_from_encoder: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        w = weights if weights is not None else self.default_weights
        losses = {}
        dev = output["ln_x2"].device
        physics = output.get("physics", {})

        # ============================================================
        # 1. Solubility loss (masked by has_solubility)
        # ============================================================
        sol_mask = targets.get("has_solubility")
        if sol_mask is None:
            sol_mask = torch.ones(
                output["ln_x2"].shape[0], dtype=torch.bool, device=dev
            )
        else:
            sol_mask = sol_mask.to(dev)
        sol_sample_weight = None
        if self.cfg.use_source_uncertainty_weights:
            maybe_weight = targets.get("source_solubility_weight")
            if isinstance(maybe_weight, Tensor):
                sol_sample_weight = maybe_weight.to(dev)[sol_mask]

        if w.get("sol", 0) > 0 and sol_mask.any():
            losses["sol"] = self.solubility_loss(
                output["ln_x2"][sol_mask],
                targets["ln_x2"].to(dev)[sol_mask],
                weight=sol_sample_weight,
            )
            if w.get("sol_variance", 0) > 0:
                losses["sol_variance"] = self.solubility_variance_loss(
                    output["ln_x2"][sol_mask],
                    targets["ln_x2"].to(dev)[sol_mask],
                )

        if (
            w.get("aux_direct_sol", 0) > 0
            and sol_mask.any()
            and isinstance(output.get("ln_x2_aux"), Tensor)
        ):
            losses["aux_direct_sol"] = self.weighted_huber_loss(
                output["ln_x2_aux"][sol_mask],
                targets["ln_x2"].to(dev)[sol_mask],
                weight=sol_sample_weight,
            )

        if (
            w.get("direct_nll", 0) > 0
            and sol_mask.any()
            and "ln_x2_direct" in output
            and "ln_x2_direct_log_sigma" in output
        ):
            mu = output["ln_x2_direct"][sol_mask]
            log_sigma = output["ln_x2_direct_log_sigma"][sol_mask]
            true = targets["ln_x2"].to(dev)[sol_mask]
            res = mu - true
            inv_var = torch.exp(-2.0 * log_sigma)
            direct_nll = (
                0.5 * inv_var * res.pow(2) + log_sigma
            )
            losses["direct_nll"] = self.weighted_mean(
                direct_nll,
                weight=sol_sample_weight,
            )

        vh_anchor_mask = targets.get("vh_anchor_mask")
        if (
            w.get("vh_anchor", 0) > 0
            and isinstance(vh_anchor_mask, Tensor)
            and bool(vh_anchor_mask.to(dev).any().item())
        ):
            mask = vh_anchor_mask.to(dev).bool()
            anchor_weight = targets.get("vh_anchor_weight")
            sample_weight = (
                anchor_weight.to(dev)[mask]
                if isinstance(anchor_weight, Tensor)
                else None
            )
            losses["vh_anchor"] = self.weighted_huber_loss(
                output["ln_x2"][mask],
                targets["vh_anchor_ln_x2"].to(dev)[mask],
                weight=sample_weight,
            )

        # ============================================================
        # 2. Melting point
        # ============================================================
        if "T_m" in targets and targets.get("T_m_mask") is not None:
            if w.get("T_m", 0) > 0:
                losses["T_m"] = self.masked_mse(
                    output["fusion_params"]["T_m"],
                    targets["T_m"].to(dev),
                    targets["T_m_mask"].to(dev),
                    scale=self.S_Tm,
                )

        # ============================================================
        # 3. Fusion enthalpy
        # ============================================================
        if "dH_fus" in targets and targets.get("dH_mask") is not None:
            if w.get("dH", 0) > 0:
                losses["dH"] = self.masked_mse(
                    output["fusion_params"]["dH_fus"],
                    targets["dH_fus"].to(dev),
                    targets["dH_mask"].to(dev),
                    scale=self.S_dH,
                )

        # ============================================================
        # 4. Hansen parameters
        # ============================================================
        if "hansen_sol" in targets and targets.get("hansen_mask") is not None:
            if w.get("hansen", 0) > 0:
                mask = targets["hansen_mask"].to(dev)
                if mask.sum() > 0:
                    pred_h = output["hansen_sol"][mask]
                    tgt_h = targets["hansen_sol"].to(dev)[mask]
                    losses["hansen"] = (
                        ((pred_h - tgt_h) / self.S_hansen).pow(2).mean()
                    )
            if (
                "timp_channel_probes" in output
                and isinstance(output["timp_channel_probes"], dict)
            ):
                mask = targets["hansen_mask"].to(dev)
                probes = output["timp_channel_probes"]
                target_hansen = targets["hansen_sol"].to(dev)
                if w.get("timp_disp_hansen", 0) > 0 and mask.sum() > 0:
                    pred_delta_d = probes["delta_d"][mask]
                    true_delta_d = target_hansen[mask, 0]
                    losses["timp_disp_hansen"] = (
                        ((pred_delta_d - true_delta_d) / self.S_hansen).pow(2).mean()
                    )
                if w.get("timp_polar_hansen", 0) > 0 and mask.sum() > 0:
                    pred_delta_p = probes["delta_p"][mask]
                    pred_delta_h = probes["delta_h"][mask]
                    true_delta_p = target_hansen[mask, 1]
                    true_delta_h = target_hansen[mask, 2]
                    losses["timp_polar_hansen"] = (
                        ((pred_delta_p - true_delta_p) / self.S_hansen).pow(2).mean()
                        + ((pred_delta_h - true_delta_h) / self.S_hansen).pow(2).mean()
                    )

        if (
            w.get("hansen_delta", 0) > 0
            and isinstance(output.get("hansen_sol"), Tensor)
            and isinstance(output.get("hansen_slv"), Tensor)
        ):
            target_sol = targets.get(
                "hansen_sol_effective",
                targets.get("hansen_sol"),
            )
            target_slv = targets.get(
                "hansen_slv_effective",
                targets.get("hansen_slv"),
            )
            pair_mask = targets.get("pair_hansen_mask")
            if (
                isinstance(target_sol, Tensor)
                and isinstance(target_slv, Tensor)
                and isinstance(pair_mask, Tensor)
            ):
                mask = pair_mask.to(dev).bool()
                if mask.sum() > 0:
                    pred_delta = output["hansen_sol"] - output["hansen_slv"]
                    true_delta = target_sol.to(dev) - target_slv.to(dev)
                    per_sample = (
                        (pred_delta[mask] - true_delta[mask]) / self.S_hansen
                    ).pow(2).mean(dim=-1)
                    pair_weight = targets.get("pair_hansen_weight")
                    sample_weight = (
                        pair_weight.to(dev)[mask].clamp_min(0.0)
                        if isinstance(pair_weight, Tensor)
                        else None
                    )
                    if sample_weight is not None and sample_weight.sum() > 0:
                        losses["hansen_delta"] = self.weighted_mean(
                            per_sample,
                            weight=sample_weight,
                        )
                    else:
                        losses["hansen_delta"] = per_sample.mean()

        # ============================================================
        # 5. Infinite dilution activity coefficient
        # ============================================================
        if "ln_gamma_inf" in targets and targets.get("gamma_mask") is not None:
            if w.get("gamma_inf", 0) > 0:
                mask = targets["gamma_mask"].to(dev)
                if mask.sum() > 0:
                    gamma_weight = targets.get("gamma_weight")
                    sample_weight = (
                        gamma_weight.to(dev)[mask]
                        if isinstance(gamma_weight, Tensor)
                        else None
                    )
                    losses["gamma_inf"] = self.weighted_mean(
                        (
                            output["physics"]["ln_gamma_inf"][mask]
                            - targets["ln_gamma_inf"].to(dev)[mask]
                        ).pow(2),
                        weight=sample_weight,
                    )

        if "ln_gamma_2_target" in targets and targets.get("gamma2_mask") is not None:
            if w.get("gamma_2", 0) > 0:
                gamma2_pred = physics.get("ln_gamma_2")
                mask = targets["gamma2_mask"].to(dev)
                if isinstance(gamma2_pred, Tensor) and mask.sum() > 0:
                    gamma2_weight = targets.get("gamma2_weight")
                    sample_weight = (
                        gamma2_weight.to(dev)[mask]
                        if isinstance(gamma2_weight, Tensor)
                        else None
                    )
                    losses["gamma_2"] = self.weighted_mean(
                        (
                            gamma2_pred[mask]
                            - targets["ln_gamma_2_target"].to(dev)[mask]
                        ).pow(2),
                        weight=sample_weight,
                    )

        # ============================================================
        # 6. Monotonicity penalty (dx₂/dT ≥ 0)
        # ============================================================
        if compute_mono and T is not None and model is not None:
            if w.get("mono", 0) > 0:
                losses["mono"] = self._monotonicity_loss(
                    model,
                    solute_data,
                    solvent_data,
                    T,
                    targets=targets,
                    solvent_type=solvent_type,
                    solute_morgan_fp=solute_morgan_fp,
                    solvent_morgan_fp=solvent_morgan_fp,
                    solute_descriptors=solute_descriptors,
                    solvent_descriptors=solvent_descriptors,
                    ionic_features=ionic_features,
                    solute_descriptor_prior_features=solute_descriptor_prior_features,
                    solvent_descriptor_prior_features=solvent_descriptor_prior_features,
                    solute_group_prior_features=solute_group_prior_features,
                    solvent_group_prior_features=solvent_group_prior_features,
                    T_m_gc=T_m_gc,
                    dH_fus_gc=dH_fus_gc,
                    dCp_fus_gc=dCp_fus_gc,
                    detach_crystal_from_encoder=detach_crystal_from_encoder,
                )

        # ============================================================
        # 7. Effective correction magnitude
        # ============================================================
        if "correction" in output:
            if w.get("res", 0) > 0:
                losses["res"] = output["correction"].pow(2).mean()

        # ============================================================
        # 8. Hansen-NRTL bridge consistency
        # ============================================================
        if ("hansen_sol" in output and "hansen_slv" in output
                and T is not None):
            if w.get("bridge", 0) > 0:
                losses["bridge"] = self._bridge_loss(output, T.to(dev))

        # ============================================================
        # 9. τ regularization
        # ============================================================
        if "tau_12" in physics:
            if w.get("tau_reg", 0) > 0:
                losses["tau_reg"] = (
                    physics["tau_12"].pow(2).mean()
                    + physics["tau_21"].pow(2).mean()
                )

        # ============================================================
        # 10. Physics preference: encourage high confidence
        # ============================================================
        if "confidence" in output:
            if w.get("phys_pref", 0) > 0:
                losses["phys_pref"] = (1.0 - output["confidence"]).mean()

        # ============================================================
        # 11. Direct path regularization
        # ============================================================
        if "ln_x2_direct" in output and "physics" in output:
            if w.get("direct_reg", 0) > 0:
                losses["direct_reg"] = (
                    output["ln_x2_direct"]
                    - output["physics"]["ln_x2"].detach()
                ).pow(2).mean()

        # ============================================================
        # 12. Crystal/activity decorrelation on jointly labeled rows
        # ============================================================
        if T is not None and w.get("decorr", 0) > 0:
            decorr_losses = self._crystal_activity_decorrelation_loss(
                output,
                targets,
                T=T,
                sol_mask=sol_mask,
                dev=dev,
            )
            losses["decorr"] = decorr_losses["decorr"]
            losses["decorr_corr"] = decorr_losses["decorr_corr"]
            losses["decorr_joint_rows"] = decorr_losses["decorr_joint_rows"]

        # ============================================================
        # 13. Same-pair temperature consistency
        # ============================================================
        pair_keys = targets.get("pair_key")
        if pair_keys is not None and T is not None:
            if (
                w.get("pair_temp_rank", 0) > 0
                or w.get("vant_hoff_local", 0) > 0
                or w.get("pair_temp_delta", 0) > 0
                or w.get("vant_hoff_slope", 0) > 0
                or w.get("vant_hoff_intercept", 0) > 0
            ):
                pair_losses = self._pair_temperature_losses(
                    output["ln_x2"],
                    T.to(dev),
                    pair_keys,
                    true_ln_x2=targets.get("ln_x2"),
                    sol_mask=sol_mask,
                    target_slope=targets.get("vh_fit_slope"),
                    target_intercept=targets.get("vh_fit_intercept"),
                    target_mask=targets.get("vh_fit_mask"),
                    target_weight=targets.get("vh_fit_weight"),
                    target_r2=targets.get("vh_fit_r2"),
                )
                if w.get("pair_temp_rank", 0) > 0:
                    losses["pair_temp_rank"] = pair_losses["pair_temp_rank"]
                if w.get("vant_hoff_local", 0) > 0:
                    losses["vant_hoff_local"] = pair_losses["vant_hoff_local"]
                if w.get("pair_temp_delta", 0) > 0:
                    losses["pair_temp_delta"] = pair_losses["pair_temp_delta"]
                if w.get("vant_hoff_slope", 0) > 0:
                    losses["vant_hoff_slope"] = pair_losses["vant_hoff_slope"]
                if w.get("vant_hoff_intercept", 0) > 0:
                    losses["vant_hoff_intercept"] = pair_losses["vant_hoff_intercept"]

        # ============================================================
        # 14. MoE balance (expert usage)
        # ============================================================
        moe_gate = output.get("moe_gate")
        if moe_gate is not None and w.get("moe_balance", 0) > 0:
            avg_gate = moe_gate.mean(dim=0)
            target = torch.full_like(
                avg_gate, 1.0 / max(avg_gate.numel(), 1)
            )
            losses["moe_balance"] = (avg_gate - target).pow(2).mean()

        descriptor_prior_reg = output.get("descriptor_prior_reg")
        if (
            isinstance(descriptor_prior_reg, Tensor)
            and w.get("descriptor_prior", 0) > 0
        ):
            losses["descriptor_prior"] = descriptor_prior_reg

        group_prior_reg = output.get("group_prior_reg")
        if (
            isinstance(group_prior_reg, Tensor)
            and w.get("group_prior", 0) > 0
        ):
            losses["group_prior"] = group_prior_reg

        if self.cfg.use_hansen_contrastive:
            representations = output.get("representations", {})
            if not isinstance(representations, dict):
                representations = {}
            g_sol = representations.get("g_sol_pre")
            if (
                isinstance(g_sol, Tensor)
                and w.get("hansen_contrastive_mol", 0) > 0
            ):
                hansen_target = targets.get(
                    "hansen_sol_effective",
                    targets.get("hansen_sol"),
                )
                hansen_mask = targets.get(
                    "hansen_contrastive_mask",
                    targets.get("hansen_mask"),
                )
                if isinstance(hansen_target, Tensor) and isinstance(hansen_mask, Tensor):
                    sample_weight = targets.get("hansen_sol_contrastive_weight")
                    losses["hansen_contrastive_mol"] = self.hansen_contrastive(
                        g_sol,
                        hansen_target.to(dev),
                        hansen_mask.to(dev),
                        sample_weight=(
                            sample_weight.to(dev)
                            if isinstance(sample_weight, Tensor)
                            else None
                        ),
                    )

            timp_channels = output.get("timp_channels", {})
            if not isinstance(timp_channels, dict):
                timp_channels = {}
            g_disp = timp_channels.get("solute_disp")
            g_polar = timp_channels.get("solute_polar")
            if w.get("hansen_contrastive_channel", 0) > 0:
                hansen_target = targets.get(
                    "hansen_sol_effective",
                    targets.get("hansen_sol"),
                )
                hansen_mask = targets.get(
                    "hansen_contrastive_mask",
                    targets.get("hansen_mask"),
                )
                if isinstance(hansen_target, Tensor) and isinstance(hansen_mask, Tensor):
                    sample_weight = targets.get("hansen_sol_contrastive_weight")
                    losses["hansen_contrastive_channel"] = (
                        self.channel_hansen_contrastive(
                            g_disp if isinstance(g_disp, Tensor) else None,
                            g_polar if isinstance(g_polar, Tensor) else None,
                            hansen_target.to(dev),
                            hansen_mask.to(dev),
                            sample_weight=(
                                sample_weight.to(dev)
                                if isinstance(sample_weight, Tensor)
                                else None
                            ),
                        )
                    )
            if (
                isinstance(g_disp, Tensor)
                and isinstance(g_polar, Tensor)
                and w.get("hansen_channel_orth", 0) > 0
            ):
                losses["hansen_channel_orth"] = channel_orthogonality_penalty(
                    g_disp,
                    g_polar,
                )

            g_pair = representations.get("g_pair")
            pair_Ra = targets.get("pair_Ra")
            pair_mask = targets.get("pair_hansen_mask")
            if (
                isinstance(g_pair, Tensor)
                and isinstance(pair_Ra, Tensor)
                and isinstance(pair_mask, Tensor)
                and w.get("hansen_contrastive_pair", 0) > 0
            ):
                pair_weight = targets.get("pair_hansen_weight")
                losses["hansen_contrastive_pair"] = self.pair_hansen_contrastive(
                    g_pair,
                    pair_Ra.to(dev),
                    pair_mask.to(dev),
                    sample_weight=(
                        pair_weight.to(dev)
                        if isinstance(pair_weight, Tensor)
                        else None
                    ),
                )

        walden_weight = w.get("walden", self.cfg.walden_weight)
        if self.cfg.use_walden_check and walden_weight > 0:
            losses["walden"] = self._walden_loss(output, targets, dev)

        # ============================================================
        # Weighted sum
        # ============================================================
        anchor = self._find_grad_anchor(output)
        if anchor is not None:
            total = anchor.sum() * 0.0
        else:
            total = torch.zeros((), device=dev)
        for key, val in losses.items():
            wt = w.get(key, 0.0)
            if wt > 0:
                total = total + wt * val
        if self.cfg.use_walden_check and walden_weight > 0:
            total = total + walden_weight * losses["walden"]

        return total, {k: v.item() for k, v in losses.items()}

    def _bridge_loss(self, output: dict[str, object], T: Tensor) -> Tensor:
        """Consistency: Hansen-estimated γ∞ ≈ NRTL γ∞."""
        h_sol = output["hansen_sol"]
        h_slv = output["hansen_slv"]
        diff_sq = (h_sol - h_slv).pow(2).sum(dim=-1)

        V2 = output.get("aux_sol", {}).get(
            "V_m", torch.full_like(T, 100.0)
        )
        lng_hansen = V2 * diff_sq / (self.cfg.R * T)
        lng_nrtl = output["physics"]["ln_gamma_inf"]

        return ((lng_nrtl - lng_hansen) / self.S_bridge).pow(2).mean()

    def _walden_loss(
        self,
        output: dict[str, object],
        targets: dict[str, object],
        dev: torch.device,
    ) -> Tensor:
        """Penalize implausible fusion entropy only on unsupervised samples."""
        fusion_params = output.get("fusion_params", {})
        T_m_pred = fusion_params.get("T_m")
        dH_fus_pred = fusion_params.get("dH_fus")
        dS_fus_pred = fusion_params.get("dS_fus")
        if (
            not isinstance(T_m_pred, Tensor)
            or not isinstance(dH_fus_pred, Tensor)
        ):
            return torch.zeros((), device=dev)

        T_m_mask = targets.get("T_m_mask")
        if T_m_mask is None:
            T_m_mask = targets.get("has_T_m")
        dH_mask = targets.get("dH_mask")
        if dH_mask is None:
            dH_mask = targets.get("has_dH_fus")

        if isinstance(T_m_mask, Tensor) and isinstance(dH_mask, Tensor):
            mask_unsup = ~(T_m_mask.to(dev).bool() & dH_mask.to(dev).bool())
        else:
            mask_unsup = torch.ones_like(T_m_pred, dtype=torch.bool, device=dev)

        if not mask_unsup.any():
            return torch.zeros((), device=dev)

        if isinstance(dS_fus_pred, Tensor):
            dS_fus = dS_fus_pred.to(device=dev, dtype=T_m_pred.dtype)
        else:
            dS_fus = dH_fus_pred / T_m_pred.clamp_min(self.cfg.eps)
        deviation = (dS_fus - self.cfg.walden_target).abs()
        walden_penalty = (
            torch.clamp(deviation - self.cfg.walden_tolerance, min=0.0) ** 2
        )
        mask_unsup_f = mask_unsup.float()
        return (walden_penalty * mask_unsup_f).sum() / mask_unsup_f.sum().clamp_min(1.0)

    def _pair_temperature_losses(
        self,
        pred_ln_x2: Tensor,
        T: Tensor,
        pair_keys: list[str] | tuple[str, ...],
        true_ln_x2: object | None = None,
        sol_mask: Tensor | None = None,
        target_slope: object | None = None,
        target_intercept: object | None = None,
        target_mask: object | None = None,
        target_weight: object | None = None,
        target_r2: object | None = None,
    ) -> dict[str, Tensor]:
        """Same-pair temperature ranking and Van't Hoff curve consistency."""
        groups: dict[str, list[int]] = {}
        for idx, key in enumerate(pair_keys):
            groups.setdefault(str(key), []).append(idx)

        rank_losses = []
        vant_hoff_losses = []
        delta_losses = []
        slope_losses = []
        intercept_losses = []
        true_tensor = true_ln_x2 if isinstance(true_ln_x2, Tensor) else None
        usable_mask = sol_mask.bool() if isinstance(sol_mask, Tensor) else None
        slope_target_tensor = target_slope if isinstance(target_slope, Tensor) else None
        intercept_target_tensor = (
            target_intercept if isinstance(target_intercept, Tensor) else None
        )
        target_mask_tensor = target_mask.bool() if isinstance(target_mask, Tensor) else None
        target_weight_tensor = target_weight if isinstance(target_weight, Tensor) else None
        target_r2_tensor = target_r2 if isinstance(target_r2, Tensor) else None

        for raw_indices in groups.values():
            indices = list(raw_indices)
            if usable_mask is not None:
                indices = [
                    idx for idx in indices
                    if bool(usable_mask[idx].detach().cpu().item())
                ]
            if len(indices) >= 2:
                idx_tensor = torch.tensor(indices, device=pred_ln_x2.device, dtype=torch.long)
                T_group = T[idx_tensor]
                pred_group = pred_ln_x2[idx_tensor]
                order = torch.argsort(T_group)
                T_sorted = T_group[order]
                pred_sorted = pred_group[order]

                delta_pred = pred_sorted[1:] - pred_sorted[:-1]
                rank_losses.append(F.relu(-delta_pred).mean())

                if len(indices) >= 3:
                    inv_T = 1.0 / T_sorted.clamp(min=self.cfg.eps)
                    inv_T_diff = inv_T[1:] - inv_T[:-1]
                    inv_T_sign = torch.where(
                        inv_T_diff < 0,
                        -torch.ones_like(inv_T_diff),
                        torch.ones_like(inv_T_diff),
                    )
                    inv_T_diff_safe = inv_T_sign * torch.clamp(
                        inv_T_diff.abs(),
                        min=MIN_VANT_HOFF_INV_T_DIFF,
                    )
                    local_slopes = (
                        pred_sorted[1:] - pred_sorted[:-1]
                    ) / inv_T_diff_safe
                    per_pair_vh = (local_slopes[1:] - local_slopes[:-1]).pow(2)
                    per_pair_vh = torch.clamp(
                        per_pair_vh,
                        max=MAX_VANT_HOFF_PAIR_LOSS,
                    )
                    vant_hoff_losses.append(per_pair_vh.mean())

                if true_tensor is not None:
                    true_group = true_tensor.to(pred_ln_x2.device, dtype=pred_ln_x2.dtype)[
                        idx_tensor
                    ][order]
                    true_delta = true_group[1:] - true_group[:-1]
                    delta_losses.append(F.huber_loss(delta_pred, true_delta, delta=1.0))

                    inv_T = 1.0 / T_sorted.clamp(min=self.cfg.eps)
                    x_center = inv_T - inv_T.mean()
                    denom = x_center.pow(2).sum().clamp_min(self.cfg.eps)
                    pred_center = pred_sorted - pred_sorted.mean()
                    true_center = true_group - true_group.mean()
                    pred_slope = (x_center * pred_center).sum() / denom
                    true_slope = (x_center * true_center).sum() / denom
                    pred_intercept = pred_sorted.mean() - pred_slope * inv_T.mean()
                    true_intercept = true_group.mean() - true_slope * inv_T.mean()

                    slope_scale = max(float(self.cfg.vant_hoff_slope_scale), self.cfg.eps)
                    intercept_scale = max(
                        float(self.cfg.vant_hoff_intercept_scale),
                        self.cfg.eps,
                    )
                    slope_losses.append(((pred_slope - true_slope) / slope_scale).pow(2))
                    intercept_losses.append(
                        ((pred_intercept - true_intercept) / intercept_scale).pow(2)
                    )

            if slope_target_tensor is None and intercept_target_tensor is None:
                continue
            if target_mask_tensor is None:
                continue

            fit_indices = [
                idx for idx in raw_indices
                if bool(target_mask_tensor[idx].detach().cpu().item())
            ]
            if len(fit_indices) < 2:
                continue

            fit_idx_tensor = torch.tensor(
                fit_indices,
                device=pred_ln_x2.device,
                dtype=torch.long,
            )
            if target_r2_tensor is not None:
                fit_r2 = target_r2_tensor.to(
                    pred_ln_x2.device,
                    dtype=pred_ln_x2.dtype,
                )[fit_idx_tensor]
                if bool((fit_r2 < float(self.cfg.vant_hoff_fit_r2_min)).all().item()):
                    continue

            T_fit = T[fit_idx_tensor]
            pred_fit = pred_ln_x2[fit_idx_tensor]
            fit_order = torch.argsort(T_fit)
            T_fit = T_fit[fit_order]
            pred_fit = pred_fit[fit_order]
            inv_T_fit = 1.0 / T_fit.clamp(min=self.cfg.eps)
            x_fit_center = inv_T_fit - inv_T_fit.mean()
            fit_denom = x_fit_center.pow(2).sum().clamp_min(self.cfg.eps)
            pred_fit_center = pred_fit - pred_fit.mean()
            pred_fit_slope = (x_fit_center * pred_fit_center).sum() / fit_denom
            pred_fit_intercept = pred_fit.mean() - pred_fit_slope * inv_T_fit.mean()

            fit_weight = pred_ln_x2.new_tensor(1.0)
            if target_weight_tensor is not None:
                fit_weight = target_weight_tensor.to(
                    pred_ln_x2.device,
                    dtype=pred_ln_x2.dtype,
                )[fit_idx_tensor].mean().clamp_min(self.cfg.eps)

            slope_scale = max(float(self.cfg.vant_hoff_slope_scale), self.cfg.eps)
            intercept_scale = max(
                float(self.cfg.vant_hoff_intercept_scale),
                self.cfg.eps,
            )
            if slope_target_tensor is not None:
                slope_target = slope_target_tensor.to(
                    pred_ln_x2.device,
                    dtype=pred_ln_x2.dtype,
                )[fit_idx_tensor].mean()
                slope_losses.append(
                    fit_weight * ((pred_fit_slope - slope_target) / slope_scale).pow(2)
                )
            if intercept_target_tensor is not None:
                intercept_target = intercept_target_tensor.to(
                    pred_ln_x2.device,
                    dtype=pred_ln_x2.dtype,
                )[fit_idx_tensor].mean()
                intercept_losses.append(
                    fit_weight
                    * ((pred_fit_intercept - intercept_target) / intercept_scale).pow(2)
                )

        zero = pred_ln_x2.new_zeros(())
        return {
            "pair_temp_rank": torch.stack(rank_losses).mean() if rank_losses else zero,
            "vant_hoff_local": (
                torch.stack(vant_hoff_losses).mean() if vant_hoff_losses else zero
            ),
            "pair_temp_delta": (
                torch.stack(delta_losses).mean() if delta_losses else zero
            ),
            "vant_hoff_slope": (
                torch.stack(slope_losses).mean() if slope_losses else zero
            ),
            "vant_hoff_intercept": (
                torch.stack(intercept_losses).mean() if intercept_losses else zero
            ),
        }

    def _monotonicity_loss(
        self,
        model: TGNNSolv,
        solute_data: object,
        solvent_data: object,
        T: Tensor,
        targets: dict[str, object] | None = None,
        solvent_type: Tensor | None = None,
        solute_morgan_fp: Tensor | None = None,
        solvent_morgan_fp: Tensor | None = None,
        solute_descriptors: Tensor | None = None,
        solvent_descriptors: Tensor | None = None,
        ionic_features: Tensor | None = None,
        solute_descriptor_prior_features: Tensor | None = None,
        solvent_descriptor_prior_features: Tensor | None = None,
        solute_group_prior_features: Tensor | None = None,
        solvent_group_prior_features: Tensor | None = None,
        T_m_gc: Tensor | None = None,
        dH_fus_gc: Tensor | None = None,
        dCp_fus_gc: Tensor | None = None,
        detach_crystal_from_encoder: bool = False,
    ) -> Tensor:
        """Penalize dx₂/dT < 0 using the configured solver path."""
        T_var = T.detach().requires_grad_(True)
        saved = model.cfg.use_implicit_diff
        if model.cfg.monotonicity_force_explicit:
            model.cfg.use_implicit_diff = False

        try:
            with torch.enable_grad():
                out = model(
                    solute_data,
                    solvent_data,
                    T_var,
                    solvent_type=solvent_type,
                    solute_morgan_fp=solute_morgan_fp,
                    solvent_morgan_fp=solvent_morgan_fp,
                    solute_descriptors=solute_descriptors,
                    solvent_descriptors=solvent_descriptors,
                    ionic_features=ionic_features,
                    solute_descriptor_prior_features=solute_descriptor_prior_features,
                    solvent_descriptor_prior_features=solvent_descriptor_prior_features,
                    solute_group_prior_features=solute_group_prior_features,
                    solvent_group_prior_features=solvent_group_prior_features,
                    T_m_gc=T_m_gc,
                    dH_fus_gc=dH_fus_gc,
                    dCp_fus_gc=dCp_fus_gc,
                    targets=targets,
                    detach_crystal_from_encoder=detach_crystal_from_encoder,
                )
                d_lnx2_dT = torch.autograd.grad(
                    out["ln_x2"].sum(),
                    T_var,
                    create_graph=False,
                    retain_graph=False,
                )[0]
        finally:
            model.cfg.use_implicit_diff = saved

        violation = F.relu(-d_lnx2_dT)
        return violation.mean()
