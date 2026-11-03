"""
Ablation study for TGNN-Solv.

Systematically removes or modifies one component at a time
to measure its contribution.  Each ablation is a (name, config_fn,
train_fn) triple that trains a full model and evaluates on test.

Ablations:
  full              — complete TGNN-Solv (reference)
  split_late_encoder — shared early GNN + role-specific late GNN layers
  no_cross_attn     — remove cross-attention layers
  no_nrtl           — remove NRTL, predict ln(x₂) from Φ + MLP
  no_curriculum     — train all losses from epoch 0 (no phases)
  no_aux_losses     — only solubility loss, no T_m/ΔH/Hansen/γ∞
  no_correction     — remove gated residual correction
  no_implicit_diff  — use explicit successive substitution only
  small_model       — hidden_dim=128 (scaling)
  large_model       — hidden_dim=512 (scaling)

Usage::

    from tgnn_solv.ablation import run_ablation_study
    results = run_ablation_study(train_loader, val_loader, test_loader)
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TypeAlias

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .config import TGNNSolvConfig
from .data.solvent_types import SOLVENT_TYPE_OTHER_ID
from .evaluate import Evaluator
from .layers import (
    build_batch_from_lists,
    make_temperature_features,
    pad_atom_features,
)
from .model import TGNNSolv
from .trainer import TGNNSolvTrainer

AblationResult: TypeAlias = dict[str, object]
AblationDefinition: TypeAlias = tuple[
    str, TGNNSolvConfig, type[TGNNSolv], type[TGNNSolvTrainer]
]


# ================================================================== #
#  Ablated model variants                                             #
# ================================================================== #

class TGNNSolvNoCrossAttn(TGNNSolv):
    """TGNN-Solv without cross-attention: solute and solvent
    are encoded independently."""

    def forward(
        self,
        solute_data: Batch,
        solvent_data: Batch,
        T: torch.Tensor,
        solvent_type: torch.Tensor | None = None,
    ) -> AblationResult:
        t_feat = make_temperature_features(T)
        encoder_t_feat = self._encoder_temp_features(t_feat)
        # Encode without cross-attention
        h_sol_atoms, g_sol_pre_payload, sol_a_disp, sol_a_polar = self._encode_and_readout(
            solute_data, "solute", temp_feat=encoder_t_feat
        )
        h_slv_atoms, g_slv_pre_payload, slv_a_disp, slv_a_polar = self._encode_and_readout(
            solvent_data, "solvent", temp_feat=encoder_t_feat
        )
        g_sol_pre = g_sol_pre_payload["value"]
        g_slv_pre = g_slv_pre_payload["value"]

        # Skip cross-attention entirely — readout with global tokens
        sol_list = self._append_global_token(
            self._split_atoms_by_graph(h_sol_atoms, solute_data.batch),
            self.sol_token,
        )
        slv_list = self._append_global_token(
            self._split_atoms_by_graph(h_slv_atoms, solvent_data.batch),
            self.slv_token,
        )
        sol_no_token = [h[:-1] for h in sol_list]
        slv_no_token = [h[:-1] for h in slv_list]
        sol_batch = build_batch_from_lists(
            sol_no_token, dtype=solute_data.batch.dtype
        )
        slv_batch = build_batch_from_lists(
            slv_no_token, dtype=solvent_data.batch.dtype
        )
        g_sol_post_payload = self._readout_payload(
            torch.cat(sol_no_token, dim=0),
            sol_batch,
            a_disp=sol_a_disp,
            a_polar=sol_a_polar,
        )
        g_slv_post_payload = self._readout_payload(
            torch.cat(slv_no_token, dim=0),
            slv_batch,
            a_disp=slv_a_disp,
            a_polar=slv_a_polar,
        )
        g_sol_post = g_sol_post_payload["value"]
        g_slv_post = g_slv_post_payload["value"]
        g_sol_tok = torch.stack([h[-1] for h in sol_list], dim=0)
        g_slv_tok = torch.stack([h[-1] for h in slv_list], dim=0)
        g_sol_post = g_sol_post + self.sol_token_gate * self.token_proj(g_sol_tok)
        g_slv_post = g_slv_post + self.slv_token_gate * self.token_proj(g_slv_tok)

        # Auxiliary heads
        hansen_sol = self.head_hansen(g_sol_pre)
        hansen_slv = self.head_hansen(g_slv_pre)
        aux_sol = self.head_aux(g_sol_pre)
        aux_slv = self.head_aux(g_slv_pre)

        # Pair representation
        g_sol_post_payload["value"] = g_sol_post
        g_slv_post_payload["value"] = g_slv_post
        g_pair = self._build_pair_representation(
            g_sol_post_payload,
            g_slv_post_payload,
        )
        moe_gate = None
        if self.solvent_moe is not None:
            if solvent_type is None:
                solvent_type = torch.full(
                    (g_pair.shape[0],),
                    SOLVENT_TYPE_OTHER_ID,
                    device=g_pair.device,
                    dtype=torch.long,
                )
            else:
                solvent_type = solvent_type.to(g_pair.device)
            g_pair, moe_gate = self.solvent_moe(g_pair, solvent_type)

        # Prediction heads
        fusion_params = self.head_fusion(g_sol_pre)
        nrtl_params = self.head_nrtl(
            g_pair,
            temp_feat=self._nrtl_temp_features(t_feat),
        )

        # SLE solver
        with torch.amp.autocast(device_type="cpu", enabled=False):
            T_f32 = T.float()
            fus_f32 = {k: v.float() for k, v in fusion_params.items()}
            nrtl_f32 = {k: v.float() for k, v in nrtl_params.items()}
            physics_out = self.sle_solver(T_f32, fus_f32, nrtl_f32)

        Ra = self.sle_solver.hansen_layer(hansen_sol, hansen_slv)

        param_summary = self._build_param_summary(
            fusion_params=fusion_params,
            nrtl_params=nrtl_params,
            physics_out=physics_out,
            temp_feat=t_feat,
            Ra=Ra,
            dtype=T.dtype,
        )

        confidence, param_deltas, proposal_log_sigma = self.correction(
            g_pair,
            param_summary,
        )
        corrected_fusion_params = self._build_corrected_fusion_params(
            fusion_params,
            param_deltas,
        )
        corrected_nrtl_state = self._build_corrected_nrtl_state(
            nrtl_params=nrtl_params,
            physics_out=physics_out,
            param_deltas=param_deltas,
        )
        with torch.amp.autocast(device_type="cpu", enabled=False):
            proposal_out = self.sle_solver(
                T.float(),
                {k: v.float() for k, v in corrected_fusion_params.items()},
                {k: v.float() for k, v in corrected_nrtl_state.items()},
                use_implicit=False,
            )

        raw_residual = proposal_out["ln_x2"].to(T.dtype) - physics_out["ln_x2"]
        bounded_residual = raw_residual.clamp(
            min=-self.cfg.correction_max_abs,
            max=self.cfg.correction_max_abs,
        )
        ln_x2_direct = physics_out["ln_x2"] + bounded_residual
        ln_x2_direct_log_sigma = proposal_log_sigma
        ln_x2 = physics_out["ln_x2"] + (1.0 - confidence) * bounded_residual
        correction = ln_x2 - physics_out["ln_x2"]
        gate = confidence.mean()

        return {
            "ln_x2": ln_x2,
            "x2": torch.exp(ln_x2).clamp(0, 1),
            "physics": physics_out,
            "proposal_physics": proposal_out,
            "fusion_params": fusion_params,
            "corrected_fusion_params": corrected_fusion_params,
            "nrtl_params": nrtl_params,
            "corrected_nrtl_state": corrected_nrtl_state,
            "hansen_sol": hansen_sol,
            "hansen_slv": hansen_slv,
            "aux_sol": aux_sol,
            "aux_slv": aux_slv,
            "Ra": Ra,
            "confidence": confidence,
            "ln_x2_direct": ln_x2_direct,
            "ln_x2_direct_log_sigma": ln_x2_direct_log_sigma,
            "ln_x2_direct_sigma": ln_x2_direct_log_sigma.exp(),
            "correction": correction,
            "gate": gate,
            "moe_gate": moe_gate,
            "attn_maps": [],
        }


class TGNNSolvNoNRTL(TGNNSolv):
    """TGNN-Solv without NRTL: uses ideal solubility + learned
    correction only.  γ₂ is always 1."""

    def forward(
        self,
        solute_data: Batch,
        solvent_data: Batch,
        T: torch.Tensor,
        solvent_type: torch.Tensor | None = None,
    ) -> AblationResult:
        t_feat = make_temperature_features(T)
        encoder_t_feat = self._encoder_temp_features(t_feat)
        interaction_t_feat = self._interaction_temp_features(t_feat)
        # ---- 1. Encode both molecules ----
        h_sol_atoms, g_sol_pre_payload, sol_a_disp, sol_a_polar = self._encode_and_readout(
            solute_data, "solute", temp_feat=encoder_t_feat
        )
        h_slv_atoms, g_slv_pre_payload, slv_a_disp, slv_a_polar = self._encode_and_readout(
            solvent_data, "solvent", temp_feat=encoder_t_feat
        )
        g_sol_pre = g_sol_pre_payload["value"]
        g_slv_pre = g_slv_pre_payload["value"]

        # ---- 2. Auxiliary heads (before cross-attention) ----
        hansen_sol = self.head_hansen(g_sol_pre)
        hansen_slv = self.head_hansen(g_slv_pre)
        aux_sol = self.head_aux(g_sol_pre)
        aux_slv = self.head_aux(g_slv_pre)

        # ---- 3. Cross-attention / Bipartite MP ----
        sol_atoms_list = self._split_atoms_by_graph(
            h_sol_atoms, solute_data.batch
        )
        slv_atoms_list = self._split_atoms_by_graph(
            h_slv_atoms, solvent_data.batch
        )

        sol_atoms_list = self._append_global_token(
            sol_atoms_list, self.sol_token
        )
        slv_atoms_list = self._append_global_token(
            slv_atoms_list, self.slv_token
        )
        sol_lengths = [h.shape[0] for h in sol_atoms_list]
        slv_lengths = [h.shape[0] for h in slv_atoms_list]

        h_sol_padded, sol_mask = pad_atom_features(sol_atoms_list)
        h_slv_padded, slv_mask = pad_atom_features(slv_atoms_list)

        attn_maps = []
        if self.interaction_mode == "cross_attn":
            for cross_layer in self.cross_attn_layers:
                sol_prev = h_sol_padded
                slv_prev = h_slv_padded
                h_sol_padded, attn_w = cross_layer(
                    sol_prev,
                    slv_prev,
                    sol_mask,
                    slv_mask,
                    interaction_t_feat,
                )
                attn_maps.append(attn_w.detach())
                h_slv_padded, _ = cross_layer(
                    slv_prev,
                    sol_prev,
                    slv_mask,
                    sol_mask,
                    interaction_t_feat,
                )
        else:
            for mp_layer in self.bipartite_layers:
                h_sol_padded, h_slv_padded = mp_layer(
                    h_sol_padded,
                    h_slv_padded,
                    sol_mask,
                    slv_mask,
                    interaction_t_feat,
                )

        # ---- 4. Post-cross-attention readout for solute ----
        sol_no_token = [
            h_sol_padded[i, :h.shape[0] - 1, :]
            for i, h in enumerate(sol_atoms_list)
        ]
        slv_no_token = [
            h_slv_padded[i, :h.shape[0] - 1, :]
            for i, h in enumerate(slv_atoms_list)
        ]
        sol_batch = build_batch_from_lists(
            sol_no_token, dtype=solute_data.batch.dtype
        )
        slv_batch = build_batch_from_lists(
            slv_no_token, dtype=solvent_data.batch.dtype
        )
        g_sol_post_payload = self._readout_payload(
            torch.cat(sol_no_token, dim=0),
            sol_batch,
            a_disp=sol_a_disp,
            a_polar=sol_a_polar,
        )
        g_slv_post_payload = self._readout_payload(
            torch.cat(slv_no_token, dim=0),
            slv_batch,
            a_disp=slv_a_disp,
            a_polar=slv_a_polar,
        )
        g_sol_post = g_sol_post_payload["value"]
        g_slv_post = g_slv_post_payload["value"]
        sol_idx = torch.tensor(
            [length - 1 for length in sol_lengths],
            device=h_sol_padded.device,
        )
        slv_idx = torch.tensor(
            [length - 1 for length in slv_lengths],
            device=h_slv_padded.device,
        )
        g_sol_tok = h_sol_padded[
            torch.arange(h_sol_padded.size(0), device=h_sol_padded.device),
            sol_idx,
        ]
        g_slv_tok = h_slv_padded[
            torch.arange(h_slv_padded.size(0), device=h_slv_padded.device),
            slv_idx,
        ]
        g_sol_post = g_sol_post + self.sol_token_gate * self.token_proj(g_sol_tok)
        g_slv_post = g_slv_post + self.slv_token_gate * self.token_proj(g_slv_tok)

        # ---- 5. Pair representation ----
        g_sol_post_payload["value"] = g_sol_post
        g_slv_post_payload["value"] = g_slv_post
        g_pair = self._build_pair_representation(
            g_sol_post_payload,
            g_slv_post_payload,
        )
        moe_gate = None
        if self.solvent_moe is not None:
            if solvent_type is None:
                solvent_type = torch.full(
                    (g_pair.shape[0],),
                    SOLVENT_TYPE_OTHER_ID,
                    device=g_pair.device,
                    dtype=torch.long,
                )
            else:
                solvent_type = solvent_type.to(g_pair.device)
            g_pair, moe_gate = self.solvent_moe(g_pair, solvent_type)

        # ---- 6. Fusion head ----
        fusion_params = self.head_fusion(g_sol_pre)

        # ---- 7. Ideal solubility only (γ₂ = 1) ----
        with torch.amp.autocast(device_type="cpu", enabled=False):
            T_f32 = T.float()
            fus_f32 = {k: v.float() for k, v in fusion_params.items()}
            Phi = self.sle_solver.ideal_layer(
                T_f32,
                fus_f32["T_m"],
                fus_f32["dH_fus"],
                fus_f32["dCp_fus"],
            )

        ln_x2_physics = -Phi
        x2 = torch.exp(ln_x2_physics).clamp(0, 1)

        zeros = torch.zeros_like(T)
        ones = torch.ones_like(T)
        nrtl_params = {
            "tau_a12": zeros,
            "tau_b12": zeros,
            "tau_c12": zeros,
            "tau_a21": zeros,
            "tau_b21": zeros,
            "tau_c21": zeros,
            "alpha_12": torch.full_like(T, 0.3),
        }

        physics_out = {
            "x2": x2,
            "ln_x2": ln_x2_physics,
            "ln_gamma_2": zeros,
            "ln_gamma_inf": zeros,
            "Phi": Phi,
            "x_ideal": x2.clamp(0, 1),
            "tau_12": zeros,
            "tau_21": zeros,
            "G_12": ones,
            "G_21": ones,
        }

        # ---- 8. Hansen distance ----
        Ra = self.sle_solver.hansen_layer(hansen_sol, hansen_slv)

        # ---- 9. Adaptive correction ----
        param_summary = self._build_param_summary(
            fusion_params=fusion_params,
            nrtl_params=nrtl_params,
            physics_out=physics_out,
            temp_feat=t_feat,
            Ra=Ra,
            dtype=T.dtype,
        )

        confidence, param_deltas, proposal_log_sigma = self.correction(
            g_pair,
            param_summary,
        )
        corrected_fusion_params = self._build_corrected_fusion_params(
            fusion_params,
            param_deltas,
        )
        corrected_nrtl_state = self._build_corrected_nrtl_state(
            nrtl_params=nrtl_params,
            physics_out=physics_out,
            param_deltas=param_deltas,
        )
        with torch.amp.autocast(device_type="cpu", enabled=False):
            proposal_out = self.sle_solver(
                T.float(),
                {k: v.float() for k, v in corrected_fusion_params.items()},
                {k: v.float() for k, v in corrected_nrtl_state.items()},
                use_implicit=False,
            )
        raw_residual = proposal_out["ln_x2"].to(T.dtype) - ln_x2_physics
        bounded_residual = raw_residual.clamp(
            min=-self.cfg.correction_max_abs,
            max=self.cfg.correction_max_abs,
        )
        ln_x2_direct = ln_x2_physics + bounded_residual
        ln_x2_direct_log_sigma = proposal_log_sigma
        ln_x2 = ln_x2_physics + (1.0 - confidence) * bounded_residual
        correction = ln_x2 - ln_x2_physics

        return {
            "ln_x2": ln_x2,
            "x2": torch.exp(ln_x2).clamp(0, 1),
            "physics": physics_out,
            "proposal_physics": proposal_out,
            "fusion_params": fusion_params,
            "corrected_fusion_params": corrected_fusion_params,
            "nrtl_params": nrtl_params,
            "corrected_nrtl_state": corrected_nrtl_state,
            "hansen_sol": hansen_sol,
            "hansen_slv": hansen_slv,
            "aux_sol": aux_sol,
            "aux_slv": aux_slv,
            "Ra": Ra,
            "confidence": confidence,
            "ln_x2_direct": ln_x2_direct,
            "ln_x2_direct_log_sigma": ln_x2_direct_log_sigma,
            "ln_x2_direct_sigma": ln_x2_direct_log_sigma.exp(),
            "correction": correction,
            "gate": confidence.mean(),
            "moe_gate": moe_gate,
            "attn_maps": attn_maps,
        }


class TGNNSolvNoCorrection(TGNNSolv):
    """TGNN-Solv without gated residual correction.
    ln(x₂) = physics only."""

    def forward(
        self,
        solute_data: Batch,
        solvent_data: Batch,
        T: torch.Tensor,
        solvent_type: torch.Tensor | None = None,
    ) -> AblationResult:
        out = super().forward(
            solute_data, solvent_data, T, solvent_type=solvent_type
        )

        # Remove correction
        out["ln_x2"] = out["physics"]["ln_x2"]
        out["x2"] = torch.exp(out["ln_x2"]).clamp(0, 1)
        out["correction"] = torch.zeros_like(out["correction"])
        out["gate"] = torch.tensor(0.0, device=T.device)

        return out


# ================================================================== #
#  Ablation trainer variants                                          #
# ================================================================== #

class NoCurriculumTrainer(TGNNSolvTrainer):
    """Train all losses from epoch 0 — no phase separation."""

    def train_full(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Single phase with all weights active from start
        self.phase_weights[1] = {
            "sol": 1.0, "T_m": 0.3, "dH": 0.3, "hansen": 0.2,
            "gamma_inf": 0.5, "mono": 0.1,
            "res": 0.05, "bridge": 0.05, "tau_reg": 0.001,
            "direct_nll": 0.1,
        }
        total_epochs = (
            self.cfg.epochs_phase1
            + self.cfg.epochs_phase2
            + self.cfg.epochs_phase3
        )
        # Unfreeze correction from start
        self._freeze_correction(False)
        self.train_phase(1, train_loader, val_loader, total_epochs)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"\nRestored best model (val MAE = {self.best_val_loss:.4f})")


class NoAuxLossTrainer(TGNNSolvTrainer):
    """Train with solubility loss only — no auxiliary targets."""

    def __init__(self, model: TGNNSolv, cfg: TGNNSolvConfig) -> None:
        super().__init__(model, cfg)
        # Override all phases: only solubility + minimal regularization
        no_aux = {
            "sol": 1.0, "T_m": 0.0, "dH": 0.0, "hansen": 0.0,
            "gamma_inf": 0.0, "mono": 0.0,
            "res": 0.05, "bridge": 0.0, "tau_reg": 0.001,
        }
        self.phase_weights = {1: no_aux, 2: no_aux, 3: no_aux}

    def train_full(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Skip phase 1 entirely (no aux targets to pretrain on)
        self._freeze_correction(True)
        total_epochs = (
            self.cfg.epochs_phase1
            + self.cfg.epochs_phase2
            + self.cfg.epochs_phase3
        )
        self.train_phase(2, train_loader, val_loader, total_epochs)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# ================================================================== #
#  Ablation definitions                                               #
# ================================================================== #

def _define_ablations(
    base_cfg: TGNNSolvConfig,
) -> list[AblationDefinition]:
    """
    Define all ablation experiments.

    Returns list of (name, config, model_class, trainer_class).
    """
    ablations = []

    # 1. Full model (reference)
    ablations.append((
        "full",
        base_cfg,
        TGNNSolv,
        TGNNSolvTrainer,
    ))

    # 2. Shared-vs-asymmetric late encoder comparison
    cfg_split_late = replace(
        base_cfg,
        encoder_role_mode="split_late",
        encoder_role_specific_layers=min(
            max(base_cfg.encoder_role_specific_layers, 1),
            max(base_cfg.n_gnn_layers - 1, 1),
        ),
    )
    ablations.append((
        "split_late_encoder",
        cfg_split_late,
        TGNNSolv,
        TGNNSolvTrainer,
    ))

    # 3. No cross-attention
    ablations.append((
        "no_cross_attn",
        base_cfg,
        TGNNSolvNoCrossAttn,
        TGNNSolvTrainer,
    ))

    # 4. No NRTL (ideal + correction only)
    ablations.append((
        "no_nrtl",
        base_cfg,
        TGNNSolvNoNRTL,
        TGNNSolvTrainer,
    ))

    # 5. No curriculum (all losses from start)
    ablations.append((
        "no_curriculum",
        base_cfg,
        TGNNSolv,
        NoCurriculumTrainer,
    ))

    # 6. No auxiliary losses
    ablations.append((
        "no_aux_losses",
        base_cfg,
        TGNNSolv,
        NoAuxLossTrainer,
    ))

    # 7. No correction
    ablations.append((
        "no_correction",
        base_cfg,
        TGNNSolvNoCorrection,
        TGNNSolvTrainer,
    ))

    # 8. No implicit differentiation
    cfg_no_impl = replace(base_cfg, use_implicit_diff=False)
    ablations.append((
        "no_implicit_diff",
        cfg_no_impl,
        TGNNSolv,
        TGNNSolvTrainer,
    ))

    # 9. Small model (scaling)
    cfg_small = replace(
        base_cfg,
        hidden_dim=128,
        pair_dim=256,
        n_gnn_layers=4,
        n_cross_attn_layers=2,
    )
    ablations.append((
        "small_128",
        cfg_small,
        TGNNSolv,
        TGNNSolvTrainer,
    ))

    # 10. Large model (scaling)
    cfg_large = replace(
        base_cfg,
        hidden_dim=512,
        pair_dim=1024,
        n_gnn_layers=8,
        n_cross_attn_layers=4,
    )
    ablations.append((
        "large_512",
        cfg_large,
        TGNNSolv,
        TGNNSolvTrainer,
    ))

    return ablations


# ================================================================== #
#  Runner                                                             #
# ================================================================== #

def run_single_ablation(
    name: str,
    cfg: TGNNSolvConfig,
    model_class: type[TGNNSolv],
    trainer_class: type[TGNNSolvTrainer],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    test_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> AblationResult:
    """
    Run one ablation experiment: build, train, evaluate.

    Returns dict with name, metrics, timing, param count.
    """
    print(f"\n{'=' * 60}")
    print(f"  ABLATION: {name}")
    print(f"{'=' * 60}")

    torch.manual_seed(seed)

    t0 = time.time()

    # Build model
    model = model_class(cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    trainer = trainer_class(model, cfg)
    trainer.train_full(train_loader, val_loader)

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.0f}s")

    # Evaluate
    evaluator = Evaluator(model, cfg)
    report = evaluator.evaluate(test_loader, test_df)
    metrics = report["overall"]

    print("\n  Results:")
    print(f"    MAE  = {metrics['mae']:.3f}")
    print(f"    RMSE = {metrics['rmse']:.3f}")
    print(f"    R²   = {metrics['r2']:.4f}")

    return {
        "name": name,
        "n_params": n_params,
        "train_time_s": train_time,
        **metrics,
    }


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    base_cfg: TGNNSolvConfig | None = None,
    test_df: pd.DataFrame | None = None,
    seeds: list[int] | None = None,
    skip: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run full ablation study.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoaders
    device : torch device
    base_cfg : reference config (default: TGNNSolvConfig())
    test_df : test DataFrame for stratified metrics
    seeds : list of random seeds (default [42])
    skip : list of ablation names to skip

    Returns
    -------
    DataFrame with one row per (ablation, seed), sorted by MAE.
    """
    if base_cfg is None:
        base_cfg = TGNNSolvConfig()
    if seeds is None:
        seeds = [42]
    if skip is None:
        skip = []

    ablations = _define_ablations(base_cfg)

    all_results = []

    for name, cfg, model_cls, trainer_cls in ablations:
        if name in skip:
            print(f"\n  Skipping: {name}")
            continue

        for seed in seeds:
            seed_label = f" (seed={seed})" if len(seeds) > 1 else ""
            result = run_single_ablation(
                name=f"{name}{seed_label}" if len(seeds) > 1 else name,
                cfg=cfg,
                model_class=model_cls,
                trainer_class=trainer_cls,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                test_df=test_df,
                seed=seed,
            )
            result["seed"] = seed
            result["ablation"] = name
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — SUMMARY")
    print("=" * 70)

    if len(seeds) > 1:
        # Aggregate over seeds
        summary = results_df.groupby("ablation").agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            n_params=("n_params", "first"),
        ).sort_values("mae_mean")

        print(f"\n  {'Ablation':20s} {'MAE':>14s} {'RMSE':>10s} "
              f"{'R²':>8s} {'Params':>10s}")
        print("  " + "-" * 66)
        for name, row in summary.iterrows():
            print(
                f"  {name:20s} "
                f"{row['mae_mean']:.3f}±{row['mae_std']:.3f}  "
                f"{row['rmse_mean']:8.3f}  "
                f"{row['r2_mean']:8.4f}  "
                f"{row['n_params']:10,.0f}"
            )
    else:
        results_sorted = results_df.sort_values("mae")
        print(f"\n  {'Ablation':20s} {'MAE':>8s} {'RMSE':>8s} "
              f"{'R²':>8s} {'Params':>10s} {'Time':>8s}")
        print("  " + "-" * 66)
        for _, row in results_sorted.iterrows():
            print(
                f"  {row['name']:20s} "
                f"{row['mae']:8.3f} "
                f"{row['rmse']:8.3f} "
                f"{row['r2']:8.4f} "
                f"{row['n_params']:10,d} "
                f"{row['train_time_s']:7.0f}s"
            )

    # Delta from full model
    full_mae = results_df[results_df["ablation"] == "full"]["mae"].mean()
    print(f"\n  Reference (full): MAE = {full_mae:.3f}")
    print(f"\n  {'Ablation':20s} {'ΔMAE':>10s} {'Verdict':>20s}")
    print("  " + "-" * 55)

    for name in results_df["ablation"].unique():
        abl_mae = results_df[results_df["ablation"] == name]["mae"].mean()
        delta = abl_mae - full_mae
        if name == "full":
            verdict = "(reference)"
        elif delta > 0.05:
            verdict = f"IMPORTANT (+{delta:.3f})"
        elif delta > 0.01:
            verdict = f"helpful (+{delta:.3f})"
        elif delta > -0.01:
            verdict = f"negligible ({delta:+.3f})"
        else:
            verdict = f"harmful ({delta:+.3f})"
        print(f"  {name:20s} {delta:+10.3f} {verdict:>20s}")

    return results_df
