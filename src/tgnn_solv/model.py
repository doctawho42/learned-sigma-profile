"""
TGNNSolv — main model assembling all components.

Key principle: the network predicts physical quantities and lets the
thermodynamic solver do the hard work. Temperature-dependent state enters
the NRTL / correction blocks explicitly instead of leaking into crystal
property heads.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .config import TGNNSolvConfig
from .data.solvent_types import SOLVENT_TYPE_COUNT, SOLVENT_TYPE_OTHER_ID
from .features import (
    DESCRIPTOR_Z_CLIP,
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    graph_feature_spec_from_config,
    validate_descriptor_normalization_stats,
)
from .heads import (
    AuxPropsHead,
    FusionHead,
    HansenHead,
    MorganFeatureAdapter,
    NRTLHead,
    PairRepresentation,
    SigmaProfileHead,
    SolventTypeMoE,
)
from .heads import AdaptivePhysicsCorrection
from .layers import (
    BipartiteMessagePassing,
    PhysicsAwareReadout,
    SoluteSolventCrossAttention,
    build_graph_encoder,
    make_temperature_features,
)
from .solver import SLESolver

# Molar volume unit conversion: 1 cm^3/mol = (1e24 / N_A) A^3 per molecule.
_CM3_PER_MOL_TO_A3 = 1.0e24 / 6.02214076e23  # ~= 1.660539


class TGNNSolv(nn.Module):
    """
    Thermodynamic Graph Neural Network for Solubility Prediction.

    Parameters
    ----------
    node_feat_dim : int
        Atom feature dimension (default from features.py).
    edge_feat_dim : int
        Bond feature dimension (default from features.py).
    cfg : TGNNSolvConfig
        Full model configuration.
    """

    def __init__(
        self,
        node_feat_dim: int | None = None,
        edge_feat_dim: int | None = None,
        cfg: TGNNSolvConfig = TGNNSolvConfig(),
    ) -> None:
        super().__init__()
        if cfg.use_descriptor_priors and cfg.use_group_priors:
            raise ValueError(
                "Descriptor priors and fixed group priors are mutually exclusive."
            )
        self.cfg = cfg
        feature_spec = graph_feature_spec_from_config(cfg)
        if node_feat_dim is None:
            node_feat_dim = feature_spec.node_dim
        if edge_feat_dim is None:
            edge_feat_dim = feature_spec.edge_dim
        self.node_feat_dim = int(node_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.is_timp = cfg.encoder_type == "timp"
        self.use_molecular_priors = (
            cfg.use_descriptor_priors or cfg.use_group_priors
        )
        F = cfg.hidden_dim

        # --- Encoder ---
        self.gnn = build_graph_encoder(
            self.node_feat_dim,
            self.edge_feat_dim,
            cfg,
        )

        # --- Interaction stack ---
        self.interaction_mode = cfg.interaction_mode
        self.cross_attn_layers = nn.ModuleList()
        self.bipartite_layers = nn.ModuleList()
        if cfg.interaction_mode == "cross_attn":
            self.cross_attn_layers = nn.ModuleList([
                SoluteSolventCrossAttention(
                    F,
                    cfg.n_attn_heads,
                    use_thermo_cross_attention=cfg.use_thermo_cross_attention,
                    thermo_cross_attention_beta_init=cfg.thermo_cross_attention_beta_init,
                )
                for _ in range(cfg.n_cross_attn_layers)
            ])
        elif cfg.interaction_mode == "bipartite":
            self.bipartite_layers = nn.ModuleList([
                BipartiteMessagePassing(F, dropout=cfg.dropout)
                for _ in range(cfg.n_cross_attn_layers)
            ])
        else:
            raise ValueError(
                f"Unknown interaction_mode: {cfg.interaction_mode}"
            )

        # --- Readout ---
        self.readout = PhysicsAwareReadout(
            F,
            set2set_steps=cfg.set2set_steps,
            channel_aware=self.is_timp,
            channel_dim=(F // 2 if self.is_timp else None),
        )
        D_r = self.readout.output_dim  # hidden_dim * 3

        # --- Pair representation ---
        self.pair_repr = PairRepresentation(D_r, cfg.pair_dim)
        self.timp_pair_repr_combined = None
        self.timp_pair_repr_disp = None
        self.timp_pair_repr_polar = None
        self.timp_pair_proj = None
        self.timp_disp_probe = None
        self.timp_polar_probe = None
        if self.is_timp:
            channel_dim = F // 2
            self.timp_pair_repr_combined = PairRepresentation(F * 3, cfg.pair_dim)
            self.timp_pair_repr_disp = PairRepresentation(channel_dim, cfg.pair_dim)
            self.timp_pair_repr_polar = PairRepresentation(channel_dim, cfg.pair_dim)
            self.timp_pair_proj = nn.Linear(cfg.pair_dim * 3, cfg.pair_dim)
            probe_hidden = max(channel_dim, 16)
            self.timp_disp_probe = nn.Sequential(
                nn.Linear(channel_dim, probe_hidden),
                nn.SiLU(),
                nn.Linear(probe_hidden, 1),
            )
            self.timp_polar_probe = nn.Sequential(
                nn.Linear(channel_dim, probe_hidden),
                nn.SiLU(),
                nn.Linear(probe_hidden, 2),
            )
        self.solvent_moe = None
        if cfg.use_solvent_moe:
            self.solvent_moe = SolventTypeMoE(
                cfg.pair_dim,
                SOLVENT_TYPE_COUNT,
                num_experts=cfg.solvent_moe_experts,
                type_emb_dim=cfg.solvent_type_emb_dim,
                hidden_dim=cfg.solvent_moe_hidden,
                dropout=cfg.dropout,
            )

        # --- Prediction heads ---
        self.head_fusion = FusionHead(D_r, cfg)
        self.fusion_head = self.head_fusion
        self.is_cosmo_sac = cfg.activity_model == "cosmo_sac"
        # Activity model: NRTL head on the pair vector, or a per-molecule
        # sigma-profile head feeding the differentiable COSMO-SAC layer.
        self.head_nrtl = None if self.is_cosmo_sac else NRTLHead(cfg.pair_dim, cfg)
        self.head_sigma = SigmaProfileHead(D_r, cfg) if self.is_cosmo_sac else None
        self.aux_direct_sol_head = None
        if cfg.use_aux_direct_sol_loss:
            self.aux_direct_sol_head = nn.Sequential(
                nn.Linear(cfg.pair_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )
        self.head_hansen = HansenHead(D_r, cfg)
        self.head_aux = AuxPropsHead(D_r, cfg)
        self.solute_fp_adapter = None
        self.solvent_fp_adapter = None
        self.fp_pre_scale = None
        self.fp_post_scale = None
        if cfg.use_morgan_features:
            self.solute_fp_adapter = MorganFeatureAdapter(
                cfg.morgan_n_bits,
                D_r,
                cfg.morgan_hidden_dim,
                cfg.dropout,
            )
            self.solvent_fp_adapter = MorganFeatureAdapter(
                cfg.morgan_n_bits,
                D_r,
                cfg.morgan_hidden_dim,
                cfg.dropout,
            )
            self.fp_pre_scale = nn.Parameter(torch.tensor(0.5))
            self.fp_post_scale = nn.Parameter(torch.tensor(0.5))
        self.descriptor_mlp = None
        self.pair_desc_proj = None
        self.ionic_feature_adapter = None
        self.ionic_pair_scale = None
        self.register_buffer("descriptor_mean", torch.empty(0))
        self.register_buffer("descriptor_std", torch.empty(0))
        if cfg.use_descriptor_augmentation:
            if cfg.descriptor_dim <= 0:
                raise ValueError(
                    "descriptor_dim must be positive when descriptor augmentation is enabled."
                )
            descriptor_hidden_dim = cfg.resolved_descriptor_hidden_dim
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(cfg.descriptor_dim, descriptor_hidden_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(descriptor_hidden_dim, descriptor_hidden_dim),
                nn.SiLU(),
            )
            self.pair_desc_proj = nn.Linear(
                cfg.pair_dim + 4 * descriptor_hidden_dim,
                cfg.pair_dim,
            )
            self.descriptor_mean = torch.zeros(cfg.descriptor_dim)
            self.descriptor_std = torch.ones(cfg.descriptor_dim)
        if cfg.use_ionic_features:
            if cfg.ionic_feature_dim <= 0:
                raise ValueError(
                    "ionic_feature_dim must be positive when ionic features are enabled."
                )
            self.ionic_feature_adapter = nn.Sequential(
                nn.Linear(cfg.ionic_feature_dim, cfg.ionic_feature_hidden_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.ionic_feature_hidden_dim, cfg.pair_dim),
            )
            self.ionic_pair_scale = nn.Parameter(torch.tensor(0.1))

        # --- Physics solver (0 learnable params) ---
        self.sle_solver = SLESolver(cfg)

        # --- Learned correction ---
        self.correction = AdaptivePhysicsCorrection(
            cfg.pair_dim,
            n_param_features=9,
            Tm_limit=cfg.correction_Tm_max_delta,
            dH_fraction_limit=cfg.correction_dH_fraction,
            tau_limit=cfg.correction_tau_max_delta,
        )

        # --- Global tokens for co-attention ---
        self.sol_token = nn.Parameter(torch.zeros(1, 1, F))
        self.slv_token = nn.Parameter(torch.zeros(1, 1, F))
        nn.init.normal_(self.sol_token, mean=0.0, std=0.02)
        nn.init.normal_(self.slv_token, mean=0.0, std=0.02)
        self.token_proj = nn.Linear(F, D_r)
        self.sol_token_gate = nn.Parameter(torch.tensor(0.0))
        self.slv_token_gate = nn.Parameter(torch.tensor(0.0))

    def _run_encoder(
        self,
        data: Batch,
        role: str = "solute",
        temp_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the configured encoder and normalize its return contract."""
        encoder_out = self.gnn(
            data.x,
            data.edge_index,
            data.edge_attr,
            role=role,
            batch=data.batch,
            temp_feat=temp_feat,
        )
        if isinstance(encoder_out, tuple) and len(encoder_out) == 3:
            h_atoms, a_disp, a_polar = encoder_out
            return h_atoms, a_disp, a_polar
        return encoder_out, None, None

    def _readout_payload(
        self,
        h_atoms: torch.Tensor,
        batch: torch.Tensor,
        *,
        a_disp: Optional[torch.Tensor] = None,
        a_polar: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Run graph pooling and always return a parts dict."""
        payload = self.readout(
            h_atoms,
            batch,
            a_disp=a_disp,
            a_polar=a_polar,
            return_parts=self.is_timp,
        )
        if isinstance(payload, dict):
            return payload
        return {
            "value": payload,
            "combined": payload,
            "disp": None,
            "polar": None,
        }

    def _encode_and_readout(
        self,
        data: Batch,
        role: str = "solute",
        temp_feat: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, Optional[torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Run encoder + readout on a batched graph."""
        h_atoms, a_disp, a_polar = self._run_encoder(
            data,
            role=role,
            temp_feat=temp_feat,
        )
        g_payload = self._readout_payload(
            h_atoms,
            data.batch,
            a_disp=a_disp,
            a_polar=a_polar,
        )
        return h_atoms, g_payload, a_disp, a_polar

    def _build_pair_representation(
        self,
        solute_payload: dict[str, Optional[torch.Tensor]],
        solvent_payload: dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Build the pair state, including TIMP channel fusion when enabled."""
        if (
            not self.is_timp
            or self.timp_pair_repr_combined is None
            or self.timp_pair_repr_disp is None
            or self.timp_pair_repr_polar is None
            or self.timp_pair_proj is None
            or solute_payload["disp"] is None
            or solute_payload["polar"] is None
            or solvent_payload["disp"] is None
            or solvent_payload["polar"] is None
        ):
            return self.pair_repr(
                solute_payload["value"],
                solvent_payload["value"],
            )

        g_pair_combined = self.timp_pair_repr_combined(
            solute_payload["combined"],
            solvent_payload["combined"],
        )
        g_pair_disp = self.timp_pair_repr_disp(
            solute_payload["disp"],
            solvent_payload["disp"],
        )
        g_pair_polar = self.timp_pair_repr_polar(
            solute_payload["polar"],
            solvent_payload["polar"],
        )
        return self.timp_pair_proj(
            torch.cat([g_pair_combined, g_pair_disp, g_pair_polar], dim=-1)
        )

    def _encoder_temp_features(
        self,
        temp_feat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return temperature features for the encoder path if enabled."""
        if self.cfg.use_temperature_in_encoder:
            return temp_feat
        return None

    def _interaction_temp_features(
        self,
        temp_feat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return temperature features for the interaction path if enabled."""
        if self.cfg.use_temperature_in_interaction:
            return temp_feat
        return None

    def _nrtl_temp_features(
        self,
        temp_feat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return temperature features for the NRTL head if enabled."""
        if self.cfg.use_temperature_in_nrtl_head:
            return temp_feat
        return None

    @staticmethod
    def _graph_count(data) -> int:
        """B без синхронизации хоста.

        ``batch.max().item()`` -- чтение значения тензора на CPU, то есть полный останов
        конвейера; ``ptr.numel()`` -- запрос формы, он бесплатен. Запасной путь оставлен для
        данных без ``ptr`` (не ``Batch``), где синхронизация неизбежна, как и раньше.
        """
        ptr = getattr(data, "ptr", None)
        if ptr is not None:
            return int(ptr.numel()) - 1
        return int(data.batch.max().item()) + 1

    def _to_dense_with_token(self, h_atoms: torch.Tensor, data, token: torch.Tensor):
        """Плоское (N,D) -> плотное (B, N_max+1, D) с глобальным токеном сразу за атомами.

        ЧТО ЭТО ЗАМЕНЯЕТ И ПОЧЕМУ. Раньше тот же результат собирался четырьмя питоновскими
        циклами по графам: ``_split_atoms_by_graph`` резал плоский тензор B булевыми
        индексациями, ``_append_global_token`` делал B вызовов ``cat``, ``pad_atom_features``
        присваивала B срезов в нулевой тензор, а ``build_batch_from_lists`` собирала вектор
        принадлежности из B вызовов ``full``. Профиль настоящего шага на T4 (2026-10-02,
        results/real_step_profile/summary.json) показал, что именно этот класс операций и
        составляет цену шага: 63% диспатчей растут РОВНО пропорционально батчу, то есть
        платятся за каждую молекулу отдельно -- ``lift_fresh`` 49.2 на граф, ``detach_`` 46.3,
        ``slice`` 19.5, ``Memcpy DtoD`` 14.5. Крупный батч такую цену не амортизирует по
        построению: батч x8 дал всего x1.39 строк в секунду, а загрузка карты даже упала с
        22% до 17%. Булева индексация вдобавок лоуэрится через ``nonzero`` (268 вызовов за
        шаг при батче 64) и копирует счётчик на хост -- отсюда часть из 531.5 синхронизаций.

        Возвращает ``(dense, atom_mask, full_mask, counts)``: плотный тензор, маску только по
        атомам, маску вместе с токеном и позиции токенов. ``atom_mask`` важна тем, что
        выбирает атомы в ИСХОДНОМ плоском порядке, поэтому обратный путь не восстанавливает
        вектор принадлежности графам -- он уже есть в ``data.batch``.
        """
        B = self._graph_count(data)
        dense, atom_mask = to_dense_batch(h_atoms, data.batch, batch_size=B)
        counts = atom_mask.sum(dim=1)                                  # (B,) атомов в графе
        # Один столбец под токен; его слот гарантированно нулевой (to_dense_batch заполняет
        # нулями, и дополненный столбец тоже), поэтому index_put записывает, а не смешивает.
        dense = torch.nn.functional.pad(dense, (0, 0, 0, 1))
        atom_mask = torch.nn.functional.pad(atom_mask, (0, 1))
        rows = torch.arange(B, device=dense.device)
        dense = dense.index_put(
            (rows, counts), token[0].to(dense).expand(B, -1)
        )
        full_mask = atom_mask.index_put(
            (rows, counts), torch.ones(B, dtype=atom_mask.dtype, device=atom_mask.device)
        )
        return dense, atom_mask, full_mask, counts

    @staticmethod
    def _gather_tokens(padded: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Глобальные токены из плотного тензора по их позициям.

        Заменяет ``_extract_tokens``, которая строила индекс через ``torch.tensor`` из
        питоновского списка длин -- то есть копировала B чисел с хоста на устройство каждый
        шаг. ``counts`` уже лежит на устройстве.
        """
        rows = torch.arange(padded.shape[0], device=padded.device)
        return padded[rows, counts]

    def _split_atoms_by_graph(
        self, h_atoms: torch.Tensor, batch: torch.Tensor
    ) -> List[torch.Tensor]:
        """Split flat atom tensor into per-graph list.

        Оставлено для ``ablation.py``, который им ещё пользуется; горячий путь
        ``forward`` перешёл на ``_to_dense_with_token``.
        """
        graphs = []
        for i in range(batch.max().item() + 1):
            mask = batch == i
            graphs.append(h_atoms[mask])
        return graphs

    def _append_global_token(
        self, atoms_list: List[torch.Tensor], token: torch.Tensor
    ) -> List[torch.Tensor]:
        token = token[0]  # (1, D)
        return [torch.cat([h, token.to(h)], dim=0) for h in atoms_list]

    def _slice_padded(
        self,
        padded: torch.Tensor,
        lengths: List[int],
        drop_last: bool = False,
    ) -> List[torch.Tensor]:
        out = []
        for i, length in enumerate(lengths):
            end = length - 1 if drop_last else length
            out.append(padded[i, :end, :])
        return out

    def _extract_tokens(
        self,
        padded: torch.Tensor,
        lengths: List[int],
    ) -> torch.Tensor:
        idx = torch.tensor(
            [length - 1 for length in lengths],
            device=padded.device,
        )
        return padded[torch.arange(padded.size(0), device=padded.device), idx]

    def _build_param_summary(
        self,
        *,
        fusion_params: dict[str, torch.Tensor],
        nrtl_params: dict[str, torch.Tensor],
        physics_out: dict[str, torch.Tensor],
        temp_feat: torch.Tensor,
        Ra: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a compact solver-facing summary for the residual corrector."""
        tau_12 = physics_out["tau_12"].to(dtype)
        tau_21 = physics_out["tau_21"].to(dtype)
        temp_feat = temp_feat.to(dtype)

        return torch.cat([
            ((fusion_params["T_m"] - 400.0) / 200.0).unsqueeze(-1),
            ((fusion_params["dH_fus"] - 20000.0) / 10000.0).unsqueeze(-1),
            (tau_12 / self.cfg.tau_clamp).unsqueeze(-1),
            (tau_21 / self.cfg.tau_clamp).unsqueeze(-1),
            ((nrtl_params["alpha_12"] - 0.3) / 0.15).unsqueeze(-1),
            temp_feat,
            (Ra.to(dtype) / 10.0).unsqueeze(-1),
        ], dim=-1)

    def _build_corrected_fusion_params(
        self,
        fusion_params: dict[str, torch.Tensor],
        param_deltas: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply bounded correction deltas to crystal-property parameters."""
        T_m_raw = fusion_params["T_m"] + param_deltas["delta_T_m"]
        if self.cfg.use_gc_priors_crystal:
            T_m_upper = torch.maximum(
                torch.full_like(T_m_raw, self.cfg.T_m_max),
                fusion_params["T_m"] + self.cfg.correction_Tm_max_delta,
            )
            T_m = torch.minimum(
                T_m_raw.clamp_min(self.cfg.T_m_min),
                T_m_upper,
            )
        else:
            T_m = T_m_raw.clamp(self.cfg.T_m_min, self.cfg.T_m_max)
        dH_scale = 1.0 + param_deltas["delta_dH_fraction"]
        dH_fus = (fusion_params["dH_fus"] * dH_scale).clamp(min=self.cfg.eps)
        corrected = self._attach_fusion_entropy({
            "T_m": T_m,
            "dH_fus": dH_fus,
            "dCp_fus": fusion_params["dCp_fus"],
        })
        if "Phi_override" in fusion_params:
            corrected["Phi_override"] = fusion_params["Phi_override"]
        if "direct_phi_mask" in fusion_params:
            corrected["direct_phi_mask"] = fusion_params["direct_phi_mask"]
        return corrected

    def _attach_fusion_entropy(
        self,
        fusion_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Keep dS_fus diagnostics consistent with the active T_m/dH_fus values."""
        fusion_params = dict(fusion_params)
        fusion_params["dS_fus"] = (
            fusion_params["dH_fus"] / fusion_params["T_m"].clamp_min(self.cfg.eps)
        )
        fusion_params.setdefault("T_m_unclamped", fusion_params["T_m"])
        fusion_params.setdefault("dH_fus_raw", fusion_params["dH_fus"])
        return fusion_params

    def _calibrate_gc_tm_prior(self, T_m_gc: torch.Tensor) -> torch.Tensor:
        """Apply the train-set affine calibration to the GC melting-point prior."""
        return (
            T_m_gc * float(self.cfg.gc_prior_tm_scale)
            + float(self.cfg.gc_prior_tm_bias)
        )

    def _build_sigma_activity_params(
        self,
        g_solute: torch.Tensor,
        g_solvent: torch.Tensor,
        *,
        targets=None,
        force_sigma_oracle: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build the COSMO-SAC activity-param dict from per-molecule sigma-profiles.

        When ``cosmo_sac_wire_volume`` is False (default), volume is passed as None
        (residual-only COSMO-SAC). When True, the aux-head molar volume (cm^3/mol) is
        converted to A^3/molecule via ``_CM3_PER_MOL_TO_A3`` and passed detached so
        the Staverman-Guggenheim combinatorial size term is active but cannot be
        trained by the solubility loss. ``alpha_12`` is a compatibility placeholder
        for the correction param-summary.

        When ``force_sigma_oracle`` is True and (the model is in eval mode, OR
        ``cfg.train_sigma_oracle`` is set for train-time co-adaptation) and
        ``targets`` is not None, the predicted sigma-profile is overridden with the
        oracle values (from VT-2005 or similar) for matched rows, identified by the
        boolean mask in ``targets["sigma_oracle_mask_solute"]``.
        """
        sol = self.head_sigma(g_solute)
        slv = self.head_sigma(g_solvent)
        p_sol, a_sol = sol["p_sigma"], sol["area"]
        p_slv, a_slv = slv["p_sigma"], slv["area"]
        allow_oracle = (not self.training) or bool(getattr(self.cfg, "train_sigma_oracle", False))
        if force_sigma_oracle and allow_oracle and targets is not None:
            op = targets.get("sigma_oracle_p_solute")
            oa = targets.get("sigma_oracle_area_solute")
            om = targets.get("sigma_oracle_mask_solute")
            if (
                isinstance(op, torch.Tensor)
                and isinstance(oa, torch.Tensor)
                and isinstance(om, torch.Tensor)
            ):
                m = om.to(p_sol.device).bool()
                p_sol = torch.where(m.unsqueeze(-1), op.to(p_sol), p_sol)
                a_sol = torch.where(m, oa.to(a_sol), a_sol)
            sp = targets.get("sigma_oracle_p_solvent")
            sa = targets.get("sigma_oracle_area_solvent")
            sm = targets.get("sigma_oracle_mask_solvent")
            if (
                isinstance(sp, torch.Tensor)
                and isinstance(sa, torch.Tensor)
                and isinstance(sm, torch.Tensor)
            ):
                m = sm.to(p_slv.device).bool()
                p_slv = torch.where(m.unsqueeze(-1), sp.to(p_slv), p_slv)
                a_slv = torch.where(m, sa.to(a_slv), a_slv)
        if getattr(self.cfg, "sigma_detached_from_closure", False):
            # Режется ПУТЬ, а не параметры: в замыкание уходят значения без истории, поэтому
            # градиент потери по растворимости до σ-головы не доходит. sigma_shape_* ниже
            # остаются присоединёнными, так что внешняя σ-супервизия продолжает её обучать.
            p_sol, a_sol = p_sol.detach(), a_sol.detach()
            p_slv, a_slv = p_slv.detach(), a_slv.detach()
        if getattr(self.cfg, "cosmo_sac_wire_volume", False):
            # Grounded size factor: use the aux-head molar volume (cm^3/mol),
            # converted to A^3/molecule, DETACHED so the combinatorial term cannot
            # be trained by the solubility loss (it is grounded by head_aux, mirroring
            # the P1 sigma-head freeze rationale).
            v_solute = self.head_aux(g_solute)["V_m"].detach() * _CM3_PER_MOL_TO_A3
            v_solvent = self.head_aux(g_solvent)["V_m"].detach() * _CM3_PER_MOL_TO_A3
        else:
            v_solute = None
            v_solvent = None
        return {
            "p_solute": p_sol,
            "A_solute": a_sol,
            "p_solvent": p_slv,
            "A_solvent": a_slv,
            "V_solute": v_solute,
            "V_solvent": v_solvent,
            "alpha_12": torch.full_like(sol["area"], 0.3),
            "sigma_shape_solute": sol["p_shape"],
            "sigma_shape_solvent": slv["p_shape"],
        }

    def _build_corrected_nrtl_state(
        self,
        *,
        nrtl_params: dict[str, torch.Tensor],
        physics_out: dict[str, torch.Tensor],
        param_deltas: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build a corrected solver-space NRTL state for the proposal path."""
        if "p_solute" in nrtl_params:
            # COSMO-SAC: activity is not corrected (only the crystal side is);
            # pass the sigma-profile activity params through unchanged.
            return nrtl_params
        if "ln_gamma_inf_ref" in nrtl_params or "ln_gamma_inf" in nrtl_params:
            return {
                "ln_gamma_inf": (
                    physics_out["ln_gamma_inf"] + param_deltas["delta_tau_12"]
                ).clamp(-self.cfg.tau_clamp, self.cfg.tau_clamp),
                "alpha_12": nrtl_params["alpha_12"],
            }

        return {
            "tau_12": (
                physics_out["tau_12"] + param_deltas["delta_tau_12"]
            ).clamp(-self.cfg.tau_clamp, self.cfg.tau_clamp),
            "tau_21": (
                physics_out["tau_21"] + param_deltas["delta_tau_21"]
            ).clamp(-self.cfg.tau_clamp, self.cfg.tau_clamp),
            "alpha_12": nrtl_params["alpha_12"],
        }

    def _require_morgan_features(
        self,
        solute_morgan_fp: Optional[torch.Tensor],
        solvent_morgan_fp: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and return Morgan fingerprints when the feature path is enabled."""
        if solute_morgan_fp is None or solvent_morgan_fp is None:
            raise ValueError(
                "Morgan fingerprint features are enabled in the config, "
                "but solute_morgan_fp/solvent_morgan_fp were not provided."
            )
        return solute_morgan_fp, solvent_morgan_fp

    def _require_descriptor_prior_features(
        self,
        solute_descriptor_prior_features: Optional[torch.Tensor],
        solvent_descriptor_prior_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and return descriptor priors when the prior path is enabled."""
        if (
            solute_descriptor_prior_features is None
            or solvent_descriptor_prior_features is None
        ):
            raise ValueError(
                "Descriptor priors are enabled in the config, but "
                "solute_descriptor_prior_features/solvent_descriptor_prior_features "
                "were not provided."
            )
        return solute_descriptor_prior_features, solvent_descriptor_prior_features

    def set_descriptor_normalization(
        self,
        mean: torch.Tensor | list[float],
        std: torch.Tensor | list[float],
    ) -> None:
        """Attach train-set descriptor normalization statistics to the model."""
        mean_arr, std_arr = validate_descriptor_normalization_stats(
            mean,
            std,
            descriptor_dim=self.cfg.descriptor_dim if self.cfg.use_descriptor_augmentation else None,
        )
        mean_tensor = torch.as_tensor(mean_arr, dtype=torch.float32)
        std_tensor = torch.as_tensor(std_arr, dtype=torch.float32)
        buffer_device = self.descriptor_mean.device
        buffer_dtype = self.descriptor_mean.dtype
        self.descriptor_mean = mean_tensor.to(
            device=buffer_device,
            dtype=buffer_dtype,
        ).clone()
        self.descriptor_std = std_tensor.clamp_min(1.0e-8).to(
            device=buffer_device,
            dtype=buffer_dtype,
        ).clone()

    def _encode_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Normalize and project raw RDKit descriptors into the pair space."""
        if self.descriptor_mlp is None:
            raise ValueError(
                "Descriptor augmentation is disabled for this TGNNSolv instance."
            )
        if descriptors.size(-1) != self.cfg.descriptor_dim:
            raise ValueError(
                f"Descriptor tensor has dim {descriptors.size(-1)}, "
                f"expected {self.cfg.descriptor_dim}."
            )
        descriptors = descriptors.to(
            device=self.descriptor_mean.device,
            dtype=self.descriptor_mean.dtype,
        )
        normalized = (descriptors - self.descriptor_mean) / self.descriptor_std.clamp_min(
            1.0e-8
        )
        normalized = normalized.clamp(
            min=-DESCRIPTOR_Z_CLIP,
            max=DESCRIPTOR_Z_CLIP,
        )
        return self.descriptor_mlp(normalized)

    def _augment_pair_representation(
        self,
        g_pair_gnn: torch.Tensor,
        *,
        solute_descriptors: Optional[torch.Tensor] = None,
        solvent_descriptors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge normalized descriptor pair features into the learned pair state."""
        if not self.cfg.use_descriptor_augmentation:
            return g_pair_gnn
        if (
            solute_descriptors is None
            or solvent_descriptors is None
            or self.descriptor_mlp is None
            or self.pair_desc_proj is None
        ):
            return g_pair_gnn

        d_sol = self._encode_descriptors(solute_descriptors).to(g_pair_gnn)
        d_slv = self._encode_descriptors(solvent_descriptors).to(g_pair_gnn)
        d_pair = torch.cat(
            [
                d_sol,
                d_slv,
                d_sol * d_slv,
                (d_sol - d_slv).abs(),
            ],
            dim=-1,
        )
        return self.pair_desc_proj(torch.cat([g_pair_gnn, d_pair], dim=-1))

    def _augment_pair_with_ionic_features(
        self,
        g_pair: torch.Tensor,
        ionic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add coarse ionic/dielectric context to the pair representation."""
        if not self.cfg.use_ionic_features:
            return g_pair
        if ionic_features is None or self.ionic_feature_adapter is None:
            return g_pair
        if ionic_features.size(-1) != self.cfg.ionic_feature_dim:
            raise ValueError(
                f"Ionic feature tensor has dim {ionic_features.size(-1)}, "
                f"expected {self.cfg.ionic_feature_dim}."
            )
        ionic_emb = self.ionic_feature_adapter(ionic_features.to(g_pair))
        scale = (
            torch.tanh(self.ionic_pair_scale)
            if self.ionic_pair_scale is not None
            else 1.0
        )
        return g_pair + scale * ionic_emb

    def _require_group_prior_features(
        self,
        solute_group_prior_features: Optional[torch.Tensor],
        solvent_group_prior_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and return fixed group priors when the path is enabled."""
        if (
            solute_group_prior_features is None
            or solvent_group_prior_features is None
        ):
            raise ValueError(
                "Fixed group priors are enabled in the config, but "
                "solute_group_prior_features/solvent_group_prior_features were "
                "not provided."
            )
        return solute_group_prior_features, solvent_group_prior_features

    def _require_crystal_gc_priors(
        self,
        T_m_gc: Optional[torch.Tensor],
        dH_fus_gc: Optional[torch.Tensor],
        dCp_fus_gc: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and return fixed crystal GC priors when enabled."""
        if T_m_gc is None or dH_fus_gc is None or dCp_fus_gc is None:
            raise ValueError(
                "Crystal GC priors are enabled in the config, but "
                "T_m_gc/dH_fus_gc/dCp_fus_gc were not provided."
            )
        return T_m_gc, dH_fus_gc, dCp_fus_gc

    def _resolve_oracle_mask(
        self,
        targets: Optional[Dict[str, torch.Tensor | object]],
        primary_key: str,
        fallback_key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        """Resolve a boolean oracle-availability mask from the targets payload."""
        if targets is None:
            return None
        mask = targets.get(primary_key)
        if mask is None:
            mask = targets.get(fallback_key)
        if not isinstance(mask, torch.Tensor):
            return None
        return mask.to(device=reference.device).bool()

    def _build_solver_fusion_params(
        self,
        fusion_params: dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor | object]] = None,
        *,
        force_oracle_injection: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Substitute oracle crystal targets into the solver path when enabled."""
        solver_fusion_params = self._attach_fusion_entropy(fusion_params)
        zero_mask = torch.zeros_like(fusion_params["T_m"], dtype=torch.bool)
        oracle_masks = {
            "T_m": zero_mask,
            "dH_fus": zero_mask.clone(),
        }
        if (
            (not self.cfg.use_oracle_injection and not force_oracle_injection)
            or (not self.training and not force_oracle_injection)
            or targets is None
        ):
            return self._apply_direct_phi_override(
                solver_fusion_params,
                fusion_params,
                targets,
            ), oracle_masks

        T_m_target = targets.get("T_m")
        dH_fus_target = targets.get("dH_fus")
        if not isinstance(T_m_target, torch.Tensor) or not isinstance(
            dH_fus_target, torch.Tensor
        ):
            return self._apply_direct_phi_override(
                solver_fusion_params,
                fusion_params,
                targets,
            ), oracle_masks

        mask_Tm = self._resolve_oracle_mask(
            targets,
            "has_T_m",
            "T_m_mask",
            fusion_params["T_m"],
        )
        mask_dH = self._resolve_oracle_mask(
            targets,
            "has_dH_fus",
            "dH_mask",
            fusion_params["dH_fus"],
        )
        if mask_Tm is None or mask_dH is None:
            return self._apply_direct_phi_override(
                solver_fusion_params,
                fusion_params,
                targets,
            ), oracle_masks

        if force_oracle_injection:
            prob = 1.0
        else:
            prob = min(max(float(self.cfg.oracle_injection_prob), 0.0), 1.0)
        if prob <= 0.0:
            return self._apply_direct_phi_override(
                solver_fusion_params,
                fusion_params,
                targets,
            ), oracle_masks
        if prob < 1.0:
            rand = torch.rand_like(fusion_params["T_m"])
            oracle_sample_mask = rand < prob
            mask_Tm = mask_Tm & oracle_sample_mask
            mask_dH = mask_dH & oracle_sample_mask

        oracle_masks = {
            "T_m": mask_Tm,
            "dH_fus": mask_dH,
        }
        solver_fusion_params["T_m"] = torch.where(
            mask_Tm,
            T_m_target.to(
                device=fusion_params["T_m"].device,
                dtype=fusion_params["T_m"].dtype,
            ).detach(),
            fusion_params["T_m"],
        )
        solver_fusion_params["dH_fus"] = torch.where(
            mask_dH,
            dH_fus_target.to(
                device=fusion_params["dH_fus"].device,
                dtype=fusion_params["dH_fus"].dtype,
            ).detach(),
            fusion_params["dH_fus"],
        )
        solver_fusion_params.pop("T_m_unclamped", None)
        solver_fusion_params.pop("dH_fus_raw", None)
        solver_fusion_params = self._attach_fusion_entropy(solver_fusion_params)
        return self._apply_direct_phi_override(
            solver_fusion_params,
            fusion_params,
            targets,
        ), oracle_masks

    def _apply_direct_phi_override(
        self,
        solver_fusion_params: dict[str, torch.Tensor],
        fusion_params: dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor | object]],
    ) -> dict[str, torch.Tensor]:
        """Attach an effective Phi(T) override for configured no-melting rows."""
        if (
            not self.cfg.use_direct_phi_branch
            or "Phi_intercept" not in fusion_params
            or "Phi_slope" not in fusion_params
            or targets is None
        ):
            return solver_fusion_params
        T = targets.get("T")
        if not isinstance(T, torch.Tensor):
            return solver_fusion_params

        ref = fusion_params["Phi_intercept"]
        mask = torch.zeros_like(ref, dtype=torch.bool)
        if self.cfg.direct_phi_for_decomposition:
            decomp = targets.get("has_decomposition_T")
            if isinstance(decomp, torch.Tensor):
                mask = mask | decomp.to(device=ref.device).bool()
        if self.cfg.direct_phi_for_missing_tm:
            valid_tm = targets.get("has_valid_T_m", targets.get("has_T_m"))
            if isinstance(valid_tm, torch.Tensor):
                mask = mask | (~valid_tm.to(device=ref.device).bool())
        if not bool(mask.any().item()):
            return solver_fusion_params

        T = T.to(device=ref.device, dtype=ref.dtype).clamp_min(self.cfg.eps)
        phi_direct = fusion_params["Phi_intercept"] + fusion_params["Phi_slope"] / T
        phi_standard = self.sle_solver.ideal_layer(
            T,
            solver_fusion_params["T_m"],
            solver_fusion_params["dH_fus"],
            solver_fusion_params["dCp_fus"],
        )
        solver_fusion_params = dict(solver_fusion_params)
        solver_fusion_params["Phi_override"] = torch.where(mask, phi_direct, phi_standard)
        solver_fusion_params["direct_phi_mask"] = mask
        return solver_fusion_params

    def forward(
            self,
            solute_data: Batch,
            solvent_data: Batch,
            T: torch.Tensor,
            solvent_type: Optional[torch.Tensor] = None,
            solute_morgan_fp: Optional[torch.Tensor] = None,
            solvent_morgan_fp: Optional[torch.Tensor] = None,
            solute_descriptors: Optional[torch.Tensor] = None,
            solvent_descriptors: Optional[torch.Tensor] = None,
            ionic_features: Optional[torch.Tensor] = None,
            solute_descriptor_prior_features: Optional[torch.Tensor] = None,
            solvent_descriptor_prior_features: Optional[torch.Tensor] = None,
            solute_group_prior_features: Optional[torch.Tensor] = None,
            solvent_group_prior_features: Optional[torch.Tensor] = None,
            unifac_ln_gamma_inf: Optional[torch.Tensor] = None,
            unifac_gamma_mask: Optional[torch.Tensor] = None,
            T_m_gc: Optional[torch.Tensor] = None,
            dH_fus_gc: Optional[torch.Tensor] = None,
            dCp_fus_gc: Optional[torch.Tensor] = None,
            targets: Optional[Dict[str, torch.Tensor | object]] = None,
            force_oracle_injection: bool = False,
            force_sigma_oracle: bool = False,
            detach_crystal_from_encoder: Optional[bool] = None,
            gamma_only: bool = False,
            return_intermediates: bool = False,  # ✦ NEW
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        solute_data : PyG Batch of solute graphs
        solvent_data : PyG Batch of solvent graphs
        T : (B,) temperature in Kelvin
        solvent_type : (B,) optional solvent type indices for MoE
        return_intermediates : if True, return (output_dict, intermediates_dict)
            where intermediates_dict contains flat physical parameters,
            representations, and gate values for analysis.

        Returns
        -------
        If return_intermediates is False:
            dict with keys: ln_x2, x2, physics, fusion_params, nrtl_params,
                hansen_sol, hansen_slv, aux_sol, aux_slv, Ra, confidence,
                ln_x2_direct, ln_x2_direct_log_sigma, ln_x2_direct_sigma,
                correction, gate, moe_gate, attn_maps
        If return_intermediates is True:
            (output_dict, intermediates_dict)
            intermediates_dict has flat keys: T_m, dH_fus, dCp_fus,
                alpha_12, tau_12, tau_21, G_12, G_21, ln_gamma_2, gamma_2,
                ln_gamma_inf, Phi, x_ideal, ln_x2_ideal, ln_x2_physics,
                ln_x2_direct, ln_x2_final, correction_gate,
                correction_magnitude, hansen_sol, hansen_slv, Ra,
                g_sol_pre, g_slv_pre, g_sol_post, g_slv_post, g_pair,
                + all nrtl_params keys (mode-dependent)
        """
        t_feat = make_temperature_features(T)
        encoder_t_feat = self._encoder_temp_features(t_feat)
        interaction_t_feat = self._interaction_temp_features(t_feat)
        nrtl_t_feat = self._nrtl_temp_features(t_feat)

        # ---- 1. Encode both molecules ----
        h_sol_atoms, g_sol_pre_parts, sol_a_disp, sol_a_polar = self._encode_and_readout(
            solute_data, "solute", temp_feat=encoder_t_feat
        )
        h_slv_atoms, g_slv_pre_parts, slv_a_disp, slv_a_polar = self._encode_and_readout(
            solvent_data, "solvent", temp_feat=encoder_t_feat
        )
        g_sol_pre = g_sol_pre_parts["value"]
        g_slv_pre = g_slv_pre_parts["value"]
        sol_fp_emb = None
        slv_fp_emb = None
        if self.cfg.use_morgan_features:
            solute_morgan_fp, solvent_morgan_fp = self._require_morgan_features(
                solute_morgan_fp,
                solvent_morgan_fp,
            )
            sol_fp_emb = self.solute_fp_adapter(solute_morgan_fp.to(g_sol_pre))
            slv_fp_emb = self.solvent_fp_adapter(solvent_morgan_fp.to(g_slv_pre))
            g_sol_pre = g_sol_pre + self.fp_pre_scale * sol_fp_emb
            g_slv_pre = g_slv_pre + self.fp_pre_scale * slv_fp_emb
        sol_prior = None
        slv_prior = None
        sol_group_for_nrtl = None
        slv_group_for_nrtl = None
        unifac_ln_gamma_for_nrtl = None
        unifac_gamma_mask_for_nrtl = None
        if self.cfg.use_descriptor_priors:
            sol_prior, slv_prior = self._require_descriptor_prior_features(
                solute_descriptor_prior_features,
                solvent_descriptor_prior_features,
            )
            sol_prior = sol_prior.to(g_sol_pre)
            slv_prior = slv_prior.to(g_slv_pre)
        if self.cfg.requires_group_prior_features:
            sol_group_for_nrtl, slv_group_for_nrtl = self._require_group_prior_features(
                solute_group_prior_features,
                solvent_group_prior_features,
            )
            sol_group_for_nrtl = sol_group_for_nrtl.to(g_sol_pre)
            slv_group_for_nrtl = slv_group_for_nrtl.to(g_slv_pre)
            if self.cfg.use_group_priors:
                sol_prior = sol_group_for_nrtl
                slv_prior = slv_group_for_nrtl
        if self.cfg.use_unifac_gamma_prior:
            if unifac_ln_gamma_inf is None and targets is not None:
                maybe_unifac = targets.get("unifac_ln_gamma_inf")
                if isinstance(maybe_unifac, torch.Tensor):
                    unifac_ln_gamma_inf = maybe_unifac
            if unifac_gamma_mask is None and targets is not None:
                maybe_mask = targets.get("unifac_gamma_mask")
                if isinstance(maybe_mask, torch.Tensor):
                    unifac_gamma_mask = maybe_mask
            if unifac_ln_gamma_inf is not None:
                unifac_ln_gamma_for_nrtl = unifac_ln_gamma_inf.to(g_sol_pre)
            if unifac_gamma_mask is not None:
                unifac_gamma_mask_for_nrtl = unifac_gamma_mask.to(g_sol_pre.device)
        solute_T_m_gc = None
        solute_dH_fus_gc = None
        solute_dCp_fus_gc = None
        if self.cfg.use_gc_priors_crystal:
            (
                solute_T_m_gc,
                solute_dH_fus_gc,
                solute_dCp_fus_gc,
            ) = self._require_crystal_gc_priors(T_m_gc, dH_fus_gc, dCp_fus_gc)
            solute_T_m_gc = self._calibrate_gc_tm_prior(
                solute_T_m_gc.to(g_sol_pre)
            )
            solute_dH_fus_gc = solute_dH_fus_gc.to(g_sol_pre)
            solute_dCp_fus_gc = solute_dCp_fus_gc.to(g_sol_pre)

        # ---- 2. Auxiliary heads (before cross-attention) ----
        return_head_parts = return_intermediates or self.use_molecular_priors
        hansen_sol_parts = self.head_hansen(
            g_sol_pre,
            prior_features=sol_prior,
            return_parts=return_head_parts,
        )
        hansen_slv_parts = self.head_hansen(
            g_slv_pre,
            prior_features=slv_prior,
            return_parts=return_head_parts,
        )
        aux_sol_parts = self.head_aux(
            g_sol_pre,
            prior_features=sol_prior,
            return_parts=return_head_parts,
        )
        aux_slv_parts = self.head_aux(
            g_slv_pre,
            prior_features=slv_prior,
            return_parts=return_head_parts,
        )
        if isinstance(hansen_sol_parts, dict):
            hansen_sol = hansen_sol_parts["value"]
            hansen_slv = hansen_slv_parts["value"]
        else:
            hansen_sol = hansen_sol_parts
            hansen_slv = hansen_slv_parts
        aux_sol = {"V_m": aux_sol_parts["V_m"]}
        aux_slv = {"V_m": aux_slv_parts["V_m"]}

        # ---- 3. Cross-attention / Bipartite MP ----
        # Плоское -> плотное с глобальным токеном одним вызовом вместо четырёх питоновских
        # циклов по графам; см. _to_dense_with_token о том, сколько эти циклы стоили.
        h_sol_padded, sol_atom_mask, sol_mask, sol_counts = self._to_dense_with_token(
            h_sol_atoms, solute_data, self.sol_token
        )
        h_slv_padded, slv_atom_mask, slv_mask, slv_counts = self._to_dense_with_token(
            h_slv_atoms, solvent_data, self.slv_token
        )

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
        # Маска по атомам выбирает их в ИСХОДНОМ плоском порядке, поэтому вектор
        # принадлежности графам восстанавливать нечем и незачем: это и есть data.batch.
        sol_batch = solute_data.batch
        slv_batch = solvent_data.batch

        g_sol_post_parts = self._readout_payload(
            h_sol_padded[sol_atom_mask],
            sol_batch,
            a_disp=sol_a_disp,
            a_polar=sol_a_polar,
        )
        g_slv_post_parts = self._readout_payload(
            h_slv_padded[slv_atom_mask],
            slv_batch,
            a_disp=slv_a_disp,
            a_polar=slv_a_polar,
        )
        g_sol_post = g_sol_post_parts["value"]
        g_slv_post = g_slv_post_parts["value"]
        g_sol_tok = self._gather_tokens(h_sol_padded, sol_counts)
        g_slv_tok = self._gather_tokens(h_slv_padded, slv_counts)
        g_sol_post = g_sol_post + self.sol_token_gate * self.token_proj(g_sol_tok)
        g_slv_post = g_slv_post + self.slv_token_gate * self.token_proj(g_slv_tok)
        if self.cfg.use_morgan_features:
            g_sol_post = g_sol_post + self.fp_post_scale * sol_fp_emb
            g_slv_post = g_slv_post + self.fp_post_scale * slv_fp_emb

        # ---- 5. Pair representation ----
        g_sol_post_parts["value"] = g_sol_post
        g_slv_post_parts["value"] = g_slv_post
        g_pair = self._build_pair_representation(
            g_sol_post_parts,
            g_slv_post_parts,
        )
        g_pair = self._augment_pair_representation(
            g_pair,
            solute_descriptors=solute_descriptors,
            solvent_descriptors=solvent_descriptors,
        )
        g_pair = self._augment_pair_with_ionic_features(g_pair, ionic_features)
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

        # ---- 6. Prediction heads ----
        detach_crystal = (
            self.cfg.detach_crystal_from_encoder
            if detach_crystal_from_encoder is None
            else bool(detach_crystal_from_encoder)
        )
        g_sol_for_crystal = g_sol_pre.detach() if detach_crystal else g_sol_pre
        fusion_params = self.head_fusion(
            g_sol_for_crystal,
            T_m_gc=solute_T_m_gc,
            dH_fus_gc=solute_dH_fus_gc,
            dCp_fus_gc=solute_dCp_fus_gc,
        )
        solver_fusion_params, oracle_injection_masks = (
            self._build_solver_fusion_params(
                fusion_params,
                targets=targets,
                force_oracle_injection=force_oracle_injection,
            )
        )
        if getattr(self.cfg, "detach_crystal_params_in_sle", False):
            # Diagnostic mode: crystal heads still learn from auxiliary labels,
            # but the SLE loss cannot route gradients through Phi(T).
            solver_fusion_params = {
                k: (v.detach() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in solver_fusion_params.items()
            }
        if self.is_cosmo_sac:
            _force_oracle = force_sigma_oracle or (
                isinstance(targets, dict)
                and bool(targets.get("__force_sigma_oracle__", False))
            )
            g_slv_for_sigma = g_slv_pre
            if getattr(self.cfg, "sigma_role_invariant", False):
                # Второй проход энкодера по растворителю С РОЛЬЮ РАСТВОРЯЕМОГО: обе стороны
                # пары попадают в σ-голову через одну и ту же функцию молекулы, поэтому
                # соединение получает один профиль независимо от стороны.
                _, g_slv_sigma_parts, _, _ = self._encode_and_readout(
                    solvent_data, "solute", temp_feat=encoder_t_feat)
                g_slv_for_sigma = g_slv_sigma_parts["value"]
            nrtl_params = self._build_sigma_activity_params(
                g_sol_pre, g_slv_for_sigma, targets=targets,
                force_sigma_oracle=_force_oracle)
        else:
            nrtl_params = self.head_nrtl(
                g_pair,
                temp_feat=nrtl_t_feat,
                solute_group_prior_features=sol_group_for_nrtl,
                solvent_group_prior_features=slv_group_for_nrtl,
                unifac_ln_gamma_inf=unifac_ln_gamma_for_nrtl,
                unifac_gamma_mask=unifac_gamma_mask_for_nrtl,
            )
        ln_x2_aux = None
        if self.aux_direct_sol_head is not None:
            ln_x2_aux = self.aux_direct_sol_head(g_pair).squeeze(-1)

        if gamma_only and self.is_cosmo_sac:
            raise NotImplementedError(
                "gamma_only forward is not supported with activity_model='cosmo_sac'; "
                "supervise the activity via the sigma-profile aux stream instead."
            )
        if gamma_only:
            nrtl = self.sle_solver.nrtl_layer
            tau_12, tau_21, G_12, G_21 = nrtl.compute_tau_G_from_params(
                nrtl_params,
                T,
            )
            ln_gamma_inf = nrtl.ln_gamma_inf(tau_12, tau_21, G_21)
            activity_x2 = None
            activity_mask = None
            if targets is not None:
                maybe_activity_x2 = targets.get("activity_x2")
                if isinstance(maybe_activity_x2, torch.Tensor):
                    activity_x2 = maybe_activity_x2.to(T)
                maybe_activity_mask = targets.get("gamma2_mask")
                if isinstance(maybe_activity_mask, torch.Tensor):
                    activity_mask = maybe_activity_mask.to(T.device).bool()

            ln_gamma_2 = torch.zeros_like(T)
            gamma_2 = torch.ones_like(T)
            x2_out = torch.exp(torch.zeros_like(T)).clamp(0, 1)
            ln_x2_out = torch.zeros_like(T)
            if activity_x2 is not None:
                x2_eval = activity_x2.clamp(1.0e-8, 1.0 - 1.0e-8)
                x1_eval = (1.0 - x2_eval).clamp(1.0e-8, 1.0 - 1.0e-8)
                ln_gamma_2_eval = nrtl.ln_gamma_2(x1_eval, x2_eval, tau_12, tau_21, G_12, G_21)
                gamma_2_eval = torch.exp(ln_gamma_2_eval)
                if activity_mask is not None:
                    if bool(activity_mask.any().item()):
                        ln_gamma_2 = torch.where(activity_mask, ln_gamma_2_eval, ln_gamma_2)
                        gamma_2 = torch.where(activity_mask, gamma_2_eval, gamma_2)
                        x2_out = torch.where(activity_mask, x2_eval, x2_out)
                        ln_x2_out = torch.where(activity_mask, torch.log(x2_eval), ln_x2_out)
                else:
                    ln_gamma_2 = ln_gamma_2_eval
                    gamma_2 = gamma_2_eval
                    x2_out = x2_eval
                    ln_x2_out = torch.log(x2_eval)
            output = {
                "ln_x2": ln_x2_out,
                "x2": x2_out,
                "physics": {
                    "ln_gamma_inf": ln_gamma_inf,
                    "ln_gamma_2": ln_gamma_2,
                    "gamma_2": gamma_2,
                    "tau_12": tau_12,
                    "tau_21": tau_21,
                    "G_12": G_12,
                    "G_21": G_21,
                },
                "fusion_params": fusion_params,
                "solver_fusion_params": solver_fusion_params,
                "nrtl_params": nrtl_params,
                "hansen_sol": hansen_sol,
                "hansen_slv": hansen_slv,
                "aux_sol": aux_sol,
                "aux_slv": aux_slv,
                "Ra": self.sle_solver.hansen_layer(hansen_sol, hansen_slv),
                "confidence": torch.ones_like(T),
                "correction": torch.zeros_like(T),
                "gate": torch.tensor(1.0, device=T.device, dtype=T.dtype),
                "moe_gate": moe_gate,
                "attn_maps": attn_maps,
                "representations": {
                    "g_sol_pre": g_sol_pre,
                    "g_slv_pre": g_slv_pre,
                    "g_sol_post": g_sol_post,
                    "g_slv_post": g_slv_post,
                    "g_pair": g_pair,
                },
            }
            if ln_x2_aux is not None:
                output["ln_x2_aux"] = ln_x2_aux
            if (
                self.is_timp
                and g_sol_pre_parts["disp"] is not None
                and g_sol_pre_parts["polar"] is not None
                and g_slv_pre_parts["disp"] is not None
                and g_slv_pre_parts["polar"] is not None
            ):
                output["timp_channels"] = {
                    "solute_disp": g_sol_pre_parts["disp"],
                    "solute_polar": g_sol_pre_parts["polar"],
                    "solute_combined": g_sol_pre_parts["combined"],
                    "solvent_disp": g_slv_pre_parts["disp"],
                    "solvent_polar": g_slv_pre_parts["polar"],
                    "solvent_combined": g_slv_pre_parts["combined"],
                }
            if self.cfg.use_oracle_injection or force_oracle_injection:
                output["oracle_injection_masks"] = oracle_injection_masks
            if self.cfg.use_gc_priors_crystal:
                output["fusion_gc_priors"] = {
                    "T_m_gc": solute_T_m_gc,
                    "dH_fus_gc": solute_dH_fus_gc,
                    "dCp_fus_gc": solute_dCp_fus_gc,
                }
            if isinstance(hansen_sol_parts, dict):
                output["hansen_sol_prior"] = hansen_sol_parts["prior"]
                output["hansen_slv_prior"] = hansen_slv_parts["prior"]
                output["hansen_sol_residual"] = hansen_sol_parts["residual"]
                output["hansen_slv_residual"] = hansen_slv_parts["residual"]
            if "V_m_prior" in aux_sol_parts:
                output["aux_sol_prior"] = {"V_m": aux_sol_parts["V_m_prior"]}
                output["aux_slv_prior"] = {"V_m": aux_slv_parts["V_m_prior"]}
                output["aux_sol_residual"] = {"V_m": aux_sol_parts["V_m_residual"]}
                output["aux_slv_residual"] = {"V_m": aux_slv_parts["V_m_residual"]}
            return output

        # ---- 7. SLE solver (float32 for numerical stability) ----
        with torch.amp.autocast(device_type='cpu', enabled=False):
            T_f32 = T.float()
            fus_f32 = {k: v.float() for k, v in solver_fusion_params.items()}
            nrtl_f32 = {
                k: (v.float() if torch.is_tensor(v) else v)
                for k, v in nrtl_params.items()
            }
            physics_out = self.sle_solver(T_f32, fus_f32, nrtl_f32)

        # ---- 8. Hansen distance ----
        Ra = self.sle_solver.hansen_layer(hansen_sol, hansen_slv)

        # ---- 9. Adaptive physics correction ----
        param_summary = self._build_param_summary(
            fusion_params=solver_fusion_params,
            nrtl_params=nrtl_params,
            physics_out=physics_out,
            temp_feat=t_feat,
            Ra=Ra,
            dtype=T.dtype,
        )

        ln_x2_physics = physics_out["ln_x2"]

        confidence, param_deltas, proposal_log_sigma = self.correction(
            g_pair,
            param_summary,
        )

        correction_output_mode = getattr(
            self.cfg,
            "correction_output_mode",
            "parameter",
        )
        if correction_output_mode == "ln_x2_residual":
            corrected_fusion_params = solver_fusion_params
            corrected_nrtl_state = {
                "tau_12": physics_out["tau_12"],
                "tau_21": physics_out["tau_21"],
                "alpha_12": nrtl_params["alpha_12"],
            }
            tau_limit = max(float(self.cfg.correction_tau_max_delta), self.cfg.eps)
            raw_residual = (
                float(getattr(self.cfg, "correction_ln_x2_max_delta", 1.0))
                * torch.tanh(param_deltas["delta_tau_12"] / tau_limit)
            )
            proposal_ln_x2 = ln_x2_physics + raw_residual
            proposal_out = {
                **physics_out,
                "ln_x2": proposal_ln_x2,
                "x2": torch.exp(proposal_ln_x2).clamp(0, 1),
            }
        else:
            corrected_fusion_params = self._build_corrected_fusion_params(
                solver_fusion_params,
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
                    {
                        k: (v.float() if torch.is_tensor(v) else v)
                        for k, v in corrected_nrtl_state.items()
                    },
                    use_implicit=False,
                )

            raw_residual = proposal_out["ln_x2"].to(T.dtype) - ln_x2_physics
        bounded_residual = raw_residual.clamp(
            min=-self.cfg.correction_max_abs,
            max=self.cfg.correction_max_abs,
        )
        ln_x2_direct = ln_x2_physics + bounded_residual
        ln_x2_direct_log_sigma = proposal_log_sigma
        if (
            getattr(self.cfg, "correction_force_open_gate", False)
            and correction_output_mode == "ln_x2_residual"
        ):
            # v2 (Gate B): gate forced fully open — the residual is applied at full strength,
            # not throttled by (1-confidence). Tests whether an un-gated output escape valve
            # lets true-σ stop hurting (v1's confidence gate stayed ≈1 → residual ≈ 0).
            ln_x2 = ln_x2_direct
        else:
            ln_x2 = ln_x2_physics + (1.0 - confidence) * bounded_residual

        timp_channel_probes = None
        if (
            self.is_timp
            and self.timp_disp_probe is not None
            and self.timp_polar_probe is not None
            and g_sol_pre_parts["disp"] is not None
            and g_sol_pre_parts["polar"] is not None
        ):
            polar_probe = self.timp_polar_probe(g_sol_pre_parts["polar"])
            timp_channel_probes = {
                "delta_d": self.timp_disp_probe(g_sol_pre_parts["disp"]).squeeze(-1),
                "delta_p": polar_probe[:, 0],
                "delta_h": polar_probe[:, 1],
            }

        output = {
            "ln_x2": ln_x2,
            "x2": torch.exp(ln_x2).clamp(0, 1),
            "physics": physics_out,
            "proposal_physics": proposal_out,
            "fusion_params": fusion_params,
            "solver_fusion_params": solver_fusion_params,
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
            "correction": ln_x2 - ln_x2_physics,
            "gate": confidence.mean(),
            "moe_gate": moe_gate,
            "attn_maps": attn_maps,
            "representations": {
                "g_sol_pre": g_sol_pre,
                "g_slv_pre": g_slv_pre,
                "g_sol_post": g_sol_post,
                "g_slv_post": g_slv_post,
                "g_pair": g_pair,
            },
        }
        if ln_x2_aux is not None:
            output["ln_x2_aux"] = ln_x2_aux
        if (
            self.is_timp
            and g_sol_pre_parts["disp"] is not None
            and g_sol_pre_parts["polar"] is not None
            and g_slv_pre_parts["disp"] is not None
            and g_slv_pre_parts["polar"] is not None
        ):
            output["timp_channels"] = {
                "solute_disp": g_sol_pre_parts["disp"],
                "solute_polar": g_sol_pre_parts["polar"],
                "solute_combined": g_sol_pre_parts["combined"],
                "solvent_disp": g_slv_pre_parts["disp"],
                "solvent_polar": g_slv_pre_parts["polar"],
                "solvent_combined": g_slv_pre_parts["combined"],
            }
        if timp_channel_probes is not None:
            output["timp_channel_probes"] = timp_channel_probes
        if self.cfg.use_oracle_injection or force_oracle_injection:
            output["oracle_injection_masks"] = oracle_injection_masks
        if self.cfg.use_gc_priors_crystal:
            output["fusion_gc_priors"] = {
                "T_m_gc": solute_T_m_gc,
                "dH_fus_gc": solute_dH_fus_gc,
                "dCp_fus_gc": solute_dCp_fus_gc,
            }
        if isinstance(hansen_sol_parts, dict):
            output["hansen_sol_prior"] = hansen_sol_parts["prior"]
            output["hansen_slv_prior"] = hansen_slv_parts["prior"]
            output["hansen_sol_residual"] = hansen_sol_parts["residual"]
            output["hansen_slv_residual"] = hansen_slv_parts["residual"]
        if "V_m_prior" in aux_sol_parts:
            output["aux_sol_prior"] = {"V_m": aux_sol_parts["V_m_prior"]}
            output["aux_slv_prior"] = {"V_m": aux_slv_parts["V_m_prior"]}
            output["aux_sol_residual"] = {"V_m": aux_sol_parts["V_m_residual"]}
            output["aux_slv_residual"] = {"V_m": aux_slv_parts["V_m_residual"]}
            prior_reg = (
                hansen_sol_parts["residual"].pow(2).mean()
                + hansen_slv_parts["residual"].pow(2).mean()
                + aux_sol_parts["V_m_residual"].pow(2).mean()
                + aux_slv_parts["V_m_residual"].pow(2).mean()
            )
            if self.cfg.use_descriptor_priors:
                output["descriptor_prior_reg"] = prior_reg
            if self.cfg.use_group_priors:
                output["group_prior_reg"] = prior_reg

        # ✦ NEW — flat intermediates for analysis scripts
        if return_intermediates:
            intermediates = {
                # Crystal / fusion properties
                "T_m": fusion_params["T_m"],
                "dH_fus": fusion_params["dH_fus"],
                "dS_fus": fusion_params["dS_fus"],
                "dCp_fus": fusion_params["dCp_fus"],
                "T_m_solver": solver_fusion_params["T_m"],
                "dH_fus_solver": solver_fusion_params["dH_fus"],
                "dS_fus_solver": solver_fusion_params["dS_fus"],
                "dCp_fus_solver": solver_fusion_params["dCp_fus"],
                "Phi_intercept": fusion_params.get(
                    "Phi_intercept",
                    torch.zeros_like(fusion_params["T_m"]),
                ),
                "Phi_slope": fusion_params.get(
                    "Phi_slope",
                    torch.zeros_like(fusion_params["T_m"]),
                ),
                "direct_phi_mask": solver_fusion_params.get(
                    "direct_phi_mask",
                    torch.zeros_like(fusion_params["T_m"], dtype=torch.bool),
                ).to(dtype=torch.float32),
                "T_m_gc": (
                    solute_T_m_gc
                    if solute_T_m_gc is not None
                    else torch.zeros_like(fusion_params["T_m"])
                ),
                "dH_fus_gc": (
                    solute_dH_fus_gc
                    if solute_dH_fus_gc is not None
                    else torch.zeros_like(fusion_params["dH_fus"])
                ),
                "dCp_fus_gc": (
                    solute_dCp_fus_gc
                    if solute_dCp_fus_gc is not None
                    else torch.zeros_like(fusion_params["dCp_fus"])
                ),
                "T_m_corrected": corrected_fusion_params["T_m"],
                "dH_fus_corrected": corrected_fusion_params["dH_fus"],
                "dS_fus_corrected": corrected_fusion_params["dS_fus"],
                "dCp_fus_corrected": corrected_fusion_params["dCp_fus"],
                # NRTL binary interaction (all mode-dependent keys)
                "alpha_12": nrtl_params["alpha_12"],
                "tau_12": physics_out["tau_12"],
                "tau_21": physics_out["tau_21"],
                "tau_12_corrected": proposal_out["tau_12"],
                "tau_21_corrected": proposal_out["tau_21"],
                "G_12": physics_out["G_12"],
                "G_21": physics_out["G_21"],
                # Activity coefficients
                "ln_gamma_2": physics_out["ln_gamma_2"],
                "gamma_2": torch.exp(physics_out["ln_gamma_2"]),
                "ln_gamma_inf": physics_out["ln_gamma_inf"],
                "ln_gamma_2_corrected": proposal_out["ln_gamma_2"],
                "ln_gamma_inf_corrected": proposal_out["ln_gamma_inf"],
                # Ideal solubility contribution
                "Phi": physics_out["Phi"],
                "x_ideal": physics_out["x_ideal"],
                "ln_x2_ideal": torch.log(
                    physics_out["x_ideal"] + self.cfg.eps
                ),
                # Physics path vs learned path vs final
                "ln_x2_physics": physics_out["ln_x2"],
                "ln_x2_direct": ln_x2_direct,
                "ln_x2_aux": (
                    ln_x2_aux
                    if ln_x2_aux is not None
                    else torch.zeros_like(ln_x2)
                ),
                "ln_x2_final": ln_x2,
                "ln_x2_proposal": proposal_out["ln_x2"],
                "correction_raw_residual": raw_residual,
                # Correction gate: σ(w) per sample
                "correction_gate": confidence,
                "correction_magnitude": ln_x2 - physics_out["ln_x2"],
                "delta_T_m": param_deltas["delta_T_m"],
                "delta_dH_fraction": param_deltas["delta_dH_fraction"],
                "delta_tau_12": param_deltas["delta_tau_12"],
                "delta_tau_21": param_deltas["delta_tau_21"],
                # Hansen solubility parameters
                "hansen_sol": hansen_sol,
                "hansen_slv": hansen_slv,
                "hansen_sol_prior": (
                    hansen_sol_parts["prior"]
                    if isinstance(hansen_sol_parts, dict)
                    else torch.zeros_like(hansen_sol)
                ),
                "hansen_slv_prior": (
                    hansen_slv_parts["prior"]
                    if isinstance(hansen_slv_parts, dict)
                    else torch.zeros_like(hansen_slv)
                ),
                "hansen_sol_residual": (
                    hansen_sol_parts["residual"]
                    if isinstance(hansen_sol_parts, dict)
                    else hansen_sol
                ),
                "hansen_slv_residual": (
                    hansen_slv_parts["residual"]
                    if isinstance(hansen_slv_parts, dict)
                    else hansen_slv
                ),
                "Ra": Ra,
                # Auxiliary properties
                "aux_sol": aux_sol,
                "aux_slv": aux_slv,
                "aux_sol_prior": aux_sol_parts.get(
                    "V_m_prior", torch.zeros_like(aux_sol["V_m"])
                ),
                "aux_slv_prior": aux_slv_parts.get(
                    "V_m_prior", torch.zeros_like(aux_slv["V_m"])
                ),
                "aux_sol_residual": aux_sol_parts.get(
                    "V_m_residual", aux_sol["V_m"]
                ),
                "aux_slv_residual": aux_slv_parts.get(
                    "V_m_residual", aux_slv["V_m"]
                ),
                # Learned representations (detached — no grad needed)
                "g_sol_pre": g_sol_pre.detach(),
                "g_slv_pre": g_slv_pre.detach(),
                "g_sol_post": g_sol_post.detach(),
                "g_slv_post": g_slv_post.detach(),
                "g_sol_combined_pre": g_sol_pre_parts["combined"].detach(),
                "g_slv_combined_pre": g_slv_pre_parts["combined"].detach(),
                "g_sol_combined_post": g_sol_post_parts["combined"].detach(),
                "g_slv_combined_post": g_slv_post_parts["combined"].detach(),
                "g_pair": g_pair.detach(),
            }
            if g_sol_pre_parts["disp"] is not None and g_sol_pre_parts["polar"] is not None:
                intermediates.update(
                    {
                        "g_sol_disp_pre": g_sol_pre_parts["disp"].detach(),
                        "g_slv_disp_pre": g_slv_pre_parts["disp"].detach(),
                        "g_sol_polar_pre": g_sol_pre_parts["polar"].detach(),
                        "g_slv_polar_pre": g_slv_pre_parts["polar"].detach(),
                        "g_sol_disp_post": g_sol_post_parts["disp"].detach(),
                        "g_slv_disp_post": g_slv_post_parts["disp"].detach(),
                        "g_sol_polar_post": g_sol_post_parts["polar"].detach(),
                        "g_slv_polar_post": g_slv_post_parts["polar"].detach(),
                    }
                )
            if timp_channel_probes is not None:
                intermediates.update(
                    {
                        "timp_delta_d": timp_channel_probes["delta_d"].detach(),
                        "timp_delta_p": timp_channel_probes["delta_p"].detach(),
                        "timp_delta_h": timp_channel_probes["delta_h"].detach(),
                    }
                )
            # Include all raw NRTL head outputs (mode-dependent keys)
            for k, v in nrtl_params.items():
                if k not in intermediates:
                    intermediates[f"nrtl_{k}"] = v

            return output, intermediates

        return output
