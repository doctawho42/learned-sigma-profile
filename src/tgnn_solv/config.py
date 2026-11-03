"""
Model and training configuration.

All physical constants are fixed (not learnable).
Scale factors control the magnitude of predicted quantities
to keep them in a numerically stable range for the network.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TGNNSolvConfig:
    """Complete configuration for TGNN-Solv model."""

    # --- Architecture ---
    hidden_dim: int = 256
    n_gnn_layers: int = 6
    encoder_type: str = "mpnn"  # "mpnn", "gps", or "timp"
    encoder_role_mode: str = "shared_residual"  # "shared_residual" or "split_late"
    encoder_role_specific_layers: int = 2
    gps_num_heads: int = 8
    gps_use_edge_attr: bool = True
    gps_positional_encoding: str = "laplacian"  # "laplacian" or "rwse"
    gps_pe_dim: int = 16
    use_gasteiger_charges: bool = False
    use_phys_edge_features: bool = False
    explicit_h_small_molecules: bool = False
    explicit_h_max_heavy_atoms: int = 3
    n_cross_attn_layers: int = 3
    n_attn_heads: int = 8
    dropout: float = 0.1
    pair_dim: int = 512
    interaction_mode: str = "cross_attn"  # "cross_attn" or "bipartite"
    use_thermo_cross_attention: bool = False
    thermo_cross_attention_beta_init: float = 0.1
    activity_model: str = "nrtl"  # "nrtl" or "cosmo_sac" (wilson/uniquac are phantom)
    nrtl_tau_mode: str = "ref_invT"  # "ref_invT", "legacy", "abc", or "gamma_inf"
    set2set_steps: int = 3
    use_solvent_moe: bool = True
    solvent_moe_experts: int = 6
    solvent_moe_hidden: int = 256
    solvent_type_emb_dim: int = 16
    use_temperature_in_encoder: bool = False
    use_temperature_in_interaction: bool = False
    use_temperature_in_nrtl_head: bool = True
    use_morgan_features: bool = False
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    morgan_hidden_dim: int = 256
    use_descriptor_augmentation: bool = False
    descriptor_dim: int = 200
    descriptor_hidden_dim: int = 128
    descriptor_augmentation_hidden_dim: Optional[int] = None
    use_ionic_features: bool = False
    ionic_feature_dim: int = 10
    ionic_feature_hidden_dim: int = 64
    use_descriptor_priors: bool = False
    descriptor_prior_hidden_dim: int = 128
    descriptor_prior_hansen_residual_max: float = 5.0
    descriptor_prior_vm_residual_max: float = 30.0
    descriptor_prior_reg_weight: float = 0.0
    use_group_priors: bool = False
    group_prior_hansen_residual_max: float = 5.0
    group_prior_vm_residual_max: float = 30.0
    group_prior_reg_weight: float = 0.0
    use_nrtl_group_prior: bool = False
    nrtl_group_prior_tau_scale: float = 0.35
    nrtl_group_prior_tau_clamp: float = 1.5
    nrtl_group_prior_ra_scale: float = 20.0
    use_unifac_gamma_prior: bool = False
    unifac_gamma_prior_tau_scale: float = 1.0
    unifac_gamma_prior_tau_clamp: float = 3.0
    use_gc_priors_crystal: bool = False
    gc_prior_Tm_residual_max: float = 50.0
    gc_prior_dH_residual_factor: tuple[float, float] = (0.3, 3.0)
    gc_prior_tm_scale: float = 1.0
    gc_prior_tm_bias: float = 0.0
    gc_prior_residual_freeze_epochs: int = 5
    use_oracle_injection: bool = False
    oracle_injection_prob: float = 1.0
    predict_dCp_fus: bool = False
    fixed_dCp_fus: float = 0.0
    use_direct_phi_branch: bool = False
    direct_phi_for_missing_tm: bool = False
    direct_phi_for_decomposition: bool = True
    direct_phi_intercept_scale: float = 20.0
    direct_phi_slope_scale: float = 5000.0
    direct_phi_slope_init: float = 2500.0
    bridge_loss_weight: float = 0.0
    use_walden_check: bool = False
    walden_target: float = 56.5
    walden_tolerance: float = 30.0
    walden_weight: float = 0.1
    use_hansen_contrastive: bool = False
    hansen_contrastive_mol_weight: float = 0.05
    hansen_contrastive_channel_weight: float = 0.05
    hansen_contrastive_pair_weight: float = 0.03
    hansen_contrastive_orth_weight: float = 0.01
    use_pseudo_hansen: bool = True
    pseudo_hansen_weight_discount: float = 0.3
    hansen_contrastive_temperature: float = 0.1
    use_hansen_delta_loss: bool = False
    hansen_delta_loss_phase1_weight: float = 0.0
    hansen_delta_loss_phase2_weight: float = 0.0
    hansen_delta_loss_phase3_weight: float = 0.0
    use_aux_direct_sol_loss: bool = False
    aux_direct_sol_loss_weight: float = 0.1
    aux_direct_sol_loss_phase3_weight: float = 0.01
    branch_training_mode: str = "standard"  # "standard" or "coordinate_descent"
    detach_crystal_from_encoder: bool = False
    detach_crystal_params_in_sle: bool = False
    use_smooth_phase2_curriculum: bool = False
    smooth_phase2_crystal_start: float = 1.0
    smooth_phase2_crystal_end: float = 0.3
    smooth_phase2_pair_start: float = 0.1
    smooth_phase2_pair_end: float = 1.0
    smooth_phase2_idac_weight: float = 0.5

    # --- Physics constants (NOT learnable) ---
    R: float = 8.314          # Gas constant, J/(mol·K)
    T_ref: float = 298.15     # Reference temperature, K

    # --- Constrained activation ranges ---
    T_m_min: float = 100.0    # Min melting point, K
    T_m_max: float = 700.0    # Max melting point, K
    alpha_min: float = 0.1    # Min NRTL non-randomness
    alpha_max: float = 0.6    # Max NRTL non-randomness
    fusion_output_mode: str = "direct"  # "direct" or "entropy_coupled"
    fusion_entropy_min: float = 20.0    # J/(mol*K)
    fusion_entropy_max: float = 150.0   # J/(mol*K)
    fusion_entropy_init: float = 56.5   # Walden-like starting point, J/(mol*K)
    fusion_enthalpy_init: float = 20_000.0  # J/mol

    # --- Scale factors for prediction heads ---
    S_H: float = 5000.0       # dH_fus scale, J/mol
    S_Cp: float = 100.0       # dCp_fus scale, J/(mol·K)
    S_g: float = 5000.0      # NRTL energy scale, J/mol
    S_aT: float = 10.0        # NRTL temperature coeff scale
    S_tau_a: float = 1.0      # NRTL tau offset scale (dimensionless)
    S_tau_b: float = 300.0    # NRTL tau 1/T scale (K)
    S_tau_c: float = 1.0      # NRTL tau log(T/T_ref) scale (dimensionless)
    S_tau_ref: float = 1.0    # NRTL tau(T_ref) scale (dimensionless)
    S_tau_inv: float = 1.0    # NRTL inverse-temperature slope scale (dimensionless)
    S_gamma_inf_ref: float = 5.0  # Direct ln(gamma_inf)(T_ref) scale
    S_gamma_inf_inv: float = 5.0  # Direct ln(gamma_inf) inverse-T slope scale
    use_activity_global_bias: bool = False
    activity_global_bias_init: float = 0.0
    S_delta: float = 10.0     # Hansen parameter scale, MPa^0.5

    # --- SLE solver ---
    n_iter_train: int = 5
    n_iter_eval: int = 20
    damping: float = 0.7
    solver_min_damping: float = 0.1
    solver_tol_train: float = 1e-5
    solver_tol_eval: float = 1e-7
    solver_adaptive_damping: bool = True
    # Early exit on solver_tol, COSMO-SAC path ONLY. OFF because its residual.max().item()
    # is a blocking CUDA host sync n_iter times per forward and it never fires there. Worth
    # 1.03x on its own, measured, not the large win the first draft of this comment implied:
    # run_solver_break_audit.py traced 8103 real profile pairs and all 144 non-degenerate
    # cells finish 3-6 orders of magnitude above tol, so turning it off is bit-identical
    # (tests/test_solver_launch_cost_rewrites.py). The NRTL path KEEPS its break: at the
    # shipped counts it does not fire there either (nearest cell 4.8x tol), but it DOES at the
    # non-default n_iter_eval=30 / tol=1e-8 that test_physics_verification.py uses -- iteration
    # 16 of 30, moving x2 by 2.3e-09. A NRTL outer step has no segment solve in it, so the
    # sync is cheap relative to the loop, and leaving it keeps every NRTL number bit-exact
    # even off the default counts. Not worth trading that for a saving measured elsewhere.
    solver_cosmo_break_on_tol: bool = False
    use_implicit_diff: bool = True

    # --- COSMO-SAC activity model (activity_model="cosmo_sac") ---
    # Neural sigma-profile head -> differentiable COSMO-SAC layer. Mirrors the
    # crystal grounding: the sigma-profile is a single-component transferable
    # object supervised by an external sigma-profile database.
    cosmo_sac_n_bins: int = 51
    cosmo_sac_sigma_min: float = -0.025
    cosmo_sac_sigma_max: float = 0.025
    cosmo_sac_a_eff: float = 7.5
    cosmo_sac_alpha_prime: float = 16466.72
    cosmo_sac_c_hb: float = 85580.0
    cosmo_sac_sigma_hb: float = 0.0084
    cosmo_sac_R_kcal: float = 1.987204e-3
    cosmo_sac_coord_z: float = 10.0
    cosmo_sac_q0: float = 79.53
    cosmo_sac_r0: float = 66.69
    cosmo_sac_use_combinatorial: bool = True
    cosmo_sac_wire_volume: bool = False  # feed (detached) molar volume so the SG combinatorial term is active
    cosmo_sac_gamma_iter_train: int = 16  # bumped from 8: n=8 failed convergence test (gap 5.86 ln-units at 273K); n=16 converges to within 6e-5 of n=30
    cosmo_sac_gamma_iter_eval: int = 30
    # ПРОВЕРКА СХОДИМОСТИ, ВЫКЛЮЧЕНА ПО УМОЛЧАНИЮ. Счётчики выше фиксированы, и до 2026-09-30
    # сходимость к неподвижной точке нигде не проверялась -- она ПРЕДПОЛАГАЛАСЬ. Комментарий у
    # iter_train («n=16 сходится с точностью 6e-5 до n=30») сравнивает два счётчика МЕЖДУ СОБОЙ,
    # а не с пределом. Измерено (results/segment_convergence): предел достигается к n~300, а при
    # n=30 медианная строка сошлась до машинной точности, но 39% строк несут >0.01 ln-единиц
    # ошибки итерации, худшая 0.073 на депонированном профиле и 0.162 на выученном.
    # Флаг включает запись невязки в layer.last_convergence и НЕ меняет арифметику.
    cosmo_sac_track_convergence: bool = False
    # Tier-3 closure-fix crossover experiment (constructive complement to the grounding paradox).
    cosmo_sac_kernel_residual_rank: int = 0         # Arm C: R>0 enables the LR-SKR learnable delta_w residual (K=52R)
    cosmo_sac_kernel_residual_penalty: float = 0.0  # optional 2nd-difference smoothness on columns of kernel_B
    sigma_head_adapter_rank: int = 0                # Arm I: r>0 enables the nonlinear rank-r adapter on SigmaProfileHead (K=320r)
    arm_trainable: Optional[str] = None             # {"kernel","sigma_adapter","correction",None}: freeze all but this arm's K params in phase>=2
    # Sigma-profile head output scaling (cavity surface area, Å²).
    sigma_area_scale: float = 75.0  # ~pool sigma_area std; see P0 loss rebalance
    sigma_area_min: float = 20.0
    sigma_shape_weight: float = 1.0  # weight on the SUM-EMD shape term
    # G4: ПОЛ на члене формы sigma-супервизии -- shape_loss := relu(shape_loss - floor).
    # Выше пола градиент в точности прежний, на полу и ниже -- в точности нуль, то есть это
    # буквально "спускаться до заданной глубины и остановиться". 0.0 отключает ветвь целиком.
    #
    # ОТКУДА ЧИСЛО. results/g4_descent_depth: спуск к депонированному профилю улучшает AAD на
    # IDAC с 1.3142 до 0.8201, но у всех пяти сидов есть ВНУТРЕННИЙ оптимум по глубине (ни
    # одного на краю сетки), и дальше оценка портится. Выигрыш от глубины +0.5490 +- 0.1129,
    # положителен 5/5. Оптимум лежит при EMD 0.3986 +- 0.0442 (разброс 11%); перенос через
    # долю снятого даёт 0.4457, то есть выбор внутри собственного разброса порога и потому не
    # несущий. Единицы совпадают: в обучающем потоке мишень нормирована на сумму 1 (проверено
    # -- все строки ровно 1.0), предсказание есть softmax, и G4 мерил ровно это.
    #
    # НАПРАВЛЕНИЕ ПРОВЕРЕНО НУЛЯМИ ПО МИШЕНИ, а не предположено: спуск на то же расстояние к
    # мишени с той же гистограммой масс и перемешанными бинами ПОРТИТ AAD монотонно (до
    # 2.6987), случайная мишень тоже -- у обоих нулей лучшее, что они могут, это не двигаться.
    sigma_shape_floor: float = 0.0
    # Sigma-profile external aux supervision stream.
    sigma_aux_steps_per_epoch: int = 0
    sigma_aux_phase1_weight: Optional[float] = None
    sigma_aux_phase2_weight: Optional[float] = None
    sigma_aux_phase3_weight: Optional[float] = None
    sigma_aux_symmetrize: bool = True  # ground solute AND solvent role embeddings
    sigma_profile_loss: str = "emd"  # "emd" (1-D Wasserstein) or "mse"
    # T1: путь к .npy с вектором ПО-БИНОВЫХ весов форменного члена sigma-супервизии (длина
    # n_bins). None -- невзвешенный член, битово прежний.
    #
    # ОТКУДА ВЕКТОР. results/per_bin_sensitivity_weights: цена замыкания за размещение массы
    # измерена ПО КОНЕЧНОЙ РАЗНОСТИ по каждому бину (градиентная карта для этого запрещена её
    # же выводом -- завышает вчетверо и сидонеустойчива), и вес есть дискретный градиент этого
    # потенциала. Взвешенный член остаётся 1-D Вассерштейном -- по метрике, в которой
    # расстояние между соседними узлами сетки равно w_k, -- а невзвешенный оказывается частным
    # случаем w = const.
    #
    # ПОВЕРХНОСТЬ ВЕСА И ПОВЕРХНОСТЬ ЧЛЕНА. Вектор выведен под режим "emd" (цена ПЕРЕНОСА через
    # границу бинов). В режиме "mse" слагаемые живут на другой поверхности -- цене РАЗМЕЩЕНИЯ в
    # бине, -- и тот же вектор там стоит не на своей поверхности.
    sigma_shape_bin_weights: Optional[str] = None
    sigma_val_data: Optional[str] = None  # path to sigma-profile val set for warmup early-stop
    freeze_sigma_head_during_sle: bool = False  # freeze head_sigma in SLE phases 2/3 to hold the σ-manifold
    # ПЛЕЧО "ЗАМОРОЖЕННОЕ ПРОМЕЖУТОЧНОЕ". Профиль и площадь уходят в замыкание ДЕТАЧЕННЫМИ, то
    # есть градиент потери по растворимости до σ-головы не доходит вовсе, тогда как внешняя
    # σ-супервизия продолжает её обучать. Это и есть конструкция ближайшего предшественника
    # (yang2025tennet), где предсказатель профиля обучен по квантовой химии и заморожен, и
    # единственное место, где статья даёт рекомендацию без измерения.
    #
    # ПОЧЕМУ НЕ freeze_sigma_head_during_sle. Тот замораживает ВЕСА головы, но энкодер под ней
    # продолжает учиться от растворимости, так что профиль всё равно движется. Детач режет путь,
    # а не параметры, и потому отвечает на поставленный вопрос.
    #
    # Прецедент в этом же файле: объём уходит в комбинаторный член детаченным по той же причине.
    sigma_detached_from_closure: bool = False
    # ПЛЕЧО "РОЛЬ-ИНВАРИАНТНАЯ ГОЛОВА". σ-голова читает представление, посчитанное с ОДНОЙ ролью
    # для обеих молекул, поэтому соединение получает один профиль независимо от того, с какой
    # стороны пары оно стоит. Закрывает возражение "нарушение референсного состояния есть
    # следствие вашего же выбора": под этим переключателем γ(i в i) = 1 выполняется тождественно.
    # Стоит одного лишнего прохода энкодера на батч.
    sigma_role_invariant: bool = False
    # Train-time TRUE-σ injection (disambiguates the eval-only oracle floor: does the
    # crystal/correction branch co-adapt to correct σ, or is COSMO-SAC misspecified?).
    train_sigma_oracle: bool = False            # inject VT-2005 σ during SLE phases 2/3 (not just eval)
    sigma_oracle_artifact: Optional[str] = None  # sigma_profiles.csv for train-time injection
    sigma_oracle_side: str = "both"             # "solute" | "solvent" | "both"
    # Sigma-profile warmup pretraining (run_sigma_warmup_pretraining).
    # Pretrain head_sigma + encoder on the σ pool BEFORE the SLE curriculum.
    sigma_warmup_epochs: int = 0               # 0 = skip warmup
    sigma_warmup_lr: float = 3e-4
    sigma_warmup_patience: int = 20
    sigma_warmup_min_epochs: int = 5
    sigma_area_anchor_mae_tol: float = 30.0    # raw Å² mean abs area error gate
    sigma_area_anchor_strict: bool = False     # raise (vs warn) if gate fails

    # --- Numerical stability ---
    eps: float = 1e-10
    tau_clamp: float = 30.0
    grad_clip: float = 1.0
    weight_decay: float = 5e-4
    direct_weight_decay: float = 1e-5
    correction_max_abs: float = 2.0
    correction_Tm_max_delta: float = 60.0
    correction_dH_fraction: float = 0.25
    correction_tau_max_delta: float = 2.0
    correction_output_mode: str = "parameter"  # "parameter" or "ln_x2_residual"
    correction_ln_x2_max_delta: float = 1.0
    # v2 (Gate B): force the confidence gate fully open in ln_x2_residual mode so the output
    # residual is NOT throttled by (1-confidence). Default off (v1 behavior preserved).
    correction_force_open_gate: bool = False
    monotonicity_force_explicit: bool = False

    # --- Training ---
    batch_size: int = 64
    num_workers: int = 0  # DataLoader workers; 0 = featurize in main process (see make_loader)
    include_water_solubility: bool = True
    use_pair_temperature_batching: bool = True
    pair_temperature_min_group_size: int = 2
    pair_temperature_group_chunk_size: int = 4
    idac_aux_steps_per_epoch: int = 0
    idac_aux_phase1_weight: Optional[float] = None
    idac_aux_phase2_weight: Optional[float] = None
    idac_aux_phase3_weight: Optional[float] = None
    # External single-component crystal auxiliary stream (T_m / dH_fus pool).
    # Grounds the crystal branch with a much larger pure-component label pool than
    # the solubility pairs, enabling scaffold transfer of Phi(T). Mirrors the IDAC
    # aux stream above. 0 disables it.
    crystal_aux_steps_per_epoch: int = 0
    crystal_aux_phase1_weight: Optional[float] = None
    crystal_aux_phase2_weight: Optional[float] = None
    crystal_aux_phase3_weight: Optional[float] = None
    vant_hoff_slope_scale: float = 5000.0
    vant_hoff_intercept_scale: float = 10.0
    vant_hoff_fit_r2_min: float = 0.95
    vh_anchor_default_weight: float = 1.0
    temperature_rescue_ramp_epochs: int = 0
    use_source_uncertainty_weights: bool = False
    source_uncertainty_csv: str = (
        "results/source_uncertainty_audit_reviewed/"
        "supervised_rows_with_source_uncertainty.csv"
    )
    source_uncertainty_weight_mode: str = "inverse_variance"
    source_uncertainty_default_sigma_ln_x2: float = 0.75
    source_uncertainty_min_sigma_ln_x2: float = 0.20
    source_uncertainty_min_weight: float = 0.25
    source_uncertainty_max_weight: float = 4.0
    sol_loss_type: str = "huber"  # "huber", "mae", "weighted_huber", "weighted_mae"
    sol_bin_weight_mode: str = "none"  # "none", "inverse_frequency", or "fixed"
    sol_bin_edges: tuple[float, ...] = (-25.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0)
    sol_bin_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    sol_bin_weight_min: float = 0.25
    sol_bin_weight_max: float = 8.0
    sol_variance_loss_weight: float = 0.0
    sol_variance_eps: float = 1.0e-6
    decorr_min_samples: int = 3
    decorr_eps: float = 1.0e-8
    lr_phase1: float = 3e-4
    lr_phase2: float = 1e-4
    lr_phase3: float = 1e-6
    epochs_phase1: int = 50
    epochs_phase2: int = 200
    epochs_phase3: int = 50
    phase2_correction_unfreeze_epoch: int = 20
    warmup_epochs: int = 5
    patience: int = 20
    early_stopping_patience: Optional[int] = None
    early_stopping_phase3_patience: Optional[int] = None
    early_stopping_min_epochs: int = 10
    probe_every: int = 0

    phase1_loss_weights: Optional[Dict[str, float]] = None
    phase2_loss_weights: Optional[Dict[str, float]] = None
    phase3_loss_weights: Optional[Dict[str, float]] = None

    @property
    def resolved_descriptor_hidden_dim(self) -> int:
        """Return the effective hidden dim for shared descriptor augmentation."""
        if self.descriptor_augmentation_hidden_dim is not None:
            return int(self.descriptor_augmentation_hidden_dim)
        return int(self.descriptor_hidden_dim)

    @property
    def requires_group_prior_features(self) -> bool:
        """Return True when any active path needs fixed group-count features."""
        return bool(self.use_group_priors or self.use_nrtl_group_prior)

    @classmethod
    def from_yaml(cls, path: str) -> "TGNNSolvConfig":
        """
        Load configuration from a YAML file.

        Supports nested YAML structure (model, training, data, loss_weights
        sections).  Flattens nested dicts and filters to only fields present
        in TGNNSolvConfig.  The ``loss_weights`` section receives special
        handling: its ``phase1``/``phase2``/``phase3`` sub-dicts are mapped
        to ``phase1_loss_weights`` etc.

        Unknown keys are silently ignored.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for from_yaml(). "
                "Install with: pip install pyyaml"
            )

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        # ✦ CHANGED — two-level flattening with phase-weight mapping
        _PHASE_MAP = {
            "phase1": "phase1_loss_weights",
            "phase2": "phase2_loss_weights",
            "phase3": "phase3_loss_weights",
        }

        flat_dict: dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                if key in {"phase1_loss_weights", "phase2_loss_weights", "phase3_loss_weights"}:
                    flat_dict[key] = value
                    continue
                for nested_key, nested_value in value.items():
                    mapped = _PHASE_MAP.get(nested_key)
                    if mapped is not None and isinstance(nested_value, dict):
                        flat_dict[mapped] = nested_value
                    elif not isinstance(nested_value, dict):
                        flat_dict[nested_key] = nested_value
                    # else: deeper nesting without mapping → skip
            else:
                flat_dict[key] = value

        field_map = {field.name: field for field in fields(cls)}
        filtered_dict = {}
        for key, value in flat_dict.items():
            field = field_map.get(key)
            if field is None:
                continue
            if isinstance(field.default, tuple) and isinstance(value, list):
                value = tuple(value)
            filtered_dict[key] = value

        return cls(**filtered_dict)

    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to output YAML file. Parent directories are created if needed.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for to_yaml(). "
                "Install with: pip install pyyaml"
            )

        def _make_yaml_safe(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _make_yaml_safe(item)
                    for key, item in value.items()
                }
            if isinstance(value, tuple):
                return [_make_yaml_safe(item) for item in value]
            if isinstance(value, list):
                return [_make_yaml_safe(item) for item in value]
            return value

        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict and save to YAML
        config_dict = _make_yaml_safe(asdict(self))
        
        with open(path, 'w') as f:
            yaml.safe_dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
