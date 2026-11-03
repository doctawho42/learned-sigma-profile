#!/usr/bin/env bash
# P3 run_e5: decisive lever-C comparison — NRTL / DirectGNN-h64 / cosmo {ungrounded,
# grounded-A residual-only, grounded-B +SG} / oracle — on the corrected split, >=3 seeds,
# metrics intersection-locked by run_e5_comparison.py. REAL metrics need GPU; CPU = smoke.
#
# Usage (GPU) — name the device, so a box whose CUDA broke stops instead of falling to CPU:
#   DEVICE=cuda bash scripts/experiments/run_e5_sigma_grounding.sh
# 1-seed pilot (validate pipeline + read ungrounded std(lng) to calibrate --lngamma-band):
#   DEVICE=cuda SEEDS=42 bash scripts/experiments/run_e5_sigma_grounding.sh
# Nothing typed: --device is not passed and each child reads this box for itself.
# CPU smoke (no meaningful metrics — just checks the wiring):
#   DEVICE=cpu SEEDS=42 WARMUP_EPOCHS=1 SIGMA_STEPS=2 \
#     EXTRA_TRAIN_ARGS="--epochs-phase1 1 --epochs-phase2 1 --epochs-phase3 1" \
#     bash scripts/experiments/run_e5_sigma_grounding.sh
#
# PREEMPTION-SAFE (Colab / Studio Lab): re-running resumes automatically —
#   * an arm whose predictions.csv already exists is SKIPPED;
#   * an arm with a partial checkpoint RESUMES mid-training (--checkpoint-every/--resume).
# Point CKPT_DIR and OUT_DIR at a persistent location (e.g. mounted Drive) so progress
# survives a disconnect.
#
# Env overrides (defaults):
#   PY               $HOME/anaconda3/envs/tgnn-solv/bin/python
#   DEVICE           (unset)         torch device for training/export; see below
#   DATA_DIR         notebooks/data/processed       (the CORRECTED split)
#   SIGMA_DIR        notebooks/data/processed_sigma_aux_stream
#   OUT_DIR          results/e5_sigma_grounding
#   CKPT_DIR         checkpoints/e5
#   SEEDS            "42 43 44"      space-separated list of random seeds
#   ARMS             all six         space-separated arm subset (smoke: ARMS="ungrounded")
#                                    E2 (кристалл закреплён снаружи): ARMS="grounded_a grounded_a_detachcrystal"
#                                    G4 (пол на глубине sigma): ARMS="grounded_a grounded_a_shapefloor"
#   DIRECT_EPOCHS    "" (config 110) directgnn --epochs override; smoke: DIRECT_EPOCHS=1
#                                    (EXTRA_TRAIN_ARGS is TGNN phase-epochs — directgnn ignores it)
#   NUM_WORKERS      "" (0)          DataLoader workers for every training arm (e.g. 8 on a many-core box)
#   SIGMA_STEPS      21              sigma aux steps per epoch
#   WARMUP_EPOCHS    40              sigma warmup epochs before SLE
#   CHECKPOINT_EVERY 5               save a resumable training checkpoint every N epochs
#   EXTRA_TRAIN_ARGS ""              passed verbatim to TGNN train.py calls
#   SIGMA_ARTIFACT   results/sigma_profile_artifact/sigma_profiles.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
export KMP_DUPLICATE_LIB_OK=TRUE   # rdkit/torch/sklearn libomp clash on macOS

PY="${PY:-$HOME/anaconda3/envs/tgnn-solv/bin/python}"
# Путь к интерпретатору вшит по умолчанию и на другой машине промахивается. При промахе шелл печатал
# "No such file or directory" и шёл дальше, а наружу уходил код 0 -- то есть десятичасовой арм мог
# "закончиться" мгновенно и успешно. Проверяем ОДИН раз, здесь, и падаем громко.
if ! "${PY}" -c 'import sys' >/dev/null 2>&1; then
  echo "ОСТАНОВ: интерпретатор '${PY}' не исполняется." >&2
  echo "  Задайте PY=<путь к python> -- например PY=/mnt/storage/\$USER/tgnn-solv/env/bin/python" >&2
  exit 2
fi
# DEVICE is a demand, not a default. Unset, --device is not passed at all and each child
# reads this box through tgnn_solv.device.default_device(); set, the string reaches every
# child, where resolve_device refuses to substitute CPU for an accelerator you named. That
# refusal is the point of typing DEVICE=cuda on a GPU box: a broken driver stops the run
# instead of quietly costing it ten hours. `DEVICE=cuda` used to be this line's default,
# which made a Mac with nothing typed fail at the first arm.
DEVICE="${DEVICE:-}"
DEV_ARGS=(); [ -n "${DEVICE}" ] && DEV_ARGS=(--device "${DEVICE}")
DATA_DIR="${DATA_DIR:-notebooks/data/processed}"
SIGMA_DIR="${SIGMA_DIR:-notebooks/data/processed_sigma_aux_stream}"
OUT_DIR="${OUT_DIR:-results/e5_sigma_grounding}"
CKPT_DIR="${CKPT_DIR:-checkpoints/e5}"
SEEDS="${SEEDS:-42 43 44}"
ARMS="${ARMS:-nrtl directgnn ungrounded grounded_a grounded_b oracle}"  # subset for smoke/targeting
SIGMA_STEPS="${SIGMA_STEPS:-21}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"   # e.g. --epochs-phase1 1 --epochs-phase2 1 --epochs-phase3 1 for CPU smoke
DIRECT_EPOCHS="${DIRECT_EPOCHS:-}"         # directgnn --epochs override; empty = config's matched 110 (smoke: 1)
NUM_WORKERS="${NUM_WORKERS:-}"             # DataLoader workers for all training arms; empty = 0 (main process)
NW_ARGS=(); [ -n "${NUM_WORKERS}" ] && NW_ARGS=(--num-workers "${NUM_WORKERS}")

# DataLoader workers pass tensors between processes as file descriptors, and several concurrent
# arms exhaust the default soft limit of 1024 during validation: one arm of the 2026-08-08 gate
# died with "RuntimeError: received 0 items of ancdata", which reads like a data bug and is not
# one. Raise the soft limit toward the hard one whenever workers are in play.
if [ -n "${NUM_WORKERS}" ] && [ "${NUM_WORKERS}" != "0" ]; then
  _fd_hard=$(ulimit -Hn 2>/dev/null || echo 1024)
  [ "${_fd_hard}" = "unlimited" ] && _fd_hard=65536
  _fd_want=65536; [ "${_fd_hard}" -lt "${_fd_want}" ] 2>/dev/null && _fd_want="${_fd_hard}"
  ulimit -n "${_fd_want}" 2>/dev/null || true
  echo "   file descriptors: soft=$(ulimit -n) hard=${_fd_hard} (workers=${NUM_WORKERS})"
fi
SIGMA_ARTIFACT="${SIGMA_ARTIFACT:-results/sigma_profile_artifact/sigma_profiles.csv}"

TRAIN="${DATA_DIR}/train.csv"; VAL="${DATA_DIR}/val.csv"; TEST="${DATA_DIR}/test.csv"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}"

echo "================================================================"
echo " run_e5: sigma-grounding decisive comparison"
echo "   device=${DEVICE:-per-script default (this box)}  data=${DATA_DIR}  sigma=${SIGMA_DIR}"
echo "   seeds=${SEEDS}  out=${OUT_DIR}  ckpt=${CKPT_DIR}  checkpoint_every=${CHECKPOINT_EVERY}"
echo "================================================================"

# --- 0. Guard: refuse to run on corrupted (+273 K) T_m labels ---
"${PY}" - "$TEST" <<'PYG'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
if "T_m" in df.columns:
    tm = pd.to_numeric(df["T_m"], errors="coerce")
    if "has_T_m" in df.columns:
        tm = tm[df["has_T_m"].astype(str).str.lower().isin({"true", "1", "1.0"})]
    med = float(tm.median())
    print(f"   test T_m median = {med:.1f} K", "(corrected)" if med < 560 else "(!! CORRUPTED)")
    assert med < 560, f"T_m median {med} looks +273 K corrupted -- run prepare_data with the fixed loader first."
print("T_m guard ok")
PYG

# --- Prereq: build the scaffold-disjoint sigma-VAL split if missing ---
if [ ! -f "${SIGMA_DIR}/sigma_val.csv" ]; then
  echo "== building sigma-VAL split (scaffold-disjoint from test+val) =="
  "${PY}" scripts/data/build_sigma_profile_aux_stream.py \
    --output-csv "${SIGMA_DIR}/sigma_train.csv" --output-val-csv "${SIGMA_DIR}/sigma_val.csv" \
    --val-fraction 0.1 --split-seed 0 \
    --exclude-scaffolds-from "${TEST}" "${VAL}"
fi

# --- Guard: oracle arm needs the VT-2005 sigma-profile artifact ---
if [ ! -f "${SIGMA_ARTIFACT}" ]; then
  echo "run_e5: missing VT-2005 oracle artifact ${SIGMA_ARTIFACT} — run scripts/data/ingest_vt2005_sigma_profiles.py first" >&2
  exit 1
fi

# --- Shared grounding args for the two grounded TGNN arms ---
# NOTE: any --set ... group must be LAST in the calling command (argparse nargs='*' is greedy).
COSMO_GROUND=(--sigma-train-data "${SIGMA_DIR}/sigma_train.csv" --sigma-val-data "${SIGMA_DIR}/sigma_val.csv" \
              --sigma-steps-per-epoch "${SIGMA_STEPS}" --sigma-warmup-epochs "${WARMUP_EPOCHS}" \
              --freeze-sigma-head-during-sle)

# --- Per-seed × per-arm loop ---
for SEED in ${SEEDS}; do
  SOUT="${OUT_DIR}/seed_${SEED}"; mkdir -p "${SOUT}"
  declare -a RUN_ARGS=()

  for arm in ${ARMS}; do
    ckpt="${CKPT_DIR}/${arm}_seed${SEED}.pt"
    pred="${SOUT}/${arm}_predictions.csv"
    echo "-- seed=${SEED}  arm=${arm}"

    # Arm-level resume: a finished arm (predictions written) is never retrained.
    if [ -f "${pred}" ]; then
      echo "   skip (predictions already exist: ${pred})"
      RUN_ARGS+=("--run" "${arm}=${pred}")
      continue
    fi

    # Mid-arm resume: continue a partial training checkpoint if present. Training args
    # for the heavy (TGNN/DirectGNN) arms; oracle is eval-only and ignores these.
    RESUME=()
    if [ -f "${ckpt}" ]; then echo "   resuming training from ${ckpt}"; RESUME=(--resume "${ckpt}"); fi
    # NW_ARGS (--num-workers) rides along here so it reaches all 5 training arms but not eval-only oracle.
    CKPT_ARGS=(--checkpoint "${ckpt}" --checkpoint-every "${CHECKPOINT_EVERY}" ${RESUME[@]+"${RESUME[@]}"} ${NW_ARGS[@]+"${NW_ARGS[@]}"})

    case "${arm}" in
      nrtl)
        "${PY}" scripts/train.py --config configs/paper_config_nrtl_h64L3.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" ${EXTRA_TRAIN_ARGS}
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      directgnn)
        "${PY}" scripts/train_directgnn.py --config configs/paper_config_directgnn_h64L3.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" ${DIRECT_EPOCHS:+--epochs ${DIRECT_EPOCHS}}
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type direct ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      ungrounded)
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          --sigma-steps-per-epoch 0 --sigma-warmup-epochs 0 ${EXTRA_TRAIN_ARGS}
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_a)
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS}
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_b)
        # --set cosmo_sac_wire_volume=true must be LAST (argparse nargs='*' is greedy)
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} --set cosmo_sac_wire_volume=true
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_a_shapefloor)
        # G4: sigma-супервизия спускается до ЗАДАННОЙ ГЛУБИНЫ и останавливается.
        # shape_loss := relu(shape_loss - 0.40); выше пола градиент прежний, на полу и ниже --
        # ровно нуль. Контроль -- плечо grounded_a, то есть та же конфигурация без одного флага.
        #
        # ОТКУДА 0.40. results/g4_descent_depth, пять сидов: спуск к депонированному профилю
        # улучшает AAD на IDAC с 1.3142 до 0.8201, но внутренний оптимум по глубине есть у всех
        # пяти сидов (ни одного на краю сетки), и дальше оценка портится. Выигрыш от глубины
        # +0.5490 +- 0.1129, положителен 5/5 -- это 3.6 сид-sd. Оптимум при EMD 0.3986 +- 0.0442;
        # перенос через долю снятого даёт 0.4457, внутри собственного разброса порога.
        # Единицы сверены: в обучающем потоке мишень нормирована на сумму 1 (все строки ровно
        # 1.0), предсказание есть softmax, и G4 мерил ровно эту величину.
        #
        # НАПРАВЛЕНИЕ ПРОВЕРЕНО НУЛЯМИ ПО МИШЕНИ: спуск на то же расстояние к мишени с той же
        # гистограммой масс и перемешанными бинами портит AAD монотонно (до 2.6987), случайная
        # мишень тоже; у обоих нулей лучшее -- не двигаться. Значит работает РАСПОЛОЖЕНИЕ массы
        # в депозите, а не факт смещения.
        #
        # ЧЕГО ЭТО ПЛЕЧО НЕ ЗНАЕТ ЗАРАНЕЕ. G4 мерил только ось ФИЗИЧНОСТИ (AAD на IDAC) на
        # подвыборке в 2875 записей. Ось ЗАДАЧИ (ln x2 MAE) из спуска по логитам недостижима --
        # она требует обучения, то есть ровно этого прогона. Поэтому ln x2 здесь измеряется
        # ПЕРВЫЙ раз, и ухудшение задачи -- допустимый исход, который надо прочитать, а не
        # объявить неудачей.
        #
        # --set обязан идти ПОСЛЕДНИМ: argparse с nargs='*' съедает всё до конца строки.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} --set sigma_shape_floor=0.40
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_a_detachcrystal)
        # E2: кристаллическая ветвь ЗАКРЕПЛЕНА СНАРУЖИ -- параметры T_m и dH_fus не получают
        # градиента из решателя SLE, то есть растворимость больше не может прятать ошибку
        # активностной ветви в кристаллическом члене. Контроль -- плечо grounded_a, то есть
        # ровно та же конфигурация без одного флага.
        #
        # ОСНОВАНИЕ -- ТЕОРЕМА, А НЕ ДОГАДКА. При dCp = 0 член Phi аффинен по 1/T,
        # активностный класс эту плоскость натягивает, и профилированная информация Фишера по
        # dH_fus равна НУЛЮ (SI.tex:697). Эмпирически на оценочной поверхности dH_fus измерен
        # у ОДНОГО растворяемого, T_m у 31 из 147, и ветвь на двух сидах из пяти хуже
        # константы. Бесплатные ворота пройдены (коммит 8126156): градиентное давление из
        # потери растворимости на T_m превышает давление кристаллической супервизии в 4.7
        # раза, то есть замораживать есть что.
        #
        # --set обязан идти ПОСЛЕДНИМ: argparse с nargs='*' съедает всё до конца строки.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} --set detach_crystal_params_in_sle=true
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_a_detachencoder)
        # КОНТРОЛЬ НА АТРИБУЦИЮ ЦЕНЫ E2. E2 забанковал цену внешнего закрепления кристаллической
        # ветви (+0.2763 и +0.9275 ln x2 на двух парах против парного пола 0.0744), но приписать
        # её компенсации НЕ УДАЛОСЬ: заявленная ось delta_phi_mean сменила знак между парами
        # (+0.8481 против -1.0791, обе за порогом), а разрез по давлению не сосредоточен и точечно
        # смотрит против компенсации. Остаётся возражение, которое иначе назовёт референт: цена
        # может быть ценой отрезания ЛЮБОГО кристаллического градиентного пути, а не именно
        # идентифицируемости.
        #
        # ЭТОТ ФЛАГ РЕЖЕТ ДРУГОЕ РЕБРО, и это ИЗМЕРЕНО, а не предположено:
        # g_sol_for_crystal = g_sol_pre.detach() (model.py:1267) -- кристаллическая голова читает
        # отцеплённое вложение, то есть её потеря больше не формирует общий энкодер, НО путь
        # растворимость -> параметры кристалла остаётся ОТКРЫТ. tests/test_e2_detach_encoder_gate.py
        # прибивает все четыре условия, включая решающее: при этом флаге градиент из ln_x2 в
        # параметры головы НЕ нуль, тогда как при detach_crystal_params_in_sle он ровно нуль.
        # Без этого утверждения контроль был бы дубликатом арма записи, и 16 GPU-часов ушли бы на
        # повторение известного.
        #
        # Контроль для сравнения -- УЖЕ ОБУЧЕННОЕ плечо grounded_a сида 42 из того же бандла:
        # тот же поток (sha 7721b661), тот же сплит, то же расписание, совпадение входов по
        # sha256 проверено. Поэтому это ОДНО плечо, а не пара.
        #
        # --set обязан идти ПОСЛЕДНИМ: argparse с nargs='*' съедает всё до конца строки.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} --set detach_crystal_from_encoder=true
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ;;
      grounded_a_truetrain)
        # DISAMBIGUATOR: inject TRUE (VT-2005) σ during TRAINING so the crystal/correction
        # branches co-adapt to correct σ (not the σ-head's prediction). Export WITH oracle σ
        # (true σ at test too) → does it now beat DirectGNN (co-adaptation artifact) or still
        # trail (COSMO-SAC misspecification)? Coverage: solvent ~99%, solute ~5% (read the
        # both-matched subset via sigma_oracle_mask_solute&solvent columns). --set must be LAST.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} \
          --set train_sigma_oracle=true sigma_oracle_side=both sigma_oracle_artifact="${SIGMA_ARTIFACT}"
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
          --sigma-oracle --sigma-oracle-side both --sigma-artifact "${SIGMA_ARTIFACT}" ;;
      grounded_a_truetrain_residual)
        # GATE B v1 (moonshot actionability): train-time TRUE σ (as grounded_a_truetrain) BUT with
        # an OUTPUT-SPACE residual escape valve enabled (correction_output_mode=ln_x2_residual →
        # ln_x2 = ln_x2_physics + (1-confidence)*bounded_residual). Tests the "faithful bottleneck"
        # claim: does giving physical σ a post-decoder residual to lean on make true-σ STOP hurting
        # (R² recover from truetrain 0.283 toward grounded_a 0.370 / direct 0.385)? KILL if ≈ truetrain
        # (residual can't rescue physical σ → the misspecified analytic closure is truly binding).
        # Eval injects true σ too (apples-to-apples vs truetrain; ONLY difference = the residual valve).
        # --set must be LAST (greedy nargs); the residual keys ride the same group as the oracle keys.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} \
          --set train_sigma_oracle=true sigma_oracle_side=both sigma_oracle_artifact="${SIGMA_ARTIFACT}" correction_output_mode=ln_x2_residual correction_ln_x2_max_delta=3.0
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
          --sigma-oracle --sigma-oracle-side both --sigma-artifact "${SIGMA_ARTIFACT}" ;;
      grounded_a_truetrain_residual_v2)
        # GATE B v2: as _residual but FORCE the confidence gate fully open
        # (correction_force_open_gate=true) so the output residual is applied at full strength.
        # v1's residual stayed gated SHUT (confidence≈1 → applied residual ~0, R² 0.236); this
        # genuinely tests the faithful-bottleneck fix — does an un-throttled output escape valve
        # let true-σ stop hurting? GREEN if R² recovers toward grounded 0.370 / direct 0.385;
        # NULL if ≈ truetrain 0.283 (closure binding even with an open valve). --set must be LAST.
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} \
          --set train_sigma_oracle=true sigma_oracle_side=both sigma_oracle_artifact="${SIGMA_ARTIFACT}" correction_output_mode=ln_x2_residual correction_ln_x2_max_delta=3.0 correction_force_open_gate=true
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
          --sigma-oracle --sigma-oracle-side both --sigma-artifact "${SIGMA_ARTIFACT}" ;;
      channel_swap)
        # CHANNEL-SWAP corroboration: coordinate_descent freezes Φ (crystal head + encoder)
        # in phase 2 while phase 1 grounds Φ on crystal-aux — so only the activity/correction
        # branch is refit, here against TRUE σ. Isolates whether the COSMO-SAC σ→γ map is the
        # ceiling independent of the crystal branch. --set must be LAST (greedy nargs).
        "${PY}" scripts/train.py --config configs/cosmo_sac.yaml \
          --train-data "${TRAIN}" --val-data "${VAL}" --test-data "${TEST}" \
          --seed "${SEED}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} "${CKPT_ARGS[@]}" \
          "${COSMO_GROUND[@]}" ${EXTRA_TRAIN_ARGS} \
          --set train_sigma_oracle=true sigma_oracle_side=both sigma_oracle_artifact="${SIGMA_ARTIFACT}" branch_training_mode=coordinate_descent
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${ckpt}" --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
          --sigma-oracle --sigma-oracle-side both --sigma-artifact "${SIGMA_ARTIFACT}" ;;
      oracle)
        # Reuse the grounded_a checkpoint; only the eval path changes (oracle sigma injection).
        # Measures the ceiling if the COSMO-SAC head had perfect sigma profiles.
        # Coverage ~5% of test rows (molecules with oracle profiles); rest are masked.
        # NOTE: comparison.json["per_arm"]["oracle"] is DILUTED (~grounded_a); the true
        # ceiling is the masked-subset block in this arm's *_predictions.summary.json.
        "${PY}" scripts/analysis/export_checkpoint_predictions.py \
          --checkpoint "${CKPT_DIR}/grounded_a_seed${SEED}.pt" \
          --data "${TEST}" --output "${pred}" \
          --model-type tgnn ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} \
          --sigma-oracle --sigma-oracle-side both \
          --sigma-artifact "${SIGMA_ARTIFACT}" ;;
      *)
        # ИМЯ ПЛЕЧА НЕ СОВПАЛО НИ С ОДНОЙ ВЕТВЬЮ -- ЭТО ОШИБКА, А НЕ ПРОПУСК.
        #
        # 2026-10-19 отсутствие этой ветви стоило запуска и трёх с половиной часов ожидания.
        # Плечо grounded_a_detachencoder было добавлено в ЭТОТ файл локально, но ядро Kaggle
        # копирует код из БАНДЛА (датасет tgnn-solv-e5, снимок от 2026-10-03), где его нет.
        # case провалился молча, обучение не запускалось (hours 0.0), и упал только агрегатор --
        # на отсутствующем CSV, то есть в двух шагах от настоящей причины. В логе между строкой
        # «-- seed=42 arm=...» и агрегатором не было НИЧЕГО, и прочитать это можно было только
        # заметив пустоту.
        #
        # Это тот же класс, что уже дважды стоил ядра: недостача, читаемая как норма.
        echo "run_e5: НЕИЗВЕСТНОЕ ПЛЕЧО '${arm}'." >&2
        echo "  Ветви case в этом файле: nrtl directgnn ungrounded grounded_a grounded_b" >&2
        echo "  grounded_a_shapefloor grounded_a_detachcrystal grounded_a_detachencoder" >&2
        echo "  grounded_a_truetrain grounded_a_truetrain_residual channel_swap oracle" >&2
        echo "  ЕСЛИ ПЛЕЧО ЕСТЬ В РЕПОЗИТОРИИ, НО НЕ ЗДЕСЬ -- запущен СТАРЫЙ КОД: ядро Kaggle" >&2
        echo "  берёт его из бандла, и бандл надо пересобрать" >&2
        echo "  (scripts/kaggle/build_kaggle_bundle.py) и выложить новой версией датасета." >&2
        exit 2 ;;
    esac

    RUN_ARGS+=("--run" "${arm}=${pred}")
  done

  echo "== seed ${SEED}: running aggregator =="
  "${PY}" scripts/analysis/run_e5_comparison.py "${RUN_ARGS[@]}" \
    --direct-label directgnn --out-json "${SOUT}/comparison.json"
done

echo ""
echo "================================================================"
echo " run_e5 complete -> ${OUT_DIR}"
echo " Per-seed comparison.json written to seed_*/comparison.json."
echo " Aggregate across seeds for the decisive verdict."
echo "================================================================"
