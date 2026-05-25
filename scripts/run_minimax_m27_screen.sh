#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/checkpoints/logs"
if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
  # checkpoints/ may be root-owned when created by docker; fall back to user-writable dirs.
  LOG_DIR="${ROOT_DIR}/logs"
  if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
    LOG_DIR="${HOME}/automodel-logs"
    mkdir -p "${LOG_DIR}"
  fi
  echo "Warning: checkpoints is not writable; writing logs to ${LOG_DIR}"
fi

SESSION_NAME="${SESSION_NAME:-minimax_m27_1600}"
MODE="${MODE:-train}" # train | prep | sweep
MAX_STEPS="${MAX_STEPS:-1600}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EP_SIZE="${EP_SIZE:-4}"
LR="${LR:-1e-5}"
PEFT_DIM="${PEFT_DIM:-8}"
PEFT_ALPHA="${PEFT_ALPHA:-32}"
PRINT_PARAM_GRADS="${PRINT_PARAM_GRADS:-0}"
PARAM_GRAD_MAX_LINES="${PARAM_GRAD_MAX_LINES:-0}"
ENABLE_LR_SCHEDULER="${ENABLE_LR_SCHEDULER:-0}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-16}"
LR_MIN="${LR_MIN:-2e-6}"
HF_OFFLINE="${HF_OFFLINE:-0}"
TRAIN_DISABLE_CHECKPOINT="${TRAIN_DISABLE_CHECKPOINT:-0}"
PREP_CKPT_EVERY_STEPS="${PREP_CKPT_EVERY_STEPS:-1}"
PREP_VAL_EVERY_STEPS="${PREP_VAL_EVERY_STEPS:-1000000}"
RESTORE_FROM_PATH="${RESTORE_FROM_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/sweep_cache}"
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}.log"

SCHEDULER_OVERRIDES=""
if [[ "${ENABLE_LR_SCHEDULER}" == "1" || "${ENABLE_LR_SCHEDULER}" == "true" ]]; then
  SCHEDULER_OVERRIDES="--lr_scheduler.lr_decay_style ${LR_DECAY_STYLE} --lr_scheduler.lr_warmup_steps ${LR_WARMUP_STEPS} --lr_scheduler.min_lr ${LR_MIN}"
fi

CKPT_DIR_EFFECTIVE="${CHECKPOINT_DIR}"
if ! mkdir -p "${CKPT_DIR_EFFECTIVE}" 2>/dev/null; then
  CKPT_DIR_EFFECTIVE="${ROOT_DIR}/local_checkpoints/sweep_cache"
  mkdir -p "${CKPT_DIR_EFFECTIVE}"
fi

MODE_OVERRIDES=""
case "${MODE}" in
  train)
    if [[ "${TRAIN_DISABLE_CHECKPOINT}" == "1" || "${TRAIN_DISABLE_CHECKPOINT}" == "true" ]]; then
      MODE_OVERRIDES="--checkpoint.enabled false"
    fi
    ;;
  prep)
    MODE_OVERRIDES="--checkpoint.enabled true --checkpoint.checkpoint_dir ${CKPT_DIR_EFFECTIVE} --step_scheduler.ckpt_every_steps ${PREP_CKPT_EVERY_STEPS} --step_scheduler.val_every_steps ${PREP_VAL_EVERY_STEPS}"
    ;;
  sweep)
    if [[ -z "${RESTORE_FROM_PATH}" ]]; then
      echo "Error: MODE=sweep requires RESTORE_FROM_PATH to be set."
      exit 1
    fi
    MODE_OVERRIDES="--restore_from.path ${RESTORE_FROM_PATH}"
    ;;
  *)
    echo "Error: unsupported MODE='${MODE}'. Use train, prep, or sweep."
    exit 1
    ;;
esac

DOCKER_CMD=$(cat <<EOF
cd "${ROOT_DIR}" && docker run --rm --gpus all --network host --shm-size=64g \
  -v "${ROOT_DIR}:/opt/Automodel" \
  -v "${ROOT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
  -w /opt/Automodel \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e HF_HUB_OFFLINE=${HF_OFFLINE} \
  -e TRANSFORMERS_OFFLINE=${HF_OFFLINE} \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e AUTOMODEL_PRINT_PARAM_GRAD_INFO=${PRINT_PARAM_GRADS} \
  -e AUTOMODEL_PRINT_PARAM_GRAD_MAX_LINES=${PARAM_GRAD_MAX_LINES} \
  "${IMAGE}" \
  automodel --nproc-per-node=4 \
  examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_lora.yaml \
  --distributed.ep_size ${EP_SIZE} \
  --distributed.activation_checkpointing true \
  --optimizer.lr ${LR} \
  --peft.dim ${PEFT_DIM} \
  --peft.alpha ${PEFT_ALPHA} \
  --step_scheduler.local_batch_size ${LOCAL_BATCH_SIZE} \
  --step_scheduler.global_batch_size ${GLOBAL_BATCH_SIZE} \
  --step_scheduler.max_steps ${MAX_STEPS} \
  ${SCHEDULER_OVERRIDES} \
  ${MODE_OVERRIDES} \
  2>&1 | tee "${LOG_FILE}"
EOF
)

if ! command -v screen >/dev/null 2>&1; then
  echo "Error: 'screen' is not installed. Install screen and retry."
  exit 1
fi

if screen -list 2>/dev/null | grep -q "\\.${SESSION_NAME}[[:space:]]"; then
  echo "Error: screen session '${SESSION_NAME}' already exists."
  echo "Use: screen -r ${SESSION_NAME}"
  exit 1
fi

screen -dmS "${SESSION_NAME}" bash -lc "${DOCKER_CMD}"

echo "Started training in detached screen session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Mode: ${MODE}"
if [[ "${MODE}" == "prep" ]]; then
  echo "Checkpoint dir: ${CKPT_DIR_EFFECTIVE}"
fi
echo
echo "Attach to session:"
echo "  screen -r ${SESSION_NAME}"
echo
echo "Watch logs without attaching:"
echo "  tail -f \"${LOG_FILE}\""
