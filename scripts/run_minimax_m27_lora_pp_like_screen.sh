#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/checkpoints/logs"
if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
  LOG_DIR="${ROOT_DIR}/logs"
  if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
    LOG_DIR="${HOME}/automodel-logs"
    mkdir -p "${LOG_DIR}"
  fi
  echo "Warning: checkpoints is not writable; writing logs to ${LOG_DIR}"
fi

# PP-like profile adapted to 4 GPUs while keeping LoRA recipe.
SESSION_NAME="${SESSION_NAME:-minimax_m27_lora_pp_like}"
MAX_STEPS="${MAX_STEPS:-100}" # Match original LoRA recipe step budget.
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}" # Match original LoRA effective global batch size.
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}.log"

DOCKER_CMD=$(cat <<EOF
cd "${ROOT_DIR}" && docker run --rm --gpus all --network host --shm-size=64g \
  -v "${ROOT_DIR}:/opt/Automodel" \
  -v "${ROOT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
  -w /opt/Automodel \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE}" \
  automodel --nproc-per-node=4 \
  examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_lora.yaml \
  --distributed.tp_size 1 \
  --distributed.cp_size 1 \
  --distributed.pp_size 2 \
  --distributed.ep_size 2 \
  --distributed.sequence_parallel false \
  --distributed.pipeline.pp_schedule interleaved1f1b \
  --distributed.pipeline.pp_microbatch_size 1 \
  --distributed.pipeline.round_virtual_stages_to_pp_multiple down \
  --distributed.pipeline.scale_grads_in_schedule false \
  --distributed.pipeline.patch_inner_model false \
  --distributed.pipeline.patch_causal_lm_model false \
  --distributed.pipeline.layers_per_stage 2 \
  --distributed.moe.reshard_after_forward false \
  --distributed.moe.wrap_outer_model false \
  --model.backend.linear te \
  --model.backend.rope_fusion false \
  --model.backend.experts torch_mm \
  --step_scheduler.local_batch_size ${LOCAL_BATCH_SIZE} \
  --step_scheduler.global_batch_size ${GLOBAL_BATCH_SIZE} \
  --step_scheduler.max_steps ${MAX_STEPS} \
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

echo "Started PP-like LoRA training in detached screen session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo
echo "Attach to session:"
echo "  screen -r ${SESSION_NAME}"
echo
echo "Watch logs without attaching:"
echo "  tail -f \"${LOG_FILE}\""
