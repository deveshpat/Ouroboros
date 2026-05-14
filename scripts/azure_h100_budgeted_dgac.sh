#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export OUROBOROS_CUDA_BUILD_MAX_JOBS="${OUROBOROS_CUDA_BUILD_MAX_JOBS:-4}"
export OUROBOROS_FLASH_ATTN_NVCC_THREADS="${OUROBOROS_FLASH_ATTN_NVCC_THREADS:-4}"

export AZURE_REGION="${AZURE_REGION:-southcentralus}"
export AZURE_H100_SKU="${AZURE_H100_SKU:-Standard_NC40ads_H100_v5}"
export AZURE_HOURLY_USD="${AZURE_HOURLY_USD:-8.38}"
export AZURE_PLANNED_HOURS="${AZURE_PLANNED_HOURS:-15.75}"
export AZURE_BUDGET_USD="${AZURE_BUDGET_USD:-190}"
export AZURE_SAFETY_BUFFER="${AZURE_SAFETY_BUFFER:-0.25}"
export AZURE_OVERHEAD_USD="${AZURE_OVERHEAD_USD:-4}"
export AZURE_FULL_TIMEOUT="${AZURE_FULL_TIMEOUT:-15h}"
export AZURE_SESSION_TIMEOUT_HOURS="${AZURE_SESSION_TIMEOUT_HOURS:-14.75}"

export OUROBOROS_DILOCO_STATE_REPO="${OUROBOROS_DILOCO_STATE_REPO:-WeirdRunner/Ouroboros}"
export OUROBOROS_WANDB_PROJECT="${OUROBOROS_WANDB_PROJECT:-ouroboros-stage3-jamba}"
export OUROBOROS_AZURE_BATCH_SIZE="${OUROBOROS_AZURE_BATCH_SIZE:-4}"
export OUROBOROS_AZURE_GRAD_ACCUM="${OUROBOROS_AZURE_GRAD_ACCUM:-8}"
export OUROBOROS_AZURE_VAL_BATCH_SIZE="${OUROBOROS_AZURE_VAL_BATCH_SIZE:-2}"
export OUROBOROS_AZURE_MAX_SEQ_LEN="${OUROBOROS_AZURE_MAX_SEQ_LEN:-2048}"
export OUROBOROS_AZURE_CANARY_MAX_SAMPLES="${OUROBOROS_AZURE_CANARY_MAX_SAMPLES:-512}"
export OUROBOROS_AZURE_CANARY_STEPS="${OUROBOROS_AZURE_CANARY_STEPS:-5}"
export OUROBOROS_AZURE_FULL_OUTPUT_DIR="${OUROBOROS_AZURE_FULL_OUTPUT_DIR:-runs/azure_h100_dgac}"
export OUROBOROS_AZURE_CANARY_OUTPUT_DIR="${OUROBOROS_AZURE_CANARY_OUTPUT_DIR:-runs/azure_h100_dgac_canary}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[azure-entrypoint] FATAL: HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required." >&2
  exit 2
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"

if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_KEY:-}" ]]; then
  echo "[azure-entrypoint] FATAL: WANDB_API_KEY or WANDB_KEY is required for online metrics." >&2
  exit 2
fi
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY}}"

echo "[azure-entrypoint] Ouroboros Azure H100 DGAC run"
echo "[azure-entrypoint] region=${AZURE_REGION} sku=${AZURE_H100_SKU} planned_hours=${AZURE_PLANNED_HOURS} budget_usd=${AZURE_BUDGET_USD}"
echo "[azure-entrypoint] diloco_state_repo=${OUROBOROS_DILOCO_STATE_REPO} wandb_project=${OUROBOROS_WANDB_PROJECT}"
echo "[azure-entrypoint] batch=${OUROBOROS_AZURE_BATCH_SIZE} grad_accum=${OUROBOROS_AZURE_GRAD_ACCUM} max_seq_len=${OUROBOROS_AZURE_MAX_SEQ_LEN}"

python -m ouroboros.azure_cost_guard \
  --region "${AZURE_REGION}" \
  --sku "${AZURE_H100_SKU}" \
  --hourly_usd "${AZURE_HOURLY_USD}" \
  --planned_hours "${AZURE_PLANNED_HOURS}" \
  --budget_usd "${AZURE_BUDGET_USD}" \
  --safety_buffer "${AZURE_SAFETY_BUFFER}" \
  --overhead_usd "${AZURE_OVERHEAD_USD}" \
  --require_budget_fit

python -m pip install -q --upgrade pip
if python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("torch")
if spec is None:
    raise SystemExit(3)

import torch

cuda_ok = bool(torch.cuda.is_available())
version_ok = str(torch.__version__).startswith("2.6.0")
if not (cuda_ok and version_ok):
    raise SystemExit(3)

print(f"[azure-entrypoint] Torch already ready: {torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
then
  :
else
  echo "[azure-entrypoint] Installing PyTorch 2.6.0 CUDA 12.4..."
  python -m pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
fi

nvidia-smi
python - <<'PY'
import torch
print(f"[azure-entrypoint] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on the Azure worker.")
print(f"[azure-entrypoint] gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

COMMON_ARGS=(
  --use_halt_gate
  --resume_from_diloco_anchor
  --diloco_state_repo "${OUROBOROS_DILOCO_STATE_REPO}"
  --data_dir data/coconut_v1
  --use_4bit
  --latent_cache
  --epochs_per_stage 3
  --max_stage 10
  --max_seq_len "${OUROBOROS_AZURE_MAX_SEQ_LEN}"
  --max_grad_norm 0.3
  --batch_size "${OUROBOROS_AZURE_BATCH_SIZE}"
  --grad_accum "${OUROBOROS_AZURE_GRAD_ACCUM}"
  --val_batch_size "${OUROBOROS_AZURE_VAL_BATCH_SIZE}"
  --val_skip_buffer_minutes 720
  --graceful_exit_buffer_minutes 20
  --log_every 5
  --no-gen_every_stage
  --keep_checkpoints_per_stage 1
  --wandb_project "${OUROBOROS_WANDB_PROJECT}"
  --wandb_mode online
)

if [[ -n "${OUROBOROS_WANDB_ENTITY:-}" ]]; then
  COMMON_ARGS+=(--wandb_entity "${OUROBOROS_WANDB_ENTITY}")
fi

echo "[azure-entrypoint] Starting 5-step H100 canary..."
python -m torch.distributed.run --standalone --nproc_per_node=1 jamba_coconut_finetune.py \
  "${COMMON_ARGS[@]}" \
  --epochs_per_stage 1 \
  --session_timeout_hours 0.75 \
  --graceful_exit_buffer_minutes 10 \
  --max_samples "${OUROBOROS_AZURE_CANARY_MAX_SAMPLES}" \
  --max_train_steps "${OUROBOROS_AZURE_CANARY_STEPS}" \
  --log_every 1 \
  --output_dir "${OUROBOROS_AZURE_CANARY_OUTPUT_DIR}" \
  --hf_stage_subdir "${OUROBOROS_AZURE_CANARY_OUTPUT_DIR}" \
  --wandb_run_name "Azure H100 SCUS DGAC canary"

echo "[azure-entrypoint] Canary passed. Starting budgeted full run..."
timeout --preserve-status "${AZURE_FULL_TIMEOUT}" \
  python -m torch.distributed.run --standalone --nproc_per_node=1 jamba_coconut_finetune.py \
    "${COMMON_ARGS[@]}" \
    --session_timeout_hours "${AZURE_SESSION_TIMEOUT_HOURS}" \
    --push_to_hub \
    --output_dir "${OUROBOROS_AZURE_FULL_OUTPUT_DIR}" \
    --hf_stage_subdir "${OUROBOROS_AZURE_FULL_OUTPUT_DIR}" \
    --wandb_run_name "Azure H100 SCUS DGAC full budgeted"
