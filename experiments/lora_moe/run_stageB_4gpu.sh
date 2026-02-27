#!/bin/bash
# Stage B: Qwen2.5-14B / UltraChat-200k on 4× H200 SXM (FSDP)
# Uses accelerate + FSDP for sharded training across 4 GPUs.
#
# FSDP shards model weights + optimizer states across GPUs, reducing per-GPU
# memory from ~88 GB (single GPU) to ~25 GB → allows batch_size=8 per device
# with gradient_checkpointing=false for extra throughput.
#
# Effective batch size: 4 GPUs × 8 per-device × 1 accum = 32  (same as Stage B 1-GPU)
#
# Usage:
#   cd /path/to/lora_moe
#   bash experiments/lora_moe/run_stageB_4gpu.sh
#
#   # Debug run:
#   bash experiments/lora_moe/run_stageB_4gpu.sh --debug

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_FLAG="--debug"
  echo ">>> DEBUG MODE"
fi

ACCEL_CFG="experiments/lora_moe/configs/accelerate_fsdp_stageB.yaml"
SPARSE_CFG="experiments/lora_moe/configs/stageB_sparse_moe.yaml"
SOFT_CFG="experiments/lora_moe/configs/stageB_soft_lora_moe.yaml"

echo "============================================================"
echo "  Stage B (4× H200 FSDP): Qwen2.5-14B / UltraChat-200k"
echo "============================================================"

# ---- [1/2] Train Sparse-MoE ----
echo ""
echo "[1/2] Training Sparse-MoE (stageB, 4-GPU FSDP)..."
accelerate launch \
    --config_file "${ACCEL_CFG}" \
    --num_processes 4 \
    -m experiments.lora_moe.run_experiment \
    --config "${SPARSE_CFG}" \
    --gradient_accumulation_steps 1 \
    ${DEBUG_FLAG}

# ---- [2/2] Train Soft-LoRA-MoE ----
echo ""
echo "[2/2] Training Soft-LoRA-MoE (stageB, 4-GPU FSDP)..."
accelerate launch \
    --config_file "${ACCEL_CFG}" \
    --num_processes 4 \
    -m experiments.lora_moe.run_experiment \
    --config "${SOFT_CFG}" \
    --gradient_accumulation_steps 1 \
    ${DEBUG_FLAG}

# ---- Benchmark both variants (4 workloads each) ----
echo ""
echo "Benchmarking trained models (single-GPU inference)..."

for method in sparse_moe soft_lora_moe; do
    dir="outputs/stageB_${method}"
    if [[ "${DEBUG_FLAG}" == "--debug" ]]; then
        dir="${dir}_debug"
    fi

    for wl in homogeneous heterogeneous bursty mixed_semantic; do
        echo "  Benchmarking ${method} / ${wl}..."
        python3 -m experiments.lora_moe.benchmark_stageBC \
            --model_dir "${dir}" \
            --method "${method}" \
            --workload "${wl}" \
            --num_requests 200
    done
done

# ---- Analyze ----
echo ""
echo "Running analysis..."

SPARSE_DIR="outputs/stageB_sparse_moe"
SOFT_DIR="outputs/stageB_soft_lora_moe"
if [[ "${DEBUG_FLAG}" == "--debug" ]]; then
    SPARSE_DIR="${SPARSE_DIR}_debug"
    SOFT_DIR="${SOFT_DIR}_debug"
fi

python3 -m experiments.lora_moe.analyze_stageBC \
    --dirs "${SPARSE_DIR}" "${SOFT_DIR}" \
    --labels "Sparse-MoE (B)" "Soft-LoRA-MoE (B)" \
    --out_dir figures/stageB/

echo ""
echo "============================================================"
echo "  Stage B (4-GPU) complete. Figures saved to figures/stageB/"
echo "============================================================"
