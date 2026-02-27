#!/bin/bash
# Stage C: Qwen2.5-32B / UltraChat-200k on 4-8× H200 with FSDP
# Trains Sparse-MoE and Soft-LoRA-MoE under multi-GPU load,
# benchmarks all 4 workloads, then runs analysis.
#
# Usage:
#   cd /path/to/lora_moe
#   bash experiments/lora_moe/run_stageC.sh
#
#   # Override GPU count (default: 8):
#   NUM_GPUS=4 bash experiments/lora_moe/run_stageC.sh
#
#   # Debug run (fast, small subset, 2 GPUs):
#   NUM_GPUS=2 bash experiments/lora_moe/run_stageC.sh --debug

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_GPUS="${NUM_GPUS:-8}"
DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_FLAG="--debug"
  echo ">>> DEBUG MODE (NUM_GPUS=${NUM_GPUS})"
fi

ACCEL_CONFIG="experiments/lora_moe/configs/accelerate_fsdp_stageC.yaml"

echo "============================================================"
echo "  Stage C: Qwen2.5-32B / UltraChat-200k (${NUM_GPUS}× GPU, FSDP)"
echo "============================================================"

# ---- [1/2] Train Sparse-MoE with torchrun + FSDP ----
echo ""
echo "[1/2] Training Sparse-MoE (stageC) with FSDP..."

# Option A: use accelerate launch (recommended for FSDP)
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageC_sparse_moe.yaml \
    ${DEBUG_FLAG}

# Option B (alternative — uncomment if accelerate is not available):
# torchrun --nproc_per_node="${NUM_GPUS}" --standalone \
#     -m experiments.lora_moe.run_experiment \
#     --config experiments/lora_moe/configs/stageC_sparse_moe.yaml \
#     ${DEBUG_FLAG}

# ---- [2/2] Train Soft-LoRA-MoE with torchrun + FSDP ----
echo ""
echo "[2/2] Training Soft-LoRA-MoE (stageC) with FSDP..."

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageC_soft_lora_moe.yaml \
    ${DEBUG_FLAG}

# ---- Benchmark both variants (4 workloads each) ----
echo ""
echo "Benchmarking trained models..."

for method in sparse_moe soft_lora_moe; do
    dir="outputs/stageC_${method}"
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

# ---- Analyze (includes rank-skew plot for Stage C) ----
echo ""
echo "Running analysis..."

SPARSE_DIR="outputs/stageC_sparse_moe"
SOFT_DIR="outputs/stageC_soft_lora_moe"
if [[ "${DEBUG_FLAG}" == "--debug" ]]; then
    SPARSE_DIR="${SPARSE_DIR}_debug"
    SOFT_DIR="${SOFT_DIR}_debug"
fi

python3 -m experiments.lora_moe.analyze_stageBC \
    --dirs "${SPARSE_DIR}" "${SOFT_DIR}" \
    --labels "Sparse-MoE (C)" "Soft-LoRA-MoE (C)" \
    --out_dir figures/stageC/

echo ""
echo "============================================================"
echo "  Stage C complete. Figures saved to figures/stageC/"
echo "============================================================"
