#!/bin/bash
# Stage B: Qwen2.5-14B / UltraChat-200k on 1× H200
# Trains Sparse-MoE and Soft-LoRA-MoE, benchmarks all 4 workloads,
# then runs analysis.
#
# Usage:
#   cd /path/to/lora_moe
#   bash experiments/lora_moe/run_stageB.sh
#
#   # Debug run (fast, small subset):
#   bash experiments/lora_moe/run_stageB.sh --debug

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_FLAG="--debug"
  echo ">>> DEBUG MODE"
fi

echo "============================================================"
echo "  Stage B: Qwen2.5-14B / UltraChat-200k"
echo "============================================================"

# ---- [1/2] Train Sparse-MoE ----
echo ""
echo "[1/2] Training Sparse-MoE (stageB)..."
python3 -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml \
    ${DEBUG_FLAG}

# ---- [2/2] Train Soft-LoRA-MoE ----
echo ""
echo "[2/2] Training Soft-LoRA-MoE (stageB)..."
python3 -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_soft_lora_moe.yaml \
    ${DEBUG_FLAG}

# ---- Benchmark both variants (4 workloads each) ----
echo ""
echo "Benchmarking trained models..."

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
echo "  Stage B complete. Figures saved to figures/stageB/"
echo "============================================================"
