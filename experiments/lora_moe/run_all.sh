#!/bin/bash
# Run all three MoE routing experiments then analyze.
# Primary comparison: Soft MoE vs Sparse MoE vs ABMIL MoE
# Standard LoRA is kept as a background sanity-check baseline.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_DIR="experiments/lora_moe/configs"

cd "$REPO_ROOT"

# ── Debug check (uncomment to validate the full pipeline in ~3 min) ────────
# for method in soft_moe sparse_moe abmil_moe; do
#     echo ">>> DEBUG: $method"
#     python3 -m experiments.lora_moe.run_experiment \
#         --config $CONFIG_DIR/${method}.yaml --debug
# done

echo "================================================================"
echo " [1/4] Standard LoRA (sanity-check baseline)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/standard_lora.yaml

echo "================================================================"
echo " [2/4] Soft MoE LoRA (dense softmax routing)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/soft_moe.yaml

echo "================================================================"
echo " [3/4] Sparse MoE LoRA (top-2 hard routing)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/sparse_moe.yaml

echo "================================================================"
echo " [4/4] ABMIL MoE LoRA (gated-attention sigmoid routing)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/abmil_moe.yaml

echo "================================================================"
echo " Analyzing results..."
echo "================================================================"
python3 -m experiments.lora_moe.analyze_results \
    --dirs  outputs/standard_lora_r16 \
            outputs/soft_moe_e8_r4 \
            outputs/sparse_moe_e8_r4_top2 \
            outputs/abmil_moe_e8_r4 \
    --labels "Standard LoRA (r=16)" \
             "Soft MoE (E=8, r=4, dense)" \
             "Sparse MoE (E=8, r=4, top-2)" \
             "ABMIL MoE (E=8, r=4, sigmoid)" \
    --out_dir figures/

echo "Done! Figures in: figures/"
