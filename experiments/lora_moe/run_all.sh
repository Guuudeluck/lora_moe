#!/bin/bash
# Stage A: Run all four MoE routing experiments on Qwen2.5-7B then analyze.
#
# Variants:
#   qlora          – NF4 4-bit QLoRA baseline (r=16)
#   soft_moe       – Dense softmax + FFN experts (E=8, H=4)
#   sparse_moe     – Top-k=2 + FFN experts     (E=8, H=4)
#   soft_lora_moe  – ABMIL sigmoid + LoRA experts (E=8, r=4, att_dim=32)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_DIR="experiments/lora_moe/configs"

cd "$REPO_ROOT"

# ── Debug check (uncomment to validate the full pipeline in ~5 min) ─────────
# for cfg in qlora soft_moe sparse_moe soft_lora_moe; do
#     echo ">>> DEBUG: $cfg"
#     python3 -m experiments.lora_moe.run_experiment \
#         --config $CONFIG_DIR/${cfg}.yaml --debug
# done

echo "================================================================"
echo " [1/4] QLoRA baseline (NF4 4-bit + LoRA r=16)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/qlora.yaml

echo "================================================================"
echo " [2/4] Soft-MoE  (dense softmax + FFN experts, E=8, H=4)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/soft_moe.yaml

echo "================================================================"
echo " [3/4] Sparse-MoE  (top-k=2 + FFN experts, E=8, H=4)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/sparse_moe.yaml

echo "================================================================"
echo " [4/4] Soft-LoRA-MoE  (ABMIL sigmoid routing, E=8, r=4)"
echo "================================================================"
python3 -m experiments.lora_moe.run_experiment \
    --config $CONFIG_DIR/soft_lora_moe.yaml

echo "================================================================"
echo " Analyzing Stage A results..."
echo "================================================================"
python3 -m experiments.lora_moe.analyze_results \
    --dirs  outputs/qlora_7b \
            outputs/soft_moe_e8_h4 \
            outputs/sparse_moe_e8_h4_top2 \
            outputs/soft_lora_moe_e8_r4 \
    --labels "QLoRA (r=16)" \
             "Soft-MoE (E=8,H=4)" \
             "Sparse-MoE (E=8,H=4,top2)" \
             "Soft-LoRA-MoE (ABMIL)" \
    --out_dir figures/stageA/

echo "Done! Figures in: figures/stageA/"
