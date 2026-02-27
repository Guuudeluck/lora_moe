# Claude Code Context — LoRA MoE Experiment (Stage B / C)

> **Purpose**: Onboarding document for a new Claude Code session on the 4× H200 SXM pod.
> Contains everything needed to continue the Stage B experiment without reviewing history.

---

## 1. Research Goal

Compare **Sparse-MoE** vs **Soft-LoRA-MoE (ABMIL)** on Qwen2.5-14B / UltraChat-200k.

- **Stage A** (complete): Established Soft-LoRA-MoE beats Sparse-MoE and QLoRA on Qwen2.5-7B / Alpaca.
- **Stage B** (in progress): Validate the quality–stability gap at 14B scale with richer data.
  - Primary question: Does Sparse-MoE exhibit load-balance pathologies at 14B scale?
- **Stage C** (pending): Full system study on Qwen2.5-32B with 4–8× H200 FSDP.
  - Primary question: Does Sparse-MoE become fragile under multi-GPU load?

---

## 2. Repository Layout

```
lora_moe/
├── experiments/lora_moe/
│   ├── configs/
│   │   ├── stageB_sparse_moe.yaml          # Qwen2.5-14B, E=8 H=8 top_k=2 capacity_factor=1.25
│   │   ├── stageB_soft_lora_moe.yaml       # Qwen2.5-14B, E=8 r=8 attention_dim=64
│   │   ├── stageC_sparse_moe.yaml          # Qwen2.5-32B, E=16 H=8
│   │   ├── stageC_soft_lora_moe.yaml       # Qwen2.5-32B, E=16 r=8
│   │   ├── accelerate_fsdp_stageB.yaml     # FSDP 4-GPU config (Stage B on 4× H200)
│   │   └── accelerate_fsdp_stageC.yaml     # FSDP 8-GPU config (Stage C)
│   ├── modules/
│   │   ├── lora_layers.py                  # LoRAConfig, MoELoRAConfig, ABMILMoELoRALinear
│   │   └── ffn_moe_layers.py               # MoEFFNConfig, SoftMoEFFNLinear, SparseMoEFFNLinear
│   ├── data.py                             # Alpaca dataset loader
│   ├── data_mixture.py                     # UltraChat-200k loader (Stage B/C)
│   ├── injection.py                        # inject_lora() dispatcher
│   ├── metrics.py                          # collect_routing_metrics(), snapshot_expert_weights()
│   ├── timing_callback.py                  # StepTimingCallback (p50/p95/p99 per-step timing)
│   ├── trainer.py                          # LoRAMoETrainer (extends HF Trainer)
│   ├── run_experiment.py                   # Main training entry point
│   ├── benchmark_inference.py              # Stage A inference benchmarking
│   ├── benchmark_stageBC.py               # Stage B/C benchmarking (mixed_semantic workload)
│   ├── analyze_results.py                  # Stage A analysis + plots
│   ├── analyze_stageBC.py                 # Stage B/C analysis (7 plots)
│   ├── run_stageB.sh                       # Stage B on 1× H200
│   ├── run_stageB_4gpu.sh                  # Stage B on 4× H200 SXM (FSDP) ← USE THIS
│   └── run_stageC.sh                       # Stage C on 4–8× H200 FSDP
└── README.md                               # Project overview
```

---

## 3. Current Experiment State

| | Details |
|---|---|
| **Single-GPU run** | PID 601570 on original pod, step ~1783/20145 (8.8% complete) |
| **Original pod** | 1× H200 (143 GB), 88.6 GB VRAM used (62%) |
| **Effective batch** | 4 per-device × 8 accum = 32 (original) |
| **Status** | Very slow (~16 sec/step → ~89 hours remaining). Move to 4× H200. |
| **Stage B Sparse-MoE** | Partially trained — can restart from scratch on 4× H200 |
| **Stage B Soft-LoRA-MoE** | Not yet started |

### Training health (single-GPU run, validated)
- `overflow_fraction ≈ 0.10` — capacity enforcement fires correctly at capacity_factor=1.25
- `max_over_mean_load ≈ 1.06` — near-balanced expert routing
- `active_experts = 8/8` — no expert collapse
- `train_loss ≈ 7.1–7.5` — healthy, decreasing

---

## 4. Running Stage B on 4× H200 SXM

### Prerequisites
```bash
cd /path/to/lora_moe
pip install -e .   # if not already installed
pip install accelerate transformers datasets torch
```

### Quick launch
```bash
bash experiments/lora_moe/run_stageB_4gpu.sh
```

This script:
1. Trains Sparse-MoE (Qwen2.5-14B) with FSDP across 4 GPUs
2. Trains Soft-LoRA-MoE with FSDP across 4 GPUs
3. Benchmarks both (4 workloads: homogeneous, heterogeneous, bursty, mixed_semantic)
4. Runs analysis → saves 7 figures to `figures/stageB/`

### Debug smoke test first
```bash
bash experiments/lora_moe/run_stageB_4gpu.sh --debug
# Runs 50 steps, 512 samples — completes in ~10 min
```

### Manual launch (if accelerate has issues)
```bash
# Option A: torchrun (DDP, no FSDP)
torchrun --nproc_per_node=4 \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml

# Option B: accelerate
accelerate launch \
    --config_file experiments/lora_moe/configs/accelerate_fsdp_stageB.yaml \
    --num_processes 4 \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml \
    --gradient_accumulation_steps 1
```

### VRAM math on 4× H200 (143 GB each)
| Mode | Per-GPU VRAM |
|------|-------------|
| Single GPU (baseline) | 88.6 GB: 28 GB weights + 56 GB optimizer + 4.6 GB activations |
| 4-GPU DDP | Same per GPU (~88 GB each) — parallelizes gradients only |
| 4-GPU FSDP FULL_SHARD | ~22 GB: weights/optimizer sharded across 4 GPUs → large batch headroom |

With FSDP, each GPU uses ~22 GB → 121 GB free for activations → can disable `gradient_checkpointing` and use `per_device_train_batch_size=8` (effective batch = 4 × 8 × 1 = 32).

### Key config differences (stageB_4gpu vs stageB_1gpu)
```
# run_stageB_4gpu.sh passes --gradient_accumulation_steps 1 at CLI
# This overrides the YAML value (4) → effective batch stays at 4 GPUs × 8 per-device × 1 = 32
```

---

## 5. All Files Modified/Created in Stage B Implementation

### New files (created from scratch)

| File | What it does |
|------|-------------|
| `data_mixture.py` | UltraChat-200k loader: flattens multi-turn conversations to instruction+response, tokenizes with label masking, filters all-masked examples |
| `timing_callback.py` | `StepTimingCallback`: records wall-clock time per step, computes p50/p95/p99 over rolling 100-step window, logs `step_time_ms/rank{r}` in DDP |
| `configs/stageB_sparse_moe.yaml` | Sparse-MoE config: Qwen2.5-14B, E=8 H=8 top_k=2 capacity_factor=1.25, bs=8 accum=4 |
| `configs/stageB_soft_lora_moe.yaml` | Soft-LoRA-MoE config: Qwen2.5-14B, E=8 r=8 attention_dim=64, bs=8 accum=4 |
| `configs/stageC_sparse_moe.yaml` | Stage C Sparse-MoE: Qwen2.5-32B, E=16, bs=2 accum=16 |
| `configs/stageC_soft_lora_moe.yaml` | Stage C Soft-LoRA-MoE: Qwen2.5-32B, E=16, bs=2 accum=16 |
| `configs/accelerate_fsdp_stageB.yaml` | FSDP config for 4× H200 (Stage B) |
| `configs/accelerate_fsdp_stageC.yaml` | FSDP config for 8× H200 (Stage C) |
| `run_stageB.sh` | Stage B pipeline on 1× H200 |
| `run_stageB_4gpu.sh` | Stage B pipeline on 4× H200 with FSDP ← main script for new pod |
| `run_stageC.sh` | Stage C pipeline with torchrun/accelerate |
| `benchmark_stageBC.py` | Extends benchmark_inference.py: adds `mixed_semantic` workload (math/code/QA/chat), `tail_amplification` stat, `stall_ratio`, `--compare_dir` flag |
| `analyze_stageBC.py` | 7 new plots: latency CDF, expert load histogram, step time distribution, rank step skew, router contribution, tail amplification, overflow over time |

### Modified files

| File | What changed |
|------|-------------|
| `modules/ffn_moe_layers.py` | Added `capacity_factor` to `MoEFFNConfig`; capacity enforcement in `SparseMoEFFNLinear.forward()` (argsort rank trick + NaN-safe router weights via `torch.where` + `has_expert`); extended `get_routing_stats()` with `token_count_per_expert`, `max_over_mean_load`, `hot_expert_count`, `load_variance`, `overflow_fraction` |
| `modules/lora_layers.py` | Added `router_weight_std` and `collapse_score` to `ABMILMoELoRALinear.get_routing_stats()` |
| `metrics.py` | Surfaces new stats from `get_routing_stats()`: per-layer Sparse-MoE stats (`max_over_mean_load`, `hot_expert_count`, `overflow_fraction`), per-layer ABMIL stats (`collapse_score`, `router_weight_std`), global aggregates (`routing/global_max_over_mean_load`, `routing/global_overflow_fraction`) |
| `trainer.py` | Registers `StepTimingCallback`; `log()` override flushes `state._timing_log_buffer` |
| `run_experiment.py` | Added `--dataset_type` CLI arg (alpaca/ultrachat), routes to `load_ultrachat()`, `capacity_factor` passthrough to `MoEFFNConfig`, `local_rank`/`world_size` logging for torchrun |

---

## 6. Key Bug Fixes (Do Not Reintroduce)

### Bug 1: NaN in routing weights under capacity enforcement
**Symptom**: `loss=NaN`, `grad_norm=NaN`, `active_experts=0.0`

**Root cause**: When all top-k assignments for a token exceed capacity, the dispatch row is all zeros.
`masked_fill(dispatch==0, -inf)` → `softmax(all -inf)` → NaN.

**Fix** in `ffn_moe_layers.py:SparseMoEFFNLinear.forward()`:
```python
has_expert = dispatch.sum(dim=-1, keepdim=True) > 0   # [B, N, 1]
router_weights = torch.where(
    has_expert.expand_as(cap_weights), cap_weights, torch.zeros_like(cap_weights)
)
```
Tokens with no assigned expert contribute zero MoE output (base linear only).

### Bug 2: NaN CE loss from all-masked labels
**Symptom**: `train_loss=NaN` during first eval on UltraChat-200k

**Root cause**: Some UltraChat instructions are very long. When `prompt_len >= max_length`,
all `labels` tokens are `-100` (masked). Cross-entropy of all-masked sequence = NaN.

**Fix** in `data_mixture.py`:
```python
has_labels = any(l != -100 for l in labels)
# ... filter step:
tokenized.filter(lambda ex: ex["_has_labels"] and len(ex["input_ids"]) > 0, ...)
```

### Bug 3: Malformed exception handler in benchmark_stageBC.py
**Symptom**: `AttributeError: str object has no attribute '_formatter_field_name_split'`

**Fix**: Simplified the exception fallback to `prompts.append(tmpl)`.

---

## 7. Monitoring

### Real-time training log
```bash
tail -f /tmp/stageB_master.log     # or whatever log path you set
```

### Key metrics to watch
- `train_loss` — should decrease from ~7.5 to ~2-3 over training
- `routing/global_max_over_mean_load` — should stay < 2.0 for Sparse-MoE (pathology = > 5)
- `routing/global_overflow_fraction` — should be ~0.09-0.11 (capacity_factor=1.25)
- `step_time_ms/p99` — watch for outliers indicating GPU stalls
- `routing/layer*/collapse_score` — should be low (< 0.8) for ABMIL; high = collapse

### TensorBoard
```bash
tensorboard --logdir outputs/stageB_sparse_moe/logs --port 6006
tensorboard --logdir outputs/stageB_soft_lora_moe/logs --port 6007
```

### GPU monitoring
```bash
watch -n 2 nvidia-smi
# Or for multi-GPU:
nvidia-smi dmon -s u -d 2    # utilization + memory, 2s interval
```

---

## 8. Accelerate Setup (first time on new pod)

```bash
# Option A: use the pre-written config
accelerate launch --config_file experiments/lora_moe/configs/accelerate_fsdp_stageB.yaml ...

# Option B: interactive setup
accelerate config
# Choose: FSDP, 4 processes, bf16, FULL_SHARD, TRANSFORMER_BASED_WRAP, SHARDED_STATE_DICT

# Verify setup
accelerate env
```

---

## 9. Expected Training Times (estimates)

| Setup | Sparse-MoE | Soft-LoRA-MoE |
|-------|-----------|---------------|
| 1× H200, bs=8 accum=4 | ~35–45 hours | ~30–40 hours |
| 4× H200 FSDP, bs=8 accum=1 | ~9–12 hours | ~8–10 hours |
| 4× H200 DDP, bs=8 accum=1 | ~9–12 hours | ~8–10 hours |

Qwen2.5-14B, 3 epochs, UltraChat-200k (~190k examples after filtering).

---

## 10. Pending Tasks

1. **Stage B training** — Run `run_stageB_4gpu.sh` on 4× H200 pod
2. **Stage B benchmark + analysis** — Handled automatically by `run_stageB_4gpu.sh`
3. **Stage C** — Qwen2.5-32B on 4–8× H200, use `run_stageC.sh`
4. **Write up results** — Compare Sparse-MoE vs Soft-LoRA-MoE on:
   - Training stability (overflow_fraction, collapse_score)
   - Inference latency (latency_cdf.png, tail_amplification.png)
   - Expert load balance (expert_load_histogram.png)
   - Step time consistency (step_time_distribution.png)

---

## 11. Common Commands

```bash
# From repo root: /path/to/lora_moe/

# Smoke test (50 steps, sanity check)
bash experiments/lora_moe/run_stageB_4gpu.sh --debug

# Full Stage B on 4× H200
bash experiments/lora_moe/run_stageB_4gpu.sh

# Train only (no benchmark/analysis)
accelerate launch \
    --config_file experiments/lora_moe/configs/accelerate_fsdp_stageB.yaml \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml

# Benchmark only (after training)
python3 -m experiments.lora_moe.benchmark_stageBC \
    --model_dir outputs/stageB_sparse_moe \
    --method sparse_moe \
    --workload mixed_semantic

# Analyze only
python3 -m experiments.lora_moe.analyze_stageBC \
    --dirs outputs/stageB_sparse_moe outputs/stageB_soft_lora_moe \
    --labels "Sparse-MoE (B)" "Soft-LoRA-MoE (B)" \
    --out_dir figures/stageB/
```
