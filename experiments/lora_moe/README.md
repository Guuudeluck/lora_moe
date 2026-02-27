# LoRA MoE Experiments

Research comparing **Sparse-MoE** vs **Soft-LoRA-MoE (ABMIL)** adapter architectures for
efficient LLM fine-tuning, from 7B to 32B scale.

## Quick Start

```bash
# From repo root
cd /path/to/lora_moe

# Stage A: 7B sanity check (Alpaca, single GPU)
python3 -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/abmil_moe.yaml --debug

# Stage B: 14B main comparison (UltraChat-200k, single H200)
bash experiments/lora_moe/run_stageB.sh

# Stage B: 14B on 4× H200 SXM (recommended — 4× faster)
bash experiments/lora_moe/run_stageB_4gpu.sh

# Stage C: 32B system study (4–8× H200 FSDP)
bash experiments/lora_moe/run_stageC.sh
```

---

## Methods

| Method | Key | Description |
|--------|-----|-------------|
| `standard` | LoRA | Standard low-rank adapter (baseline) |
| `qlora` | QLoRA | 4-bit quantized LoRA |
| `sparse_moe` | Sparse-MoE | Top-k hard routing with FFN experts; Switch-Transformer style |
| `soft_moe` | Soft-MoE | Dense softmax routing with FFN experts |
| `soft_lora_moe` / `abmil_moe` | Soft-LoRA-MoE | ABMIL gated attention routing with LoRA experts (primary proposal) |
| `softmax_lora_moe` | Softmax-LoRA-MoE | Softmax routing with LoRA experts |

---

## Architecture Overview

### Sparse-MoE (`SparseMoEFFNLinear`)

```
output = W·x  +  Σ_{e ∈ top_k}  w_e · GELU(x @ W1_e) @ W2_e
```

- Router: linear projection → top-k selection → sparse softmax weights
- **Capacity enforcement**: `capacity = ceil(capacity_factor × B × N / E)` tokens per expert
  - Overflow tokens (rank ≥ capacity) receive only base output (`W·x`)
  - NaN-safe: all-overflow tokens handled with `torch.where` + zero fallback
- Balance loss (Switch-Transformer): `L_bal = E × Σ_e frac_e × mean_prob_e`

### Soft-LoRA-MoE / ABMIL (`ABMILMoELoRALinear`)

```
output = W·x  +  Σ_e  w_e · (x @ A_e^T) @ B_e^T
```

- Router: ABMIL attention mechanism (sigmoid-gated, independent expert scoring)
- All experts participate (no hard routing, no capacity issues)
- Balance loss (dense): `L_bal = E × Σ_e p_e²`

---

## Config Keys

### Common to all methods

| Key | Default | Description |
|-----|---------|-------------|
| `model_name_or_path` | — | HuggingFace model ID |
| `method` | — | One of the method names above |
| `dataset_type` | `alpaca` | `alpaca` or `ultrachat` |
| `max_length` | `512` | Sequence truncation length |
| `num_train_epochs` | `3` | Training epochs |
| `per_device_train_batch_size` | `4` | Batch size per GPU |
| `gradient_accumulation_steps` | `8` | Steps before optimizer update |
| `learning_rate` | `1e-4` | Peak learning rate |
| `lr_scheduler_type` | `cosine` | LR schedule |
| `warmup_steps` | `50` | Linear warmup steps |
| `gradient_checkpointing` | `true` | Trade compute for memory |
| `bf16` | `true` | BFloat16 mixed precision |
| `target_modules` | `[q_proj, v_proj]` | Which linear layers to replace |
| `balance_loss_coeff` | `0.01` | λ for MoE balance loss |
| `seed` | `42` | Random seed |

### Sparse-MoE specific

| Key | Default | Description |
|-----|---------|-------------|
| `num_experts` | `8` | Number of experts E |
| `expert_hidden_dim` | `8` | FFN expert bottleneck dim H |
| `top_k` | `2` | Experts per token K |
| `capacity_factor` | `0.0` | Capacity per expert (0 = disabled, 1.25 = standard) |

### Soft-LoRA-MoE specific

| Key | Default | Description |
|-----|---------|-------------|
| `num_experts` | `8` | Number of LoRA experts E |
| `lora_rank` | `8` | LoRA rank r |
| `attention_dim` | `64` | ABMIL attention dimension |

---

## Stages

### Stage A — Proof of Concept (complete)
- **Model**: Qwen2.5-7B
- **Dataset**: Alpaca (52k instruction pairs)
- **Hardware**: 1× H100/H200
- **Question**: Does Soft-LoRA-MoE outperform Sparse-MoE and QLoRA?
- **Result**: Yes — better eval loss, no routing pathologies, cleaner expert specialization

### Stage B — Medium Scale (in progress)
- **Model**: Qwen2.5-14B
- **Dataset**: HuggingFaceH4/ultrachat_200k (~190k multi-turn conversations)
- **Hardware**: 1× H200 (slow) or 4× H200 SXM with FSDP (recommended)
- **Question**: Does the quality–stability gap persist at 14B scale with richer data?
- **Configs**: `stageB_sparse_moe.yaml`, `stageB_soft_lora_moe.yaml`

### Stage C — Full System Study (pending)
- **Model**: Qwen2.5-32B
- **Dataset**: UltraChat-200k
- **Hardware**: 4–8× H200 with FSDP
- **Question**: Does Sparse-MoE become fragile under multi-GPU load?
- **Configs**: `stageC_sparse_moe.yaml`, `stageC_soft_lora_moe.yaml`

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_experiment.py` | Main training entry point; supports CLI config overrides |
| `run_stageB.sh` | Stage B pipeline (1× H200): train → benchmark → analyze |
| `run_stageB_4gpu.sh` | Stage B pipeline (4× H200 FSDP): faster, recommended |
| `run_stageC.sh` | Stage C pipeline (4–8× H200 FSDP) |
| `benchmark_inference.py` | Stage A inference latency benchmarking |
| `benchmark_stageBC.py` | Stage B/C benchmarking with mixed_semantic workload |
| `analyze_results.py` | Stage A analysis and plots |
| `analyze_stageBC.py` | Stage B/C analysis: 7 plots including latency CDF, expert load histogram |

---

## Routing Metrics (logged to TensorBoard)

### Sparse-MoE (per layer)
| Metric | Description |
|--------|-------------|
| `routing/layerNN/entropy` | Router weight entropy (higher = more uniform) |
| `routing/layerNN/cv` | Coefficient of variation of expert weights |
| `routing/layerNN/active_experts` | Experts with mean weight > 0.01 |
| `routing/layerNN/max_over_mean_load` | Max token count / mean token count (1.0 = perfectly balanced) |
| `routing/layerNN/hot_expert_count` | Experts receiving > 2× mean tokens |
| `routing/layerNN/overflow_fraction` | Fraction of tokens dropped by capacity enforcement |
| `routing/global_max_over_mean_load` | Max across all layers |
| `routing/global_overflow_fraction` | Mean across all layers |

### Soft-LoRA-MoE / ABMIL (per layer)
| Metric | Description |
|--------|-------------|
| `routing/layerNN/weight_entropy` | Attention weight entropy |
| `routing/layerNN/weight_cv` | Coefficient of variation |
| `routing/layerNN/active_experts` | Experts with weight > 0.01 |
| `routing/layerNN/collapse_score` | 1 − CV; high = all experts equal weight (collapse) |
| `routing/layerNN/router_weight_std` | Std of mean expert weights; near 0 = collapse |

### Step timing
| Metric | Description |
|--------|-------------|
| `step_time_ms/p50` | Median step time (ms) over last 100 steps |
| `step_time_ms/p95` | 95th percentile step time |
| `step_time_ms/p99` | 99th percentile (GPU stall indicator) |
| `step_time_ms/rankN/p50` | Per-rank timing (multi-GPU, DDP/FSDP) |

---

## Data Modules

### `data.py` — Alpaca
- Loads `tatsu-lab/alpaca`
- Formats: `"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"`
- Label masking: only response tokens contribute to loss

### `data_mixture.py` — UltraChat-200k
- Loads `HuggingFaceH4/ultrachat_200k` (train_sft + test_sft splits)
- Flattens multi-turn conversations: takes first user message as instruction, first assistant message as response
- Filters all-masked examples (instruction exceeds max_length → NaN CE loss)

---

## Multi-GPU Training

### DDP (simpler, same per-GPU memory)
```bash
torchrun --nproc_per_node=4 \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml
```

### FSDP (shards weights + optimizer → lower per-GPU memory, larger batch)
```bash
accelerate launch \
    --config_file experiments/lora_moe/configs/accelerate_fsdp_stageB.yaml \
    --num_processes 4 \
    -m experiments.lora_moe.run_experiment \
    --config experiments/lora_moe/configs/stageB_sparse_moe.yaml \
    --gradient_accumulation_steps 1
```

FSDP reduces per-GPU memory from ~88 GB (14B single-GPU) to ~22 GB (14B / 4-GPU),
leaving room for `gradient_checkpointing=false` and larger per-device batch sizes.

---

## Known Issues and Fixes

1. **NaN routing weights under capacity enforcement**: When capacity_factor > 0 and all top-k
   experts for a token exceed capacity, the token gets all-zero dispatch. Fixed with
   `torch.where(has_expert, softmax_weights, zeros)` in `ffn_moe_layers.py`.

2. **NaN CE loss from all-masked labels**: Long UltraChat instructions can fill max_length,
   leaving no room for the response → all labels are -100. Fixed by filtering these examples
   in `data_mixture.py` using the `_has_labels` field.

3. **Transformers 5.x compatibility**: PyTorch < 2.5 lacks `nn.Module.set_submodule`. Fixed
   with a monkey-patch in `run_experiment.py` for QLoRA mode.
