# Stage A Experiment Results — LoRA MoE on Qwen2.5-7B

**Model**: Qwen/Qwen2.5-7B
**Dataset**: Alpaca (~49k train / 2.6k val)
**Epochs**: 3 | **Effective batch size**: 32 | **Max seq len**: 512
**Targets**: q/k/v/o_proj (all 28 layers)

Figures: [`figures/stageA/`](figures/stageA/)

---

## Summary Table

| Method | Trainable Params | Train Loss | Eval Loss | Eval PPL | Train Time |
|---|---|---|---|---|---|
| QLoRA (r=16) | 10.1M (0.13%*) | 2.211 | 1.172 | 3.23 | 1.75 h |
| Soft-MoE (E=8, H=4) | 23.4M (0.31%) | 2.327 | 1.143 | 3.14 | 2.08 h |
| Sparse-MoE (E=8, H=4, top-2) | 23.4M (0.31%) | 2.341 | 1.155 | 3.17 | 2.15 h |
| **Soft-LoRA-MoE / ABMIL (E=8, r=4)** | 23.4M (0.31%) | **2.286** | **1.134** | **3.11** | **1.46 h** |

\* QLoRA backbone is 4-bit quantized (NF4); trainable LoRA adapters are full precision.

**Winner: Soft-LoRA-MoE (ABMIL)** — best eval loss and perplexity, and fastest to train.

---

## Per-Method Details

### 1. QLoRA Baseline (r=16)

| Metric | Value |
|---|---|
| Architecture | 4-bit NF4 backbone + standard LoRA |
| LoRA rank | r=16, α=16 |
| Trainable params | 10,092,544 (lora_A + lora_B) |
| Train loss | 2.2110 |
| Eval loss | 1.1722 |
| Eval perplexity | 3.23 |
| Training time | 1.75 h |
| Balance loss | N/A (no routing) |

Classic QLoRA baseline. Fewest trainable parameters (~10M vs 23M for MoE variants). No routing overhead. Lower perplexity than Sparse-MoE but worse than both dense-routing methods.

---

### 2. Soft-MoE (E=8, H=4) — Dense Softmax Routing + FFN Experts

| Metric | Value |
|---|---|
| Architecture | Linear router → softmax → 8 FFN experts (all participate per token) |
| Expert hidden dim | H=4 |
| Trainable params | 23,396,352 (router + W1 + W2) |
| Train loss | 2.3269 |
| Eval loss | 1.1434 |
| Eval perplexity | 3.14 |
| Training time | 2.08 h |
| Balance loss (final) | 0.01016 |
| Mean routing entropy | 2.0701 |
| Mean load CV | 0.1125 |

Dense routing means all 8 experts contribute to every token (weighted by softmax scores). The balance loss (Switch-Transformer style: E·Σp²) kept routing moderately balanced (CV=0.11). Outperforms QLoRA and Sparse-MoE on eval perplexity.

---

### 3. Sparse-MoE (E=8, H=4, top-2)

| Metric | Value |
|---|---|
| Architecture | Linear router → top-2 gate → 8 FFN experts (2 active per token) |
| Expert hidden dim | H=4 |
| Top-k | 2 |
| Trainable params | 23,396,352 |
| Train loss | 2.3406 |
| Eval loss | 1.1549 |
| Eval perplexity | 3.17 |
| Training time | 2.15 h |
| Balance loss (final) | 0.02026 |
| Mean routing entropy | 2.0460 |
| Mean load CV | 0.2178 |

Top-2 sparse routing activates only 2 of 8 experts per token. Despite having the same parameter budget as Soft-MoE, it underperforms: lower routing entropy (2.046 vs 2.070) and much higher load imbalance (CV=0.218 vs 0.113), suggesting some experts are over-used and others idle. The balance loss at 0.020 is the highest of any MoE variant, indicating the load-balancing regularizer is working harder but still not achieving dense-routing uniformity.

---

### 4. Soft-LoRA-MoE / ABMIL (E=8, r=4) ← Best

| Metric | Value |
|---|---|
| Architecture | ABMIL sigmoid attention routing + LoRA experts |
| LoRA rank | r=4 per expert |
| Num experts | 8 |
| Trainable params | 23,396,352 |
| Train loss | 2.2859 |
| Eval loss | **1.1343** |
| Eval perplexity | **3.11** |
| Training time | **1.46 h** |
| Balance loss (final) | 0.0000035 (effectively zero) |
| Mean routing entropy | **2.0785** |
| Mean load CV | **0.0243** |

ABMIL uses attention-based multiple instance learning as a routing mechanism, with sigmoid (not softmax) scores — experts do not compete, allowing simultaneous activation. Key advantages over the other methods:

- **Best eval perplexity (3.11)** — 3.7% lower PPL than QLoRA, 0.9% better than Soft-MoE.
- **Most balanced routing** — CV=0.024, nearly 10× lower than Sparse-MoE (0.218) and 4.6× lower than Soft-MoE (0.113). The sigmoid routing eliminates the winner-takes-all collapse seen in top-k sparse routing.
- **Highest entropy (2.079)** — experts are used near-uniformly across layers.
- **Near-zero balance loss** — the ABMIL mechanism is inherently well-balanced without needing a strong regularization penalty.
- **Fastest MoE training (1.46 h)** — faster than both FFN-based MoE variants despite identical param count, likely because LoRA experts are lower-rank matrix products (cheaper forward/backward) compared to H=4 FFN layers.

---

## Routing Analysis (MoE methods)

| Method | Mean Entropy | Mean Load CV | Balance Loss |
|---|---|---|---|
| Soft-MoE (E=8, H=4) | 2.0701 | 0.1125 | 0.01016 |
| Sparse-MoE (E=8, H=4, top-2) | 2.0460 | 0.2178 | 0.02026 |
| Soft-LoRA-MoE / ABMIL | **2.0785** | **0.0243** | ~0.0 |

- **Entropy** (higher = more uniform expert usage across the routing distribution)
- **Load CV** (coefficient of variation, lower = more balanced expert load)
- **Balance loss** (auxiliary loss penalising uneven routing; lower = routing is already balanced without needing regularisation)

ABMIL's sigmoid routing distributes load nearly uniformly by design, requiring virtually no auxiliary loss to maintain balance — a structural advantage over softmax/top-k routing.

---

## Figures

| File | Description |
|---|---|
| `figures/stageA/loss_curves.png` | Train and validation loss curves for all 4 methods |
| `figures/stageA/routing_metrics.png` | Routing entropy and load CV over training |
| `figures/stageA/balance_loss.png` | Auxiliary balance loss curves (MoE methods) |
| `figures/stageA/expert_utilisation.png` | Per-expert utilisation heatmap from end-of-training snapshot |
| `figures/stageA/summary.csv` | Raw numbers for all metrics |

---

## Key Takeaways

1. **All three MoE variants outperform QLoRA** on eval perplexity (3.11–3.17 vs 3.23) with 2.3× more trainable parameters.
2. **ABMIL routing is the clear winner**: best perplexity, best load balance, and fastest wall-clock training time.
3. **Sparse top-k routing is the weakest MoE variant**: highest load imbalance (CV=0.218) and worst perplexity (3.17), barely beating QLoRA. The competition between experts in top-k gating causes collapse that even the balance loss cannot fully correct.
4. **Dense softmax (Soft-MoE) is a solid middle ground**: good balance (CV=0.113) and 3.14 PPL, with no risk of expert collapse.
5. **ABMIL's near-zero balance loss** confirms that sigmoid multi-instance routing is intrinsically load-balanced — no trade-off between routing quality and regularisation pressure.
