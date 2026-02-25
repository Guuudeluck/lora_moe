"""
LoRA variant implementations as drop-in replacements for nn.Linear.

Three variants for comparison:
  1. StandardLoRALinear  – single low-rank adapter (baseline)
  2. SoftmaxMoELoRALinear – multiple adapters with softmax / top-k routing
  3. ABMILMoELoRALinear   – proposed: gated-attention (tanh*sigmoid) routing + sigmoid scoring

All variants:
  - Wrap a frozen pretrained nn.Linear
  - Store routing_weights in self._last_router_weights for metric collection
  - Expose get_routing_stats() → dict
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class MoELoRAConfig(LoRAConfig):
    num_experts: int = 8
    # softmax-moe only
    top_k: int = 0           # 0 = dense softmax; >0 = sparse top-k
    # abmil-moe only
    attention_dim: int = 128
    # shared
    balance_loss_coeff: float = 0.01


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BaseLoRALinear(nn.Module):
    """Common interface for all LoRA variants."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Frozen base weight (share storage — no copy)
        self.weight = nn.Parameter(linear.weight.data, requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data, requires_grad=False)
        else:
            self.bias = None

        # Routing weights from the last forward pass (detached, for logging)
        self._last_router_weights: Optional[torch.Tensor] = None

    def get_routing_stats(self) -> dict:
        return {}

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
        )


# ---------------------------------------------------------------------------
# 1. Standard LoRA
# ---------------------------------------------------------------------------

class StandardLoRALinear(_BaseLoRALinear):
    """
    Standard LoRA (Hu et al. 2022).
    output = W*x + scaling * (x @ A) @ B
    """

    def __init__(self, linear: nn.Linear, cfg: LoRAConfig):
        super().__init__(linear, cfg.rank, cfg.alpha, cfg.dropout)

        self.lora_A = nn.Parameter(
            torch.randn(self.in_features, cfg.rank) * (1.0 / math.sqrt(cfg.rank))
        )
        self.lora_B = nn.Parameter(torch.zeros(cfg.rank, self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.lora_A @ self.lora_B
        return base + self.scaling * lora


# ---------------------------------------------------------------------------
# 2. Softmax MoE LoRA  (standard-MoE baseline)
# ---------------------------------------------------------------------------

class SoftmaxMoELoRALinear(_BaseLoRALinear):
    """
    MoE LoRA with a simple linear router and softmax (optionally top-k sparse).

    Router:  scores = x @ W_r  →  softmax  →  optional top-k mask
    MoE out: sum_i( w_i * (x @ A_i) @ B_i )

    Auxiliary balance loss (Switch-Transformer style, stored, added by trainer):
        L_bal = num_experts * sum(mean_prob_i * frac_dispatched_i)
    """

    def __init__(self, linear: nn.Linear, cfg: MoELoRAConfig):
        super().__init__(linear, cfg.rank, cfg.alpha, cfg.dropout)
        E = cfg.num_experts
        self.num_experts = E
        self.top_k = cfg.top_k  # 0 → dense

        # Router: single linear projection onto expert scores
        self.router = nn.Linear(self.in_features, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # Expert parameters
        self.experts_A = nn.Parameter(
            torch.randn(E, self.in_features, cfg.rank) * (1.0 / math.sqrt(cfg.rank))
        )
        self.experts_B = nn.Parameter(torch.zeros(E, cfg.rank, self.out_features))

        # Stored for balance-loss computation
        self._last_router_logits: Optional[torch.Tensor] = None
        self._last_dispatch_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        B, N, D = x.shape
        E = self.num_experts

        base = F.linear(x, self.weight, self.bias)

        # Router
        logits = self.router(x)          # [B, N, E]
        self._last_router_logits = logits.detach()

        if self.top_k > 0 and self.top_k < E:
            # Sparse top-k: zero out non-selected experts, renormalise
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            router_weights = torch.full_like(logits, float('-inf'))
            router_weights.scatter_(-1, topk_idx, topk_vals)
            router_weights = F.softmax(router_weights, dim=-1)     # [B, N, E]

            dispatch = torch.zeros_like(logits)
            dispatch.scatter_(-1, topk_idx, 1.0)
            self._last_dispatch_mask = dispatch.detach()
        else:
            router_weights = F.softmax(logits, dim=-1)             # [B, N, E]
            self._last_dispatch_mask = None

        self._last_router_weights = router_weights.detach()

        # Efficient MoE forward  (same einsum as ABMIL variant)
        lora_down = torch.einsum('bnd,edr->bner', self.dropout(x), self.experts_A)
        weighted = lora_down * router_weights.unsqueeze(-1)
        moe_out = torch.einsum('bner,erd->bnd', weighted, self.experts_B)

        out = base + self.scaling * moe_out
        return out.squeeze(1) if squeeze else out

    def compute_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer auxiliary load-balance loss."""
        if self._last_router_logits is None:
            return torch.tensor(0.0)
        E = self.num_experts
        probs = F.softmax(self._last_router_logits, dim=-1)   # [B, N, E]
        mean_prob = probs.mean(dim=(0, 1))                     # [E]
        if self._last_dispatch_mask is not None:
            frac = self._last_dispatch_mask.mean(dim=(0, 1))  # [E]
        else:
            frac = mean_prob
        return E * (frac * mean_prob).sum()

    def get_routing_stats(self) -> dict:
        if self._last_router_weights is None:
            return {}
        w = self._last_router_weights               # [B, N, E]
        mean_w = w.mean(dim=(0, 1)).cpu()           # [E]
        p = mean_w / (mean_w.sum() + 1e-8)
        entropy = -(p * (p + 1e-9).log()).sum().item()
        return {
            "mean_expert_weights": mean_w,
            "weight_entropy": entropy,
            "weight_cv": (mean_w.std() / (mean_w.mean() + 1e-8)).item(),
            "active_experts": (mean_w > 0.01).float().sum().item(),
        }


# ---------------------------------------------------------------------------
# 3. ABMIL MoE LoRA  (proposed)
# ---------------------------------------------------------------------------

class ABMILMoELoRALinear(_BaseLoRALinear):
    """
    ABMIL (Attention-Based Multiple Instance Learning) LoRA MoE.

    Router uses GATED ATTENTION mechanism:
        V(x) = tanh(W_V · x)          [B, N, attention_dim]
        U(x) = sigmoid(W_U · x)       [B, N, attention_dim]
        scores = W_out · (V ⊙ U)      [B, N, num_experts]
        w_i = sigmoid(scores_i)        ← INDEPENDENT (not competitive)

    MoE out: sum_i( w_i * (x @ A_i) @ B_i )

    Key properties vs Softmax-MoE:
      - Sigmoid scoring: experts are independent; multiple can be fully active
      - Gated features: richer router representation
      - No top-k: purely soft (dense) but with varying activation magnitude
    """

    def __init__(self, linear: nn.Linear, cfg: MoELoRAConfig):
        super().__init__(linear, cfg.rank, cfg.alpha, cfg.dropout)
        E = cfg.num_experts
        self.num_experts = E
        D_att = cfg.attention_dim

        # ABMIL Router
        self.router_V = nn.Linear(self.in_features, D_att, bias=False)
        self.router_U = nn.Linear(self.in_features, D_att, bias=False)
        self.router_W = nn.Linear(D_att, E, bias=False)

        nn.init.normal_(self.router_V.weight, std=0.02)
        nn.init.normal_(self.router_U.weight, std=0.02)
        nn.init.zeros_(self.router_W.weight)      # zero-init → all sigmoid(0)=0.5 at start

        # Expert parameters
        self.experts_A = nn.Parameter(
            torch.randn(E, self.in_features, cfg.rank) * 0.01
        )
        self.experts_B = nn.Parameter(torch.zeros(E, cfg.rank, self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        # ---- Base ----
        base = F.linear(x, self.weight, self.bias)    # [B, N, out]

        # ---- ABMIL Router ----
        v = torch.tanh(self.router_V(x))              # [B, N, D_att]
        u = torch.sigmoid(self.router_U(x))           # [B, N, D_att]
        gated = v * u                                  # [B, N, D_att]
        scores = self.router_W(gated)                  # [B, N, E]
        router_weights = torch.sigmoid(scores)         # [B, N, E]  (0‥1, independent)
        self._last_router_weights = router_weights.detach()

        # ---- Efficient LoRA MoE ----
        # x @ A:  [B,N,D] × [E,D,r] → [B,N,E,r]
        lora_down = torch.einsum('bnd,edr->bner', self.dropout(x), self.experts_A)
        # weight:  [B,N,E,r] * [B,N,E,1]
        weighted = lora_down * router_weights.unsqueeze(-1)
        # × B:    [B,N,E,r] × [E,r,D] → [B,N,D]
        moe_out = torch.einsum('bner,erd->bnd', weighted, self.experts_B)

        out = base + self.scaling * moe_out
        return out.squeeze(1) if squeeze else out

    def compute_balance_loss(self) -> torch.Tensor:
        """
        Variance-based balance loss for sigmoid routing.
        Minimises variance of per-expert mean activation (encourages uniform use).
        """
        if self._last_router_weights is None:
            return torch.tensor(0.0)
        mean_w = self._last_router_weights.mean(dim=(0, 1))   # [E]
        return mean_w.var()

    def get_routing_stats(self) -> dict:
        if self._last_router_weights is None:
            return {}
        w = self._last_router_weights                # [B, N, E]
        mean_w = w.mean(dim=(0, 1)).cpu()            # [E]  in (0,1)
        p = mean_w / (mean_w.sum() + 1e-8)
        entropy = -(p * (p + 1e-9).log()).sum().item()
        return {
            "mean_expert_weights": mean_w,
            "weight_entropy": entropy,
            "weight_cv": (mean_w.std() / (mean_w.mean() + 1e-8)).item(),
            "active_experts": (mean_w > 0.1).float().sum().item(),
            "max_weight": mean_w.max().item(),
            "min_weight": mean_w.min().item(),
        }
