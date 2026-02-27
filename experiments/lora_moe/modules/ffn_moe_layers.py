"""
FFN-based MoE layers for Soft-MoE and Sparse-MoE experiments.

Unlike LoRA experts (low-rank adapter), FFN experts use a two-layer MLP delta:
    expert_i(x) = GELU(x @ W1_i) @ W2_i

Dense (Soft-MoE):  all experts participate with softmax weights.
Sparse (Sparse-MoE): top-k experts participate (Switch-Transformer style).

Balance loss is computed inline with the live computation graph so gradients
flow to the router during training.  This is the key correctness fix vs. the
previous implementation that detached logits before computing the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lora_moe.modules.lora_layers import _BaseLoRALinear


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MoEFFNConfig:
    num_experts: int = 8
    expert_hidden_dim: int = 4   # H; set equal to lora_rank for param-budget matching
    top_k: int = 0               # 0 = dense (Soft-MoE); >0 = sparse (Sparse-MoE)
    dropout: float = 0.05
    balance_loss_coeff: float = 0.01
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    capacity_factor: float = 0.0   # 0 = disabled; 1.25 = standard
    expert_capacity: int = 0       # computed from capacity_factor if > 0


# ---------------------------------------------------------------------------
# 1. Soft-MoE (dense softmax + FFN experts)
# ---------------------------------------------------------------------------

class SoftMoEFFNLinear(_BaseLoRALinear):
    """
    Dense Soft-MoE with two-layer FFN experts.

    output = W*x  +  Σ_e  w_e · GELU(x @ W1_e) @ W2_e

    All E experts participate for every token (competitive softmax weights).
    Balance loss computed inline with live graph (Switch-Transformer dense):
        L_bal = E * Σ_e (p_e)^2   where p_e = mean softmax probability
    """

    def __init__(self, linear: nn.Linear, cfg: MoEFFNConfig):
        # Pass expert_hidden_dim as rank so _BaseLoRALinear stores it;
        # alpha = H → scaling = 1.0 (FFN delta has no separate scaling).
        super().__init__(
            linear,
            rank=cfg.expert_hidden_dim,
            alpha=float(cfg.expert_hidden_dim),
            dropout=cfg.dropout,
        )
        E = cfg.num_experts
        H = cfg.expert_hidden_dim
        self.num_experts = E
        self.expert_hidden_dim = H

        # Router: projects hidden state onto per-expert logits
        self.router = nn.Linear(self.in_features, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # FFN expert weights
        # W1: [E, in_features, H]  (down-projection per expert)
        # W2: [E, H, out_features] (up-projection per expert)
        self.experts_W1 = nn.Parameter(
            torch.randn(E, self.in_features, H) * 0.01
        )
        self.experts_W2 = nn.Parameter(torch.zeros(E, H, self.out_features))

        # Cached for compute_balance_loss(); set in forward(), live graph
        self._balance_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)          # [B, 1, D]

        B, N, D = x.shape
        E = self.num_experts

        # Base (frozen pretrained linear)
        base = F.linear(x, self.weight, self.bias)   # [B, N, out]

        # Router – keep in live graph for balance loss gradient
        logits = self.router(x)                         # [B, N, E]
        router_weights = F.softmax(logits, dim=-1)      # [B, N, E]

        # Balance loss (dense Switch-Transformer: f_i = p_i, so L = E * Σ p_i^2)
        mean_prob = router_weights.mean(dim=(0, 1))     # [E], live graph
        self._balance_loss = E * (mean_prob * mean_prob).sum()

        # Detach for stats-only logging
        self._last_router_weights = router_weights.detach()

        # FFN MoE forward (all einsum, no Python loops)
        x_dropped = self.dropout(x)
        # [B,N,D] × [E,D,H] → [B,N,E,H]
        ffn_hidden = torch.einsum("bnd,edh->bneh", x_dropped, self.experts_W1)
        ffn_hidden = F.gelu(ffn_hidden)
        # [B,N,E,H] × [E,H,out] → [B,N,E,out]
        ffn_out = torch.einsum("bneh,eho->bneo", ffn_hidden, self.experts_W2)
        # Weight by router and sum over experts: [B,N,E,out] * [B,N,E,1] → [B,N,out]
        moe_out = (ffn_out * router_weights.unsqueeze(-1)).sum(dim=2)

        out = base + moe_out
        return out.squeeze(1) if squeeze else out

    def compute_balance_loss(self) -> torch.Tensor:
        """Return the inline-computed balance loss (has gradient to router)."""
        if self._balance_loss is None:
            return torch.tensor(0.0)
        return self._balance_loss

    def get_routing_stats(self) -> dict:
        if self._last_router_weights is None:
            return {}
        w = self._last_router_weights           # [B, N, E]
        mean_w = w.mean(dim=(0, 1)).cpu()       # [E]
        p = mean_w / (mean_w.sum() + 1e-8)
        entropy = -(p * (p + 1e-9).log()).sum().item()
        return {
            "mean_expert_weights": mean_w,
            "weight_entropy": entropy,
            "weight_cv": (mean_w.std() / (mean_w.mean() + 1e-8)).item(),
            "active_experts": (mean_w > 0.01).float().sum().item(),
        }


# ---------------------------------------------------------------------------
# 2. Sparse-MoE (top-k + FFN experts)
# ---------------------------------------------------------------------------

class SparseMoEFFNLinear(_BaseLoRALinear):
    """
    Sparse top-k MoE with two-layer FFN experts (Switch-Transformer style).

    output = W*x  +  Σ_{e ∈ topk}  w_e · GELU(x @ W1_e) @ W2_e

    Each token is hard-routed to exactly top_k of E experts.
    Balance loss (Switch-Transformer sparse):
        L_bal = E * Σ_e  frac_e * mean_prob_e
        frac_e   = fraction of tokens dispatched to expert e  (non-differentiable)
        mean_prob_e = mean full softmax prob for expert e     (differentiable)
    Only mean_prob carries gradient; frac acts as a constant coefficient.
    """

    def __init__(self, linear: nn.Linear, cfg: MoEFFNConfig):
        super().__init__(
            linear,
            rank=cfg.expert_hidden_dim,
            alpha=float(cfg.expert_hidden_dim),
            dropout=cfg.dropout,
        )
        E = cfg.num_experts
        H = cfg.expert_hidden_dim
        self.num_experts = E
        self.expert_hidden_dim = H
        self.top_k = cfg.top_k if cfg.top_k > 0 else 2   # default to top-2
        self.capacity_factor = cfg.capacity_factor

        self.router = nn.Linear(self.in_features, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        self.experts_W1 = nn.Parameter(
            torch.randn(E, self.in_features, H) * 0.01
        )
        self.experts_W2 = nn.Parameter(torch.zeros(E, H, self.out_features))

        self._balance_loss: Optional[torch.Tensor] = None
        self._token_counts_per_expert: Optional[torch.Tensor] = None
        self._token_counts_raw: Optional[torch.Tensor] = None
        self._overflow_fraction: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        B, N, D = x.shape
        E = self.num_experts
        K = self.top_k

        base = F.linear(x, self.weight, self.bias)

        # Full logits with live graph
        logits = self.router(x)                         # [B, N, E]

        # Full softmax probs (live graph) – used for balance loss only
        probs_for_loss = F.softmax(logits, dim=-1)      # [B, N, E]

        # Top-k selection → sparse router weights
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)   # [B, N, K]
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, topk_idx, topk_vals)
        router_weights = F.softmax(masked_logits, dim=-1)      # [B, N, E], sparse

        # Dispatch mask (non-differentiable) for frac computation
        dispatch = torch.zeros_like(logits)
        dispatch.scatter_(-1, topk_idx, 1.0)

        # Store raw token counts before capacity enforcement
        token_counts_raw = dispatch.sum(dim=(0, 1)).detach()   # [E], no grad

        # Capacity enforcement (if capacity_factor > 0)
        if self.capacity_factor > 0:
            import math as _math
            capacity = _math.ceil(self.capacity_factor * B * N / E)
            # For each expert, rank tokens by their router priority (logit value)
            # Shape of logits after topk masking: we use topk_vals for priority
            # Build per-expert priority tensor from masked_logits (non-selected = -inf)
            # rank trick: argsort(argsort(...)) gives rank of each token per expert
            # dispatch shape: [B, N, E] → reshape to [B*N, E] for ranking
            dispatch_flat = dispatch.view(B * N, E)            # [T, E]
            priority_flat = masked_logits.view(B * N, E)       # [T, E]

            # For each expert, rank tokens by descending priority among dispatched ones
            # Mask non-dispatched tokens to -inf before ranking
            priority_masked = priority_flat.masked_fill(dispatch_flat == 0, float("-inf"))
            # ranks[i, e] = position of token i in the queue for expert e (0 = highest priority)
            sorted_idx = torch.argsort(priority_masked, dim=0, descending=True)
            ranks = torch.argsort(sorted_idx, dim=0)           # [T, E], rank per (token, expert)

            overflow_mask = (dispatch_flat == 1) & (ranks >= capacity)   # [T, E]
            self._overflow_fraction = overflow_mask.float().mean().item()

            # Zero overflowed tokens in dispatch
            dispatch_flat = dispatch_flat.clone()
            dispatch_flat[overflow_mask] = 0.0
            dispatch = dispatch_flat.view(B, N, E)

            # Recompute sparse router_weights with updated dispatch.
            # Tokens that had ALL their top-k assignments dropped (dispatch row all zeros)
            # produce all-inf masked logits → softmax NaN. Use zeros for such tokens
            # (they receive no MoE contribution; base output only).
            masked_logits_cap = logits.masked_fill(dispatch == 0, float("-inf"))
            cap_weights = F.softmax(masked_logits_cap, dim=-1)   # [B, N, E]
            # has_expert[b, n] = True if token (b,n) still has at least one expert
            has_expert = dispatch.sum(dim=-1, keepdim=True) > 0   # [B, N, 1]
            router_weights = torch.where(
                has_expert.expand_as(cap_weights), cap_weights, torch.zeros_like(cap_weights)
            )
        else:
            self._overflow_fraction = 0.0

        # Balance loss: frac is non-diff (OK), mean_prob has grad → gradient flows
        frac = dispatch.detach().mean(dim=(0, 1))        # [E], no grad
        mean_prob = probs_for_loss.mean(dim=(0, 1))      # [E], live graph
        self._balance_loss = E * (frac * mean_prob).sum()

        # Store final token counts for logging
        self._token_counts_per_expert = dispatch.sum(dim=(0, 1)).detach()  # [E]
        self._token_counts_raw = token_counts_raw                          # [E] pre-capacity

        # Stats only (detached)
        self._last_router_weights = router_weights.detach()

        # FFN MoE forward (sparse weights have zeros for non-selected experts)
        x_dropped = self.dropout(x)
        ffn_hidden = torch.einsum("bnd,edh->bneh", x_dropped, self.experts_W1)
        ffn_hidden = F.gelu(ffn_hidden)
        ffn_out = torch.einsum("bneh,eho->bneo", ffn_hidden, self.experts_W2)
        moe_out = (ffn_out * router_weights.unsqueeze(-1)).sum(dim=2)

        out = base + moe_out
        return out.squeeze(1) if squeeze else out

    def compute_balance_loss(self) -> torch.Tensor:
        if self._balance_loss is None:
            return torch.tensor(0.0)
        return self._balance_loss

    def get_routing_stats(self) -> dict:
        if self._last_router_weights is None:
            return {}
        w = self._last_router_weights           # [B, N, E]
        mean_w = w.mean(dim=(0, 1)).cpu()       # [E] (sparse: zeros for non-selected)
        p = mean_w / (mean_w.sum() + 1e-8)
        entropy = -(p * (p + 1e-9).log()).sum().item()

        stats = {
            "mean_expert_weights": mean_w,
            "weight_entropy": entropy,
            "weight_cv": (mean_w.std() / (mean_w.mean() + 1e-8)).item(),
            "active_experts": (mean_w > 0.01).float().sum().item(),
            "overflow_fraction": self._overflow_fraction,
        }

        # Token count stats (from dispatch tensor)
        if self._token_counts_per_expert is not None:
            token_counts = self._token_counts_per_expert.cpu().float()
            mean_count = token_counts.mean()
            stats["token_count_per_expert"] = token_counts.tolist()
            stats["max_over_mean_load"] = (token_counts.max() / (mean_count + 1e-8)).item()
            stats["hot_expert_count"] = (token_counts > 2 * mean_count).sum().item()
            stats["load_variance"] = token_counts.var().item()

        return stats
