"""
Routing metrics collected from all injected LoRA MoE layers.

Collected after every forward pass (the trainer calls these):
  - per_layer_entropy    : routing entropy per layer
  - per_layer_cv         : coefficient-of-variation (load imbalance) per layer
  - per_layer_active     : # active experts per layer
  - global_entropy       : mean entropy across layers
  - global_cv            : mean CV across layers
  - balance_loss         : aggregate auxiliary balance loss (scalar tensor)

For Standard LoRA these all return zeros/None so the trainer can be uniform.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from experiments.lora_moe.modules.lora_layers import (
    ABMILMoELoRALinear,
    SoftmaxMoELoRALinear,
)
from experiments.lora_moe.modules.ffn_moe_layers import (
    SoftMoEFFNLinear,
    SparseMoEFFNLinear,
)

_MoELayer = (ABMILMoELoRALinear, SoftmaxMoELoRALinear, SoftMoEFFNLinear, SparseMoEFFNLinear)


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_routing_metrics(
    model: nn.Module,
    balance_loss_coeff: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Walk the model, gather routing stats from every MoE LoRA layer,
    and compute an aggregate auxiliary balance loss.

    Returns:
        balance_loss  : scalar tensor (0 if no MoE layers)
        log_metrics   : flat dict of scalars suitable for trainer.log()
    """
    balance_terms: List[torch.Tensor] = []
    entropies: List[float] = []
    cvs: List[float] = []
    actives: List[float] = []

    layer_idx = 0
    log: Dict[str, float] = {}

    for name, module in model.named_modules():
        if not isinstance(module, _MoELayer):
            continue

        # Per-layer balance loss
        bl = module.compute_balance_loss()
        if bl.requires_grad or bl.item() != 0.0:
            balance_terms.append(bl)

        # Per-layer routing stats
        stats = module.get_routing_stats()
        if stats:
            h = stats.get("weight_entropy", 0.0)
            cv = stats.get("weight_cv", 0.0)
            act = stats.get("active_experts", 0.0)

            entropies.append(h)
            cvs.append(cv)
            actives.append(act)

            log[f"routing/layer{layer_idx:02d}/entropy"] = h
            log[f"routing/layer{layer_idx:02d}/cv"] = cv
            log[f"routing/layer{layer_idx:02d}/active_experts"] = act
            layer_idx += 1

    # Aggregate scalars
    if entropies:
        log["routing/global_entropy"] = sum(entropies) / len(entropies)
        log["routing/global_cv"] = sum(cvs) / len(cvs)
        log["routing/global_active_experts"] = sum(actives) / len(actives)

    # Aggregate balance loss
    if balance_terms:
        balance_loss = balance_loss_coeff * torch.stack(balance_terms).mean()
    else:
        balance_loss = torch.tensor(0.0)

    log["routing/balance_loss"] = balance_loss.item()
    return balance_loss, log


# ---------------------------------------------------------------------------
# Expert utilisation snapshot (for analysis script)
# ---------------------------------------------------------------------------

def snapshot_expert_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return {layer_name: mean_expert_weights tensor} for all MoE layers.
    Useful for post-training analysis.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, _MoELayer):
            stats = module.get_routing_stats()
            if "mean_expert_weights" in stats:
                out[name] = stats["mean_expert_weights"]
    return out
