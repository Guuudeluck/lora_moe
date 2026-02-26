"""
Model injection: replace target nn.Linear layers in any HuggingFace model
with our custom LoRA variant modules.

Usage:
    from injection import inject_lora, print_trainable_params

    model = AutoModelForCausalLM.from_pretrained(...)
    inject_lora(model, method="abmil_moe", cfg=MoELoRAConfig(...))
    print_trainable_params(model)
"""

from __future__ import annotations

import re
from typing import Union

import torch.nn as nn

from experiments.lora_moe.modules.lora_layers import (
    ABMILMoELoRALinear,
    LoRAConfig,
    MoELoRAConfig,
    QLoRALinear,
    SoftmaxMoELoRALinear,
    StandardLoRALinear,
)
from experiments.lora_moe.modules.ffn_moe_layers import (
    MoEFFNConfig,
    SoftMoEFFNLinear,
    SparseMoEFFNLinear,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_lora(
    model: nn.Module,
    method: str,                              # "standard" | "softmax_lora_moe" | "abmil_moe"
                                              # | "soft_moe" | "sparse_moe"
                                              # | "soft_lora_moe" | "qlora"
    cfg: Union[LoRAConfig, MoELoRAConfig, MoEFFNConfig],
) -> nn.Module:
    """
    Traverse `model` and replace every nn.Linear whose name matches
    cfg.target_modules with the chosen LoRA variant.

    For standard/MoE methods: copy the frozen weight into the new module,
    then move it to GPU after injection.
    For QLoRA: keep the original quantized module as base_linear; only
    freeze non-quantized parameters (quantized params are already frozen).

    Returns the modified model (in-place).
    """
    if method == "qlora":
        # For QLoRA the backbone is already 4-bit quantized (requires_grad=False
        # on quantized params).  Only freeze the non-quantized layers that are
        # trainable by default (e.g. layer norms, embeddings).
        for name, p in model.named_parameters():
            p.requires_grad_(False)
    else:
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)

    _replace_linears(model, method, cfg, prefix="")
    return model


def get_lora_moe_layers(model: nn.Module):
    """Yield (name, module) for every injected LoRA/MoE layer."""
    for name, module in model.named_modules():
        if isinstance(module, (
            QLoRALinear,
            StandardLoRALinear,
            SoftmaxMoELoRALinear,
            ABMILMoELoRALinear,
            SoftMoEFFNLinear,
            SparseMoEFFNLinear,
        )):
            yield name, module


def print_trainable_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\nParameter summary:")
    print(f"  Total      : {total:>15,}")
    print(f"  Frozen     : {frozen:>15,}")
    print(f"  Trainable  : {trainable:>15,}  ({100 * trainable / total:.2f}%)")

    # Break down by layer type
    counts: dict[str, int] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = _param_kind(name)
            counts[key] = counts.get(key, 0) + p.numel()
    if counts:
        print("\n  Trainable breakdown:")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {k:<30s}: {v:>12,}")
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _param_kind(name: str) -> str:
    for tag in ("experts_W1", "experts_W2", "experts_A", "experts_B",
                "lora_A", "lora_B",
                "router_V", "router_U", "router_W", "router"):
        if tag in name:
            return tag
    return "other"


def _module_matches_target(child_name: str, target_modules: list[str]) -> bool:
    """
    Check if child_name matches any target.
    Supports exact strings ("q_proj") and regex (".*proj").
    """
    for target in target_modules:
        if re.fullmatch(target, child_name):
            return True
    return False


def _is_linear(module: nn.Module) -> bool:
    """Return True for nn.Linear and bitsandbytes Linear4bit/Linear8bit."""
    if isinstance(module, nn.Linear):
        return True
    # bitsandbytes quantized linears are not subclasses of nn.Linear
    cls_name = type(module).__name__
    return cls_name in ("Linear4bit", "Linear8bitLt")


def _replace_linears(
    parent: nn.Module,
    method: str,
    cfg: Union[LoRAConfig, MoELoRAConfig, MoEFFNConfig],
    prefix: str,
) -> None:
    """Recursively replace matching linear children."""
    for child_name, child in list(parent.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if _is_linear(child) and _module_matches_target(child_name, cfg.target_modules):
            new_module = _build_lora_linear(child, method, cfg)
            setattr(parent, child_name, new_module)
        else:
            # Recurse
            _replace_linears(child, method, cfg, full_name)


def _build_lora_linear(
    linear: nn.Linear,
    method: str,
    cfg: Union[LoRAConfig, MoELoRAConfig, MoEFFNConfig],
) -> nn.Module:
    if method == "standard":
        return StandardLoRALinear(linear, cfg)
    elif method == "qlora":
        return QLoRALinear(linear, cfg)
    elif method == "softmax_lora_moe":
        assert isinstance(cfg, MoELoRAConfig), "softmax_lora_moe requires MoELoRAConfig"
        return SoftmaxMoELoRALinear(linear, cfg)
    elif method in ("abmil_moe", "soft_lora_moe"):
        assert isinstance(cfg, MoELoRAConfig), f"{method} requires MoELoRAConfig"
        return ABMILMoELoRALinear(linear, cfg)
    elif method == "soft_moe":
        assert isinstance(cfg, MoEFFNConfig), "soft_moe requires MoEFFNConfig"
        return SoftMoEFFNLinear(linear, cfg)
    elif method == "sparse_moe":
        assert isinstance(cfg, MoEFFNConfig), "sparse_moe requires MoEFFNConfig"
        return SparseMoEFFNLinear(linear, cfg)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Choose: standard | qlora | softmax_lora_moe | abmil_moe | "
            f"soft_lora_moe | soft_moe | sparse_moe"
        )
