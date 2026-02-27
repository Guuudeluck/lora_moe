#!/usr/bin/env python3
"""
LoRA MoE Experiment Runner
==========================

Trains a Qwen model with one of three LoRA variants and logs results.

Usage:
    # Quick sanity check (tiny model, few steps)
    python -m experiments.lora_moe.run_experiment --config configs/abmil_moe.yaml --debug

    # Full experiment
    python -m experiments.lora_moe.run_experiment --config configs/abmil_moe.yaml

    # Override any yaml key on the CLI
    python -m experiments.lora_moe.run_experiment \
        --config configs/abmil_moe.yaml \
        --num_train_epochs 1 \
        --learning_rate 2e-4

    # Run all three methods sequentially
    for cfg in standard_lora softmax_lora_moe abmil_moe; do
        python -m experiments.lora_moe.run_experiment --config configs/${cfg}.yaml
    done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Make sure the repo root is in sys.path when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # lora_moe/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.lora_moe.data import load_alpaca
from experiments.lora_moe.data_mixture import load_ultrachat
from experiments.lora_moe.injection import inject_lora, print_trainable_params
from experiments.lora_moe.modules.lora_layers import LoRAConfig, MoELoRAConfig
from experiments.lora_moe.modules.ffn_moe_layers import MoEFFNConfig
from experiments.lora_moe.trainer import LoRAMoETrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA MoE experiment runner")

    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true",
                        help="Run a tiny debug trial (50 steps, small subset)")

    # Any yaml key can be overridden on the CLI
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--method", type=str,
                        choices=["standard", "softmax_lora_moe", "abmil_moe",
                                 "qlora", "soft_moe", "sparse_moe", "soft_lora_moe"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--expert_hidden_dim", type=int)
    parser.add_argument("--balance_loss_coeff", type=float)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--report_to", type=str, help="wandb | tensorboard | none")
    parser.add_argument("--dataset_type", type=str, choices=["alpaca", "ultrachat"],
                        help="Dataset to use for training")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Merge YAML file → CLI overrides."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to CWD first, then fall back to this file's directory
        if not config_path.exists():
            config_path = Path(__file__).parent / Path(args.config).name

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in [
        "model_name_or_path", "method", "output_dir", "num_train_epochs",
        "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
        "max_length", "max_samples", "lora_rank", "num_experts", "expert_hidden_dim",
        "balance_loss_coeff", "seed", "report_to", "dataset_type",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if args.debug:
        cfg["max_samples"] = cfg.get("debug_max_samples", 512)
        cfg["num_train_epochs"] = 1
        cfg["max_steps"] = 50
        cfg["eval_steps"] = 25
        cfg["save_steps"] = 50
        cfg["logging_steps"] = 5
        cfg["output_dir"] = cfg["output_dir"] + "_debug"
        print(">>> DEBUG MODE: max_samples=512, max_steps=50")

    return cfg


# ---------------------------------------------------------------------------
# Build LoRA config from experiment config
# ---------------------------------------------------------------------------

def build_lora_cfg(cfg: dict) -> LoRAConfig | MoELoRAConfig | MoEFFNConfig:
    method = cfg["method"]
    target = cfg.get("target_modules", ["q_proj", "v_proj"])
    rank = cfg.get("lora_rank", 16)
    alpha = cfg.get("lora_alpha", float(rank))
    dropout = cfg.get("lora_dropout", 0.05)

    if method in ("standard", "qlora"):
        return LoRAConfig(rank=rank, alpha=alpha, dropout=dropout, target_modules=target)

    if method in ("soft_moe", "sparse_moe"):
        return MoEFFNConfig(
            num_experts=cfg.get("num_experts", 8),
            expert_hidden_dim=cfg.get("expert_hidden_dim", 4),
            top_k=cfg.get("top_k", 0) if method == "soft_moe" else cfg.get("top_k", 2),
            dropout=dropout,
            balance_loss_coeff=cfg.get("balance_loss_coeff", 0.01),
            target_modules=target,
            capacity_factor=cfg.get("capacity_factor", 0.0),
        )

    # MoELoRAConfig: softmax_lora_moe | abmil_moe | soft_lora_moe
    return MoELoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target,
        num_experts=cfg.get("num_experts", 8),
        top_k=cfg.get("top_k", 0),
        attention_dim=cfg.get("attention_dim", 128),
        balance_loss_coeff=cfg.get("balance_loss_coeff", 0.01),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    method = cfg["method"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config for reproducibility
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  LoRA MoE Experiment")
    print(f"  Method      : {method}")
    print(f"  Model       : {cfg['model_name_or_path']}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*70}\n")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # causal LM needs right-padding during training

    # ---- Dataset ----
    dataset_type = cfg.get("dataset_type", "alpaca")
    if dataset_type == "ultrachat":
        print("Loading UltraChat-200k dataset...")
        datasets = load_ultrachat(
            tokenizer=tokenizer,
            max_length=cfg.get("max_length", 1024),
            val_split=cfg.get("val_split", 0.05),
            max_samples=cfg.get("max_samples", None),
            seed=seed,
            num_proc=cfg.get("dataloader_num_workers", 4),
        )
    else:
        print("Loading Alpaca dataset...")
        datasets = load_alpaca(
            tokenizer=tokenizer,
            max_length=cfg.get("max_length", 512),
            val_split=cfg.get("val_split", 0.05),
            max_samples=cfg.get("max_samples", None),
            seed=seed,
            num_proc=cfg.get("dataloader_num_workers", 4),
        )
    print(f"  Train : {len(datasets['train']):,} examples")
    print(f"  Val   : {len(datasets['validation']):,} examples\n")

    # ---- Model ----
    print("Loading base model...")
    skip_to_device = False

    if method == "qlora" or cfg.get("load_in_4bit", False):
        # PyTorch < 2.5 is missing nn.Module.set_submodule; patch it so that
        # Transformers 5.x bitsandbytes integration works on PyTorch 2.4.x.
        import torch.nn as _nn
        if not hasattr(_nn.Module, "set_submodule"):
            def _set_submodule(self, target: str, module):
                atoms = target.split(".")
                mod = self
                for item in atoms[:-1]:
                    mod = getattr(mod, item)
                setattr(mod, atoms[-1], module)
            _nn.Module.set_submodule = _set_submodule

        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        skip_to_device = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )

    model.config.use_cache = False          # required for gradient checkpointing

    # ---- Inject LoRA ----
    print(f"Injecting {method} LoRA...")
    lora_cfg = build_lora_cfg(cfg)
    inject_lora(model, method=method, cfg=lora_cfg)
    print_trainable_params(model)

    # Signal to Transformers 5.x Trainer that trainable adapters have been attached.
    # Without this flag, the quantization validator rejects training of 4-bit models
    # that are not wrapped via peft.PeftModel (our custom injection bypasses PEFT).
    if skip_to_device:
        model._hf_peft_config_loaded = True

    if not skip_to_device:
        # Move the full model (frozen weights + newly injected LoRA params) to GPU.
        # Cast to bfloat16 as well: inject_lora creates new float32 parameters, which
        # would cause dtype mismatches with the bfloat16 backbone.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Moving model to {device} (bfloat16)...")
        model = model.to(device=device, dtype=torch.bfloat16)

    if not skip_to_device and cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ---- Multi-GPU: read torchrun environment variables ----
    # TrainingArguments reads LOCAL_RANK / RANK / WORLD_SIZE automatically
    # from the environment when launched via torchrun or accelerate.
    # We expose them here for logging purposes only.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        print(f"  Distributed training: world_size={world_size}, local_rank={local_rank}")

    # ---- Training args ----
    report_to = cfg.get("report_to", "tensorboard")
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Training schedule
        num_train_epochs=cfg.get("num_train_epochs", 3),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        # Optimiser
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 10),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "adamw_torch_fused"),
        # Precision
        bf16=True,
        # Eval / save
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_strategy="steps",
        logging_steps=cfg.get("logging_steps", 10),
        report_to=report_to if report_to != "none" else [],
        run_name=cfg.get("run_name", f"{method}_{Path(output_dir).name}"),
        # Misc
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        # Multi-GPU: DDP/FSDP parameters are handled by TrainingArguments
        # automatically via LOCAL_RANK/WORLD_SIZE env vars set by torchrun.
        ddp_find_unused_parameters=False,
    )
    # Point TensorBoard at our log dir (v5.x style)
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(output_dir, "logs")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    balance_coeff = (
        lora_cfg.balance_loss_coeff
        if isinstance(lora_cfg, (MoELoRAConfig, MoEFFNConfig))
        else 0.0
    )

    trainer = LoRAMoETrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        # Custom kwargs
        method=method,
        balance_loss_coeff=balance_coeff,
    )

    # ---- Train ----
    print(f"\nStarting training  [{method}]...")
    train_result = trainer.train()

    # ---- Save ----
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # Final eval
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"\n{'='*70}")
    print(f"  Done!  Train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"         Eval  loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Outputs saved to : {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
