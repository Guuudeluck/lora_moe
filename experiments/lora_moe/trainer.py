"""
Custom HuggingFace Trainer that:
  1. Adds auxiliary balance loss to the CE loss (for MoE methods)
  2. Logs routing metrics (entropy, CV, active experts) every logging_steps
  3. Saves a routing-stats JSON snapshot at evaluation time
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Trainer
from transformers.utils import logging
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

from experiments.lora_moe.metrics import collect_routing_metrics, snapshot_expert_weights
from experiments.lora_moe.timing_callback import StepTimingCallback

logger = logging.get_logger(__name__)


class LoRAMoETrainer(Trainer):
    """
    Drop-in Trainer subclass.  Pass extra kwargs:
        balance_loss_coeff  (float, default 0.01)
        method              (str)   – "standard" | "softmax_lora_moe" | "abmil_moe"
    """

    def __init__(self, *args, balance_loss_coeff: float = 0.01, method: str = "abmil_moe", **kwargs):
        super().__init__(*args, **kwargs)
        self.balance_loss_coeff = balance_loss_coeff
        self.method = method
        self._routing_log_buffer: Dict[str, float] = {}
        # Register step-timing callback for wall-clock profiling
        self.add_callback(StepTimingCallback())
        # _hf_peft_config_loaded=True is set as an INSTANCE attribute only for
        # QLoRA to bypass validate_quantization_for_training() in __init__().
        # Leaving it set causes save_pretrained to call PEFT-specific methods
        # (active_adapters, get_adapter_state_dict) that don't exist on our
        # custom-injected model.  Safe to remove now that __init__ is done.
        # NOTE: PreTrainedModel has this as a CLASS attribute (defaults False),
        # so we must check __dict__ (instance attrs only), not hasattr.
        if "_hf_peft_config_loaded" in self.model.__dict__:
            del self.model._hf_peft_config_loaded

    # ------------------------------------------------------------------
    # Override compute_loss to inject balance loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:

        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Collect routing metrics and balance loss
        balance_loss, routing_log = collect_routing_metrics(
            model, balance_loss_coeff=self.balance_loss_coeff
        )

        # Stash for logging (we log on the trainer's schedule)
        self._routing_log_buffer = routing_log

        # Add balance loss only for MoE methods
        if self.method != "standard" and balance_loss.item() != 0.0:
            loss = ce_loss + balance_loss.to(ce_loss.device)
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Override log to append routing metrics
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:
        if self._routing_log_buffer:
            logs.update(self._routing_log_buffer)
            self._routing_log_buffer = {}
        # Flush timing stats collected by StepTimingCallback
        timing_buf = getattr(self.state, "_timing_log_buffer", {})
        if timing_buf:
            logs.update(timing_buf)
            self.state._timing_log_buffer = {}
        super().log(logs, start_time, **kwargs)

    # ------------------------------------------------------------------
    # Save only adapter (trainable) weights — keeps checkpoints tiny
    # (~50 MB) instead of saving the full frozen 7B backbone (~14 GB).
    # ------------------------------------------------------------------

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Collect only the trainable parameters (adapter / router / expert weights).
        adapter_state = {
            name: param.data.cpu().contiguous()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        save_safetensors(adapter_state, os.path.join(output_dir, "adapter_model.safetensors"))

        # Also persist the model config so the checkpoint is self-describing.
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)

        logger.info(
            f"Saved adapter weights ({len(adapter_state)} tensors, "
            f"{sum(t.numel() for t in adapter_state.values()) / 1e6:.1f}M params) "
            f"to {output_dir}"
        )

    def _load_best_model(self) -> None:
        best_dir = self.state.best_model_checkpoint
        if best_dir is None:
            return
        adapter_path = os.path.join(best_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state_dict = load_safetensors(adapter_path, device="cpu")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if unexpected:
                logger.warning(f"Unexpected keys when loading adapter: {unexpected[:5]}")
            logger.info(f"Loaded best adapter from {adapter_path}")
        else:
            # Fallback: full checkpoint saved before this change was applied.
            super()._load_best_model()

    # ------------------------------------------------------------------
    # Save routing snapshot on evaluation
    # ------------------------------------------------------------------

    def evaluate(self, *args, **kwargs):
        results = super().evaluate(*args, **kwargs)

        # Save per-expert weight snapshot
        snapshot = snapshot_expert_weights(self.model)
        if snapshot:
            out_path = os.path.join(self.args.output_dir, "routing_snapshot.json")
            serialisable = {k: v.tolist() for k, v in snapshot.items()}
            with open(out_path, "w") as f:
                json.dump(serialisable, f, indent=2)
            logger.info(f"Routing snapshot saved to {out_path}")

        return results
