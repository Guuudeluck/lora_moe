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

from experiments.lora_moe.metrics import collect_routing_metrics, snapshot_expert_weights

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
        # _hf_peft_config_loaded=True is needed only to pass
        # validate_quantization_for_training() in super().__init__() above.
        # Leaving it set causes save_pretrained to call PEFT-specific methods
        # (active_adapters, get_adapter_state_dict) that don't exist on our
        # custom-injected model.  Safe to remove now that __init__ is done.
        if hasattr(self.model, "_hf_peft_config_loaded"):
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
        super().log(logs, start_time, **kwargs)

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
