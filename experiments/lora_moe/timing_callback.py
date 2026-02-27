"""
Step-timing TrainerCallback for wall-clock latency tracking.

Records per-step elapsed time, maintains a rolling buffer of the last
100 steps, and computes p50/p95/p99 percentiles.  In a distributed
(DDP/FSDP) run each rank logs its own timing under a rank-suffixed key
so rank skew is visible in TensorBoard.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _percentile(buf: list, p: float) -> float:
    if not buf:
        return float("nan")
    s = sorted(buf)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class StepTimingCallback(TrainerCallback):
    """
    Measures wall-clock time for each optimizer step (gradient accumulation
    micro-steps are excluded — only the outer step boundary is timed).

    Percentile stats (p50/p95/p99) over the last `window` steps are stored
    in `trainer._timing_log_buffer` and picked up by LoRAMoETrainer.log().

    In a distributed run, timing is logged per rank so that rank-level skew
    (a symptom of load imbalance) is visible:
        step_time_ms/rank0, step_time_ms/rank1, ...
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._t0: Optional[float] = None
        self._buffer: Deque[float] = deque(maxlen=window)
        self._rank: int = 0
        self._is_distributed: bool = False

    # ------------------------------------------------------------------
    # Determine distributed rank on first step_begin
    # ------------------------------------------------------------------

    def _init_rank(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._is_distributed = True
        else:
            self._rank = 0
            self._is_distributed = False

    # ------------------------------------------------------------------
    # Trainer callback hooks
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._t0 is None:
            # First call — resolve rank
            self._init_rank()
        self._t0 = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._t0 is None:
            return
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._buffer.append(elapsed_ms)

        buf = list(self._buffer)
        p50 = _percentile(buf, 50)
        p95 = _percentile(buf, 95)
        p99 = _percentile(buf, 99)

        rank_suffix = f"/rank{self._rank}" if self._is_distributed else ""
        timing_metrics: Dict[str, float] = {
            f"step_time_ms{rank_suffix}/p50": p50,
            f"step_time_ms{rank_suffix}/p95": p95,
            f"step_time_ms{rank_suffix}/p99": p99,
        }

        # Stash for LoRAMoETrainer.log() to pick up
        trainer = kwargs.get("model", None)  # trainer is not passed directly
        # Fallback: attach to the TrainerState (accessible in log())
        if not hasattr(state, "_timing_log_buffer"):
            state._timing_log_buffer = {}
        state._timing_log_buffer.update(timing_metrics)
