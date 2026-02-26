#!/usr/bin/env python3
"""
Analyze and compare results from the three LoRA MoE experiments.

Reads TensorBoard event logs from each output_dir and produces:
  1. Loss curves (train + eval)              → loss_curves.png
  2. Routing entropy over time               → routing_entropy.png
  3. Load imbalance (CV) over time           → load_balance.png
  4. Per-expert utilisation heatmap          → expert_utilisation.png  (from routing_snapshot.json)
  5. Summary table                           → summary.csv

Usage:
    python -m experiments.lora_moe.analyze_results \
        --dirs outputs/standard_lora_r16 outputs/softmax_lora_moe_e8_r4 outputs/abmil_moe_e8_r4 \
        --labels "Standard LoRA" "Softmax MoE" "ABMIL MoE" \
        --out_dir figures/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("WARNING: matplotlib / pandas / tensorboard not installed. "
          "Install them to generate plots:\n"
          "  pip install matplotlib pandas tensorboard")


# ---------------------------------------------------------------------------
# TensorBoard reader
# ---------------------------------------------------------------------------

def read_tb_scalar(log_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Return (steps, values) for a TensorBoard scalar tag."""
    acc = EventAccumulator(log_dir)
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [], []
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def load_run(run_dir: str) -> Dict[str, Tuple[List[int], List[float]]]:
    """Load key scalars from a run's TensorBoard log directory."""
    log_dir = os.path.join(run_dir, "logs")
    tags = {
        "train_loss":       "train/loss",
        "eval_loss":        "eval/loss",
        "routing_entropy":  "train/routing/global_entropy",
        "routing_cv":       "train/routing/global_cv",
        "routing_active":   "train/routing/global_active_experts",
        "balance_loss":     "train/routing/balance_loss",
    }
    data: Dict[str, Tuple[List[int], List[float]]] = {}
    for key, tag in tags.items():
        steps, vals = read_tb_scalar(log_dir, tag)
        if steps:
            data[key] = (steps, vals)
    return data


def load_routing_snapshot(run_dir: str) -> Optional[Dict[str, List[float]]]:
    path = os.path.join(run_dir, "routing_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_eval_metrics(run_dir: str) -> Dict[str, float]:
    path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]   # blue, orange, green, purple
LINESTYLES = ["-", "--", "-.", ":"]


def smooth(values: List[float], window: int = 5) -> List[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="same"))


def plot_scalar(
    ax,
    run_data_list: List[Dict],
    labels: List[str],
    key: str,
    title: str,
    ylabel: str,
    smooth_window: int = 5,
) -> None:
    for i, (data, label) in enumerate(zip(run_data_list, labels)):
        if key not in data:
            continue
        steps, vals = data[key]
        smoothed = smooth(vals, smooth_window)
        ax.plot(steps, smoothed, color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=label, linewidth=1.8)
        # Raw values in faint background
        ax.plot(steps, vals, color=COLORS[i % len(COLORS)], alpha=0.2, linewidth=0.8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_expert_heatmap(ax, snapshot: Dict[str, List[float]], title: str) -> None:
    """Heatmap of mean expert weights per layer."""
    layer_names = list(snapshot.keys())
    data = np.array([snapshot[k] for k in layer_names])  # [num_layers, num_experts]

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(0, len(layer_names), max(1, len(layer_names) // 10)))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean activation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Run output directories")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels (default: directory names)")
    parser.add_argument("--out_dir", default="figures",
                        help="Directory to save figures")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("Install matplotlib, pandas, and tensorboard to run analysis.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    dirs = args.dirs
    labels = args.labels if args.labels else [Path(d).name for d in dirs]
    assert len(labels) == len(dirs)

    print(f"Loading runs: {dict(zip(labels, dirs))}")

    run_data = [load_run(d) for d in dirs]
    snapshots = [load_routing_snapshot(d) for d in dirs]
    eval_metrics = [load_eval_metrics(d) for d in dirs]

    # -------------------------------------------------------------------
    # Figure 1: Loss curves
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_scalar(axes[0], run_data, labels, "train_loss",
                "Training Loss", "Loss", smooth_window=10)
    plot_scalar(axes[1], run_data, labels, "eval_loss",
                "Validation Loss", "Loss", smooth_window=1)
    fig.suptitle("Stage A Loss Comparison: QLoRA / Soft-MoE / Sparse-MoE / Soft-LoRA-MoE",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    # -------------------------------------------------------------------
    # Figure 2: Routing metrics (MoE only; skip pure-LoRA baselines)
    # -------------------------------------------------------------------
    # Exclude standard LoRA / QLoRA rows that have no routing metrics
    _no_routing = {"standard", "qlora", "soft_moe", "sparse_moe"}
    moe_data = [
        d for d, l in zip(run_data, labels)
        if not any(tag in l.lower() for tag in ("qlora", "standard lora"))
    ]
    moe_labels = [
        l for l in labels
        if not any(tag in l.lower() for tag in ("qlora", "standard lora"))
    ]
    # Fallback: use all
    if not moe_data:
        moe_data, moe_labels = run_data, labels

    has_routing = any("routing_entropy" in d for d in moe_data)
    if has_routing:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_scalar(axes[0], moe_data, moe_labels, "routing_entropy",
                    "Routing Entropy\n(higher = more balanced)", "Entropy (nats)")
        plot_scalar(axes[1], moe_data, moe_labels, "routing_cv",
                    "Load Imbalance (CV)\n(lower = more balanced)",
                    "Coefficient of Variation")
        plot_scalar(axes[2], moe_data, moe_labels, "routing_active",
                    "Active Experts\n(higher = better utilisation)", "# Active Experts")
        fig.suptitle("Routing / Load Balance Metrics", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(args.out_dir, "routing_metrics.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # -------------------------------------------------------------------
    # Figure 3: Expert utilisation heatmaps
    # -------------------------------------------------------------------
    moe_snapshots = [(s, l) for s, l in zip(snapshots, labels) if s is not None]
    if moe_snapshots:
        ncols = len(moe_snapshots)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ax, (snap, label) in zip(axes, moe_snapshots):
            plot_expert_heatmap(ax, snap, f"Expert Utilisation\n{label}")
        fig.suptitle("Per-Layer Expert Activation (final checkpoint)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(args.out_dir, "expert_utilisation.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # -------------------------------------------------------------------
    # Figure 4: Balance loss over time
    # -------------------------------------------------------------------
    has_balance = any("balance_loss" in d for d in run_data)
    if has_balance:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_scalar(ax, run_data, labels, "balance_loss",
                    "Auxiliary Balance Loss", "Balance Loss")
        fig.tight_layout()
        path = os.path.join(args.out_dir, "balance_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    rows = []
    for label, data, emets in zip(labels, run_data, eval_metrics):
        row = {"Method": label}
        row["Final Train Loss"] = (
            round(data["train_loss"][1][-1], 4) if "train_loss" in data else "N/A"
        )
        row["Final Eval Loss"] = emets.get("eval_loss", "N/A")
        row["Final Eval Perplexity"] = (
            round(np.exp(emets["eval_loss"]), 2) if "eval_loss" in emets else "N/A"
        )
        if "routing_entropy" in data:
            row["Mean Routing Entropy"] = round(
                np.mean(data["routing_entropy"][1][-50:]), 4
            )
        if "routing_cv" in data:
            row["Mean Load CV"] = round(np.mean(data["routing_cv"][1][-50:]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
