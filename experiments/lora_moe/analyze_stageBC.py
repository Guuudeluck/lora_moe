#!/usr/bin/env python3
"""
Stage B/C analysis: compare Sparse-MoE vs Soft-LoRA-MoE at scale.

Generates 7 figures:
  1. latency_cdf.png          — CDF of per-request latency per workload × method
  2. expert_load_histogram.png — token count per expert (Sparse-MoE only)
  3. step_time_distribution.png — violin/box of step times per method
  4. rank_step_skew.png        — per-rank step-time skew (Stage C only; skipped gracefully)
  5. router_contribution.png   — per-expert mean weight (Soft-LoRA-MoE only)
  6. tail_amplification.png    — p99/p50 latency ratio per workload, bar chart
  7. overflow_over_time.png    — overflow_fraction per training step (Sparse-MoE only)

Also writes a summary CSV combining Stage B/C eval metrics.

Reuses: read_tb_scalar, load_run, load_eval_metrics, smooth, plot_scalar
from analyze_results.py.

Usage:
    python -m experiments.lora_moe.analyze_stageBC \\
        --dirs outputs/stageB_sparse_moe outputs/stageB_soft_lora_moe \\
        --labels "Sparse-MoE (B)" "Soft-LoRA-MoE (B)" \\
        --out_dir figures/stageB/

    # Stage C (rank-skew plot enabled automatically if data present)
    python -m experiments.lora_moe.analyze_stageBC \\
        --dirs outputs/stageC_sparse_moe outputs/stageC_soft_lora_moe \\
        --labels "Sparse-MoE (C)" "Soft-LoRA-MoE (C)" \\
        --out_dir figures/stageC/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("WARNING: matplotlib / pandas / tensorboard not installed.")


# ---------------------------------------------------------------------------
# Re-use helpers from analyze_results.py
# ---------------------------------------------------------------------------

def read_tb_scalar(log_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    try:
        acc = EventAccumulator(log_dir)
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            return [], []
        events = acc.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]
    except Exception:
        return [], []


def read_tb_scalar_tags_matching(log_dir: str, prefix: str) -> Dict[str, Tuple[List[int], List[float]]]:
    """Return all scalar tags that start with `prefix`."""
    result = {}
    try:
        acc = EventAccumulator(log_dir)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if tag.startswith(prefix):
                events = acc.Scalars(tag)
                result[tag] = ([e.step for e in events], [e.value for e in events])
    except Exception:
        pass
    return result


def load_run(run_dir: str) -> Dict[str, Tuple[List[int], List[float]]]:
    log_dir = os.path.join(run_dir, "logs")
    tags = {
        "train_loss":             "train/loss",
        "eval_loss":              "eval/loss",
        "routing_entropy":        "train/routing/global_entropy",
        "routing_cv":             "train/routing/global_cv",
        "routing_active":         "train/routing/global_active_experts",
        "balance_loss":           "train/routing/balance_loss",
        "global_max_over_mean":   "train/routing/global_max_over_mean_load",
        "global_overflow":        "train/routing/global_overflow_fraction",
    }
    data: Dict[str, Tuple[List[int], List[float]]] = {}
    for key, tag in tags.items():
        steps, vals = read_tb_scalar(log_dir, tag)
        if steps:
            data[key] = (steps, vals)
    return data


def load_eval_metrics(run_dir: str) -> Dict[str, float]:
    path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_benchmark(run_dir: str, workload: str) -> Optional[Dict]:
    path = os.path.join(run_dir, f"benchmark_{workload}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def smooth(values: List[float], window: int = 5) -> List[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="same"))


COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
LINESTYLES = ["-", "--", "-.", ":"]


def plot_scalar(ax, run_data_list, labels, key, title, ylabel, smooth_window=5):
    for i, (data, label) in enumerate(zip(run_data_list, labels)):
        if key not in data:
            continue
        steps, vals = data[key]
        smoothed = smooth(vals, smooth_window)
        ax.plot(steps, smoothed, color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=label, linewidth=1.8)
        ax.plot(steps, vals, color=COLORS[i % len(COLORS)], alpha=0.2, linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 1: Latency CDF
# ---------------------------------------------------------------------------

def plot_latency_cdf(dirs, labels, out_dir):
    workloads = ["homogeneous", "heterogeneous", "bursty", "mixed_semantic"]
    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4))
    if len(workloads) == 1:
        axes = [axes]

    for ax, wl in zip(axes, workloads):
        has_data = False
        for i, (d, label) in enumerate(zip(dirs, labels)):
            bm = load_benchmark(d, wl)
            if bm is None:
                continue
            has_data = True
            # Reconstruct approximate latency distribution from percentile stats
            # (actual per-request data not stored; use known percentiles to build CDF)
            p50 = bm.get("latency_p50_ms", 0)
            p95 = bm.get("latency_p95_ms", 0)
            p99 = bm.get("latency_p99_ms", 0)
            p999 = bm.get("latency_p999_ms", p99 * 1.1)

            # Approximate CDF from 4 quantile points
            lat_vals = [0, p50, p95, p99, p999]
            cdf_vals = [0, 0.50, 0.95, 0.99, 0.999]
            ax.plot(lat_vals, cdf_vals,
                    color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    marker="o", markersize=4, label=label, linewidth=1.8)

        ax.set_title(f"Latency CDF\n({wl})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    fig.suptitle("Latency CDF by Workload", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "latency_cdf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Expert load histogram (Sparse-MoE only)
# ---------------------------------------------------------------------------

def plot_expert_load_histogram(dirs, labels, out_dir):
    workloads = ["homogeneous", "heterogeneous", "bursty", "mixed_semantic"]

    # Collect all sparse-moe entries that have per_expert_token_counts
    sparse_entries = []
    for d, label in zip(dirs, labels):
        for wl in workloads:
            bm = load_benchmark(d, wl)
            if bm and "per_expert_token_counts" in bm:
                sparse_entries.append((label, wl, bm["per_expert_token_counts"]))

    if not sparse_entries:
        print("  Skipping expert_load_histogram: no per_expert_token_counts found.")
        return

    ncols = min(4, len(sparse_entries))
    nrows = (len(sparse_entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for idx, (label, wl, counts) in enumerate(sparse_entries):
        ax = axes_flat[idx]
        expert_idx = list(range(len(counts)))
        mean_count = np.mean(counts)
        colors = ["#FF5722" if c > 2 * mean_count else "#2196F3" for c in counts]
        ax.bar(expert_idx, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(mean_count, color="black", linestyle="--", linewidth=1,
                   label=f"Mean={mean_count:.1f}")
        ax.axhline(2 * mean_count, color="red", linestyle=":", linewidth=1,
                   label="2× mean (hot)")
        ax.set_title(f"{label}\n{wl}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Token count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    # Hide unused subplots
    for ax in axes_flat[len(sparse_entries):]:
        ax.set_visible(False)

    fig.suptitle("Expert Token Load Distribution (Sparse-MoE)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "expert_load_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Step-time distribution (violin/box)
# ---------------------------------------------------------------------------

def plot_step_time_distribution(dirs, labels, run_data_list, out_dir):
    """
    Reads step_time_ms/p50 per step from TensorBoard and draws a violin plot.
    """
    step_times_per_run = []
    valid_labels = []

    for d, label in zip(dirs, labels):
        log_dir = os.path.join(d, "logs")
        # Try both "step_time_ms/p50" (distributed) and plain "step_time_ms/p50"
        steps, vals = read_tb_scalar(log_dir, "train/step_time_ms/p50")
        if not vals:
            steps, vals = read_tb_scalar(log_dir, "step_time_ms/p50")
        if vals:
            step_times_per_run.append(vals)
            valid_labels.append(label)

    if not step_times_per_run:
        print("  Skipping step_time_distribution: no step_time_ms data found.")
        return

    fig, ax = plt.subplots(figsize=(max(6, 3 * len(valid_labels)), 5))
    positions = list(range(1, len(valid_labels) + 1))
    parts = ax.violinplot(step_times_per_run, positions=positions,
                          showmedians=True, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i % len(COLORS)])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(valid_labels, rotation=15, ha="right")
    ax.set_ylabel("Step time p50 (ms)")
    ax.set_title("Step-Time Distribution\n(p50 per step, rolling window of 100)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "step_time_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Rank step skew (Stage C only)
# ---------------------------------------------------------------------------

def plot_rank_step_skew(dirs, labels, out_dir):
    """
    For each run, find all step_time_ms/rank* tags in TensorBoard and plot
    the distribution of per-rank p50 step times to reveal load skew.
    Skipped gracefully if no rank-specific data is found.
    """
    rank_data_found = False
    fig, axes = plt.subplots(1, len(dirs), figsize=(6 * len(dirs), 4))
    if len(dirs) == 1:
        axes = [axes]

    for ax, (d, label) in zip(axes, zip(dirs, labels)):
        log_dir = os.path.join(d, "logs")
        rank_tags = read_tb_scalar_tags_matching(log_dir, "train/step_time_ms/rank")
        if not rank_tags:
            rank_tags = read_tb_scalar_tags_matching(log_dir, "step_time_ms/rank")

        if not rank_tags:
            ax.text(0.5, 0.5, "No rank timing data\n(single-GPU run)",
                    transform=ax.transAxes, ha="center", va="center", color="gray",
                    fontsize=11)
            ax.set_title(f"{label}", fontsize=10)
            continue

        rank_data_found = True
        rank_medians = []
        rank_names = []
        for tag, (steps, vals) in sorted(rank_tags.items()):
            if vals:
                rank_medians.append(np.median(vals))
                # Extract rank number from tag name
                rank_names.append(tag.split("rank")[-1].split("/")[0])

        ax.bar(rank_names, rank_medians,
               color=[COLORS[i % len(COLORS)] for i in range(len(rank_names))])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Median step time (ms)")
        ax.set_title(f"Per-Rank Step Time\n{label}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate skew ratio
        if len(rank_medians) > 1:
            skew = max(rank_medians) / (min(rank_medians) + 1e-8)
            ax.text(0.98, 0.95, f"Skew: {skew:.2f}×",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if not rank_data_found:
        plt.close(fig)
        print("  Skipping rank_step_skew: no multi-GPU rank data found (Stage C only).")
        return

    fig.suptitle("Per-Rank Step-Time Skew (Stage C)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "rank_step_skew.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Router contribution (Soft-LoRA-MoE expert weights)
# ---------------------------------------------------------------------------

def plot_router_contribution(dirs, labels, out_dir):
    """
    For Soft-LoRA-MoE runs, load the routing_snapshot.json and plot
    the per-expert mean weight distribution across layers.
    """
    abmil_entries = []
    for d, label in zip(dirs, labels):
        if "soft_lora" in label.lower() or "abmil" in label.lower() or "soft-lora" in label.lower():
            snap_path = os.path.join(d, "routing_snapshot.json")
            if os.path.exists(snap_path):
                with open(snap_path) as f:
                    snap = json.load(f)
                abmil_entries.append((label, snap))

    if not abmil_entries:
        print("  Skipping router_contribution: no ABMIL routing snapshots found.")
        return

    ncols = len(abmil_entries)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, (label, snap) in zip(axes, abmil_entries):
        layer_names = list(snap.keys())
        data = np.array([snap[k] for k in layer_names])   # [L, E]
        num_experts = data.shape[1] if data.ndim == 2 else 0

        if num_experts == 0:
            ax.text(0.5, 0.5, "Empty snapshot", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        # Box plot: one box per expert showing distribution across layers
        bp = ax.boxplot(data, patch_artist=True, notch=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.6)

        ax.set_xlabel("Expert index")
        ax.set_ylabel("Mean activation weight")
        ax.set_title(f"Router Contribution\n{label}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add collapse score annotation
        all_means = data.mean(axis=0)   # [E] mean over layers
        std = all_means.std()
        mean = all_means.mean()
        collapse = 1.0 - std / (mean + 1e-8)
        ax.text(0.98, 0.95, f"Collapse score: {collapse:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    fig.suptitle("Soft-LoRA-MoE Expert Contribution Distribution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "router_contribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Tail amplification bar chart
# ---------------------------------------------------------------------------

def plot_tail_amplification(dirs, labels, out_dir):
    workloads = ["homogeneous", "heterogeneous", "bursty", "mixed_semantic"]

    # Build data: {workload: {label: tail_amp}}
    wl_data: Dict[str, Dict[str, float]] = {wl: {} for wl in workloads}
    for d, label in zip(dirs, labels):
        for wl in workloads:
            bm = load_benchmark(d, wl)
            if bm and "tail_amplification" in bm:
                wl_data[wl][label] = bm["tail_amplification"]
            elif bm and "latency_p50_ms" in bm and "latency_p99_ms" in bm:
                p50 = bm["latency_p50_ms"]
                p99 = bm["latency_p99_ms"]
                wl_data[wl][label] = p99 / (p50 + 1e-8)

    # Check if any data
    has_any = any(wl_data[wl] for wl in workloads)
    if not has_any:
        print("  Skipping tail_amplification: no benchmark data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(workloads))
    n_methods = len(labels)
    width = 0.8 / max(n_methods, 1)

    for i, label in enumerate(labels):
        vals = [wl_data[wl].get(label, float("nan")) for wl in workloads]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=COLORS[i % len(COLORS)], alpha=0.8,
                      edgecolor="white")
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}×", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.set_ylabel("Tail amplification (p99/p50)")
    ax.set_title("Tail Latency Amplification\n(lower = better; 1.0 = no tail)",
                 fontsize=12, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "tail_amplification.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Overflow fraction over training time (Sparse-MoE only)
# ---------------------------------------------------------------------------

def plot_overflow_over_time(dirs, labels, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False

    for i, (d, label) in enumerate(zip(dirs, labels)):
        log_dir = os.path.join(d, "logs")
        steps, vals = read_tb_scalar(log_dir, "train/routing/global_overflow_fraction")
        if not vals:
            steps, vals = read_tb_scalar(log_dir, "routing/global_overflow_fraction")
        if vals:
            has_data = True
            smoothed = smooth(vals, window=5)
            ax.plot(steps, smoothed, color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=label, linewidth=1.8)
            ax.plot(steps, vals, color=COLORS[i % len(COLORS)], alpha=0.2, linewidth=0.8)

    if not has_data:
        plt.close(fig)
        print("  Skipping overflow_over_time: no overflow_fraction data found.")
        return

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Overflow fraction")
    ax.set_title("Token Overflow Fraction Over Training\n(Sparse-MoE; capacity_factor=1.25)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "overflow_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(dirs, labels, run_data_list, out_dir):
    workloads = ["homogeneous", "heterogeneous", "bursty", "mixed_semantic"]
    rows = []

    for label, d, data in zip(labels, dirs, run_data_list):
        emets = load_eval_metrics(d)
        row: Dict = {"Method": label}

        # Training metrics
        row["Final Train Loss"] = (
            round(data["train_loss"][1][-1], 4) if "train_loss" in data else "N/A"
        )
        row["Final Eval Loss"] = emets.get("eval_loss", "N/A")
        if "eval_loss" in emets:
            import math
            row["Eval Perplexity"] = round(math.exp(emets["eval_loss"]), 2)

        # Routing metrics (last 50 steps)
        for key, col in [
            ("routing_cv", "Mean Load CV"),
            ("global_max_over_mean", "Max/Mean Load"),
            ("global_overflow", "Overflow Fraction"),
        ]:
            if key in data:
                row[col] = round(np.mean(data[key][1][-50:]), 4)

        # Benchmark stats
        for wl in workloads:
            bm = load_benchmark(d, wl)
            if bm:
                row[f"P99 latency ({wl})"] = round(bm.get("latency_p99_ms", float("nan")), 1)
                row[f"Tail amp ({wl})"] = round(
                    bm.get("tail_amplification",
                           bm.get("latency_p99_ms", 0) / (bm.get("latency_p50_ms", 1) + 1e-8)),
                    3,
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary_stageBC.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("\n" + df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B/C analysis")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out_dir", default="figures/stageBC")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("Install matplotlib, pandas, and tensorboard to generate plots:\n"
              "  pip install matplotlib pandas tensorboard")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    dirs = args.dirs
    labels = args.labels if args.labels else [Path(d).name for d in dirs]
    assert len(labels) == len(dirs), "Number of labels must match number of dirs"

    print(f"Loading runs: {dict(zip(labels, dirs))}")
    run_data = [load_run(d) for d in dirs]

    # ---- Figure 1: Latency CDF ----
    print("\n[1/7] Latency CDF...")
    plot_latency_cdf(dirs, labels, args.out_dir)

    # ---- Figure 2: Expert load histogram ----
    print("[2/7] Expert load histogram...")
    plot_expert_load_histogram(dirs, labels, args.out_dir)

    # ---- Figure 3: Step-time distribution ----
    print("[3/7] Step-time distribution...")
    plot_step_time_distribution(dirs, labels, run_data, args.out_dir)

    # ---- Figure 4: Rank step skew (Stage C only) ----
    print("[4/7] Rank step skew...")
    plot_rank_step_skew(dirs, labels, args.out_dir)

    # ---- Figure 5: Router contribution (Soft-LoRA-MoE) ----
    print("[5/7] Router contribution...")
    plot_router_contribution(dirs, labels, args.out_dir)

    # ---- Figure 6: Tail amplification ----
    print("[6/7] Tail amplification...")
    plot_tail_amplification(dirs, labels, args.out_dir)

    # ---- Figure 7: Overflow over time ----
    print("[7/7] Overflow over time...")
    plot_overflow_over_time(dirs, labels, args.out_dir)

    # ---- Summary CSV ----
    print("\nWriting summary CSV...")
    write_summary_csv(dirs, labels, run_data, args.out_dir)

    print(f"\nAll outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
