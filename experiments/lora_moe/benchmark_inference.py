#!/usr/bin/env python3
"""
Inference benchmark for trained LoRA MoE models.

Measures latency, throughput, and expert utilisation for three workloads:
  homogeneous  – fixed prompt/gen lengths
  heterogeneous – exponentially distributed lengths
  bursty        – Poisson-arrival sequential dispatch

Usage:
    python -m experiments.lora_moe.benchmark_inference \
        --model_dir outputs/soft_lora_moe_e8_r4 \
        --method soft_lora_moe \
        --workload homogeneous

    # Run all workloads
    for w in homogeneous heterogeneous bursty; do
        python -m experiments.lora_moe.benchmark_inference \
            --model_dir outputs/soft_lora_moe_e8_r4 \
            --method soft_lora_moe \
            --workload $w
    done
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lora_moe.injection import inject_lora, get_lora_moe_layers
from experiments.lora_moe.run_experiment import build_lora_cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA MoE inference benchmark")
    p.add_argument("--model_dir", required=True,
                   help="Path to trained model output directory")
    p.add_argument("--method", required=True,
                   choices=["standard", "qlora", "soft_moe", "sparse_moe",
                            "soft_lora_moe", "abmil_moe", "softmax_lora_moe"],
                   help="LoRA method used for this model")
    p.add_argument("--workload", default="homogeneous",
                   choices=["homogeneous", "heterogeneous", "bursty"],
                   help="Request workload type")
    p.add_argument("--num_requests", type=int, default=200,
                   help="Number of requests to benchmark")
    p.add_argument("--prompt_len", type=int, default=128,
                   help="Fixed prompt length (homogeneous) or mean (heterogeneous)")
    p.add_argument("--gen_len", type=int, default=128,
                   help="Fixed gen length (homogeneous) or mean (heterogeneous)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum total sequence length")
    p.add_argument("--arrival_rate", type=float, default=5.0,
                   help="Poisson arrival rate (req/s) for bursty workload")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def _sample_exp(mean: int, rng: random.Random, lo: int = 16, hi: int = 512) -> int:
    val = int(rng.expovariate(1.0 / mean))
    return max(lo, min(hi, val))


def gen_requests(
    workload: str,
    num_requests: int,
    prompt_len: int,
    gen_len: int,
    max_length: int,
    arrival_rate: float,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    requests = []
    t = 0.0

    for i in range(num_requests):
        if workload == "homogeneous":
            pl, gl = prompt_len, gen_len
            arrival = i / arrival_rate  # not used
        elif workload == "heterogeneous":
            pl = _sample_exp(prompt_len, rng, lo=16, hi=max_length - 16)
            gl = _sample_exp(gen_len, rng, lo=16, hi=max_length - pl)
            arrival = i / arrival_rate  # not used
        else:  # bursty
            pl = prompt_len
            gl = gen_len
            # Poisson inter-arrival: exponential with rate=arrival_rate
            arrival = t
            t += rng.expovariate(arrival_rate)

        requests.append({
            "id": i,
            "prompt_len": pl,
            "gen_len": min(gl, max_length - pl),
            "arrival": arrival,
        })
    return requests


# ---------------------------------------------------------------------------
# Expert load tracking hook
# ---------------------------------------------------------------------------

def attach_router_hooks(model: nn.Module, store: Dict[str, List[torch.Tensor]]):
    """Attach forward hooks to collect _last_router_weights from each MoE layer."""
    handles = []
    for name, module in get_lora_moe_layers(model):
        def make_hook(n):
            def hook(mod, inp, out):
                w = getattr(mod, "_last_router_weights", None)
                if w is not None:
                    store.setdefault(n, []).append(w.cpu().float())
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = args.model_dir
    method = args.method

    # ---- Load experiment config (if available) ----
    cfg_path = os.path.join(model_dir, "experiment_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            exp_cfg = json.load(f)
    else:
        exp_cfg = {"method": method, "model_name_or_path": model_dir}

    exp_cfg["method"] = method

    # ---- Tokenizer ----
    model_name = exp_cfg.get("model_name_or_path", model_dir)
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation

    # ---- Model ----
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=exp_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = True  # enable KV cache for generation

    # Inject LoRA adapters
    print(f"Injecting {method} adapters...")
    lora_cfg = build_lora_cfg(exp_cfg)
    inject_lora(model, method=method, cfg=lora_cfg)

    # Load saved adapter weights if they exist
    adapter_path = os.path.join(model_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded adapter weights from {safetensors_path}")
    elif os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded adapter weights from {adapter_path}")
    else:
        print("  WARNING: No saved adapter weights found; using random init.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model on {device}\n")

    # ---- Workload ----
    requests = gen_requests(
        workload=args.workload,
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        max_length=args.max_length,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
    )

    # ---- Router weight store ----
    router_weights_store: Dict[str, List[torch.Tensor]] = {}
    handles = attach_router_hooks(model, router_weights_store)

    # ---- Benchmark loop ----
    latencies: List[float] = []
    output_tokens: List[int] = []
    peak_memory_mb: float = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Running {args.workload} benchmark ({args.num_requests} requests)...")
    for req in requests:
        pl = req["prompt_len"]
        gl = req["gen_len"]

        # Synthesize a prompt (random token IDs within vocab range)
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(
            100, vocab_size - 1, (1, pl), device=device
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=gl,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        n_generated = out.shape[1] - pl
        latencies.append(t1 - t0)
        output_tokens.append(n_generated)

    # ---- Peak memory ----
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # Remove hooks
    for h in handles:
        h.remove()

    # ---- Latency percentiles ----
    lat_p50  = percentile(latencies, 50)  * 1000  # ms
    lat_p95  = percentile(latencies, 95)  * 1000
    lat_p99  = percentile(latencies, 99)  * 1000
    lat_p999 = percentile(latencies, 99.9) * 1000
    total_tokens = sum(output_tokens)
    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    # ---- Expert load stats ----
    expert_stats: Dict[str, float] = {}
    if router_weights_store:
        per_expert_means = []
        for layer_name, weight_list in router_weights_store.items():
            # Each tensor: [B, N, E], mean over batch/seq
            stacked = torch.cat(weight_list, dim=0)          # [total_tokens, N, E] approx
            layer_mean = stacked.mean(dim=(0, 1))            # [E]
            per_expert_means.append(layer_mean)

        all_means = torch.stack(per_expert_means)            # [L, E]
        global_mean = all_means.mean(dim=0)                  # [E]
        max_load = global_mean.max().item()
        mean_load = global_mean.mean().item()
        p = global_mean / (global_mean.sum() + 1e-8)
        entropy = -(p * (p + 1e-9).log()).sum().item()

        expert_stats = {
            "max_mean_expert_weight": max_load,
            "mean_expert_weight": mean_load,
            "max_mean_ratio": max_load / (mean_load + 1e-8),
            "expert_weight_entropy": entropy,
        }

    # ---- Results ----
    results = {
        "method": method,
        "workload": args.workload,
        "num_requests": args.num_requests,
        "latency_p50_ms": lat_p50,
        "latency_p95_ms": lat_p95,
        "latency_p99_ms": lat_p99,
        "latency_p999_ms": lat_p999,
        "throughput_tokens_per_sec": throughput,
        "total_output_tokens": total_tokens,
        "peak_gpu_memory_mb": peak_memory_mb,
        **expert_stats,
    }

    # ---- Console table ----
    print("\n" + "="*60)
    print(f"  Benchmark results: {method} / {args.workload}")
    print("="*60)
    print(f"  Requests          : {args.num_requests}")
    print(f"  Latency p50       : {lat_p50:.1f} ms")
    print(f"  Latency p95       : {lat_p95:.1f} ms")
    print(f"  Latency p99       : {lat_p99:.1f} ms")
    print(f"  Latency p99.9     : {lat_p999:.1f} ms")
    print(f"  Throughput        : {throughput:.1f} output tokens/sec")
    print(f"  Peak GPU memory   : {peak_memory_mb:.0f} MB")
    if expert_stats:
        print(f"  Expert max/mean   : {expert_stats['max_mean_ratio']:.3f}")
        print(f"  Expert entropy    : {expert_stats['expert_weight_entropy']:.3f}")
    print("="*60 + "\n")

    # ---- Save JSON ----
    out_path = os.path.join(model_dir, f"benchmark_{args.workload}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
