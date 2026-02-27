#!/usr/bin/env python3
"""
Stage B/C inference benchmark — extends benchmark_inference.py with:

  1. mixed_semantic workload: 4 template sets × 50 prompts each (200 total).
     Tests whether semantic shift causes routing skew in Sparse-MoE.

  2. New output stats:
       tail_amplification   : latency_p99 / latency_p50
       per_expert_token_counts : token histogram (Sparse-MoE only)
       stall_ratio          : fraction of requests with latency > 2× p50

  3. --compare_dir flag: loads a second model, runs the same workload,
     outputs side-by-side JSON for analyze_stageBC.py.

Usage:
    python -m experiments.lora_moe.benchmark_stageBC \\
        --model_dir outputs/stageB_sparse_moe \\
        --method sparse_moe \\
        --workload mixed_semantic

    # Compare two models side-by-side
    python -m experiments.lora_moe.benchmark_stageBC \\
        --model_dir outputs/stageB_sparse_moe \\
        --method sparse_moe \\
        --compare_dir outputs/stageB_soft_lora_moe \\
        --compare_method soft_lora_moe \\
        --workload mixed_semantic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lora_moe.injection import inject_lora, get_lora_moe_layers
from experiments.lora_moe.modules.ffn_moe_layers import SparseMoEFFNLinear
from experiments.lora_moe.run_experiment import build_lora_cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B/C LoRA MoE inference benchmark")
    p.add_argument("--model_dir", required=True,
                   help="Path to trained model output directory")
    p.add_argument("--method", required=True,
                   choices=["standard", "qlora", "soft_moe", "sparse_moe",
                            "soft_lora_moe", "abmil_moe", "softmax_lora_moe"])
    p.add_argument("--workload", default="homogeneous",
                   choices=["homogeneous", "heterogeneous", "bursty", "mixed_semantic"])
    p.add_argument("--num_requests", type=int, default=200)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--arrival_rate", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    # Compare flag
    p.add_argument("--compare_dir", type=str, default=None,
                   help="Optional second model dir for side-by-side comparison")
    p.add_argument("--compare_method", type=str, default=None,
                   choices=["standard", "qlora", "soft_moe", "sparse_moe",
                            "soft_lora_moe", "abmil_moe", "softmax_lora_moe"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mixed-semantic workload templates
# ---------------------------------------------------------------------------

_MATH_TEMPLATES = [
    "A farmer has {a} apples and gives away {b}. How many remain?",
    "If a train travels {a} km in {b} hours, what is its speed?",
    "What is the area of a rectangle with width {a} and height {b}?",
    "Solve: {a}x + {b} = {c}. What is x?",
    "A store sells items for ${a}. After a {b}% discount, what is the price?",
]

_CODE_TEMPLATES = [
    "Write a Python function that returns the {a}th Fibonacci number.",
    "Implement a binary search algorithm in Python for a list of {a} elements.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Create a Python decorator that retries a function up to {a} times.",
    "Write a Python generator that yields prime numbers up to {a}.",
]

_QA_TEMPLATES = [
    "What is the capital of {country}?",
    "Who invented {invention}?",
    "When did {event} occur?",
    "What is the chemical formula for {substance}?",
    "Explain the concept of {concept} in simple terms.",
]

_CHAT_TEMPLATES = [
    "I'm feeling {emotion} today. Can you help me feel better?",
    "Can you recommend a {genre} book for someone who enjoys {topic}?",
    "What are the pros and cons of {option_a} vs {option_b}?",
    "Help me write a short {format} about {subject}.",
    "What advice would you give someone starting to learn {skill}?",
]

_MATH_FILLS = [
    dict(a=10, b=3), dict(a=150, b=2), dict(a=7, b=5),
    dict(a=100, b=25), dict(a=3, b=4, c=19),
    dict(a=12, b=6), dict(a=200, b=3), dict(a=9, b=7),
    dict(a=50, b=10), dict(a=8, b=3, c=29),
]

_CODE_FILLS = [
    dict(a=10), dict(a=100), dict(a=50), dict(a=3), dict(a=1000),
    dict(a=15), dict(a=200), dict(a=5), dict(a=7), dict(a=500),
]

_QA_FILLS = [
    dict(country="France"), dict(invention="the telephone"),
    dict(event="World War II"), dict(substance="water"),
    dict(concept="entropy"), dict(country="Japan"),
    dict(invention="the printing press"), dict(event="the French Revolution"),
    dict(substance="carbon dioxide"), dict(concept="quantum entanglement"),
]

_CHAT_FILLS = [
    dict(emotion="anxious", genre="mystery", topic="travel",
         option_a="Python", option_b="JavaScript",
         format="poem", subject="autumn", skill="guitar"),
    dict(emotion="excited", genre="sci-fi", topic="science",
         option_a="macOS", option_b="Linux",
         format="essay", subject="innovation", skill="cooking"),
    dict(emotion="tired", genre="self-help", topic="productivity",
         option_a="remote work", option_b="office work",
         format="story", subject="friendship", skill="painting"),
    dict(emotion="curious", genre="history", topic="culture",
         option_a="cats", option_b="dogs",
         format="letter", subject="gratitude", skill="photography"),
    dict(emotion="happy", genre="thriller", topic="adventure",
         option_a="tea", option_b="coffee",
         format="blog post", subject="sustainability", skill="chess"),
]


def _build_mixed_semantic_prompts(num_per_category: int = 50, seed: int = 42) -> List[str]:
    """Build a mixed pool of semantically diverse prompts."""
    rng = random.Random(seed)
    prompts: List[str] = []

    template_groups = [
        (_MATH_TEMPLATES, _MATH_FILLS),
        (_CODE_TEMPLATES, _CODE_FILLS),
        (_QA_TEMPLATES, _QA_FILLS),
        (_CHAT_TEMPLATES, _CHAT_FILLS),
    ]

    for templates, fills in template_groups:
        for i in range(num_per_category):
            tmpl = templates[i % len(templates)]
            fill = fills[i % len(fills)]
            try:
                prompts.append(tmpl.format(**fill))
            except KeyError:
                # Fall back to the raw template if fill is missing a key
                prompts.append(tmpl)

    rng.shuffle(prompts)
    return prompts


# ---------------------------------------------------------------------------
# Workload generation
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
) -> Tuple[List[Dict], Optional[List[str]]]:
    """
    Returns (requests, semantic_prompts).
    semantic_prompts is non-None only for mixed_semantic workload.
    """
    rng = random.Random(seed)
    requests = []
    t = 0.0
    semantic_prompts: Optional[List[str]] = None

    if workload == "mixed_semantic":
        # 200 prompts (50 per category), truncated/padded to num_requests
        all_prompts = _build_mixed_semantic_prompts(
            num_per_category=max(50, (num_requests + 3) // 4),
            seed=seed,
        )
        semantic_prompts = all_prompts[:num_requests]
        for i, prompt_text in enumerate(semantic_prompts):
            requests.append({
                "id": i,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "arrival": i / arrival_rate,
                "prompt_text": prompt_text,
            })
        return requests, semantic_prompts

    for i in range(num_requests):
        if workload == "homogeneous":
            pl, gl = prompt_len, gen_len
            arrival = i / arrival_rate
        elif workload == "heterogeneous":
            pl = _sample_exp(prompt_len, rng, lo=16, hi=max_length - 16)
            gl = _sample_exp(gen_len, rng, lo=16, hi=max_length - pl)
            arrival = i / arrival_rate
        else:  # bursty
            pl = prompt_len
            gl = gen_len
            arrival = t
            t += rng.expovariate(arrival_rate)

        requests.append({
            "id": i,
            "prompt_len": pl,
            "gen_len": min(gl, max_length - pl),
            "arrival": arrival,
        })
    return requests, None


# ---------------------------------------------------------------------------
# Expert load tracking
# ---------------------------------------------------------------------------

def attach_router_hooks(model: nn.Module, store: Dict[str, List[torch.Tensor]]):
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


def collect_token_counts(model: nn.Module) -> Optional[List[float]]:
    """Collect token counts per expert from SparseMoEFFNLinear layers."""
    counts_per_layer = []
    for name, module in get_lora_moe_layers(model):
        if isinstance(module, SparseMoEFFNLinear):
            tc = getattr(module, "_token_counts_per_expert", None)
            if tc is not None:
                counts_per_layer.append(tc.cpu().float().tolist())
    if not counts_per_layer:
        return None
    # Average across layers
    import numpy as np
    arr = np.array(counts_per_layer)   # [L, E]
    return arr.mean(axis=0).tolist()


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
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str, method: str):
    cfg_path = os.path.join(model_dir, "experiment_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            exp_cfg = json.load(f)
    else:
        exp_cfg = {"method": method, "model_name_or_path": model_dir}
    exp_cfg["method"] = method

    model_name = exp_cfg.get("model_name_or_path", model_dir)
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=exp_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = True

    print(f"Injecting {method} adapters...")
    lora_cfg = build_lora_cfg(exp_cfg)
    inject_lora(model, method=method, cfg=lora_cfg)

    # Load saved adapter weights
    adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
    fallback_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(adapter_path):
        from safetensors.torch import load_file
        state = load_file(adapter_path)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded adapter weights from {adapter_path}")
    elif os.path.exists(fallback_path):
        from safetensors.torch import load_file
        state = load_file(fallback_path)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded adapter weights from {fallback_path}")
    else:
        print("  WARNING: No saved adapter weights found; using random init.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model on {device}\n")
    return model, tokenizer, exp_cfg, device


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    requests: List[Dict],
    workload: str,
    args: argparse.Namespace,
    method: str,
) -> Dict:
    router_weights_store: Dict[str, List[torch.Tensor]] = {}
    handles = attach_router_hooks(model, router_weights_store)

    latencies: List[float] = []
    output_tokens: List[int] = []
    peak_memory_mb: float = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Running {workload} benchmark ({len(requests)} requests)...")
    for req in requests:
        pl = req["prompt_len"]
        gl = req["gen_len"]

        if workload == "mixed_semantic" and "prompt_text" in req:
            # Tokenize actual semantic prompt
            enc = tokenizer(
                req["prompt_text"],
                return_tensors="pt",
                truncation=True,
                max_length=pl,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
        else:
            vocab_size = tokenizer.vocab_size
            input_ids = torch.randint(100, vocab_size - 1, (1, pl), device=device)

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

        n_generated = out.shape[1] - input_ids.shape[1]
        latencies.append(t1 - t0)
        output_tokens.append(n_generated)

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    for h in handles:
        h.remove()

    # Latency stats
    lat_p50  = percentile(latencies, 50)  * 1000
    lat_p95  = percentile(latencies, 95)  * 1000
    lat_p99  = percentile(latencies, 99)  * 1000
    lat_p999 = percentile(latencies, 99.9) * 1000
    total_tokens = sum(output_tokens)
    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    # Tail amplification and stall ratio
    tail_amplification = lat_p99 / (lat_p50 + 1e-8)
    stall_threshold = 2.0 * lat_p50
    stall_ratio = sum(1 for l in latencies if l * 1000 > stall_threshold) / max(len(latencies), 1)

    # Expert load stats from router hooks
    expert_stats: Dict = {}
    if router_weights_store:
        per_expert_means = []
        for layer_name, weight_list in router_weights_store.items():
            stacked = torch.cat(weight_list, dim=0)
            layer_mean = stacked.mean(dim=(0, 1))
            per_expert_means.append(layer_mean)

        all_means = torch.stack(per_expert_means)
        global_mean = all_means.mean(dim=0)
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

    # Per-expert token counts (Sparse-MoE only)
    per_expert_token_counts = collect_token_counts(model)

    results = {
        "method": method,
        "workload": workload,
        "num_requests": len(requests),
        "latency_p50_ms": lat_p50,
        "latency_p95_ms": lat_p95,
        "latency_p99_ms": lat_p99,
        "latency_p999_ms": lat_p999,
        "tail_amplification": tail_amplification,
        "stall_ratio": stall_ratio,
        "throughput_tokens_per_sec": throughput,
        "total_output_tokens": total_tokens,
        "peak_gpu_memory_mb": peak_memory_mb,
        **expert_stats,
    }
    if per_expert_token_counts is not None:
        results["per_expert_token_counts"] = per_expert_token_counts

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Benchmark: {method} / {workload}")
    print(f"{'='*60}")
    print(f"  Requests          : {len(requests)}")
    print(f"  Latency p50       : {lat_p50:.1f} ms")
    print(f"  Latency p95       : {lat_p95:.1f} ms")
    print(f"  Latency p99       : {lat_p99:.1f} ms")
    print(f"  Tail amplification: {tail_amplification:.2f}x  (p99/p50)")
    print(f"  Stall ratio       : {stall_ratio:.3f}  (latency > 2×p50)")
    print(f"  Throughput        : {throughput:.1f} tokens/sec")
    print(f"  Peak GPU memory   : {peak_memory_mb:.0f} MB")
    if expert_stats:
        print(f"  Expert max/mean   : {expert_stats['max_mean_ratio']:.3f}")
        print(f"  Expert entropy    : {expert_stats['expert_weight_entropy']:.3f}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Primary model ----
    model, tokenizer, exp_cfg, device = load_model_and_tokenizer(
        args.model_dir, args.method
    )

    requests, _ = gen_requests(
        workload=args.workload,
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        max_length=args.max_length,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
    )

    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        requests=requests,
        workload=args.workload,
        args=args,
        method=args.method,
    )

    out_path = os.path.join(args.model_dir, f"benchmark_{args.workload}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    # ---- Optional comparison model ----
    if args.compare_dir:
        compare_method = args.compare_method or args.method
        print(f"\n--- Running comparison: {compare_method} ---")

        # Free primary model memory before loading second
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model2, tokenizer2, _, device2 = load_model_and_tokenizer(
            args.compare_dir, compare_method
        )

        # Re-generate same requests for fair comparison
        requests2, _ = gen_requests(
            workload=args.workload,
            num_requests=args.num_requests,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            max_length=args.max_length,
            arrival_rate=args.arrival_rate,
            seed=args.seed,
        )

        results2 = run_benchmark(
            model=model2,
            tokenizer=tokenizer2,
            device=device2,
            requests=requests2,
            workload=args.workload,
            args=args,
            method=compare_method,
        )

        compare_path = os.path.join(args.compare_dir, f"benchmark_{args.workload}.json")
        with open(compare_path, "w") as f:
            json.dump(results2, f, indent=2)
        print(f"Comparison results saved to: {compare_path}")

        # Side-by-side JSON
        side_by_side = {
            args.method: results,
            compare_method: results2,
        }
        sb_path = os.path.join(
            args.model_dir, f"comparison_{args.workload}.json"
        )
        with open(sb_path, "w") as f:
            json.dump(side_by_side, f, indent=2)
        print(f"Side-by-side comparison saved to: {sb_path}")


if __name__ == "__main__":
    main()
