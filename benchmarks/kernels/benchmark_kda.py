#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for the KDA (Kimi Delta Attention) Triton kernels used in Kimi-K3.

This script exercises two execution paths:
  - **prefill** : ``chunk_kda`` / ``chunk_kda_with_fused_gate`` — processes an
    entire context in one shot (chunk-recurrent algorithm).
  - **decode**  : ``fused_recurrent_kda`` — updates the SSM state one token at
    a time (the per-step decode path).

All inputs and weights are randomly initialised; no real model checkpoint or
network access is required. Set ``--use-real-weights`` and point
``--model-path`` to a local directory if you want to plug in real weights
(currently this flag is a pass-through for user extension; the kernel itself
only needs q/k/v/g/beta tensors regardless of weight origin).

Requirements
------------
* CUDA GPU (the Triton kernels target CUDA/ROCm). The script exits cleanly
  on CPU-only environments with an informative message.
* The vLLM package installed in the current Python environment.

Example – quick smoke run (no GPU needed for --help):
    python benchmarks/kernels/benchmark_kda.py --help

Example – prefill benchmark with custom shapes:
    python benchmarks/kernels/benchmark_kda.py \\
        --batch-size 4 --seq-len 2048 --num-heads 32 --head-dim 128 \\
        --warmup 5 --iters 20

Example – decode benchmark:
    python benchmarks/kernels/benchmark_kda.py \\
        --mode decode --batch-size 8 --num-heads 32 --head-dim 128

Example – sweep over multiple batch sizes and sequence lengths:
    python benchmarks/kernels/benchmark_kda.py \\
        --batch-sizes 1 4 8 --seq-lens 512 1024 4096 --mode both
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import NamedTuple

import torch
from torch.nn.functional import logsigmoid

# ---------------------------------------------------------------------------
# Source-tree import fallback for vllm KDA ops
# ---------------------------------------------------------------------------


def _ensure_kda_importable() -> None:
    """Extend vllm's package search paths with the repo source tree if needed.

    When the installed vllm wheel pre-dates the KDA ops (e.g.
    ``vllm.third_party.flash_linear_attention`` is absent), this function
    patches ``vllm.__path__`` and related subpackage paths so that the
    source tree copies are discoverable without reinstalling.
    """
    try:
        import vllm.third_party.flash_linear_attention  # noqa: F401
        return
    except ImportError:
        pass

    # This script lives at benchmarks/kernels/<name>.py; repo root is ../../
    vllm_src = Path(__file__).resolve().parent.parent.parent / "vllm"
    if not (vllm_src / "third_party" / "flash_linear_attention").is_dir():
        return

    import vllm

    if str(vllm_src) not in vllm.__path__:
        vllm.__path__.insert(0, str(vllm_src))

    for key in list(sys.modules):
        if sys.modules[key] is None and key.startswith("vllm."):
            del sys.modules[key]

    try:
        import vllm.utils as _vu
        utils_src = str(vllm_src / "utils")
        if utils_src not in _vu.__path__:
            _vu.__path__.insert(0, utils_src)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class BenchResult(NamedTuple):
    mode: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: str
    latency_mean_ms: float
    latency_median_ms: float
    latency_min_ms: float
    latency_max_ms: float
    tokens_per_sec: float
    peak_mem_mb: float | None


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _make_prefill_inputs(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Create random q/k/v/g/beta and initial state for the prefill (chunk) path."""
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate = logsigmoid(
        torch.randn(batch, seq_len, num_heads, head_dim,
                    dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.randn(batch, seq_len, num_heads, dtype=dtype, device=device).sigmoid()
    # initial_state layout for chunk_kda at the call site is (K, V);
    # bench_prefill transposes to (V, K) before passing to the kernel.
    initial_state = torch.zeros(
        batch, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    # cu_seqlens: one segment per batch element
    cu_seqlens_list = [i * seq_len for i in range(batch + 1)]
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    return dict(q=q, k=k, v=v, g=gate, beta=beta,
                initial_state=initial_state, cu_seqlens=cu_seqlens)


def _make_decode_inputs(
    batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Create random inputs for the decode (fused-recurrent, single-token) path."""
    # decode: seq_len == 1 per token step
    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate = logsigmoid(
        torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.randn(batch, 1, num_heads, dtype=dtype, device=device).sigmoid()
    # fused_recurrent_kda expects initial_state: [B, H, K, V] float32
    initial_state = torch.zeros(
        batch, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    return dict(q=q, k=k, v=v, g=gate, beta=beta, initial_state=initial_state)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_timed(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    """Run *fn* for *warmup* iterations then measure *iters* iterations.

    Returns a list of per-iteration wall-clock times in milliseconds.
    """
    for _ in range(warmup):
        fn()
    _sync(device)

    latencies: list[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1e3)
    return latencies


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


def bench_prefill(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> BenchResult:
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda

    inp = _make_prefill_inputs(batch_size, seq_len, num_heads, head_dim,
                                dtype, device)

    # chunk_kda expects initial_state in (V, K) layout; flip the last two dims.
    h0 = inp["initial_state"].transpose(-1, -2).contiguous()

    def fn():
        chunk_kda(
            q=inp["q"].clone(),
            k=inp["k"].clone(),
            v=inp["v"].clone(),
            g=inp["g"].clone(),
            beta=inp["beta"].clone(),
            initial_state=h0.clone(),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inp["cu_seqlens"],
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = _run_timed(fn, warmup, iters, device)

    peak_mb: float | None = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    tokens = batch_size * seq_len
    return BenchResult(
        mode="prefill",
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=str(dtype).replace("torch.", ""),
        latency_mean_ms=mean(latencies),
        latency_median_ms=median(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        tokens_per_sec=tokens / (mean(latencies) * 1e-3),
        peak_mem_mb=peak_mb,
    )


def bench_decode(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> BenchResult:
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

    inp = _make_decode_inputs(batch_size, num_heads, head_dim, dtype, device)

    def fn():
        fused_recurrent_kda(
            q=inp["q"].clone(),
            k=inp["k"].clone(),
            v=inp["v"].clone(),
            g=inp["g"].clone(),
            beta=inp["beta"].clone(),
            initial_state=inp["initial_state"].clone(),
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = _run_timed(fn, warmup, iters, device)

    peak_mb: float | None = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    tokens = batch_size * 1  # one token per step
    return BenchResult(
        mode="decode",
        batch_size=batch_size,
        seq_len=1,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=str(dtype).replace("torch.", ""),
        latency_mean_ms=mean(latencies),
        latency_median_ms=median(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        tokens_per_sec=tokens / (mean(latencies) * 1e-3),
        peak_mem_mb=peak_mb,
    )


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def _print_result(r: BenchResult) -> None:
    mem_str = f"{r.peak_mem_mb:.1f} MB" if r.peak_mem_mb is not None else "N/A"
    print(
        f"[{r.mode:>7}] bs={r.batch_size:>4}  seqlen={r.seq_len:>6}  "
        f"H={r.num_heads}  D={r.head_dim}  dtype={r.dtype:<10}"
        f"  mean={r.latency_mean_ms:>8.3f}ms  "
        f"median={r.latency_median_ms:>8.3f}ms  "
        f"min={r.latency_min_ms:>8.3f}ms  "
        f"max={r.latency_max_ms:>8.3f}ms  "
        f"throughput={r.tokens_per_sec:>12.0f} tok/s  "
        f"peak_mem={mem_str}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark KDA (Kimi Delta Attention) Triton kernels.\n"
            "Runs on CUDA GPU. Uses random inputs — no real model weights needed.\n"
            "Pass --use-real-weights to extend with your own weight-loading logic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Shape arguments (single values, used when --batch-sizes / --seq-lens
    # are not provided)
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--seq-len", type=int, default=1024,
        help="Context/sequence length for prefill (default: 1024).",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32,
        help="Number of attention heads (default: 32).",
    )
    parser.add_argument(
        "--head-dim", type=int, default=128,
        help="Head dimension K=V (default: 128).",
    )
    # Sweep arguments
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=None,
        help="Sweep over multiple batch sizes (overrides --batch-size).",
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=None,
        help="Sweep over multiple sequence lengths (overrides --seq-len).",
    )
    # Measurement parameters
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Number of timed iterations (default: 20).",
    )
    # Mode
    parser.add_argument(
        "--mode", choices=["prefill", "decode", "both"], default="both",
        help="Which execution path to benchmark (default: both).",
    )
    # dtype
    parser.add_argument(
        "--dtype", choices=list(_DTYPE_MAP), default="bfloat16",
        help="Floating-point dtype for q/k/v (default: bfloat16).",
    )
    # Device
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device string (default: cuda).",
    )
    # Real-weights flag (pass-through; no-op in default mode)
    parser.add_argument(
        "--use-real-weights", action="store_true",
        help=(
            "Placeholder flag: when set, the script does NOT automatically "
            "download weights. Extend the script with your own "
            "weight-loading logic. Default: use random weights."
        ),
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a local model directory (only used when --use-real-weights).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _ensure_kda_importable()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "ERROR: CUDA device requested but no CUDA GPU is available. "
            "The KDA Triton kernels require a GPU. "
            "Run with --device cpu is not supported; exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    if device.type != "cuda":
        print(
            "WARNING: KDA Triton kernels target CUDA/ROCm. "
            "Non-CUDA devices are not supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.use_real_weights:
        print(
            "INFO: --use-real-weights is set. "
            "The script will still use random tensors for the kernel inputs. "
            "Extend the weight-loading section below to load from "
            f"--model-path={args.model_path!r} and derive q/k/v projections."
        )

    dtype = _DTYPE_MAP[args.dtype]
    batch_sizes = (
        args.batch_sizes if args.batch_sizes is not None else [args.batch_size]
    )
    seq_lens = args.seq_lens if args.seq_lens is not None else [args.seq_len]

    print(
        f"KDA benchmark  mode={args.mode}  dtype={args.dtype}  "
        f"device={device}  warmup={args.warmup}  iters={args.iters}"
    )
    print("-" * 120)

    results: list[BenchResult] = []

    for bs in batch_sizes:
        for sl in seq_lens:
            if args.mode in ("prefill", "both"):
                r = bench_prefill(
                    batch_size=bs,
                    seq_len=sl,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(r)
                _print_result(r)

            if args.mode in ("decode", "both"):
                r = bench_decode(
                    batch_size=bs,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(r)
                _print_result(r)

    print("-" * 120)
    print(f"Done. {len(results)} configuration(s) benchmarked.")


if __name__ == "__main__":
    main()
