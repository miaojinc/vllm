#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone correctness & micro-benchmark test for KDA (Kimi Delta Attention).

This script can be run from **any directory** as long as vLLM is installed in
the active Python environment.  It does NOT require the vLLM source tree to be
present on sys.path.

Requirements
------------
* Python 3.10+
* torch
* vllm installed
  (``pip install vllm`` or ``VLLM_USE_PRECOMPILED=1 uv pip install -e .``)
* A CUDA GPU (Triton kernels are CUDA/ROCm only)

Usage
-----
# Correctness tests only (default)
python standalone_test_kda.py

# Correctness + benchmark sweep
python standalone_test_kda.py --bench

# Tune tolerance / shapes
python standalone_test_kda.py --num-heads 16 --head-dim 64

# Full help
python standalone_test_kda.py --help

Exit codes
----------
0  All tests passed (and optional benchmarks completed).
1  One or more tests failed, or CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import sys
import time
from statistics import mean, median
from typing import NamedTuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failed: list[str] = []


def _check(name: str, cond: bool, msg: str = "") -> None:
    status = PASS if cond else FAIL
    suffix = f" — {msg}" if msg else ""
    print(f"  [{status}] {name}{suffix}")
    if not cond:
        _failed.append(name)


def _assert_close(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    rel_tol: float,
    abs_atol: float = 1e-6,
) -> None:
    abs_err = (ref.detach() - tri.detach()).abs().max().item()
    rmse_diff = (
        (ref.detach() - tri.detach()).float().square().mean().sqrt().item()
    )
    rmse_base = ref.detach().float().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    has_nan_ref = torch.isnan(ref).any().item()
    has_nan_tri = torch.isnan(tri).any().item()
    ok = (
        not has_nan_ref
        and not has_nan_tri
        and (abs_err <= abs_atol or rel_err < rel_tol)
    )
    msg = (
        f"abs={abs_err:.6f}  rmse_rel={rel_err:.6f}  tol={rel_tol}"
        f"{'  NaN in ref!' if has_nan_ref else ''}"
        f"{'  NaN in tri!' if has_nan_tri else ''}"
    )
    _check(name, ok, msg)


# ---------------------------------------------------------------------------
# Naive reference implementation (float32, CPU or GPU)
# ---------------------------------------------------------------------------


def _naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch recurrent KDA reference (ported from FLA's naive.py).

    Args:
        q: (B, T, H, K)
        k: (B, T, H, K)
        v: (B, T, H, V)
        g: (B, T, H, K)  — log-sigmoid gate
        beta: (B, T, H)
        scale: head-dim scale; defaults to K**-0.5
        initial_state: (H, K, V) float32
        output_final_state: whether to return final state

    Returns:
        o: (B, T, H, V), same dtype as v
        ht: (H, K, V) float32 or None
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = (x.float() for x in [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i = q[:, i]        # (B, H, K)
        k_i = k[:, i]
        v_i = v[:, i]        # (B, H, V)
        g_i = g[:, i]        # (B, H, K)
        b_i = beta[:, i]     # (B, H)
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)

    final_state = S if output_final_state else None
    return o.to(dtype), final_state


# ---------------------------------------------------------------------------
# Import vLLM KDA ops (requires vllm installed)
# ---------------------------------------------------------------------------


def _import_kda():
    try:
        from vllm.third_party.flash_linear_attention.ops.kda import (
            chunk_kda,
            chunk_kda_with_fused_gate,
            fused_kda_gate,
            fused_recurrent_kda,
        )
        from vllm.third_party.flash_linear_attention.ops.l2norm import (
            l2norm_fwd,
        )
    except ImportError as exc:
        print(
            f"ERROR: Could not import vLLM KDA ops: {exc}\n"
            "Make sure vLLM is installed: pip install vllm",
            file=sys.stderr,
        )
        sys.exit(1)
    return (
        chunk_kda, chunk_kda_with_fused_gate,
        fused_kda_gate, fused_recurrent_kda, l2norm_fwd,
    )


# ---------------------------------------------------------------------------
# Test: chunk_kda correctness
# ---------------------------------------------------------------------------


def test_chunk_kda(
    chunk_kda,
    l2norm_fwd,
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    label = f"chunk_kda H={H} D={D} cu={cu_seqlens} {dtype}"
    print(f"\n--- {label} ---")

    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    torch.manual_seed(42)
    B = 1
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(
        torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)

    # Reference: naive recurrent with l2-normalised q/k per segment
    ref_outputs, ref_states = [], []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        o_i, ht_i = _naive_recurrent_kda(
            q_i, k_i, v[:, s:e], g[:, s:e], beta[:, s:e],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)
    ref_ht = torch.cat(ref_states, dim=0)   # (N, H, K, V)

    # Triton: initial_state in (V, K) layout
    tri_o, tri_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.transpose(-1, -2).contiguous().clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close(f"{label} / o", ref_o, tri_o, rel_tol=0.005)
    _assert_close(
        f"{label} / ht",
        ref_ht,
        tri_ht.transpose(-1, -2).contiguous(),
        rel_tol=0.005,
    )


# ---------------------------------------------------------------------------
# Test: fused gate matches unfused
# ---------------------------------------------------------------------------


def test_chunk_kda_fused_gate(
    chunk_kda,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    label = f"chunk_kda_fused_gate H={H} D={D} cu={cu_seqlens} {dtype}"
    print(f"\n--- {label} ---")

    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    torch.manual_seed(123)
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn(1, T, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    raw_g = torch.randn(1, T, H, D, dtype=dtype, device=device)
    beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()
    A_log = (torch.randn(H, dtype=torch.float32, device=device) * 0.5).contiguous()
    dt_bias = (
        torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
    ).contiguous()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    initial_state = h0.transpose(-1, -2).contiguous()

    gate = fused_kda_gate(
        raw_g.reshape(T, H * D), A_log, D, g_bias=dt_bias
    ).unsqueeze(0)
    old_o, old_ht = chunk_kda(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g=gate, beta=beta.clone(),
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )
    new_o, new_ht = chunk_kda_with_fused_gate(
        q=q.clone(), k=k.clone(), v=v.clone(),
        raw_g=raw_g, beta=beta.clone(),
        A_log=A_log, g_bias=dt_bias,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close(f"{label} / o", old_o, new_o, rel_tol=1e-3, abs_atol=1e-3)
    _assert_close(f"{label} / ht", old_ht, new_ht, rel_tol=1e-3, abs_atol=1e-3)


# ---------------------------------------------------------------------------
# Test: fused_recurrent_kda (decode path) produces finite output
# ---------------------------------------------------------------------------


def test_fused_recurrent_kda(
    fused_recurrent_kda,
    H: int,
    D: int,
    B: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    label = f"fused_recurrent_kda B={B} H={H} D={D} {dtype}"
    print(f"\n--- {label} ---")
    torch.manual_seed(7)

    q = torch.randn(B, 1, H, D, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(
        torch.randn(B, 1, H, D, dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.rand(B, 1, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)

    o, ht = fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=h0.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    _check(f"{label} / o finite", not torch.isnan(o).any().item())
    _check(f"{label} / ht finite", not torch.isnan(ht).any().item())
    _check(f"{label} / o shape", o.shape == (B, 1, H, D))


# ---------------------------------------------------------------------------
# Optional micro-benchmark
# ---------------------------------------------------------------------------


class BenchResult(NamedTuple):
    mode: str
    batch: int
    seq: int
    H: int
    D: int
    dtype: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    tok_per_s: float
    peak_mb: float | None


def _timed(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    lats: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1e3)
    return lats


def bench_prefill(
    chunk_kda,
    batch: int, seq: int, H: int, D: int,
    dtype: torch.dtype, device: torch.device,
    warmup: int = 3, iters: int = 10,
) -> BenchResult:
    q = torch.randn(batch, seq, H, D, dtype=dtype, device=device)
    k, v = torch.randn_like(q), torch.randn_like(q)
    g = F.logsigmoid(
        torch.randn(batch, seq, H, D, dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.rand(batch, seq, H, dtype=dtype, device=device).sigmoid()
    cu_seqlens = torch.tensor(
        [i * seq for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    h0 = torch.zeros(batch, H, D, D, dtype=torch.float32, device=device)
    h0t = h0.transpose(-1, -2).contiguous()

    def fn():
        chunk_kda(
            q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
            initial_state=h0t.clone(), output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lats = _timed(fn, warmup, iters, device)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6
        if device.type == "cuda" else None
    )

    tokens = batch * seq
    return BenchResult(
        mode="prefill", batch=batch, seq=seq, H=H, D=D,
        dtype=str(dtype).replace("torch.", ""),
        mean_ms=mean(lats), median_ms=median(lats),
        min_ms=min(lats), max_ms=max(lats),
        tok_per_s=tokens / (mean(lats) * 1e-3),
        peak_mb=peak_mb,
    )


def bench_decode(
    fused_recurrent_kda,
    batch: int, H: int, D: int,
    dtype: torch.dtype, device: torch.device,
    warmup: int = 3, iters: int = 10,
) -> BenchResult:
    q = torch.randn(batch, 1, H, D, dtype=dtype, device=device)
    k, v = torch.randn_like(q), torch.randn_like(q)
    g = F.logsigmoid(
        torch.randn(batch, 1, H, D, dtype=torch.float32, device=device)
    ).to(dtype)
    beta = torch.rand(batch, 1, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.zeros(batch, H, D, D, dtype=torch.float32, device=device)

    def fn():
        fused_recurrent_kda(
            q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
            initial_state=h0.clone(), inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lats = _timed(fn, warmup, iters, device)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6
        if device.type == "cuda" else None
    )

    return BenchResult(
        mode="decode", batch=batch, seq=1, H=H, D=D,
        dtype=str(dtype).replace("torch.", ""),
        mean_ms=mean(lats), median_ms=median(lats),
        min_ms=min(lats), max_ms=max(lats),
        tok_per_s=batch / (mean(lats) * 1e-3),
        peak_mb=peak_mb,
    )


def _print_bench(r: BenchResult) -> None:
    mem = f"{r.peak_mb:.1f} MB" if r.peak_mb is not None else "N/A"
    print(
        f"  [{r.mode:>7}] bs={r.batch:>3}  seq={r.seq:>6}  H={r.H}  D={r.D}"
        f"  {r.dtype:<10}"
        f"  mean={r.mean_ms:>7.3f}ms  median={r.median_ms:>7.3f}ms"
        f"  min={r.min_ms:>7.3f}ms  max={r.max_ms:>7.3f}ms"
        f"  {r.tok_per_s:>12.0f} tok/s  peak={mem}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-heads", type=int, default=32, help="Number of heads (default: 32)"
    )
    p.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension (default: 128)"
    )
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                   help="dtype for q/k/v (default: float16)")
    p.add_argument("--bench", action="store_true",
                   help="Run micro-benchmarks after correctness tests")
    p.add_argument("--bench-batch-sizes", type=int, nargs="+", default=[1, 4],
                   help="Batch sizes for benchmarks (default: 1 4)")
    p.add_argument(
        "--bench-seq-lens", type=int, nargs="+", default=[512, 2048],
        help="Sequence lengths for prefill benchmark (default: 512 2048)",
    )
    p.add_argument(
        "--warmup", type=int, default=3, help="Warmup iters (default: 3)"
    )
    p.add_argument(
        "--iters", type=int, default=10, help="Timed iters (default: 10)"
    )
    p.add_argument(
        "--device", type=str, default="cuda", help="torch device (default: cuda)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if device.type == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but no GPU found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("  KDA standalone test")
    print(f"  device={device}  dtype={args.dtype}"
          f"  H={args.num_heads}  D={args.head_dim}")
    print("=" * 72)

    chunk_kda, chunk_kda_with_fused_gate, fused_kda_gate, fused_recurrent_kda, l2norm_fwd = (  # noqa: E501
        _import_kda()
    )

    H, D = args.num_heads, args.head_dim

    # --- correctness: chunk_kda ---
    print("\n[1] chunk_kda correctness")
    correctness_cases: list[tuple[int, int, list[int], torch.dtype]] = [
        (H, D, [0, 64], dtype),
        (H, D, [0, 256, 512], dtype),
        (H, D, [0, 15], dtype),
        (H, D, [0, 15, 100, 300], dtype),
    ]
    # Add a longer sequence if head-dim allows
    if D <= 128:
        correctness_cases.append((H, D, [0, 1024], dtype))
    for case in correctness_cases:
        test_chunk_kda(chunk_kda, l2norm_fwd, *case, device=device)

    # --- correctness: fused gate ---
    print("\n[2] chunk_kda_with_fused_gate correctness")
    for cu in [[0, 64], [0, 15, 100, 300]]:
        test_chunk_kda_fused_gate(
            chunk_kda, chunk_kda_with_fused_gate, fused_kda_gate,
            H=8, D=min(D, 64), cu_seqlens=cu, dtype=dtype, device=device,
        )

    # --- correctness: fused_recurrent_kda (decode) ---
    print("\n[3] fused_recurrent_kda (decode) correctness")
    for B in [1, 4]:
        test_fused_recurrent_kda(
            fused_recurrent_kda, H=H, D=D, B=B, dtype=dtype, device=device
        )

    # --- summary ---
    print("\n" + "=" * 72)
    if _failed:
        print(f"  FAILED: {len(_failed)} test(s)")
        for name in _failed:
            print(f"    • {name}")
    else:
        print("  All correctness tests PASSED.")
    print("=" * 72)

    # --- optional benchmarks ---
    if args.bench:
        print("\n[4] Micro-benchmarks")
        print("-" * 72)
        for bs in args.bench_batch_sizes:
            for sl in args.bench_seq_lens:
                r = bench_prefill(
                    chunk_kda, batch=bs, seq=sl, H=H, D=D,
                    dtype=dtype, device=device,
                    warmup=args.warmup, iters=args.iters,
                )
                _print_bench(r)
            r = bench_decode(
                fused_recurrent_kda, batch=bs, H=H, D=D,
                dtype=dtype, device=device,
                warmup=args.warmup, iters=args.iters,
            )
            _print_bench(r)
        print("-" * 72)

    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
