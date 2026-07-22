# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for benchmarks/kernels/benchmark_kda.py.

Tests:
- CLI argument parsing produces expected defaults and overrides.
- Input construction (_make_prefill_inputs / _make_decode_inputs) returns
  tensors with the expected shapes.
- Output finiteness: the kernel produces finite values when run on CUDA.
- CPU/no-CUDA environments are skipped for GPU-only tests.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

# We import the benchmark module directly so we can test its internals
# without going through subprocess.
_BM_MODULE_PATH = "benchmarks.kernels.benchmark_kda"


def _import_bm() -> types.ModuleType:
    """Import the benchmark module, adding the repo root to sys.path if needed."""
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module(_BM_MODULE_PATH)


bm = _import_bm()

# ---------------------------------------------------------------------------
# Argument-parsing tests (CPU-safe)
# ---------------------------------------------------------------------------


def test_defaults():
    parser = bm._build_parser()
    args = parser.parse_args([])
    assert args.batch_size == 4
    assert args.seq_len == 1024
    assert args.num_heads == 32
    assert args.head_dim == 128
    assert args.warmup == 5
    assert args.iters == 20
    assert args.mode == "both"
    assert args.dtype == "bfloat16"
    assert args.use_real_weights is False
    assert args.model_path is None


def test_overrides():
    parser = bm._build_parser()
    args = parser.parse_args([
        "--batch-size", "8",
        "--seq-len", "512",
        "--num-heads", "16",
        "--head-dim", "64",
        "--warmup", "2",
        "--iters", "5",
        "--mode", "prefill",
        "--dtype", "float16",
        "--use-real-weights",
        "--model-path", "/tmp/model",
    ])
    assert args.batch_size == 8
    assert args.seq_len == 512
    assert args.num_heads == 16
    assert args.head_dim == 64
    assert args.warmup == 2
    assert args.iters == 5
    assert args.mode == "prefill"
    assert args.dtype == "float16"
    assert args.use_real_weights is True
    assert args.model_path == "/tmp/model"


def test_sweep_args():
    parser = bm._build_parser()
    args = parser.parse_args(["--batch-sizes", "1", "4", "8",
                               "--seq-lens", "256", "1024"])
    assert args.batch_sizes == [1, 4, 8]
    assert args.seq_lens == [256, 1024]


# ---------------------------------------------------------------------------
# Input-construction tests (CPU-safe, no Triton)
# ---------------------------------------------------------------------------


def test_prefill_input_shapes():
    batch, seq, H, D = 2, 64, 4, 32
    dtype = torch.float16
    device = torch.device("cpu")
    inp = bm._make_prefill_inputs(batch, seq, H, D, dtype, device)
    assert inp["q"].shape == (batch, seq, H, D)
    assert inp["k"].shape == (batch, seq, H, D)
    assert inp["v"].shape == (batch, seq, H, D)
    assert inp["g"].shape == (batch, seq, H, D)
    assert inp["beta"].shape == (batch, seq, H)
    assert inp["initial_state"].shape == (batch, H, D, D)
    assert inp["cu_seqlens"].shape == (batch + 1,)
    assert inp["cu_seqlens"][-1].item() == batch * seq


def test_decode_input_shapes():
    batch, H, D = 3, 4, 32
    dtype = torch.bfloat16
    device = torch.device("cpu")
    inp = bm._make_decode_inputs(batch, H, D, dtype, device)
    assert inp["q"].shape == (batch, 1, H, D)
    assert inp["k"].shape == (batch, 1, H, D)
    assert inp["v"].shape == (batch, 1, H, D)
    assert inp["g"].shape == (batch, 1, H, D)
    assert inp["beta"].shape == (batch, 1, H)
    assert inp["initial_state"].shape == (batch, H, D, D)


def test_prefill_input_finiteness():
    batch, seq, H, D = 1, 16, 2, 16
    dtype = torch.float16
    device = torch.device("cpu")
    inp = bm._make_prefill_inputs(batch, seq, H, D, dtype, device)
    for name, t in inp.items():
        if isinstance(t, torch.Tensor):
            assert torch.isfinite(t).all(), f"{name} contains non-finite values"


# ---------------------------------------------------------------------------
# GPU kernel tests — skipped on CPU-only environments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bench_prefill_output_finite():
    device = torch.device("cuda")
    r = bm.bench_prefill(
        batch_size=1,
        seq_len=64,
        num_heads=4,
        head_dim=64,
        dtype=torch.float16,
        device=device,
        warmup=1,
        iters=2,
    )
    assert r.mode == "prefill"
    assert r.latency_mean_ms > 0
    assert r.tokens_per_sec > 0
    assert r.peak_mem_mb is not None and r.peak_mem_mb >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bench_decode_output_finite():
    device = torch.device("cuda")
    r = bm.bench_decode(
        batch_size=1,
        num_heads=4,
        head_dim=64,
        dtype=torch.float16,
        device=device,
        warmup=1,
        iters=2,
    )
    assert r.mode == "decode"
    assert r.latency_mean_ms > 0
    assert r.tokens_per_sec > 0
    assert r.peak_mem_mb is not None and r.peak_mem_mb >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64, 256])
def test_bench_prefill_parametric(batch_size: int, seq_len: int):
    device = torch.device("cuda")
    r = bm.bench_prefill(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=4,
        head_dim=64,
        dtype=torch.bfloat16,
        device=device,
        warmup=1,
        iters=2,
    )
    assert r.batch_size == batch_size
    assert r.seq_len == seq_len
    assert r.latency_mean_ms > 0
