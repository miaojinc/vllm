# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Triton and DeepKlox Kimi-K3 dense MLA decode outputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--deepklox-root", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=3)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kimi_k3_mla_decode_comparison"),
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser.parse_args()


def run_probe(args: argparse.Namespace, backend: str) -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    probe = script_dir / "kimi_k3_xpu_layer_probe.py"
    output = args.output_dir / f"{backend}_output.pt"
    mla_output = args.output_dir / f"{backend}_mla_output.pt"
    report = args.output_dir / f"{backend}_report.json"
    command = [
        sys.executable,
        str(probe),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--layer-index",
        str(args.layer_index),
        "--num-tokens",
        "1",
        "--context-length",
        str(args.context_length),
        "--warmup-iters",
        "0",
        "--benchmark-iters",
        "0",
        "--save-output",
        str(output),
        "--save-mla-decode-output",
        str(mla_output),
        "--report",
        str(report),
    ]
    env = os.environ.copy()
    env["VLLM_XPU_MLA_DECODE_BACKEND"] = backend
    if backend == "deepklox":
        pythonpath = env.get("PYTHONPATH")
        roots = [str(args.deepklox_root)]
        if pythonpath:
            roots.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(roots)
    subprocess.run(command, env=env, check=True)
    return output, mla_output, report


def tensor_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference_float = reference.float()
    actual_float = actual.float()
    difference = (actual_float - reference_float).abs()
    denominator = reference_float.abs().clamp_min(1e-6)
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_rel_error": (difference / denominator).max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            reference_float.flatten(), actual_float.flatten(), dim=0
        ).item(),
        "allclose": torch.allclose(
            reference_float, actual_float, atol=atol, rtol=rtol
        ),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    triton_output, triton_mla_output, triton_report = run_probe(args, "triton")
    deepklox_output, deepklox_mla_output, deepklox_report = run_probe(
        args, "deepklox"
    )

    triton_state = torch.load(triton_output, map_location="cpu", weights_only=True)
    deepklox_state = torch.load(
        deepklox_output, map_location="cpu", weights_only=True
    )
    triton_state["mla_decode"] = torch.load(
        triton_mla_output, map_location="cpu", weights_only=True
    )
    deepklox_state["mla_decode"] = torch.load(
        deepklox_mla_output, map_location="cpu", weights_only=True
    )
    metrics: dict[str, Any] = {}
    for name, reference in triton_state.items():
        actual = deepklox_state[name]
        if isinstance(reference, dict):
            metrics[name] = {
                tensor_name: tensor_metrics(
                    tensor_reference,
                    actual[tensor_name],
                    args.atol,
                    args.rtol,
                )
                for tensor_name, tensor_reference in reference.items()
                if tensor_reference is not None
            }
            continue
        if reference is None:
            metrics[name] = {"both_none": actual is None}
            continue
        if not isinstance(reference, torch.Tensor) or not isinstance(
            actual, torch.Tensor
        ):
            raise TypeError(f"Unsupported output state value for {name}")
        metrics[name] = tensor_metrics(reference, actual, args.atol, args.rtol)

    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "layer_index": args.layer_index,
        "context_length": args.context_length,
        "atol": args.atol,
        "rtol": args.rtol,
        "triton_report": str(triton_report),
        "deepklox_report": str(deepklox_report),
        "metrics": metrics,
        "passed": all(
            (
                all(item["allclose"] for item in value.values())
                if name == "mla_decode"
                else value.get("allclose", value.get("both_none", False))
            )
            for name, value in metrics.items()
        ),
    }
    result_path = args.output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())