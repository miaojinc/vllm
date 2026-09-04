# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run any Kimi-K3 XPU decoder layer with real checkpoint weights.

Layer indices are zero-based. The default ``--layer-index 3`` therefore runs
the fourth transformer layer. Activations can be synthetic or loaded from a
``torch.save`` file containing ``hidden_states`` and, when attn-res is enabled,
``prefix_sum`` and ``residual`` tensors. Set ``--benchmark-iters`` to measure
steady-state, host-observed decoder-layer forward latency. Set
``--context-length`` to populate cache before one-token decode. Set
``--profile-output`` to export a trace for Perfetto.

Block 0 of the block table is deliberately left unused. vLLM reserves it as
``NULL_BLOCK_ID``, and the mamba conv / KDA state kernels silently skip any
request whose state slot is 0 -- which previously made KDA layers report empty
``conv_state`` / ``recurrent_state`` and produce non-prefix-invariant output.
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import torch

from kimi_k3_xpu_moe_weight_probe import (
    ProbeError,
    default_dtype,
    load_checkpoint_config,
    load_moe_weights,
    load_quant_config,
    load_text_config,
    load_weight_map,
    read_tensor,
)
from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.kimi_k3.xpu.linear import KimiDecoderLayer
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.workspace import init_workspace_manager

try:
    from vllm.models.kimi_k3.xpu.kda import KimiK3DeltaAttention
except ImportError:
    # Branches without the XPU KDA backend only build MLA layers; keep the
    # isinstance checks below working by matching nothing.
    class KimiK3DeltaAttention:  # type: ignore[no-redef]
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Populate this many historical tokens before the timed forward.",
    )
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        help=(
            "Treat the timed forward as a prefill chunk that follows "
            "--context-length already-computed tokens, instead of a decode "
            "step. Requires --num-tokens > 1."
        ),
    )
    parser.add_argument(
        "--context-chunk-size",
        type=int,
        default=0,
        help=(
            "Populate the context in chunks of this many tokens instead of a "
            "single forward; zero keeps the single-shot behaviour."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help=(
            "KV cache block size. Serving configures 64 for Kimi-K3 dense MLA, "
            "and the DeepKlox MLA decode backend rejects anything not a "
            "multiple of 32."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup forwards before benchmarking.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=0,
        help="Timed forwards; zero disables benchmarking.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Write a Perfetto-compatible PyTorch profiler trace.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help=(
            "Record allocator events in the trace; expensive and inflates the "
            "trace badly at long sequence lengths."
        ),
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        help="Load only experts [0, N) for a smaller development run.",
    )
    parser.add_argument(
        "--input-state",
        type=Path,
        help=(
            "Optional torch file with hidden_states, prefix_sum, and residual; "
            "cached decode requires prefill and decode sub-dictionaries."
        ),
    )
    parser.add_argument("--save-output", type=Path)
    parser.add_argument(
        "--save-mla-decode-output",
        type=Path,
        help="Save the latent MLA decode output and LSE for differential tests.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/kimi_k3_xpu_layer_probe.json"),
    )
    return parser.parse_args()


def initialize_single_rank() -> None:
    file_descriptor, init_file = tempfile.mkstemp(prefix="kimi_layer_probe_")
    os.close(file_descriptor)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1, 1)


def make_model_config(
    source_config: Any,
    checkpoint_dir: Path,
    max_model_len: int,
) -> ModelConfig:
    config_dir = Path(tempfile.mkdtemp(prefix="kimi_linear_config_"))
    config_dict = source_config.to_dict()
    config_dict.update(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
    )
    (config_dir / "config.json").write_text(
        json.dumps(config_dict),
        encoding="utf-8",
    )
    return ModelConfig(
        model=str(config_dir),
        model_weights=str(checkpoint_dir),
        dtype=torch.bfloat16,
        max_model_len=max_model_len,
        enforce_eager=True,
    )


def make_common_metadata(
    query_length: int,
    context_length: int,
    block_size: int,
    device: torch.device,
    is_prefilling: bool,
) -> CommonAttentionMetadata:
    """手工构造"单请求、已算 context_length 个 token、本次算 query_length 个"的元数据。

    正常这些由 scheduler / GPUModelRunner 生成，探针里只能自己搭。三种形态：纯
    prefill（ctx=0）、chunked prefill（ctx>0 且 is_prefilling）、解码。
    """
    sequence_length = context_length + query_length
    query_start_loc = torch.tensor(
        [0, query_length], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([sequence_length], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    num_blocks = cdiv(sequence_length, block_size)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.tensor(
            [context_length], dtype=torch.int32
        ),
        num_reqs=1,
        num_actual_tokens=query_length,
        max_query_len=query_length,
        max_seq_len=sequence_length,
        # Block 0 is vLLM's reserved NULL_BLOCK_ID: mamba/conv kernels silently
        # skip any request whose state slot is 0, so real blocks start at 1.
        block_table_tensor=torch.arange(
            1, num_blocks + 1, dtype=torch.int32, device=device
        ).view(1, num_blocks),
        slot_mapping=torch.arange(
            context_length + block_size,
            sequence_length + block_size,
            dtype=torch.int64,
            device=device,
        ),
        causal=True,
        # KDA 的 metadata builder 靠它区分真解码和 prefill 分块，chunked prefill
        # 下 context_length>0 但仍在 prefill 中，所以不能由 context_length 推。
        is_prefilling=torch.tensor([is_prefilling], dtype=torch.bool),
    )


class ForwardPlan(NamedTuple):
    """一次前向所需的元数据构建材料。"""

    builder: Any
    layer_name: str
    common: CommonAttentionMetadata
    offset: int
    length: int

    def build(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """现场构建元数据。

        MLA 的 prefill backend 在 build() 里把 metadata 存到 backend 自身上，后一次
        build 会覆盖前一次，所以必须紧贴着对应的前向建，不能提前批量建好。
        """
        return (
            {
                self.layer_name: self.builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=self.common,
                )
            },
            {self.layer_name: self.common.slot_mapping},
        )


def build_context_steps(
    builder: Any,
    layer_name: str,
    context_length: int,
    chunk_size: int,
    block_size: int,
    device: torch.device,
) -> list[ForwardPlan]:
    """把 context 切成若干 prefill 步，chunk_size <= 0 时一步铺完。

    KDA 的递归状态是逐块演进的，长 context 下一次铺完会把激活显存拉到峰值。
    """
    if context_length <= 0:
        return []
    step_size = chunk_size if chunk_size > 0 else context_length
    steps: list[ForwardPlan] = []
    for offset in range(0, context_length, step_size):
        length = min(step_size, context_length - offset)
        steps.append(
            ForwardPlan(
                builder=builder,
                layer_name=layer_name,
                common=make_common_metadata(
                    length, offset, block_size, device, True
                ),
                offset=offset,
                length=length,
            )
        )
    return steps


def bind_mla_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    query_length: int,
    context_length: int,
    device: torch.device,
    is_prefilling: bool,
    context_chunk_size: int,
) -> tuple[ForwardPlan, list[ForwardPlan]]:
    """为 MLA 层分配 KV cache 并构建 attention 元数据。

    返回（计时前向的计划, 铺 cache 的 prefill 步列表）。
    """
    mla = layer.self_attn.mla_attn.mla_attn
    layer_name = mla.layer_name
    backend = mla.get_attn_backend()
    cache_spec = mla.get_kv_cache_spec(vllm_config)
    builder = backend.get_builder_cls()(
        kv_cache_spec=cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )
    timed_common = make_common_metadata(
        query_length,
        context_length,
        vllm_config.cache_config.block_size,
        device,
        is_prefilling,
    )
    sequence_length = context_length + query_length
    # +1 for the reserved block 0 that the block table skips
    num_cache_blocks = cdiv(sequence_length, cache_spec.block_size) + 1
    cache_shape = backend.get_kv_cache_shape(
        num_cache_blocks,
        cache_spec.block_size,
        cache_spec.num_kv_heads,
        cache_spec.head_size,
    )
    mla.kv_cache = torch.zeros(cache_shape, dtype=cache_spec.dtype, device=device)
    return (
        ForwardPlan(
            builder=builder,
            layer_name=layer_name,
            common=timed_common,
            offset=context_length,
            length=query_length,
        ),
        build_context_steps(
            builder,
            layer_name,
            context_length,
            context_chunk_size,
            vllm_config.cache_config.block_size,
            device,
        ),
    )


def bind_kda_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    query_length: int,
    context_length: int,
    device: torch.device,
    is_prefilling: bool,
    context_chunk_size: int,
) -> tuple[ForwardPlan, list[ForwardPlan]]:
    """为 KDA（线性注意力）层分配状态缓存并构建元数据。

    KDA 用的是 Mamba 风格的状态缓存（conv 状态 + 递归状态），走 MambaSpec 按字节数
    分配 raw buffer，再由 bind_kv_cache 切成各个状态视图。
    """
    kda = layer.self_attn
    if not isinstance(kda, KimiK3DeltaAttention):
        raise ProbeError("Selected layer does not use XPU KDA")
    layer_name = kda.prefix
    backend = kda.get_attn_backend()
    cache_spec = kda.get_kv_cache_spec(vllm_config)
    if not isinstance(cache_spec, MambaSpec):
        raise ProbeError("KDA layer did not produce a Mamba cache spec")
    builder = backend.get_builder_cls()(
        kv_cache_spec=cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )
    timed_common = make_common_metadata(
        query_length,
        context_length,
        cache_spec.block_size,
        device,
        is_prefilling,
    )
    sequence_length = context_length + query_length
    # +1 for the reserved block 0 that the block table skips
    num_cache_blocks = cache_spec.max_num_blocks_per_req(
        vllm_config,
        sequence_length,
    ) + 1
    raw_cache = torch.zeros(
        num_cache_blocks,
        1,
        1,
        cache_spec.page_size_bytes,
        dtype=torch.uint8,
        device=device,
    )
    kda.bind_kv_cache(raw_cache)
    return (
        ForwardPlan(
            builder=builder,
            layer_name=layer_name,
            common=timed_common,
            offset=context_length,
            length=query_length,
        ),
        build_context_steps(
            builder,
            layer_name,
            context_length,
            context_chunk_size,
            cache_spec.block_size,
            device,
        ),
    )


def bind_attention_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    query_length: int,
    context_length: int,
    device: torch.device,
    is_prefilling: bool,
    context_chunk_size: int,
) -> tuple[ForwardPlan, list[ForwardPlan]]:
    """根据层的注意力类型派发到 KDA 或 MLA 的绑定逻辑。"""
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        return bind_kda_cache_and_metadata(
            layer,
            vllm_config,
            query_length,
            context_length,
            device,
            is_prefilling,
            context_chunk_size,
        )
    return bind_mla_cache_and_metadata(
        layer,
        vllm_config,
        query_length,
        context_length,
        device,
        is_prefilling,
        context_chunk_size,
    )


def load_direct_parameter(
    parameters: dict[str, torch.nn.Parameter],
    target_name: str,
    tensor: torch.Tensor,
    shard_id: int | None = None,
) -> None:
    parameter = parameters[target_name]
    loader = getattr(parameter, "weight_loader", default_weight_loader)
    if shard_id is None:
        loader(parameter, tensor)
    else:
        loader(parameter, tensor, shard_id)


def load_layer_weights(
    layer: KimiDecoderLayer,
    weight_map: dict[str, Path],
    checkpoint_prefix: str,
    source_config: Any,
    expert_ids: list[int],
) -> tuple[set[str], list[dict[str, Any]]]:
    parameters = dict(layer.named_parameters())
    loaded: set[str] = set()
    stacked_mapping = (
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.q_proj.weight", 0),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.k_proj.weight", 1),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.v_proj.weight", 2),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.g_proj.weight", 3),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.f_a_proj.weight", 4),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.b_proj.weight", 5),
        ("self_attn.conv1d.weight", "self_attn.q_conv1d.weight", 0),
        ("self_attn.conv1d.weight", "self_attn.k_conv1d.weight", 1),
        ("self_attn.conv1d.weight", "self_attn.v_conv1d.weight", 2),
        ("self_attn.fused_qkv_a_proj.weight", "self_attn.q_a_proj.weight", 0),
        (
            "self_attn.fused_qkv_a_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            1,
        ),
        ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
        ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
    )
    moe_prefix = f"{checkpoint_prefix}.block_sparse_moe"
    for source_name in weight_map:
        if not source_name.startswith(f"{checkpoint_prefix}."):
            continue
        if source_name.startswith(f"{moe_prefix}.experts."):
            continue
        relative_name = source_name.removeprefix(f"{checkpoint_prefix}.")
        if relative_name.startswith("block_sparse_moe."):
            continue
        for target_name, source_suffix, shard_id in stacked_mapping:
            if relative_name == source_suffix and target_name in parameters:
                load_direct_parameter(
                    parameters,
                    target_name,
                    read_tensor(weight_map, source_name),
                    shard_id,
                )
                loaded.add(target_name)
                break
        else:
            if relative_name not in parameters:
                continue
            load_direct_parameter(
                parameters,
                relative_name,
                read_tensor(weight_map, source_name),
            )
            loaded.add(relative_name)

    moe_records: list[dict[str, Any]] = []
    if layer.is_moe_layer:
        moe_records = load_moe_weights(
            layer.block_sparse_moe,
            weight_map,
            moe_prefix,
            source_config,
            expert_ids,
        )
        loaded.update(
            f"block_sparse_moe.{record['target']}" for record in moe_records
        )
    return loaded, moe_records


def make_input_state(
    layer: KimiDecoderLayer,
    device: torch.device,
    num_tokens: int,
    seed: int,
    state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if state is not None:
        hidden_states = state["hidden_states"]
        expected_shape = (num_tokens, layer.hidden_size)
        if tuple(hidden_states.shape) != expected_shape:
            raise ProbeError(
                f"Input hidden_states shape {tuple(hidden_states.shape)} does not "
                f"match {expected_shape}"
            )
        prefix_sum = state.get("prefix_sum")
        residual = state.get("residual")
        if layer.use_attn_res and (prefix_sum is None or residual is None):
            raise ProbeError(
                "Attn-res layer input requires prefix_sum and residual tensors"
            )
        return (
            hidden_states,
            prefix_sum,
            residual,
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (num_tokens, layer.hidden_size)
    hidden_states = torch.randn(
        shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    prefix_sum = None
    residual = None
    if layer.use_attn_res:
        prefix_sum = torch.randn(
            shape, dtype=torch.bfloat16, device=device, generator=generator
        )
        num_blocks = layer.prev_valid_blocks + int(layer.is_block_write_layer)
        residual = torch.randn(
            num_tokens,
            num_blocks,
            layer.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    return hidden_states, prefix_sum, residual


def capture_attention_cache(layer: KimiDecoderLayer) -> tuple[torch.Tensor, ...]:
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        return tuple(state.clone() for state in layer.self_attn.kv_cache)
    mla = layer.self_attn.mla_attn.mla_attn
    return (mla.kv_cache.clone(),)


def restore_attention_cache(
    layer: KimiDecoderLayer,
    cache_state: tuple[torch.Tensor, ...],
) -> None:
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        for state, initial_state in zip(layer.self_attn.kv_cache, cache_state):
            state.copy_(initial_state)
        return
    mla = layer.self_attn.mla_attn.mla_attn
    mla.kv_cache.copy_(cache_state[0])


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def benchmark_layer_forward(
    layer: KimiDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    prefix_sum: torch.Tensor | None,
    metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    cache_state: tuple[torch.Tensor, ...],
    context_length: int,
    is_prefilling: bool,
    warmup_iters: int,
    benchmark_iters: int,
) -> dict[str, Any]:
    def make_iteration_inputs() -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        restore_attention_cache(layer, cache_state)
        iteration_hidden_states = hidden_states.clone()
        iteration_residual = None if residual is None else residual.clone()
        iteration_prefix_sum = None if prefix_sum is None else prefix_sum.clone()
        return iteration_hidden_states, iteration_residual, iteration_prefix_sum

    with torch.inference_mode():
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=hidden_states.size(0),
            slot_mapping=slot_mapping,
        ):
            for _ in range(warmup_iters):
                iteration_inputs = make_iteration_inputs()
                layer(positions, *iteration_inputs)
            torch.xpu.synchronize()

            latencies_ms: list[float] = []
            for _ in range(benchmark_iters):
                iteration_inputs = make_iteration_inputs()
                torch.xpu.synchronize()
                start_ns = time.perf_counter_ns()
                layer(positions, *iteration_inputs)
                torch.xpu.synchronize()
                latencies_ms.append(
                    (time.perf_counter_ns() - start_ns) / 1_000_000
                )

    median_ms = statistics.median(latencies_ms)
    return {
        "warmup_iters": warmup_iters,
        "benchmark_iters": benchmark_iters,
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "latency_p50_ms": median_ms,
        "latency_median_ms": median_ms,
        "latency_p90_ms": percentile(latencies_ms, 0.90),
        "latency_p99_ms": percentile(latencies_ms, 0.99),
        "tokens_per_second_median": hidden_states.size(0) * 1000 / median_ms,
        "attention_mode": (
            "chunked_prefill"
            if context_length > 0 and is_prefilling
            else "cached_decode"
            if context_length > 0
            else "cold_decode"
            if hidden_states.size(0) == 1
            else "prefill"
        ),
        "query_length": hidden_states.size(0),
        "context_length": context_length,
        "sequence_length": context_length + hidden_states.size(0),
        "cache_snapshot": (
            "post_context_population" if context_length > 0 else "initial"
        ),
        "context_population_timed": False,
        "cache_reset_between_iters": True,
        "input_reset_between_iters": True,
        "timing_method": "synchronized_host_wall_clock",
        "timing_scope": "KimiDecoderLayer.forward and device completion",
    }


MODULE_ANNOTATION_PREFIX = "module::"


@contextlib.contextmanager
def annotate_submodules(layer: KimiDecoderLayer):
    """给每个子模块的 forward 套一层 record_function，让 trace 能按模块归因。

    kernel 名区分不出 attention 投影和 MLP 的 gemm（都是同一批 GEMM kernel），
    只有模块边界能把耗时切开。
    """
    handles = []
    stack: list[Any] = []

    def make_enter(name: str):
        def enter(module: torch.nn.Module, args: Any) -> None:
            annotation = torch.profiler.record_function(
                f"{MODULE_ANNOTATION_PREFIX}{name}"
            )
            annotation.__enter__()
            stack.append(annotation)

        return enter

    def leave(module: torch.nn.Module, args: Any, output: Any) -> None:
        stack.pop().__exit__(None, None, None)

    for name, module in layer.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_pre_hook(make_enter(name)))
        handles.append(module.register_forward_hook(leave))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        while stack:
            stack.pop().__exit__(None, None, None)


def profile_layer_forward(
    layer: KimiDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    prefix_sum: torch.Tensor | None,
    metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    cache_state: tuple[torch.Tensor, ...],
    output_path: Path,
    profile_memory: bool,
) -> None:
    """采一次前向的 CPU/XPU 算子轨迹，导出 chrome trace。"""
    restore_attention_cache(layer, cache_state)
    iteration_hidden_states = hidden_states.clone()
    iteration_residual = None if residual is None else residual.clone()
    iteration_prefix_sum = None if prefix_sum is None else prefix_sum.clone()
    torch.xpu.synchronize()

    with torch.inference_mode():
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=hidden_states.size(0),
            slot_mapping=slot_mapping,
        ):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                record_shapes=True,
                profile_memory=profile_memory,
                with_stack=False,
            ) as profiler:
                with torch.profiler.record_function(
                    "kimi_decoder_layer_forward"
                ), annotate_submodules(layer):
                    layer(
                        positions,
                        iteration_hidden_states,
                        iteration_residual,
                        iteration_prefix_sum,
                    )
                torch.xpu.synchronize()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_path))


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "layer_index": args.layer_index,
        "block_size": args.block_size,
    }
    try:
        if args.layer_index < 0:
            raise ProbeError("--layer-index must be non-negative")
        if args.num_tokens < 1:
            raise ProbeError("--num-tokens must be positive")
        if args.context_length < 0:
            raise ProbeError("--context-length must be non-negative")
        if args.block_size < 1:
            raise ProbeError("--block-size must be positive")
        if args.chunked_prefill and args.num_tokens < 2:
            raise ProbeError("--chunked-prefill requires --num-tokens > 1")
        if args.chunked_prefill and args.context_length == 0:
            raise ProbeError("--chunked-prefill requires --context-length > 0")
        if (
            not args.chunked_prefill
            and args.context_length > 0
            and args.num_tokens != 1
        ):
            raise ProbeError(
                "Multi-token forwards with context require --chunked-prefill"
            )
        if args.context_chunk_size < 0:
            raise ProbeError("--context-chunk-size must be non-negative")
        if args.warmup_iters < 0 or args.benchmark_iters < 0:
            raise ProbeError("Benchmark iteration counts must be non-negative")
        raw_config = load_checkpoint_config(args.checkpoint_dir)
        source_config = load_text_config(raw_config)
        source_num_experts = source_config.num_experts
        if source_num_experts is None:
            if args.num_experts is not None:
                raise ProbeError("--num-experts requires a MoE checkpoint")
            num_experts = 0
            expert_ids: list[int] = []
        else:
            num_experts = args.num_experts or source_num_experts
            if not 16 <= num_experts <= source_num_experts:
                raise ProbeError(
                    "--num-experts must be between TopK 16 and source count"
                )
            expert_ids = list(range(num_experts))
        config_dict = source_config.to_dict()
        if source_num_experts is not None:
            config_dict.update(
                num_experts=num_experts,
                num_experts_per_token=min(
                    source_config.num_experts_per_token or num_experts,
                    num_experts,
                ),
                num_expert_group=(
                    1
                    if num_experts < source_num_experts
                    else source_config.num_expert_group
                ),
                topk_group=(
                    1
                    if num_experts < source_num_experts
                    else source_config.topk_group
                ),
            )
        config = type(source_config)(**config_dict)
        weight_map = load_weight_map(args.checkpoint_dir)
        checkpoint_prefix = f"language_model.model.layers.{args.layer_index}"
        if not any(name.startswith(f"{checkpoint_prefix}.") for name in weight_map):
            raise ProbeError(f"Checkpoint has no {checkpoint_prefix} tensors")

        device = torch.device("xpu:0")
        torch.xpu.set_device(device)
        sequence_length = args.context_length + args.num_tokens
        model_config = make_model_config(
            source_config,
            args.checkpoint_dir,
            # 128 是下限，保持短序列跑法下 mamba block_size 不变
            max(sequence_length, 128),
        )
        cache_config = CacheConfig(
            block_size=args.block_size,
            cache_dtype="auto",
            enable_prefix_caching=False,
            mamba_block_size=model_config.max_model_len,
            mamba_cache_mode="none",
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            quant_config=load_quant_config(raw_config),
        )
        with set_current_vllm_config(vllm_config):
            initialize_single_rank()
            # WorkerBase does this at startup; without it every IR op keeps an
            # empty priority list and silently dispatches to the decomposed
            # native path instead of the fused vllm_c kernels.
            vllm_config.kernel_config.ir_op_priority.set_default()
            init_workspace_manager(device)
            with default_dtype(torch.bfloat16):
                layer = KimiDecoderLayer(
                    config,
                    vllm_config,
                    prefix=f"model.layers.{args.layer_index}",
                ).to(device)
            layer.eval()
            loaded, moe_records = load_layer_weights(
                layer,
                weight_map,
                checkpoint_prefix,
                source_config,
                expert_ids,
            )
            missing = sorted(set(dict(layer.named_parameters())) - loaded)
            if missing:
                raise ProbeError(f"Unloaded layer parameters: {missing}")
            if layer.is_moe_layer:
                routed_experts = layer.block_sparse_moe.experts.routed_experts
                routed_experts.quant_method.process_weights_after_loading(
                    routed_experts
                )
                moe_runner = layer.block_sparse_moe.experts
                quant_method = moe_runner.routed_experts.quant_method
                moe_kernel = quant_method.moe_kernel
                situ_config = {
                    "source_beta": source_config.activation_situ_beta,
                    "source_linear_beta": source_config.activation_situ_linear_beta,
                    "runner_beta": moe_runner.moe_config.activation_situ_beta,
                    "runner_linear_beta": (
                        moe_runner.moe_config.activation_situ_linear_beta
                    ),
                    "kernel_beta": moe_kernel.moe_config.activation_situ_beta,
                    "kernel_linear_beta": (
                        moe_kernel.moe_config.activation_situ_linear_beta
                    ),
                }
                report["situ_config"] = situ_config
                if (
                    config.hidden_act == "situ"
                    and situ_config["kernel_beta"] is None
                ):
                    raise ProbeError("SITU beta was lost before MXFP4 kernel creation")
            is_kda = isinstance(layer.self_attn, KimiK3DeltaAttention)
            if not is_kda:
                layer.self_attn.mla_attn.mla_attn.process_weights_after_loading(
                    torch.bfloat16
                )
            mla_decode_state: dict[str, torch.Tensor | None] = {}
            if args.save_mla_decode_output is not None:
                if is_kda:
                    raise ProbeError(
                        "--save-mla-decode-output requires an MLA layer"
                    )
                mla = layer.self_attn.mla_attn.mla_attn
                original_forward_mqa = mla.impl.forward_mqa

                def capture_forward_mqa(*call_args, **call_kwargs):
                    latent_output, lse = original_forward_mqa(
                        *call_args, **call_kwargs
                    )
                    mla_decode_state["latent_output"] = latent_output.detach().clone()
                    mla_decode_state["lse"] = (
                        None if lse is None else lse.detach().clone()
                    )
                    return latent_output, lse

                mla.impl.forward_mqa = capture_forward_mqa
            (
                timed_plan,
                context_steps,
            ) = bind_attention_cache_and_metadata(
                layer,
                vllm_config,
                args.num_tokens,
                args.context_length,
                device,
                args.chunked_prefill,
                args.context_chunk_size,
            )
            loaded_input_state = None
            if args.input_state is not None:
                loaded_input_state = torch.load(
                    args.input_state,
                    map_location=device,
                    weights_only=True,
                )
            if args.context_length > 0:
                if loaded_input_state is not None:
                    if not all(
                        key in loaded_input_state for key in ("prefill", "decode")
                    ):
                        raise ProbeError(
                            "Cached decode input-state requires prefill and "
                            "decode sub-dictionaries"
                        )
                    context_input_state = loaded_input_state["prefill"]
                    decode_input_state = loaded_input_state["decode"]
                else:
                    context_input_state = None
                    decode_input_state = None
                context_hidden_states, context_prefix_sum, context_residual = (
                    make_input_state(
                        layer,
                        device,
                        args.context_length,
                        seed=17,
                        state=context_input_state,
                    )
                )
                assert context_steps
                context_positions = torch.arange(
                    args.context_length,
                    dtype=torch.int64,
                    device=device,
                )
                # 这些前向只为把历史写进 cache，不计时也不校验数值
                for step in context_steps:
                    chunk = slice(step.offset, step.offset + step.length)
                    step_metadata, step_slot_mapping = step.build()
                    with torch.inference_mode(), set_forward_context(
                        step_metadata,
                        vllm_config,
                        num_tokens=step.length,
                        slot_mapping=step_slot_mapping,
                    ):
                        context_output, _, _ = layer(
                            context_positions[chunk],
                            context_hidden_states[chunk].clone(),
                            (
                                None
                                if context_residual is None
                                else context_residual[chunk].clone()
                            ),
                            (
                                None
                                if context_prefix_sum is None
                                else context_prefix_sum[chunk].clone()
                            ),
                        )
                    torch.xpu.synchronize()
                    if not bool(torch.isfinite(context_output).all()):
                        raise ProbeError(
                            "Context population output contains non-finite values"
                        )
                benchmark_cache_state = capture_attention_cache(layer)
                # The history now lives in the cache; at 32K these activations
                # are ~470 MiB each and the timed forward never reads them.
                del context_output, context_positions
                del context_hidden_states, context_prefix_sum, context_residual
                torch.xpu.empty_cache()
            else:
                decode_input_state = loaded_input_state
                benchmark_cache_state = capture_attention_cache(layer)
            hidden_states, prefix_sum, residual = make_input_state(
                layer,
                device,
                args.num_tokens,
                seed=18 if args.context_length > 0 else 17,
                state=decode_input_state,
            )
            positions = torch.arange(
                args.context_length,
                sequence_length,
                dtype=torch.int64,
                device=device,
            )
            restore_attention_cache(layer, benchmark_cache_state)
            # 必须在铺完 context 之后再建，见 ForwardPlan.build 的说明
            metadata, slot_mapping = timed_plan.build()
            with torch.inference_mode():
                with set_forward_context(
                    metadata,
                    vllm_config,
                    num_tokens=args.num_tokens,
                    slot_mapping=slot_mapping,
                ):
                    output, output_prefix_sum, output_residual = layer(
                        positions,
                        hidden_states.clone(),
                        None if residual is None else residual.clone(),
                        None if prefix_sum is None else prefix_sum.clone(),
                    )
            torch.xpu.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise ProbeError("Layer output contains non-finite values")
            output_state = {
                "hidden_states": output.cpu(),
                "prefix_sum": (
                    None if output_prefix_sum is None else output_prefix_sum.cpu()
                ),
                "residual": (
                    None if output_residual is None else output_residual.cpu()
                ),
            }
            if is_kda:
                conv_state, recurrent_state = layer.self_attn.kv_cache
                output_state.update(
                    conv_state=conv_state.cpu(),
                    recurrent_state=recurrent_state.cpu(),
                )
            benchmark_result = None
            if args.benchmark_iters > 0:
                benchmark_result = benchmark_layer_forward(
                    layer=layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    prefix_sum=prefix_sum,
                    metadata=metadata,
                    slot_mapping=slot_mapping,
                    vllm_config=vllm_config,
                    cache_state=benchmark_cache_state,
                    context_length=args.context_length,
                    is_prefilling=args.chunked_prefill,
                    warmup_iters=args.warmup_iters,
                    benchmark_iters=args.benchmark_iters,
                )
            if args.profile_output is not None:
                profile_layer_forward(
                    layer=layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    prefix_sum=prefix_sum,
                    metadata=metadata,
                    slot_mapping=slot_mapping,
                    vllm_config=vllm_config,
                    cache_state=benchmark_cache_state,
                    output_path=args.profile_output,
                    profile_memory=args.profile_memory,
                )
            if args.save_output is not None:
                args.save_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_state, args.save_output)
            if args.save_mla_decode_output is not None:
                if "latent_output" not in mla_decode_state:
                    raise ProbeError("MLA decode output was not captured")
                args.save_mla_decode_output.parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(
                    {
                        name: None if value is None else value.cpu()
                        for name, value in mla_decode_state.items()
                    },
                    args.save_mla_decode_output,
                )
            report.update(
                status="passed",
                ordinal_layer=args.layer_index + 1,
                checkpoint_prefix=checkpoint_prefix,
                num_experts=num_experts,
                attention_type="kda" if is_kda else "mla",
                mlp_type="moe" if layer.is_moe_layer else "dense",
                loaded_parameters=len(loaded),
                loaded_moe_tensors=len(moe_records),
                input_state="file" if args.input_state else "synthetic",
                query_length=args.num_tokens,
                context_length=args.context_length,
                sequence_length=sequence_length,
                chunked_prefill=args.chunked_prefill,
                context_population_steps=len(context_steps),
                context_population_executed=args.context_length > 0,
                mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                output_all_finite=True,
                output_max_abs=float(output.abs().max()),
            )
            if benchmark_result is not None:
                report["benchmark"] = benchmark_result
            if args.profile_output is not None:
                report["profile"] = {
                    "output": str(args.profile_output),
                    "format": "chrome_trace_json",
                    "activities": ["cpu", "xpu"],
                    "forward_iters": 1,
                    "perfetto_url": "https://ui.perfetto.dev/",
                }
            if is_kda:
                report.update(
                    conv_state_shape=list(conv_state.shape),
                    conv_state_dtype=str(conv_state.dtype),
                    recurrent_state_shape=list(recurrent_state.shape),
                    recurrent_state_dtype=str(recurrent_state.dtype),
                    conv_state_max_abs=float(conv_state.abs().max()),
                    recurrent_state_max_abs=float(recurrent_state.abs().max()),
                )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())