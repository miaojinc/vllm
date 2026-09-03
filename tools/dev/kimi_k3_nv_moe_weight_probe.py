# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY: validate loading a sampled Kimi-K3 MoE layer on NVIDIA GPUs.

This development-only probe is intentionally isolated from production model
loading. It reads only the tensors needed for one layer and a selected set of
experts, then writes a JSON report. Delete this file before release.

Ported from the XPU probe: uses ``vllm.models.kimi_k3.nvidia`` and CUDA.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.kimi_k3.nvidia.model import KimiMoE
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.worker.workspace import init_workspace_manager

_MOE_PREFIX_RE = re.compile(r"^(.*\.layers\.(\d+)\.block_sparse_moe)\.")


class ProbeError(RuntimeError):
    """探针无法在该 checkpoint 上完成请求时抛出。

    与普通异常区分开，便于 main() 把它归类为"探针本身的前置条件不满足"，
    而不是 vllm 内部的真实错误。
    """


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--layer-index",
        type=int,
        help="MoE layer index. Defaults to the available layer with most keys.",
    )
    parser.add_argument(
        "--expert-ids",
        default="0",
        help="Comma-separated original checkpoint expert IDs to load.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Run a finite-value forward smoke test after loading the subset.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1,
        help="Token count for --run-forward.",
    )
    parser.add_argument(
        "--activation-override",
        choices=("silu",),
        help=(
            "Development-only activation override for --run-forward. "
            "It does not validate Kimi's original activation semantics."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/kimi_k3_nv_moe_weight_probe.json"),
    )
    return parser.parse_args()


def parse_expert_ids(value: str) -> list[int]:
    """把 ``--expert-ids`` 的逗号分隔字符串解析为整数列表并做合法性校验。

    这里的 ID 是 checkpoint 里的"原始专家编号"，与加载到模块后的局部编号
    (0..N-1) 不同，两者的对应关系记录在报告的 loaded_parameters 里。
    """
    try:
        expert_ids = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise ProbeError(f"Invalid --expert-ids value: {value}") from error
    if not expert_ids or len(expert_ids) != len(set(expert_ids)):
        raise ProbeError("--expert-ids must contain unique integer IDs")
    if min(expert_ids) < 0:
        raise ProbeError("--expert-ids cannot contain negative IDs")
    return expert_ids


def load_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    """读取 checkpoint 目录下的 config.json 原始内容。"""
    config_path = checkpoint_dir / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def load_text_config(raw_config: dict[str, Any]) -> KimiLinearConfig:
    """从多模态 config 中取出语言模型部分并构造 KimiLinearConfig。

    Kimi-K3 是多模态模型，语言模型的超参在 ``text_config`` 子对象里；
    纯语言模型 checkpoint 则退化为整个 config 本身。
    """
    text_config = raw_config.get("text_config", raw_config)
    if not isinstance(text_config, dict):
        raise ProbeError("config.json does not contain a text_config object")
    return KimiLinearConfig(**text_config)


def load_weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    """解析 safetensors 索引，返回 {张量名: 所在分片的绝对路径}。

    过滤掉索引中存在但分片文件缺失的条目，这样减层/裁剪过的 checkpoint
    也能正常工作 —— 缺失的张量会在后续 required_checkpoint_names 比对时
    被报告为 missing_tensors，而不是在读取时崩溃。
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ProbeError(f"Missing safetensors index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map")
    if not isinstance(weight_map, dict):
        raise ProbeError("safetensors index has no weight_map object")
    return {
        name: checkpoint_dir / shard
        for name, shard in weight_map.items()
        if (checkpoint_dir / shard).is_file()
    }


def available_moe_layer_prefixes(weight_map: Iterable[str]) -> dict[int, str]:
    """扫描权重名，找出所有含 MoE 张量的层，返回 {层号: 层前缀}。

    非 MoE 层（如 first_k_dense_replace 覆盖的 dense 层）不会出现在结果里。
    """
    prefixes: dict[int, str] = {}
    for name in weight_map:
        match = _MOE_PREFIX_RE.match(name)
        if match is not None:
            prefixes.setdefault(int(match.group(2)), match.group(1))
    return prefixes


def select_layer_prefix(
    weight_map: dict[str, Path], layer_index: int | None
) -> tuple[int, str]:
    """选定要探测的 MoE 层，未指定时取最小可用层号。"""
    prefixes = available_moe_layer_prefixes(weight_map)
    if not prefixes:
        raise ProbeError("No available checkpoint layer contains MoE expert tensors")
    if layer_index is not None:
        try:
            return layer_index, prefixes[layer_index]
        except KeyError as error:
            raise ProbeError(
                f"No available MLP checkpoint keys for layer {layer_index}"
            ) from error

    return min(prefixes), prefixes[min(prefixes)]


def required_checkpoint_names(
    prefix: str, config: KimiLinearConfig, expert_ids: list[int]
) -> list[str]:
    """根据 config 推导该层应当存在的 checkpoint 张量名清单。

    这份清单是探针的"预期"，main() 拿它跟实际 weight_map 做差集，
    差集非空就说明 checkpoint 不完整或命名与预期不符。
    """
    names = [
        f"{prefix}.gate.weight",
        f"{prefix}.gate.e_score_correction_bias",
    ]
    # latent MoE：路由专家在低维空间计算，前后各有一个降/升维投影
    if config.routed_expert_hidden_size is not None:
        names.extend(
            (
                f"{prefix}.routed_expert_down_proj.weight",
                f"{prefix}.routed_expert_up_proj.weight",
            )
        )
        if config.latent_moe_use_norm:
            names.append(f"{prefix}.routed_expert_norm.weight")
    if config.num_shared_experts:
        names.extend(
            (
                f"{prefix}.shared_experts.gate_proj.weight",
                f"{prefix}.shared_experts.up_proj.weight",
                f"{prefix}.shared_experts.down_proj.weight",
            )
        )
    for expert_id in expert_ids:
        for projection in ("w1", "w2", "w3"):
            # MXFP4 量化后每个投影拆成两个张量：4bit 打包权重 + 分组 scale
            names.extend(
                (
                    f"{prefix}.experts.{expert_id}.{projection}.weight_packed",
                    f"{prefix}.experts.{expert_id}.{projection}.weight_scale",
                )
            )
    return names


def read_tensor(weight_map: dict[str, Path], name: str) -> torch.Tensor:
    """按需从对应分片里读单个张量到 CPU。

    每次都重新 open 分片，换来不需要把整个 58GB checkpoint 载入内存。
    """
    shard = weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        return tensors.get_tensor(name)


def load_parameter(
    params: dict[str, torch.nn.Parameter],
    target_name: str,
    tensor: torch.Tensor,
    records: list[dict[str, Any]],
    source_name: str,
    loader_args: tuple[Any, ...] = (),
    record_data: dict[str, Any] | None = None,
    **loader_kwargs: Any,
) -> None:
    """把一个 checkpoint 张量写入目标参数，并记录一条映射明细。

    优先用参数自带的 ``weight_loader``（vllm 给融合/分片/量化参数挂上去的），
    它知道该写到哪个 shard、哪个专家位置；普通参数才回退到 default_weight_loader。
    形状/dtype 不匹配会在这里当场报错，这正是探针想抓的问题。
    """
    try:
        parameter = params[target_name]
    except KeyError as error:
        raise ProbeError(f"Missing target parameter: {target_name}") from error
    weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
    weight_loader(parameter, tensor, *loader_args, **loader_kwargs)
    # 记录源/目标的形状与 dtype，报告里可直接看出融合权重是怎么拼的
    record = {
        "source": source_name,
        "target": target_name,
        "source_shape": list(tensor.shape),
        "target_shape": list(parameter.shape),
        "source_dtype": str(tensor.dtype),
        "target_dtype": str(parameter.dtype),
        "loader_kwargs": loader_kwargs,
    }
    if record_data is not None:
        record.update(record_data)
    records.append(record)


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterable[None]:
    """临时切换 torch 默认 dtype，退出时恢复。

    模块构造时新建的参数会跟随默认 dtype，否则会建成 fp32。
    """
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def initialize_single_rank() -> None:
    """初始化单卡分布式环境。

    vllm 的层内部会调用 TP/EP 通信原语，即使单卡也必须先建进程组。
    用临时文件作为 rendezvous，gloo 后端已足够（不走真实集体通信）。
    """
    fd, init_file = tempfile.mkstemp(prefix="kimi_moe_probe_")
    os.close(fd)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend="gloo",
    )


def make_model_config(
    source_config: KimiLinearConfig,
    checkpoint_dir: Path,
) -> ModelConfig:
    """构造一份指向 kimi_linear 架构副本的 ModelConfig。

    把 architectures/model_type 改写成 kimi_linear 写到临时目录，权重仍指向真实
    checkpoint。这样可以绕过多模态包装直接构造单个层，而不必走完整的模型加载流程。
    """
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
        max_model_len=128,
        enforce_eager=True,
    )


def make_subset_config(
    source_config: KimiLinearConfig,
    expert_ids: list[int],
    activation_override: str | None,
) -> KimiLinearConfig:
    """把原始 config 裁剪成只含选中专家的小配置。

    896 个专家全加载太慢也太吃显存，开发时只建 N 个。同时关掉分组路由
    （use_grouped_topk=False、group 相关置 1），否则分组数会跟不上专家数。
    """
    config_dict = source_config.to_dict()
    config_dict.update(
        num_experts=len(expert_ids),
        # TopK 不能超过实际专家数
        num_experts_per_token=min(source_config.num_experts_per_token, len(expert_ids)),
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
    )
    if activation_override is not None:
        config_dict["hidden_act"] = activation_override
    return KimiLinearConfig(**config_dict)


def load_quant_config(raw_config: dict[str, Any]) -> CompressedTensorsConfig:
    """从 config.json 取出量化配置并构造 vllm 的 CompressedTensorsConfig。

    探针只支持 mxfp4-pack-quantized，遇到其他格式直接报错而不是静默降级。
    """
    try:
        quant_config = raw_config["text_config"]["quantization_config"]
    except KeyError as error:
        raise ProbeError("text_config.quantization_config is required") from error
    if quant_config.get("format") != "mxfp4-pack-quantized":
        raise ProbeError(
            "This temporary probe only supports mxfp4-pack-quantized experts"
        )
    return CompressedTensorsConfig.from_config(quant_config.copy())


def load_moe_weights(
    moe: KimiMoE,
    weight_map: dict[str, Path],
    prefix: str,
    config: KimiLinearConfig,
    expert_ids: list[int],
) -> list[dict[str, Any]]:
    """把一个 MoE 层的全部权重加载进 ``moe``，返回逐张量的映射记录。

    分四类处理：路由 gate、latent 投影、共享专家、路由专家。
    该函数也被 layer probe 当作库调用。
    """
    params = dict(moe.named_parameters())
    records: list[dict[str, Any]] = []

    # gate 的行数等于专家数，按选中的 expert_ids 抓行后再写入
    gate_weight = read_tensor(weight_map, f"{prefix}.gate.weight")[expert_ids]
    load_parameter(
        params,
        "gate.weight",
        gate_weight,
        records,
        f"{prefix}.gate.weight",
        record_data={"source_expert_ids": expert_ids},
    )
    correction_bias = read_tensor(
        weight_map, f"{prefix}.gate.e_score_correction_bias"
    )[expert_ids]
    load_parameter(
        params,
        "gate.e_score_correction_bias",
        correction_bias,
        records,
        f"{prefix}.gate.e_score_correction_bias",
        record_data={"source_expert_ids": expert_ids},
    )

    direct_names = (
        "routed_expert_down_proj.weight",
        "routed_expert_norm.weight",
        "routed_expert_up_proj.weight",
    )
    # latent MoE 的降/升维投影与 norm，名字与 checkpoint 一致，直连即可
    for target_name in direct_names:
        source_name = f"{prefix}.{target_name}"
        if source_name in weight_map:
            load_parameter(
                params,
                target_name,
                read_tensor(weight_map, source_name),
                records,
                source_name,
            )

    shared_mapping = (
        ("shared_experts.gate_up_proj.weight", "gate_proj", 0),
        ("shared_experts.gate_up_proj.weight", "up_proj", 1),
        ("shared_experts.down_proj.weight", "down_proj", None),
    )
    # 共享专家的 gate/up 在 vllm 侧融合为 gate_up_proj，靠 shard_id 区分写入位置
    for target_name, source_projection, shard_id in shared_mapping:
        source_name = f"{prefix}.shared_experts.{source_projection}.weight"
        if source_name not in weight_map:
            continue
        loader_args = () if shard_id is None else (shard_id,)
        load_parameter(
            params,
            target_name,
            read_tensor(weight_map, source_name),
            records,
            source_name,
            loader_args=loader_args,
        )

    # 向 vllm 要一份专家参数映射表：checkpoint 的 w1/w2/w3 对应 gate/down/up，
    # 目标侧 w1+w3 会被拼成 w13_*，w2 单独为 w2_*
    mapping = fused_moe_make_expert_params_mapping(
        moe,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=len(expert_ids),
        routed_experts_prefix="routed_experts",
    )
    expert_mapping = {
        (expert_id, source_projection): (target_name, source_identifier)
        for target_name, source_identifier, expert_id, source_projection in mapping
    }
    # 原始专家号 -> 局部专家号的重映射，两者都记入报告便于核对
    for local_expert_id, source_expert_id in enumerate(expert_ids):
        for projection in ("w1", "w2", "w3"):
            target_prefix, source_identifier = expert_mapping[
                (local_expert_id, projection)
            ]
            for suffix in ("weight_packed", "weight_scale"):
                source_name = (
                    f"{prefix}.experts.{source_expert_id}.{projection}.{suffix}"
                )
                relative_name = source_name.removeprefix(f"{prefix}.")
                target_name = relative_name.replace(source_identifier, target_prefix)
                load_parameter(
                    params,
                    target_name,
                    read_tensor(weight_map, source_name),
                    records,
                    source_name,
                    loader_args=(target_name,),
                    expert_id=local_expert_id,
                    shard_id=projection,
                    record_data={
                        "source_expert_id": source_expert_id,
                        "target_expert_id": local_expert_id,
                    },
                )

    return records


def main() -> int:
    """探针主流程：校验清单 -> 建层 -> 加载权重 -> 可选前向 -> 写报告。

    任何异常都不向外抛，而是写进报告的 error/traceback 字段，
    靠退出码 0/1 告知调用方，便于被自动化脚本驱动。
    """
    args = parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": args.device,
        "temporary_probe": True,
    }
    try:
        expert_ids = parse_expert_ids(args.expert_ids)
        raw_config = load_checkpoint_config(args.checkpoint_dir)
        source_config = load_text_config(raw_config)
        if max(expert_ids) >= source_config.num_experts:
            raise ProbeError(
                f"Requested expert exceeds num_experts={source_config.num_experts}"
            )
        weight_map = load_weight_map(args.checkpoint_dir)
        report["available_moe_layer_indices"] = sorted(
            available_moe_layer_prefixes(weight_map)
        )
        layer_index, prefix = select_layer_prefix(weight_map, args.layer_index)
        required_names = required_checkpoint_names(prefix, source_config, expert_ids)
        missing_names = [name for name in required_names if name not in weight_map]
        report.update(
            layer_index=layer_index,
            checkpoint_prefix=prefix,
            source_expert_ids=expert_ids,
            required_tensors=len(required_names),
            missing_tensors=missing_names,
        )
        if missing_names:
            raise ProbeError("Selected layer does not have all required MoE tensors")
        if args.run_forward and args.num_tokens < 1:
            raise ProbeError("--num-tokens must be positive")

        device = torch.device(
            f"{args.device}:0" if args.device == "cuda" else args.device
        )
        if device.type == "cuda":
            # torch.cuda.set_device 不接受不带索引的 "cuda"
            torch.cuda.set_device(device)
        subset_config = make_subset_config(
            source_config,
            expert_ids,
            args.activation_override,
        )
        quant_config = load_quant_config(raw_config)
        vllm_config = VllmConfig(
            model_config=make_model_config(source_config, args.checkpoint_dir),
            cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
            quant_config=quant_config,
        )
        with set_current_vllm_config(vllm_config):
            initialize_single_rank()
            initialize_model_parallel(1, 1)
            init_workspace_manager(device)
            with default_dtype(torch.bfloat16):
                moe = KimiMoE(
                    subset_config,
                    vllm_config,
                    quant_config=quant_config,
                    prefix=prefix,
                    layer_idx=layer_index,
                ).to(device)
            records = load_moe_weights(
                moe,
                weight_map,
                prefix,
                source_config,
                expert_ids,
            )
            # MXFP4 权重在这一步重排/预处理成 kernel 需要的布局
            moe.experts.routed_experts.quant_method.process_weights_after_loading(
                moe.experts.routed_experts
            )
            forward_report: dict[str, Any] | None = None
            if args.run_forward:
                hidden_states = torch.randn(
                    args.num_tokens,
                    subset_config.hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
                with set_forward_context(
                    {}, vllm_config, num_tokens=args.num_tokens
                ):
                    output = moe(hidden_states)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                # 只做有限性检查，不比对数值精度
                forward_report = {
                    "num_tokens": args.num_tokens,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "all_finite": bool(torch.isfinite(output).all()),
                }
                if not forward_report["all_finite"]:
                    raise ProbeError("Real-weight forward produced non-finite values")
        report.update(
            status="passed",
            loaded_tensors=len(records),
            expected_tensors=len(required_names),
            loaded_parameters=records,
            subset_num_experts=len(expert_ids),
            subset_num_experts_per_token=subset_config.num_experts_per_token,
            source_activation=source_config.hidden_act,
            effective_activation=subset_config.hidden_act,
            post_load_processing="passed",
        )
        if forward_report is not None:
            report["forward"] = forward_report
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
