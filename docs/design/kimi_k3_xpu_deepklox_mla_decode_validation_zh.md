# Kimi-K3 XPU DeepKlox Dense MLA Decode 对比验证

本文记录在 Intel XPU 上使用相同输入，对比 vLLM Triton MLA decode 与
DeepKlox dense MLA decode 结果的方法和实测结果。

## 测试环境

- vLLM 目录：`/home/mjc/vllm`
- DeepKlox 目录：`/home/mjc/applications.ai.gpu.deepklox`
- DeepKlox 分支：`wenbin/flash_mla`
- DeepKlox commit：`415aabd`
- 模型目录：`/home/hf_models/Kimi-K3-4layer-vllm-20260815`
- Python：`/opt/venv/bin/python`
- 测试设备：`ZE_AFFINITY_MASK=0`
- 测试层：layer 3，即checkpoint中的第4层MLA层
- Query长度：1
- KV cache block size：64
- 数值容差：`atol=0.02, rtol=0.02`

比较脚本为：

```text
tools/dev/compare_kimi_k3_mla_decode.py
```

脚本会使用固定随机种子生成相同的context和decode输入，分别启动两个独立进程：

1. `VLLM_XPU_MLA_DECODE_BACKEND=triton`
2. `VLLM_XPU_MLA_DECODE_BACKEND=deepklox`

每个进程都会保存完整decoder layer输出、MLA latent输出、LSE和probe报告，
然后统一转换为FP32计算误差。

## DeepKlox 构建

当前分支需要生成`xattention._C`扩展。容器内执行：

```bash
source /opt/intel/oneapi/setvars.sh
cd /home/mjc/applications.ai.gpu.deepklox

XATTENTION_ENABLED_KERNELS=mla_decode \
SYCL_THREADS=6 \
/root/.local/bin/uv pip install \
  --python /opt/venv/bin/python \
  --no-build-isolation \
  --no-deps \
  -e . \
  -v
```

确认扩展可以导入：

```bash
/opt/venv/bin/python -c \
  "import xattention; print(xattention.__file__)"
```

预期路径为：

```text
/home/mjc/applications.ai.gpu.deepklox/xattention/__init__.py
```

## 完整执行命令

以下命令可直接在Windows PowerShell中执行。使用Base64传递Linux脚本，避免
PowerShell、SSH和`docker exec`多层引号与换行解析问题。

### Context length 64

```powershell
$script = @'
set -e
cd /home/mjc/vllm
rm -rf /home/mjc/kimi_mla_deepklox_compare_ctx64
ZE_AFFINITY_MASK=0 /opt/venv/bin/python \
  tools/dev/compare_kimi_k3_mla_decode.py \
  --checkpoint-dir /home/hf_models/Kimi-K3-4layer-vllm-20260815 \
  --deepklox-root /home/mjc/applications.ai.gpu.deepklox \
  --layer-index 3 \
  --context-length 64 \
  --output-dir /home/mjc/kimi_mla_deepklox_compare_ctx64
'@
$bytes = [Text.Encoding]::UTF8.GetBytes($script.Replace("`r", ""))
$encoded = [Convert]::ToBase64String($bytes)
ssh -o BatchMode=yes miaojinc@10.239.11.67 "echo $encoded | base64 -d | docker exec -i mjc-kimi bash -s"
```

### Context length 2048

```powershell
$script = @'
set -e
cd /home/mjc/vllm
rm -rf /home/mjc/kimi_mla_deepklox_compare_ctx2048
ZE_AFFINITY_MASK=0 /opt/venv/bin/python \
  tools/dev/compare_kimi_k3_mla_decode.py \
  --checkpoint-dir /home/hf_models/Kimi-K3-4layer-vllm-20260815 \
  --deepklox-root /home/mjc/applications.ai.gpu.deepklox \
  --layer-index 3 \
  --context-length 2048 \
  --output-dir /home/mjc/kimi_mla_deepklox_compare_ctx2048
'@
$bytes = [Text.Encoding]::UTF8.GetBytes($script.Replace("`r", ""))
$encoded = [Convert]::ToBase64String($bytes)
ssh -o BatchMode=yes miaojinc@10.239.11.67 "echo $encoded | base64 -d | docker exec -i mjc-kimi bash -s"
```

也可以进入容器后直接执行。以context 2048为例：

```bash
cd /home/mjc/vllm
ZE_AFFINITY_MASK=0 /opt/venv/bin/python \
  tools/dev/compare_kimi_k3_mla_decode.py \
  --checkpoint-dir /home/hf_models/Kimi-K3-4layer-vllm-20260815 \
  --deepklox-root /home/mjc/applications.ai.gpu.deepklox \
  --layer-index 3 \
  --context-length 2048 \
  --output-dir /home/mjc/kimi_mla_deepklox_compare_ctx2048
```

## 输出文件

每个输出目录包含：

```text
comparison.json
triton_output.pt
deepklox_output.pt
triton_mla_output.pt
deepklox_mla_output.pt
triton_report.json
deepklox_report.json
```

其中：

- `*_output.pt`：完整decoder layer输出。
- `*_mla_output.pt`：MLA decode的512维latent输出和LSE。
- `*_report.json`：单次layer probe运行报告。
- `comparison.json`：汇总误差和最终通过状态。

## Context 64结果

最终状态：`passed=true`。

| 对比项 | Shape | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | Cosine similarity | Allclose |
|---|---:|---:|---:|---:|---:|---:|
| MLA latent output | `[1, 96, 512]` | 0.0000610352 | 0.000000001397 | 0.006494 | 1.0000011 | 通过 |
| MLA LSE | `[1, 96]` | 0.0156083 | 0.0081969 | 0.003710 | 0.9999977 | 通过 |
| Layer hidden states | `[1, 7168]` | 0.0004882812 | 0.0000468258 | 61.0352 | 0.9999937 | 通过 |
| Layer prefix sum | `[1, 7168]` | 0.0078125 | 0.0000017030 | 0.005587 | 1.0000020 | 通过 |
| Layer residual | `[1, 1, 7168]` | 0 | 0 | 0 | 1.0000014 | 通过 |

结果目录：

```text
/home/mjc/kimi_mla_deepklox_compare_ctx64
```

## Context 2048结果

最终状态：`passed=true`。

| 对比项 | Shape | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | Cosine similarity | Allclose |
|---|---:|---:|---:|---:|---:|---:|
| MLA latent output | `[1, 96, 512]` | 0.0001220703 | 0.0000057773 | 20.3978 | 1.0000004 | 通过 |
| MLA LSE | `[1, 96]` | 0.0155067 | 0.0077558 | 0.002017 | 0.9999993 | 通过 |
| Layer hidden states | `[1, 7168]` | 0.0004882812 | 0.0000426361 | 65.5 | 0.9999948 | 通过 |
| Layer prefix sum | `[1, 7168]` | 0.0078125 | 0.0000087363 | 0.019481 | 1.0000019 | 通过 |
| Layer residual | `[1, 1, 7168]` | 0 | 0 | 0 | 1.0000014 | 通过 |

结果目录：

```text
/home/mjc/kimi_mla_deepklox_compare_ctx2048
```

## 结果说明

DeepKlox与Triton的MLA latent输出、LSE和完整layer输出在两个context长度下均
通过`torch.allclose(atol=0.02, rtol=0.02)`。

Triton当前返回BF16 LSE，DeepKlox返回FP32 LSE。比较脚本统一转换为FP32后计算
误差。LSE约0.0155的最大绝对差异符合BF16量化精度范围，两个实现的LSE cosine
similarity均高于0.999997。

部分完整layer输出的最大相对误差较大，是因为参考值接近0，比较脚本使用
`abs(reference).clamp_min(1e-6)`作为相对误差分母。此时最大相对误差不具代表性，
应结合最大绝对误差、平均绝对误差、cosine similarity和allclose判断。

## 当前验证范围

已验证：

- TP1、batch 1、96个query heads；
- BF16 query和BF16 KV cache；
- block size 64；
- 单token decode；
- context length 64和2048；
- eager模式；
- Kimi-K3 NoPE配置；
- MLA latent输出、LSE和完整decoder layer输出。

尚未覆盖：

- TP2/TP4/TP8下的head padding；
- batch大于1和不等长sequence；
- DCP/PCP；
- prefix caching；
- speculative multi-token decode；
- FP8 KV cache；
- XPU Graph capture/replay。

因此当前DeepKlox路径应继续作为显式启用的实验性backend，不应直接作为所有XPU
MLA workload的默认实现。
