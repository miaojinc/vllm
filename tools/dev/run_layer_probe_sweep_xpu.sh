#!/usr/bin/env bash
# Kimi-K3 layer-probe sweep on one Arc Pro B70, for KDA / MoE time share.
#
# Mirrors the NV sweep's modes A-D. The CUDA-graph modes (E/F/G) have no XPU
# counterpart here: vLLM disables XPU graph under enforce_eager, so host-side
# launch overhead cannot be amortised the same way.
#
# Mode A (chunk sweep): the timed prefill chunk is 8K/16K/32K tokens and follows
#   an equally long already-computed context, i.e. "what should
#   max_num_batched_tokens be".
# Mode B (prompt sweep): the timed chunk is fixed at 2K and the prompt is
#   8K/16K/32K, i.e. the last chunk of a long prompt in real serving.
# Mode C (cold prefill): the whole 8K/16K/32K prompt in one chunk, no context.
# Mode D (decode): one token on top of an 8K/16K/32K context.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:-/mnt/disk2/hf_models/Kimi-K3-4L}
# Not "layer_probe": that directory holds the NV reference run.
OUT=${OUT:-"$HERE/layer_probe_xpu"}
# Layer 1 is KDA + MoE, layer 3 is MLA + MoE (zero-based) in the 4-layer build.
LAYERS=${LAYERS:-"1 3"}
SEQS=${SEQS:-"8192 16384 32768"}
DECODE_CHUNK=${DECODE_CHUNK:-2048}
# Serving runs Kimi-K3 dense MLA at 64; DeepKlox additionally rejects any block
# size that is not a multiple of 32. Kept at 16 so existing reports stay comparable.
BLOCK_SIZE=${BLOCK_SIZE:-16}
# triton | deepklox. Only affects the MLA decode path, so modes A/B/C and KDA
# layers are identical either way.
MLA_DECODE_BACKEND=${MLA_DECODE_BACKEND:-triton}
# Matches the XPU probe default; the NV run used 3, a +0.4~1.2% difference that
# sits inside the noise band, so the NV numbers are not re-collected.
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-10}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export VLLM_XPU_MLA_DECODE_BACKEND="$MLA_DECODE_BACKEND"
# vllm-kda10 is the only checkout that wires KDA into KimiDecoderLayer; $HERE/vllm
# is the unpatched miaojinc branch and silently falls back to NotImplementedError.
VLLM_SRC=${VLLM_SRC:-"$HERE/vllm-kda10"}
export PYTHONPATH="$HERE:$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

run_one() {
  local tag=$1 layer=$2 tokens=$3 context=$4 ctx_chunk=$5
  shift 5
  local extra=("$@")
  local sub
  case "$tag" in
    A_*|B_*) sub=chunked_prefill ;;
    C_*) sub=prefill ;;
    D_*) sub=decode ;;
    *) sub=other ;;
  esac
  mkdir -p "$OUT/$sub"
  # Non-default sweeps get a suffix so they never collide with, or silently skip
  # against, the block-16 triton reports collected earlier.
  local variant=""
  [[ $BLOCK_SIZE -ne 16 ]] && variant="${variant}_bs${BLOCK_SIZE}"
  [[ $MLA_DECODE_BACKEND != triton ]] && variant="${variant}_${MLA_DECODE_BACKEND}"
  local stem="$OUT/$sub/${tag}_l${layer}${variant}"
  # --chunked-prefill is only legal when a context precedes the timed forward.
  local chunked=(--chunked-prefill)
  [[ $context -eq 0 || $tokens -eq 1 ]] && chunked=()
  # Keyed off a passed report, not just its presence: failed runs must be retried.
  if [[ -s "$stem.json" ]] && grep -q '"status": "passed"' "$stem.json"; then
    echo "== skip $sub/$(basename "$stem") (passed report exists)"
    return 0
  fi
  echo "== $sub/$(basename "$stem")  chunk=$tokens context=$context ${extra[*]}"
  "$PYTHON" "$HERE/kimi_k3_xpu_layer_probe.py" \
    --checkpoint-dir "$CHECKPOINT" \
    --layer-index "$layer" \
    --num-tokens "$tokens" \
    --context-length "$context" \
    "${chunked[@]}" \
    "${extra[@]}" \
    --context-chunk-size "$ctx_chunk" \
    --block-size "$BLOCK_SIZE" \
    --warmup-iters "$WARMUP" \
    --benchmark-iters "$ITERS" \
    --profile-output "$stem.trace.json" \
    --report "$stem.json" \
    >"$stem.log" 2>&1 || { echo "   FAILED, see $stem.log"; return 0; }
  "$PYTHON" - "$stem.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
b = r.get("benchmark", {})
print(f"   {r['status']} {r.get('attention_type')}/{r.get('mlp_type')} "
      f"p50={b.get('latency_p50_ms', float('nan')):.2f} ms")
PY
}

for layer in $LAYERS; do
  for seq in $SEQS; do
    kb=$((seq / 1024))
    run_one "A_chunk${kb}k" "$layer" "$seq" "$seq" "$seq"
    run_one "B_prompt${kb}k" "$layer" "$DECODE_CHUNK" "$((seq - DECODE_CHUNK))" "$DECODE_CHUNK"
    run_one "C_cold${kb}k" "$layer" "$seq" 0 0
    run_one "D_decode${kb}k" "$layer" 1 "$seq" "$DECODE_CHUNK"
  done
done

echo "== analysing"
# Sharded parallel runs set ANALYZE=0 and let the driver analyse once at the end.
[[ ${ANALYZE:-1} == 1 ]] || exit 0
for sub in prefill chunked_prefill decode; do
  [[ -n $(compgen -G "$OUT/$sub/*.trace.json") ]] || continue
  "$PYTHON" "$HERE/analyze_layer_trace_xpu.py" "$OUT/$sub"/*.trace.json \
    --top-kernels 5 --json "$OUT/$sub/summary.json" > "$OUT/$sub/summary.txt"
done
[[ -n $(compgen -G "$OUT"/*/*.trace.json) ]] || exit 0
"$PYTHON" "$HERE/analyze_layer_trace_xpu.py" "$OUT"/*/*.trace.json \
  --top-kernels 8 --json "$OUT/summary.json" | tee "$OUT/summary.txt"
"$PYTHON" "$HERE/summarize_layer_operators_xpu.py" "$OUT"/*/*.trace.json \
  --output "$OUT/operator_tables.md"
"$PYTHON" "$HERE/plot_layer_operators_xpu.py"
