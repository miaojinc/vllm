#!/usr/bin/env bash
# Kimi-K3 layer-probe sweep on one RTX PRO 5000, for KDA / MoE time share.
#
# Mode A (chunk sweep): the timed prefill chunk is 8K/16K/32K tokens and follows
#   an equally long already-computed context, i.e. "what should
#   max_num_batched_tokens be".
# Mode B (prompt sweep): the timed chunk is fixed at 2K and the prompt is
#   8K/16K/32K, i.e. the last chunk of a long prompt in real serving.
# Mode C (cold prefill): the whole 8K/16K/32K prompt in one chunk, no context.
# Mode D (decode): one token on top of an 8K/16K/32K context.
# Mode E (decode + CUDA graph): same as D, timed as graph replays.
# Mode F/G: C/A repeated as CUDA graph replays, to size the host-side overhead
#   that prefill kernels do not amortize away.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-/data/miniforge3/envs/vllm_0_27_1_pip/bin/python}
CHECKPOINT=${CHECKPOINT:-/mnt/disk1/models/Kimi-K3-4L}
OUT=${OUT:-"$HERE/logs/layer_probe"}
# Layer 1 is KDA + MoE, layer 3 is MLA + MoE (zero-based) in the 4-layer build.
LAYERS=${LAYERS:-"1 3"}
SEQS=${SEQS:-"8192 16384 32768"}
DECODE_CHUNK=2048
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
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
    *) sub=cudagraph ;;
  esac
  mkdir -p "$OUT/$sub"
  local stem="$OUT/$sub/${tag}_l${layer}"
  # --chunked-prefill is only legal when a context precedes the timed forward.
  local chunked=(--chunked-prefill)
  [[ $context -eq 0 || $tokens -eq 1 ]] && chunked=()
  # Keyed off the report, not the trace: traces get pruned as redundant.
  if [[ -s "$stem.json" ]]; then
    echo "== skip $sub/${tag}_l${layer} (report exists)"
    return 0
  fi
  echo "== $sub/${tag}_l${layer}  chunk=$tokens context=$context ${extra[*]}"
  "$PYTHON" "$HERE/kimi_k3_nv_layer_probe.py" \
    --checkpoint-dir "$CHECKPOINT" \
    --layer-index "$layer" \
    --num-tokens "$tokens" \
    --context-length "$context" \
    "${chunked[@]}" \
    "${extra[@]}" \
    --context-chunk-size "$ctx_chunk" \
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
    run_one "E_graph${kb}k" "$layer" 1 "$seq" "$DECODE_CHUNK" --cuda-graph
    run_one "F_coldgraph${kb}k" "$layer" "$seq" 0 0 --cuda-graph
    run_one "G_chunkgraph${kb}k" "$layer" "$seq" "$seq" "$seq" --cuda-graph
  done
done

echo "== analysing"
for sub in prefill chunked_prefill decode; do
  [[ -n $(compgen -G "$OUT/$sub/*.trace.json") ]] || continue
  "$PYTHON" "$HERE/analyze_layer_trace.py" "$OUT/$sub"/*.trace.json \
    --top-kernels 5 --json "$OUT/$sub/summary.json" > "$OUT/$sub/summary.txt"
done
"$PYTHON" "$HERE/analyze_layer_trace.py" "$OUT"/*/*.trace.json \
  --top-kernels 8 --json "$OUT/summary.json" | tee "$OUT/summary.txt"
