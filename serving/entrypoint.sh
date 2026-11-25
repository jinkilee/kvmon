#!/usr/bin/env bash
# Runs the GPU exporter in the background and vLLM in the foreground.
# vLLM output is tee'd to a log file so the exporter can read the
# "Model loading took ..." / "Available KV cache memory ..." lines.
set -uo pipefail

LOG="${VLLM_LOG_PATH:-/var/log/vllm/vllm.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

python3 /app/exporter.py &
EXPORTER_PID=$!

ARGS=(
  --host 0.0.0.0
  --port "${VLLM_PORT:-8000}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
)
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")

vllm serve "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" "${ARGS[@]}" ${VLLM_EXTRA_ARGS:-} "$@" \
  > >(tee -a "$LOG") 2>&1 &
VLLM_PID=$!

trap 'kill -TERM $VLLM_PID $EXPORTER_PID 2>/dev/null' TERM INT
wait $VLLM_PID
STATUS=$?
# if a signal interrupted the wait, let vLLM finish shutting down
if kill -0 $VLLM_PID 2>/dev/null; then wait $VLLM_PID; STATUS=$?; fi
kill $EXPORTER_PID 2>/dev/null
exit $STATUS
