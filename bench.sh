#!/usr/bin/env bash
# Run `vllm bench serve` against the running serving container.
# Load shows up on the dashboard (http://localhost:8080) while it runs.
#
# Usage:  ./bench.sh                      # defaults below
#         NUM_PROMPTS=500 REQUEST_RATE=20 MAX_CONCURRENCY=64 ./bench.sh
#         ./bench.sh --ignore-eos         # extra args go straight to vllm bench serve
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER="${CONTAINER:-kvmon-serving}"
[ -f .env ] && set -a && . ./.env && set +a
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-10}"          # req/s, "inf" = send all at once
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"

docker exec "$CONTAINER" vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  "$@"
