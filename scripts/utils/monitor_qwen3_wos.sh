#!/usr/bin/env bash
set -u

INTERVAL="${1:-5}"
RUN_DIR="${2:-targets/qwen3_06b/runs/baseline_qwen3_06b_wos_fast}"
CONFIG_PATH="${3:-metapac/configs/auto_qwen3_wos_fast.yaml}"

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 1 ]]; then
  echo "Error: INTERVAL must be a positive integer (for example: 5)."
  echo "Usage: $0 [INTERVAL_SEC] [RUN_DIR] [CONFIG_PATH]"
  exit 1
fi

while true; do
  clear
  date

  echo "=== GPU ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || echo "nvidia-smi error"
  else
    echo "nvidia-smi is not available"
  fi

  echo
  echo "=== MetaPAC process ==="
  PROCESS_REGEX="python .* -m metapac --config ${CONFIG_PATH}"
  PROCESS_REGEX_ALT="python .* -m metapac --config .*/$(basename "$CONFIG_PATH")"

  if ! pgrep -af "$PROCESS_REGEX"; then
    pgrep -af "$PROCESS_REGEX_ALT" || echo "no running metapac process for this config"
  fi

  pid="$(pgrep -f "$PROCESS_REGEX" | head -n1)"
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -f "$PROCESS_REGEX_ALT" | head -n1)"
  fi
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -f "python -m metapac|metapac/__main__.py|trainer.py" | head -n1)"
  fi

  if [[ -n "$pid" ]]; then
    ps -p "$pid" -o pid,etime,%cpu,%mem,cmd
  fi

  echo
  echo "=== Latest checkpoint ==="
  ls -1dt "${RUN_DIR}"/checkpoint-* 2>/dev/null | head -n1 || echo "no checkpoint yet"

  sleep "$INTERVAL"
done
