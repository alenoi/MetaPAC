#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CFG="metapac/configs/meta_qwen3_wos_nooverlap_metarefresh.yaml"
SWEEP_CFG="metapac/configs/scenarios/meta_qwen3_nooverlap_sweep_stage1_8.yaml"

if [[ "${1:-}" == "--dry-run" ]]; then
  source .venv/bin/activate
  python scripts/utils/meta_train_sweep.py --base-config "$BASE_CFG" --sweep-config "$SWEEP_CFG" --dry-run
  exit 0
fi

source .venv/bin/activate
python scripts/utils/meta_train_sweep.py --base-config "$BASE_CFG" --sweep-config "$SWEEP_CFG"

echo
echo "Sweep done. Check outputs under: experiments/meta_train_sweep/"
echo "Learning curves are in each sweep's: results/learning_curves/"
