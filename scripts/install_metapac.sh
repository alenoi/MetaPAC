#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="metapac"
PY_VER="3.12"
INSTALL_DEV="${INSTALL_DEV:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure logs are stored in the centralized logs directory
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Example log file setup
LOG_FILE="$LOG_DIR/install_metapac.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 1) Miniconda check + installation (silent)
if ! command -v conda >/dev/null 2>&1; then
  echo "[INFO] Conda not found, installing Miniconda..."
  TMPD=$(mktemp -d)
  OS=$(uname -s)
  if [[ "$OS" == "Darwin" ]]; then
    INST="$TMPD/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o "$INST"
    bash "$INST" -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  else
    INST="$TMPD/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INST"
    bash "$INST" -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  fi
else
  # shell init
  eval "$(conda shell.bash hook)"
fi

# 2) Create environment
if conda env list | grep -q "^$ENV_NAME\s"; then
  echo "[INFO] Existing conda environment: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

# 3) Select CUDA/PyTorch build (based on driver/runtime)
CU_IDX_URL=""
TORCH_VER=""
TV_VER=""

CUDA_VER_STR=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | awk '/CUDA Version/{print $3}' | head -n1 || echo "CPU")
echo "[INFO] nvidia-smi CUDA Version: $CUDA_VER_STR"

choose_torch() {
  # Preferred: cu128 (RTX 50 series), fallback: cu126, final fallback: cu124
  # Torch 2.8.0 (cu128) / 2.7.1 (cu128) and fallback 2.6.0 (cu124)
  case "$1" in
    cu128)
      TORCH_VER="2.8.0"; TV_VER="0.23.0"; CU_IDX_URL="https://download.pytorch.org/whl/cu128"
      ;;
    cu126)
      TORCH_VER="2.7.1"; TV_VER="0.22.1"; CU_IDX_URL="https://download.pytorch.org/whl/cu126"
      ;;
    cu124)
      TORCH_VER="2.6.0"; TV_VER="0.21.0"; CU_IDX_URL="https://download.pytorch.org/whl/cu124"
      ;;
    cpu)
      TORCH_VER="2.8.0"; TV_VER="0.23.0"; CU_IDX_URL="https://download.pytorch.org/whl/cpu"
      ;;
  esac
}

if [[ "$CUDA_VER_STR" == "CPU" ]]; then
  echo "[WARN] nvidia-smi is not available, installing CPU build."
  choose_torch cpu
else
  # Simple heuristic: if runtime >= 12.8 -> cu128; if >= 12.6 -> cu126; otherwise cu124
  MAJ=$(echo "$CUDA_VER_STR" | cut -d. -f1)
  MIN=$(echo "$CUDA_VER_STR" | cut -d. -f2)
  if (( MAJ > 12 )) || (( MAJ == 12 && MIN >= 8 )); then
    choose_torch cu128
  elif (( MAJ == 12 && MIN >= 6 )); then
    choose_torch cu126
  else
    choose_torch cu124
  fi
fi

echo "[INFO] PyTorch $TORCH_VER + torchvision $TV_VER"
echo "[INFO] Index URL: $CU_IDX_URL"

# 4) Install dependencies
python -m pip install --upgrade pip
pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" --index-url "$CU_IDX_URL"
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
else
  pip install numpy pandas scikit-learn matplotlib tqdm torchmetrics
fi

if [[ "$INSTALL_DEV" == "1" && -f "requirements-dev.txt" ]]; then
  pip install -r requirements-dev.txt
fi

python - <<'PY'
import site
from pathlib import Path

repo_root = Path.cwd().resolve()
pth_name = "metapac-dev.pth"
for site_dir in site.getsitepackages():
    site_path = Path(site_dir)
    if site_path.exists():
        (site_path / pth_name).write_text(str(repo_root) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {pth_name} to {site_path}")
        break
PY

# 5) Quick verification
python - <<'PY'
import importlib
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CC:", torch.cuda.get_device_capability(0))

required = [
    "yaml", "torch", "transformers", "datasets", "pandas",
    "sklearn", "joblib", "safetensors", "pyarrow"
]
for module_name in required:
    importlib.import_module(module_name)
print("Verified imports:", ", ".join(required))
PY

if [[ "$INSTALL_DEV" == "1" ]]; then
  echo "[OK] MetaPAC environment is ready (runtime + dev): conda activate $ENV_NAME"
else
  echo "[OK] MetaPAC environment is ready (runtime): conda activate $ENV_NAME"
  echo "[INFO] For dev dependencies, rerun with: INSTALL_DEV=1 bash scripts/install_metapac.sh"
fi
