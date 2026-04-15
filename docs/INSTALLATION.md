# Installation Guide

This document describes how to bootstrap MetaPAC in a fresh environment and what is currently provided by the repository.

## Current State

The repository provides:

- `requirements.txt` for runtime dependencies
- `requirements-dev.txt` for test and developer tooling
- `scripts/install_metapac.sh` for Linux/macOS style shell environments
- `scripts/install_metapac.ps1` for Windows PowerShell

The install scripts now:

- create or reuse a Conda environment named `metapac`
- install a CUDA-appropriate PyTorch build when possible, or CPU PyTorch otherwise
- install repository runtime dependencies
- optionally install developer dependencies when `INSTALL_DEV=1`
- add a `.pth` file so the repository is importable in the environment
- verify critical imports after installation

## Prerequisites

- Git
- Conda or permission to let the installer bootstrap Miniconda
- Python 3.12
- NVIDIA driver plus `nvidia-smi` if you want automatic CUDA wheel selection

## Recommended Install: Linux/macOS

From the repository root:

```bash
bash scripts/install_metapac.sh
```

With development dependencies:

```bash
INSTALL_DEV=1 bash scripts/install_metapac.sh
```

Activate the environment:

```bash
conda activate metapac
```

## Recommended Install: Windows PowerShell

From the repository root:

```powershell
.\scripts\install_metapac.ps1
```

With development dependencies:

```powershell
$env:INSTALL_DEV=1
.\scripts\install_metapac.ps1
```

Activate the environment:

```powershell
conda activate metapac
```

## Manual Install

If you prefer to manage the environment yourself:

```bash
conda create -n metapac python=3.12
conda activate metapac
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run from the repository root, or add the repository root to `PYTHONPATH` if you want to invoke MetaPAC from elsewhere.

## Verification

Minimal verification:

```bash
python - <<'PY'
import yaml
import torch
import transformers
import datasets
import pandas
import sklearn
import joblib
import safetensors
import pyarrow
print("MetaPAC runtime imports look good")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Smoke tests:

```bash
python -m pytest tests -q
```

## Dependency Split

Runtime packages in `requirements.txt` cover the active pipeline, including:

- PyTorch and Transformers stack
- datasets and evaluation stack
- parquet and checkpoint support via `pyarrow` and `safetensors`
- numerical and ML tooling such as NumPy, pandas, SciPy, scikit-learn, and joblib
- logging and reporting helpers used by the active code paths

Developer packages in `requirements-dev.txt` cover:

- pytest
- black
- flake8
- isort
- mypy
- extra tooling used by developer and legacy utility scripts such as `psutil` and `pypdf`

## Known Scope Boundary

- The repository does not currently ship a `pyproject.toml` or wheel-based package install flow.
- The supported workflow is environment setup plus running from the checked-out repository.
- The developer dependency set also covers auxiliary utilities used by validation and maintenance scripts.