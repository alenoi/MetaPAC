#Requires -Version 5.0
$ErrorActionPreference = "Stop"

$EnvName = "metapac"
$PyVer = "3.12"
$InstallDev = if ($env:INSTALL_DEV) { $env:INSTALL_DEV } else { "0" }
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Ensure logs are stored in the centralized logs directory
$LogDir = "logs"
if (-Not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Example log file setup
$LogFile = Join-Path $LogDir "install_metapac.log"
Start-Transcript -Path $LogFile -Append

function Ensure-Conda {
  if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[INFO] Conda not found, installing Miniconda..."
    $tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath() + "\miniconda_dl") -Force
    $inst = Join-Path $tmp "Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $inst
    Start-Process -FilePath $inst -ArgumentList "/InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=$env:USERPROFILE\Miniconda3" -Wait
    $condaExe = "$env:USERPROFILE\Miniconda3\condabin\conda.bat"
    & $condaExe init powershell | Out-Null
    $env:Path += ";$env:USERPROFILE\Miniconda3\condabin"
  }
  conda --version | Out-Null
}

Ensure-Conda

# A new PowerShell session may occasionally require: . $PROFILE; try this first:
& conda.exe env list | Out-Null

# Create environment
if ((conda env list) -match "^\s*$EnvName\s") {
  Write-Host "[INFO] Conda env already exists: $EnvName"
} else {
  conda create -y -n $EnvName ("python=" + $PyVer) | Out-Null
}
conda activate $EnvName

# Detect CUDA runtime
$cudaVer = "CPU"
try {
  $nvsmi = & nvidia-smi 2>$null
  if ($LASTEXITCODE -eq 0) {
    $line = ($nvsmi | Select-String -Pattern "CUDA Version").ToString()
    if ($line) {
      $cudaVer = ($line -split "CUDA Version\s*:\s*")[1].Trim()
    }
  }
} catch {}

function Choose-Torch ($tag) {
  switch ($tag) {
    "cu128" { return @{ torch="2.8.0"; tv="0.23.0"; idx="https://download.pytorch.org/whl/cu128" } }
    "cu126" { return @{ torch="2.7.1"; tv="0.22.1"; idx="https://download.pytorch.org/whl/cu126" } }
    "cu124" { return @{ torch="2.6.0"; tv="0.21.0"; idx="https://download.pytorch.org/whl/cu124" } }
    "cpu"   { return @{ torch="2.8.0"; tv="0.23.0"; idx="https://download.pytorch.org/whl/cpu"   } }
  }
}

if ($cudaVer -eq "CPU") {
  Write-Host "[WARN] nvidia-smi not found; installing CPU build."
  $sel = Choose-Torch "cpu"
} else {
  $parts = $cudaVer.Split(".")
  $maj = [int]$parts[0]; $min = [int]$parts[1]
  if (($maj -gt 12) -or ($maj -eq 12 -and $min -ge 8)) { $sel = Choose-Torch "cu128" }
  elseif ($maj -eq 12 -and $min -ge 6) { $sel = Choose-Torch "cu126" }
  else { $sel = Choose-Torch "cu124" }
}

Write-Host "[INFO] Torch" $sel.torch " + torchvision " $sel.tv
Write-Host "[INFO] Index URL:" $sel.idx

python -m pip install --upgrade pip
pip install ("torch==" + $sel.torch) ("torchvision==" + $sel.tv) --index-url $sel.idx

if (Test-Path ".\requirements.txt") {
  pip install -r requirements.txt
} else {
  pip install numpy pandas scikit-learn matplotlib tqdm torchmetrics
}

if ($InstallDev -eq "1" -and (Test-Path ".\requirements-dev.txt")) {
  pip install -r requirements-dev.txt
}

@'
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
'@ | python -

@'
import torch
import importlib

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
'@ | python -

if ($InstallDev -eq "1") {
  Write-Host "[OK] MetaPAC env ready (runtime + dev). Use: conda activate $EnvName"
} else {
  Write-Host "[OK] MetaPAC env ready (runtime). Use: conda activate $EnvName"
  Write-Host "[INFO] For dev dependencies rerun with: `$env:INSTALL_DEV=1; .\scripts\install_metapac.ps1"
}

Stop-Transcript
