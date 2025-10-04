#!/usr/bin/env bash
set -euo pipefail

# Lightning AI Studio bootstrap for this repo
# - Upgrades pip tooling
# - Installs base deps
# - Installs a CUDA-enabled PyTorch build (override TORCH_CUDA if needed)
# - Verifies CUDA availability

PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT_DIR"

# You can opt to use Studio's default env or create a venv here
# python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# Base deps
pip install -r requirements.txt -r requirements-ml.txt

# Install CUDA PyTorch (override via: TORCH_CUDA=cu124 bash cloud/studio_bootstrap.sh)
TORCH_CUDA=${TORCH_CUDA:-cu121}

# Remove any CPU-only torch to avoid conflicts
pip uninstall -y torch torchvision torchaudio || true

# Install chosen CUDA wheels
pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch torchvision torchaudio

# Sanity check
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda version:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('device', i, torch.cuda.get_device_name(i))
PY

# Lightning quick smoke (GPU will be auto-selected if available)
python ml-model/lightning_smoke.py || true
