# Lightning AI Studio (GPU) — Quick Start

This guide helps you run this repo on Lightning AI Studio with on-demand GPUs. You get one free CPU Studio and monthly GPU credits.

## 1) Launch a Studio and get the code
- Start a new Studio from your Lightning AI account.
- In the Studio terminal or notebook, clone your repo (or use the VS Code plugin):

```bash
# In Studio terminal
git clone https://github.com/forknay/quant-hackathon.git
cd quant-hackathon
```

If your project is private, set up auth (GitHub PAT or the VS Code plugin in Studio) and then clone.

## 2) Bootstrap the environment
- Option A (recommended): run the bootstrap script that installs base deps and a CUDA-enabled PyTorch matching the Studio GPU.

```bash
bash cloud/studio_bootstrap.sh
```

- Option B (manual):

```bash
# Create/activate a virtual environment (or reuse Studio's single env)
python -m venv .venv
source .venv/bin/activate  # Windows images: .venv/Scripts/activate

# Base deps
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt -r requirements-ml.txt

# Install CUDA-enabled PyTorch matching Studio (example)
# Use https://pytorch.org/get-started/locally/ to pick the exact command for the image
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Notes:
- Studios are a single environment by design. You can use that directly or create a venv.
- If wheels conflict, uninstall CPU torch first (`pip uninstall -y torch torchvision torchaudio`) then install the CUDA variant.

## 3) Verify GPU

```bash
python inference/test_cuda.py
```
You should see `CUDA available: True`, device name(s), and a successful GPU matmul.

## 4) Lightning smoke

```bash
python ml-model/lightning_smoke.py
```
Trainer is configured with `accelerator=auto, devices=auto`, so it will use GPUs if present.

## 5) Data & next steps
- Use the Drive plugin to mount/upload large datasets (we keep big files out of git).
- All Studios share the same filesystem; preprocess in one, train in another.
- Next we’ll add a FinBERT feature extractor and XGBoost + SHAP scripts.

## Troubleshooting
- Studio has a single env; if it’s broken, start a new Studio.
- If Studio is GPU-attached but `CUDA available` is False, ensure you installed a CUDA build of PyTorch compatible with the image.
