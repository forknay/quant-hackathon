#!/usr/bin/env bash
set -euo pipefail

# Script to run Phase 1 end-to-end in Lightning AI Studio
# - Bootstraps CUDA PyTorch and deps
# - Optional warm-up limited run to build PCA and cache the model
# - Full run with --reuse-pca
# - Verification step

PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT_DIR"

echo "[phase1] Bootstrapping environment..."
# Adjust CUDA version via TORCH_CUDA=cu124 bash cloud/run_phase1_full.sh
TORCH_CUDA=${TORCH_CUDA:-cu121}
TORCH_CUDA=$TORCH_CUDA bash cloud/studio_bootstrap.sh

echo "[phase1] Warm-up (limit 200 docs) to build PCA and cache model..."
python -m nlp_features.run --limit-docs 200 --batch-size 32 || true

echo "[phase1] Full run with PCA reuse..."
python -m nlp_features.run --batch-size 32 --reuse-pca

echo "[phase1] Verifying outputs..."
python cloud/verify_phase1.py

echo "[phase1] Done. Outputs in text_features_out/"
