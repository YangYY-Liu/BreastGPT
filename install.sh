#!/usr/bin/env bash
# BreastGPT environment setup.
#
# Tested with: CUDA 12.8, PyTorch 2.10, Python 3.12, A100/H100-class GPUs.
# Run from the repository root.

set -euo pipefail

# --- System libraries (RoCE / InfiniBand for multi-node training) ----------
sudo apt-get update
sudo apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils

# --- Python ----------------------------------------------------------------
# Use a fresh Python 3.12 environment (conda example shown; venv works too).
conda install python=3.12 -y

# --- Core Python dependencies ----------------------------------------------
pip install \
    mpire nibabel SimpleITK pandas fire tqdm openai h5py decord \
    deepspeed 'ms-swift[all]'

# --- Vendored ms-swift (editable install) ----------------------------------
# We ship a pinned copy of ms-swift under thirdParty/ms-swift with BreastGPT
# integrations. Install it editable so the local copy takes precedence.
pip install -e 'thirdParty/ms-swift[all]'

# --- FlashAttention --------------------------------------------------------
# Install a prebuilt wheel that matches your CUDA / PyTorch / Python ABI.
# Browse: https://github.com/Dao-AILab/flash-attention/releases
# Example (CUDA 12.8 / PyTorch 2.10 / Python 3.12):
#   pip install flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl
pip install flash-attn --no-build-isolation

# --- vLLM (optional, for fast inference / deployment) ----------------------
pip uninstall -y ms-opencompass || true
pip install vllm

echo "BreastGPT environment ready."
