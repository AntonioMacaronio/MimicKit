#!/usr/bin/env bash
# Source from any directory before data preparation, testing or training.
export SMP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_STORAGE_ROOT="${SMP_STORAGE_ROOT:-/data/sky/projects/MimicKit}"
printf 'SMP checkout: %s\nProject storage: %s\n' "$SMP_ROOT" "$PROJECT_STORAGE_ROOT"
export TMPDIR="$PROJECT_STORAGE_ROOT/tmp"
export XDG_CACHE_HOME=/data/sky/cache
export PIP_CACHE_DIR=/data/sky/cache/pip UV_CACHE_DIR=/data/sky/cache/uv
export HF_HOME=/data/sky/cache/huggingface TORCH_HOME=/data/sky/cache/torch
export TORCH_EXTENSIONS_DIR="$PROJECT_STORAGE_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_STORAGE_ROOT/cache/triton"
export PYTHONPYCACHEPREFIX="$PROJECT_STORAGE_ROOT/cache/pycache"
export WANDB_DIR="$PROJECT_STORAGE_ROOT/runs/wandb" WANDB_CACHE_DIR=/data/sky/cache/wandb
export CONDA_PKGS_DIRS=/data/sky/cache/conda-pkgs RUFF_CACHE_DIR=/data/sky/cache/ruff
export PYTHONPATH="$SMP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
