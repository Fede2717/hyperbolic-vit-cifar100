#!/usr/bin/env bash
set -e

python scripts/train.py \
  --variant hyp-only-residual-centered \
  -c configs/train/hyp_only_residual_centered.yaml \

python scripts/eval.py \
  --variant hyp-only-residual-centered \
  --ckpt experiments/residual_only_centered/h_only_residual_centered.pth
