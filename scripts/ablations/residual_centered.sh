#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-residual-centered -c configs/train/hyp_residual_centered.yaml --epochs 100
python scripts/eval.py  --variant hyp-residual-centered --ckpt experiments/residual_centered/h_residual_centered.pth
