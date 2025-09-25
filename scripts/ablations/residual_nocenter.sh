#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-residual-nocenter -c configs/train/hyp_residual_nocenter.yaml --epochs 100
python scripts/eval.py  --variant hyp-residual-nocenter --ckpt experiments/residual_nocenter/h_residual_nocenter.pth
