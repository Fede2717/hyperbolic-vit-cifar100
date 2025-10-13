#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-residual-nocenter-0 -c configs/train/hyp_residual_nocenter_0.yaml 
python scripts/eval.py  --variant hyp-residual-nocenter-0 --ckpt experiments/residual_nocenter_0/h_residual_nocenter_0.pth
