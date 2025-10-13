#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-residual-nocenter-x -c configs/train/hyp_residual_nocenter_x.yaml 
python scripts/eval.py  --variant hyp-residual-nocenter-x --ckpt experiments/residual_nocenter_x/h_residual_nocenter_x.pth
