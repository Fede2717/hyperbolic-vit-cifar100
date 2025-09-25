#!/usr/bin/env bash
set -e

python scripts/train.py --variant hyp-mlp \
  -c configs/train/hyp_linear.yaml \
  --epochs 50 \
  --total_epochs 100

python scripts/train.py --variant hyp-mlp \
  -c configs/train/hyp_linear.yaml \
  --epochs 50 \
  --ckpt experiments/linear/h_hyper_best.pth \
  --total_epochs 100 --resume_from_epoch 50
