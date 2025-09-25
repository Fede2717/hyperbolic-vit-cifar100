#!/usr/bin/env bash
set -e

python scripts/train.py --variant hyp-attn \
  -c configs/train/hyp_attn.yaml \
  --epochs 10 \
  --ckpt experiments/linear/h_linear.pth \
  --non_strict_load \
  --attn_phase attn-only \
  --lr 5e-4

python scripts/eval.py  --variant hyp-attn \
  --ckpt experiments/attn/h_attn.pth

python scripts/train.py --variant hyp-attn \
  -c configs/train/hyp_attn.yaml \
  --epochs 10 \
  --ckpt experiments/attn/h_attn.pth \
  --attn_phase full \
  --lr 1e-4

python scripts/eval.py  --variant hyp-attn \
  --ckpt experiments/attn/h_attn.pth
