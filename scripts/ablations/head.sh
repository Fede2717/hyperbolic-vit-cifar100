#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-head -c configs/train/hyp_head.yaml
python scripts/eval.py  --variant hyp-head --ckpt experiments/head/h_head.pth
