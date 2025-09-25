#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-pos -c configs/train/hyp_pos.yaml
python scripts/eval.py  --variant hyp-pos --ckpt experiments/pos/h_pos.pth
