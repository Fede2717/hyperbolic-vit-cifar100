#!/usr/bin/env bash
set -e
python scripts/train.py --variant euclid -c configs/train/euclid.yaml
python scripts/eval.py  --variant euclid --ckpt experiments/euclid/w_euclid.pth

