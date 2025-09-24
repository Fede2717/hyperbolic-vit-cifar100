#!/usr/bin/env bash
set -e
python scripts/train.py --variant euclid -c configs/base.yaml
python scripts/eval.py  --variant euclid --ckpt checkpoints/w_vit_euclidean.pth
