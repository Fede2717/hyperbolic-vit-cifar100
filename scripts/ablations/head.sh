#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-head -c configs/base.yaml
python scripts/eval.py  --variant hyp-head --ckpt checkpoints/h_hyper_best.pth
