#!/usr/bin/env bash
set -e
python scripts/train.py --variant hyp-pos -c configs/base.yaml
python scripts/eval.py  --variant hyp-pos --ckpt checkpoints/h_hyper_best.pth
