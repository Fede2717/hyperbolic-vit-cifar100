#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

python scripts/train.py --variant hyp-mlp \
  -c configs/train/hyp_linear.yaml \
  --epochs 50 \
  --total_epochs 100

python scripts/eval.py  --variant hyp-mlp \
  --ckpt experiments/linear/h_linear.pth

python scripts/train.py --variant hyp-mlp \
  -c configs/train/hyp_linear.yaml \
  --epochs 50 \
  --ckpt experiments/linear/h_linear.pth \
  --total_epochs 100 --resume_from_epoch 50

python scripts/eval.py  --variant hyp-mlp \
  --ckpt experiments/linear/h_linear.pth

