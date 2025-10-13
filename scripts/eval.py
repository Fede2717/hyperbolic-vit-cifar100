# ============================================================
# Eval script: Top-1 / Top-5 on CIFAR-100 checkpoints
# ============================================================
import os, argparse, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from typing import Tuple
import yaml

from hypervit.data.cifar100 import get_cifar100_loaders

# Reuse your training utilities
from scripts.train import (
    Config, set_seed, build_model,
    accuracy
)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = crit(logits, yb)
        B = xb.size(0)
        total += B
        loss_sum += loss.item() * B
        a1, a5 = accuracy(logits, yb, topk=(1, 5))
        top1_sum += a1 * B
        top5_sum += a5 * B
    return loss_sum / total, top1_sum / total, top5_sum / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="euclid", choices=["euclid","hyp-head","hyp-pos", 
                                                                     "hyp-residual-centered","hyp-residual-nocenter_x","hyp-residual-nocenter_0","hyp-only-residual-centered", "hyp-mlp","hyp-attn","hyp-all"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--hamp", action="store_true")
    p.add_argument("-c", "--config", type=str, default=None)
    p.add_argument("--progressive", action="store_true", default=True)
    args = p.parse_args()
    cfg = Config()
    if args.config:
        with open("configs/base.yaml", "r") as f:
            base = yaml.safe_load(f) or {}
        with open(args.config, "r") as f:
            over = yaml.safe_load(f) or {}
        merged = {**base, **over}
        for k, v in merged.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.amp:  cfg.amp = True
    if args.hamp: cfg.hamp = True

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model with the chosen variant
    model = build_model(cfg, args.variant, progressive=args.progressive).to(device)

    # load weights (support both plain state_dict or dict with 'model')
    state = torch.load(args.ckpt, map_location="cpu")
    strict = True  
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)


    _, val_loader = get_cifar100_loaders(cfg)
    val_loss, top1, top5 = evaluate(model, val_loader, device)
    print(f"[EVAL] {args.variant} | loss={val_loss:.4f} | top1={top1:.2f} | top5={top5:.2f}")

if __name__ == "__main__":
    main()
