# ============================================================
# Import, seed, hyper-params & device
# ============================================================
#!pip install -q geoopt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import random, time, math, argparse, yaml
from dataclasses import dataclass
from typing import Tuple

import geoopt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from hypervit.data.cifar100 import get_cifar100_loaders

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#My moduls
from hypervit.models.vit_tiny import CifarViT
from hypervit.models.head import HyperbolicHead                 # t=0.96
from hypervit.models.pos import HyperbolicPositionalEmbedding   # t=0.98
from hypervit.models.h_lin_mlp import HyperbolicFeedForward     # t=0.985
from hypervit.models.h_attn import HyperbolicSelfAttention, SharedHyperbolicCentroids  # t=0.985

# ============================================================
# Hyperparameters and general
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    # data
    img_size: int = 32
    in_chans: int = 3
    num_classes: int = 100
    batch_size: int = 128
    eff_batch_size: int | None = None # if None no accomulation
    num_workers: int = 4
    # model
    patch_size: int = 4
    embed_dim: int = 192
    num_blocks: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1 # drop on the different outs
    attn_drop: float = 0.1
    drop_res: float = 0.2
    # train
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    man_grad_clip: float = 1.0
    amp: bool = True
    hamp: bool = False
    seed: int = 42
    init_cls_scale: float=0.3
    hyp_init_c: float = 1.0
    hyp_init_s: float = 1.0 
    t_head: float = 0.96
    t_pos: float = 0.98
    t_default: float = 0.985
    proto_std: float = 2e-2
    # dir
    out_dir: str = "checkpoints"

# ============================================================
# CIFAR-100 Dataset & DataLoaders
# ============================================================

train_loader, val_loader = get_cifar100_loaders(cfg)

# ============================================================
# Hyper modules factories (closures on cfg)
# ============================================================

def make_factories(cfg: Config, variant: str):
    """
    Build factories for optional hyperbolic modules.
    We only pass `clip_t` to the head if:
      - head-only:           use cfg.t_head
      - head + positional:   use cfg.t_pos
      - otherwise:           do not pass (class default)
    """
    use_head = variant in {"hyp-head", "hyp-all"}
    use_pos  = variant in {"hyp-pos", "hyp-all"}
    use_mlp  = variant in {"hyp-mlp", "hyp-all"}
    use_attn = variant in {"hyp-attn", "hyp-all"}

    head_factory = None
    pos_factory  = None
    mlp_factory  = None
    attn_factory = None

    if use_head:
        # decide which t to attempt for the head
        clip_for_head = None
        if use_pos:
            clip_for_head = cfg.t_pos      # head+pos => use positional t
        elif variant == "hyp-head":
            clip_for_head = cfg.t_head     # head-only => use head t

        def head_factory(D, C):
            kwargs = dict(
                hamp=cfg.hamp,
                dim=D,
                num_classes=C,
                init_c=cfg.hyp_init_c,
                hyp_init_s=cfg.hyp_init_s,
                init_cls_scale=cfg.init_cls_scale,
                eps=1e-6,
                proto_std=cfg.proto_std,
            )
            # pass clip_t only when requested AND if class supports it
            if clip_for_head is not None:
                try:
                    return HyperbolicHead(**kwargs, clip_t=clip_for_head)
                except TypeError:
                    pass  # class doesn't support clip_t -> fallback
            return HyperbolicHead(**kwargs)

    if use_pos:
        def pos_factory(N, D, drop):
            try:
                return HyperbolicPositionalEmbedding(
                    num_tokens=N, dim=D, dropout=drop, init_c=cfg.hyp_init_c, clip_t=cfg.t_pos
                )
            except TypeError:
                return HyperbolicPositionalEmbedding(
                    num_tokens=N, dim=D, dropout=drop, init_c=cfg.hyp_init_c
                )

    if use_mlp:
        def mlp_factory(D, ratio, drop):
            return HyperbolicFeedForward(
                dim=D, mlp_ratio=ratio, drop=drop,
                init_c=cfg.hyp_init_c, clip_t=cfg.t_default, hamp=cfg.hamp
            )

    if use_attn:
        shared = SharedHyperbolicCentroids(
            heads=cfg.num_heads,
            M=cfg.num_classes,
            head_dim=cfg.embed_dim // cfg.num_heads,
            proto_std=cfg.proto_std,
        )
        def attn_factory(D, H, attn_drop, proj_drop):
            return HyperbolicSelfAttention(
                dim=D, heads=H,
                attn_drop=attn_drop, proj_drop=proj_drop,
                shared=shared,
                init_c=cfg.hyp_init_c, init_sigma=cfg.hyp_init_s,
                clip_t=cfg.t_default, eps=1e-6, hamp=cfg.hamp
            )

    return head_factory, pos_factory, mlp_factory, attn_factory


# ============================================================
# Build model from factories
# ============================================================

def build_model(cfg: Config, variant: str) -> nn.Module:
    hf, pf, mf, af = make_factories(cfg, variant)
    kwargs = {}
    if hf is not None: kwargs["head_factory"] = hf
    if pf is not None: kwargs["pos_factory"]  = pf
    if mf is not None: kwargs["mlp_factory"]  = mf
    if af is not None: kwargs["attn_factory"] = af
    # residual_factory: usa i default euclidei finché non avrai il modulo hyper-residual
    model = CifarViT(cfg, **kwargs)
    return model



# ============================================================
# Training & Evaluation
# ============================================================
@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)) -> list:  # topk: best k indices, logits is (B,num_classes)
    maxk = max(topk)
    B = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # pred has shape (B, maxk)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # boolean (maxk, B): topk logit (=pred) == targets
    accs = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accs.append((correct_k * (100.0 / B)).item())
    return accs

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    loss_sum = n = 0
    top1_sum = 0.0
    top5_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = crit(logits, yb)
            B = xb.size(0)
            loss_sum += loss.item() * B
            n += B
            top1, top5 = accuracy(logits, yb, topk=(1, 5))
            top1_sum += top1 * B
            top5_sum += top5 * B
    return loss_sum / n, top1_sum / n, top5_sum / n

# ============================================================
# Hyper training utilities (two optimizers like your scripts)
# ============================================================

def set_hamp_flag(module: nn.Module, flag: bool):
    for m in module.modules():
        if hasattr(m, "hamp"):
            m.hamp = bool(flag)

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_attention_only(model: nn.Module):
    freeze_all(model)
    # optional shared centroids attr name could differ; guard it:
    if hasattr(model, "shared_centroids_"):
        for p in model.shared_centroids_.parameters():
            p.requires_grad = True
    from hypervit.models.h_attn import HyperbolicSelfAttention
    for m in model.modules():
        if isinstance(m, HyperbolicSelfAttention):
            for p in m.parameters():
                p.requires_grad = True

def built_opt(model: nn.Module, cfg: Config):
    manifold_params = []
    eucl_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, geoopt.ManifoldParameter):
            manifold_params.append(p)
        else:
            eucl_params.append(p)

    opt_euc = torch.optim.AdamW(eucl_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_man = geoopt.optim.RiemannianAdam(
        [{"params": manifold_params, "weight_decay": 0.0}],
        lr=cfg.lr * 0.25
    )
    return opt_euc, opt_man


# ============================================================
# Train loops (exactly your style, with accumulation support)
# ============================================================

def train_euclid(cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler(device="cuda", enabled=cfg.amp)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_acc = 0.0  # acc1
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "w_vit_euclidean.pth")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0

        start_t = time.perf_counter()
        seen_imgs = 0

        for it, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=cfg.amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            seen_imgs += xb.size(0)

            if it % 100 == 0:
                top1 = accuracy(logits, yb, topk=(1,))[0]
                curr_lr = optimizer.param_groups[0]["lr"]
                print(f"Ep {ep:03d} | it {it:04d} | loss {run_loss/100:.4f} | acc1 {top1:5.2f} | lr {curr_lr:.2e}")
                run_loss = 0.0

        epoch_sec = max(1e-9, time.perf_counter() - start_t)
        imgs_per_sec = seen_imgs / epoch_sec

        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device)
        print(f"[VAL] Ep {ep:03d} | loss {val_loss:.4f} | acc1 {val_acc1:5.2f} | acc5 {val_acc5:5.2f} | "
              f"time {epoch_sec:.2f}s | img/s {imgs_per_sec:.1f}")

        improved = (val_loss < best_val) or (val_acc1 > best_acc)
        if improved:
            best_val = min(best_val, val_loss)
            best_acc = max(best_acc, val_acc1)
            torch.save(model.state_dict(), best_path)

    print(f"Best saved to: {best_path}")
    return best_path


def train_hyper(cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                opt_euc: torch.optim.Optimizer, opt_man: geoopt.optim.RiemannianAdam):
    model.to(device)

    # accumulation (exactly like your attn2)
    if cfg.eff_batch_size is None or cfg.eff_batch_size <= cfg.batch_size:
        accum_steps = 1
    else:
        assert cfg.eff_batch_size % cfg.batch_size == 0, "eff_batch_size must be multiple of batch_size"
        accum_steps = cfg.eff_batch_size // cfg.batch_size

    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = cfg.epochs * updates_per_epoch

    scheduler_euc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_euc, T_max=total_steps)
    scheduler_man = torch.optim.lr_scheduler.CosineAnnealingLR(opt_man, T_max=total_steps)

    euc_params = [p for g in opt_euc.param_groups for p in g["params"]]
    man_params = [p for g in opt_man.param_groups for p in g["params"]]

    scaler = GradScaler(device="cuda", enabled=True)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_acc = 0.0  # acc1
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "h_hyper_best.pth")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0

        start_t = time.perf_counter()
        seen_imgs = 0

        for it, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with autocast('cuda', enabled=cfg.amp):
                logits = model(xb)
                loss = criterion(logits.float(), yb)
                loss_for_backward = loss / accum_steps

            if not torch.isfinite(loss):
                print(f"[NaN] skip batch ep={ep} it={it}")
                continue

            scaler.scale(loss_for_backward).backward()

            run_loss += loss.item()
            seen_imgs += xb.size(0)

            do_step = (it % accum_steps == 0) or (it == len(train_loader))
            if do_step:
                if cfg.grad_clip is not None:
                    scaler.unscale_(opt_euc)
                    nn.utils.clip_grad_norm_(euc_params, cfg.grad_clip)
                if cfg.man_grad_clip is not None:
                    scaler.unscale_(opt_man)
                    nn.utils.clip_grad_norm_(man_params, cfg.man_grad_clip)

                scaler.step(opt_euc)
                scaler.step(opt_man)
                scaler.update()
                scheduler_euc.step()
                scheduler_man.step()
                opt_euc.zero_grad(set_to_none=True)
                opt_man.zero_grad(set_to_none=True)

            if it % 100 == 0:
                top1 = accuracy(logits, yb, topk=(1,))[0]
                curr_lr = opt_euc.param_groups[0]["lr"]
                print(f"Ep {ep:03d} | it {it:04d} | loss {run_loss/100:.4f} | acc1 {top1:5.2f} | lr {curr_lr:.2e}")
                run_loss = 0.0

        epoch_sec = max(1e-9, time.perf_counter() - start_t)
        imgs_per_sec = seen_imgs / epoch_sec

        # validate every 5 epochs (like your attn2)
        if ep % 5 == 0:
            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device)
            print(f"[VAL] Ep {ep:03d} | loss {val_loss:.4f} | acc1 {val_acc1:5.2f} | acc5 {val_acc5:5.2f} | "
                  f"time {epoch_sec:.2f}s | img/s {imgs_per_sec:.1f}")

            improved = (val_loss < best_val) or (val_acc1 > best_acc)
            if improved:
                best_val = min(best_val, val_loss)
                best_acc = max(best_acc, val_acc1)
                torch.save(model.state_dict(), best_path)

    print(f"Best saved to: {best_path}")
    return best_path


# ============================================================
# Main (English comments, coherent with your workflow)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="euclid",
                        choices=["euclid", "hyp-head", "hyp-pos", "hyp-mlp", "hyp-attn", "hyp-all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eff_batch_size", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hamp", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)   # optional resume/init weights
    args = parser.parse_args()

    # 1) build config and apply CLI overrides
    parser.add_argument("-c","--config", type=str)
    cfg = Config()
    if args.config:
    with open("configs/base.yaml","r") as f: base = yaml.safe_load(f) or {}
    with open(args.config,"r") as f: over = yaml.safe_load(f) or {}
    merged = {**base, **over}
    for k, v in merged.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
            
    if args.epochs is not None:        cfg.epochs = args.epochs
    if args.lr is not None:            cfg.lr = args.lr
    if args.batch_size is not None:    cfg.batch_size = args.batch_size
    if args.eff_batch_size is not None: cfg.eff_batch_size = args.eff_batch_size
    if args.amp:                       cfg.amp = True
    if args.hamp:                      cfg.hamp = True

    # 2) seed, device, output dir
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 3) data
    train_loader, val_loader = get_cifar100_loaders(cfg)

    # 4) model (factories reflect the chosen variant)
    model = build_model(cfg, args.variant).to(device)

    # 5) optional: load checkpoint weights for fine-tuning or resume
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        # be strict by default for your flows
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=True)
            print(f"=> Loaded state['model'] from {args.ckpt}")
        else:
            model.load_state_dict(state, strict=True)
            print(f"=> Loaded state_dict from {args.ckpt}")

    # 6) train according to variant
    if args.variant == "euclid":
        best_path = train_euclid(cfg, model, train_loader, val_loader, device)
    else:
        # hyper training uses two optimizers (Euclidean + Riemannian), like your scripts
        opt_euc, opt_man = built_opt(model, cfg)
        best_path = train_hyper(cfg, model, train_loader, val_loader, device, opt_euc, opt_man)

    print(f"[DONE] Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
