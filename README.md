
# Hyperbolic ViT on CIFAR‑100

**Goal.** Turn a ViT‑Tiny baseline into a *hyperbolic* ViT on CIFAR‑100 and measure **where** non‑Euclidean geometry helps under a fixed parameter budget.  We keep the backbone and number of parameters constant and run a **progressive ablation**: each variant **adds** the next hyperbolic block on top of the previous ones (Head → Pos → Residual → MLP → Attention).  

**Main takeaway.** A **centered hyperbolic residual** (residual around a learnable barycenter) delivers the largest Top‑1 gain at equal params. Other hyperbolic blocks (head, positional, MLP, attention) show different accuracy/compute trade‑offs.

---

## 1. Dataset & Backbone (constant across ablations)
- **Dataset.** CIFAR‑100 with standard normalization; train: RandomCrop(32, padding=4) + HorizontalFlip; test: normalize only.
- **Backbone.** ViT‑Tiny: 12 blocks, 3 heads, embed dim 192, patch size 4. LayerNorm remains Euclidean.

## 2. Hyperbolic Modules (what we swap)
We operate on the **Poincaré ball**; each hyperbolic block has its own (fixed or learnable) curvature. We enforce **pre/post clipping** around exp/log maps and use AMP by default (optional *hAMP* inside manifold ops).

- **Residual (three modes).**
  - **No‑center (Hyp).** Mӧbius add around the origin: `x ⊕ f(x)`.
  - **Centered (Hyp).** Learn a barycenter `p`; compose via `exp_p( log_p(x) + log_p(f(x)) )`.
  - **Only‑residual (Hyp).** *Only* the residual is hyperbolic; all other blocks stay Euclidean.
- **Head (Hyp).** Class prototypes on the ball; distance‑based logits; learnable curvature/temperature.
- **Positional (Hyp).** Learnable positional vectors mapped to the ball and combined via Mӧbius addition.
- **MLP (Hyp).** Two **HyperbolicLinear** layers with centers `(p_in, p_out)` and GELU in between.
- **Attention (Hyp).** Hyperbolic MHA with shared centroids; trained in a short two‑phase schedule for stability.

## 3. Training Protocol (fixed)
- **Epochs.** 100 for all settings (best checkpoint may be saved earlier).
- **Batching.** Effective batch size 128 (gradient accumulation where needed).
- **Optimizers.** Euclidean params → AdamW; manifold params → Geoopt RiemannianAdam; separate grad clipping.
- **Precision.** AMP on; optional *hAMP* inside hyperbolic ops.
- **Clipping.** `t=0.985`.
 
## 4. Variants (what to run)
Ablations are **progressive** unless stated otherwise:

- `euclid` — pure baseline
- `hyp-head` → **Head**
- `hyp-pos` → **Head + Pos**
- `hyp-residual-nocenter` → **Head + Pos + Residual (no center)**
- `hyp-residual-centered` → **Head + Pos + Residual (centered)**
- `hyp-only-residual-centered` — **only the residual** is hyperbolic; everything else stays Euclidean
- `hyp-mlp` → **Head + Pos + Residual + MLP**
- `hyp-attn` → **Head + Pos + Residual + MLP + Attention**
- `hyp-all` — all hyperbolic blocks enabled

### Commands (use the provided .sh)
```bash
# Baseline
bash scripts/ablations/euclid.sh

# Head 
bash scripts/ablations/head.sh

# Positional
bash scripts/ablations/pos.sh

# Residual ablations
bash scripts/ablations/residual_nocenter.sh
bash scripts/ablations/residual_centered.sh
bash scripts/ablations/residual_only_centered.sh   # special branch: only residual is hyperbolic

# MLP (Hyperbolic Linear)
bash scripts/ablations/linear.sh

# Attention (two-phase schedule)
bash scripts/ablations/attn.sh
Evaluate a checkpoint:
```bash
python scripts/eval.py --variant hyp-residual-centered --ckpt experiments/residual_centered/h_residual_centered.pth
```

## 5. Results (CIFAR‑100, ViT‑Tiny, **100 epochs**)
(*This table includes your most recent runs; MLP/Attention rows are placeholders until we add the updated numbers*.)

| Setting                                  | Ep | Val Loss | Top‑1 | Top‑5 | Total time | imgs/s | pball | hball |
|------------------------------------------|:--:|:--------:|:-----:|:-----:|:----------:|:------:|:-----:|:-----:|
| **Euclidean baseline (reference)**       |100 |    —     | 53.10 |   —   |     —      |   —    |   —   |   —   |
| Residual **no center**                   |100 | 2.7884   | 55.37 | 81.55 | 4h28m      | 332.5  | 1.79  | 0.99  |
| Residual **with center**                 | 97 | 2.6093   | **57.39** | 82.99 | 8h05m      | 177.0  | 1.91  | 0.99  |
| **Only residual** (rest Euclid, centered)| 98 | 2.6113   | 57.37 | 82.94 | 7h30m      | 180.0  | 1.91  | 0.99  |
| Hyperbolic **positional** only           |100 | 3.5538   | 45.11 | 72.35 | 44.05s     | 1133.3 | 1.72  | 1.47  |
| Hyperbolic **head** only                 | 98 | 3.6272   | 50.16 | 76.46 | 40.74s     | 1225.3 |   —   | 1.54  |
| Hyperbolic **MLP** only                  |100 |    —     |   —   |   —   |     —      |   —    |   —   |   —   |
| Hyperbolic **Attention** only            |100 |    —     |   —   |   —   |     —      |   —    |   —   |   —   |

> Note: total wall‑clock time for **head** and **positional** is equal in our runs.

## 6. Repository layout
```
configs/                       # YAMLs (base + per-variant training configs)
hypervit/                      # library code (ViT backbone + hyperbolic modules)
  models/                      # head, pos, residual (center/no-center), MLP, attention
  utils/                       # manifold helpers (pre/post clipping, etc.)
scripts/                       # train/eval entrypoints + ablation shell scripts
tests/                         # tiny sanity tests (forward/eval/progressive build)
weights/                       # (empty) checkpoints are published in GitHub Releases
README.md
requirements.txt
```

## 7. Checkpoints & scripts
- **Weights** will be published in **GitHub Releases** (planned tag `v1.0.0`). Shell scripts accept `--ckpt <path>` and default to `weights/<name>.pth`.
- Make scripts executable:
  ```bash
  chmod +x scripts/*.sh scripts/ablations/*.sh
  ```

## 8. Repro notes
Single‑GPU with AMP; effective batch 128 (accumulation where needed). Hyperbolic ops are heavier and numerically sensitive; clipping margins `t` are tuned per block.
