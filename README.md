ChatGPT ha detto:
1) Overview

Goal. Understand where hyperbolic geometry helps in a ViT-Tiny on CIFAR-100 by isolating each module’s contribution.

Key idea. Use a centered hyperbolic residual (learnable barycenter) plus safe Poincaré operations with pre/post clipping, while keeping the parameter count fixed versus the Euclidean baseline.

Takeaway. The centered hyperbolic residual is the main source of accuracy improvement at equal parameters. Other hyperbolic blocks can add gains with different compute trade-offs.

2) Method (short)

All hyperbolic modules operate on the Poincaré ball with positive curvature (learnable or fixed per module). For numerical stability:

Pre/Post clipping: limit norms before and after mapping to the ball using a margin t in (0, 1).

Two optimizers: Euclidean parameters use AdamW; manifold parameters use Geoopt’s RiemannianAdam.

hAMP option: mixed precision inside hyperbolic ops, compatible with standard AMP.

Modules

HyperbolicHead: class prototypes on the ball; logits from distances; learnable curvature and temperature; default t = 0.96.

HyperbolicPositionalEmbedding: learnable positional embeddings mapped to the ball and combined via Möbius addition; default t = 0.98.

HyperbolicResidualAdd (centered): residuals taken around a learnable center p to reduce distortion; main accuracy gain at equal params.

HyperbolicLinear + FeedForward: linear layers with separate centers; default t = 0.985.

HyperbolicSelfAttention: hyperbolic multi-head attention with shared centroids across heads; trained in two short phases for stability.

3) Training setup (repro)

Dataset: CIFAR-100, standard normalization; augmentations: random crop and horizontal flip.

Backbone: ViT-Tiny (12 blocks, 3 heads, dim 192, patch size 4).

Schedules: cosine LR; 100 epochs for most variants; short 10 + 10 schedule for hyperbolic attention.

Batching: effective batch size 128 via gradient accumulation where needed (e.g., MLP and attention).

Precision: AMP enabled by default; --hamp toggles mixed precision inside hyperbolic ops.

Clipping:

Euclidean: grad_clip = 1.0

Riemannian: man_grad_clip = 1.0 for manifold parameters

Margins t: head 0.96, positional 0.98, other hyperbolic modules 0.985.
When head and positional are both hyperbolic, the head uses t = 0.98 for consistency.

4) Results (summary)

Euclidean baseline (ViT-Tiny): 53.10 Top-1.

Centered hyperbolic residual: 57.39 Top-1 at equal parameter count (+4.3), with a moderate throughput drop (compute trade-off).

Other hyperbolic blocks (head, positional, linear/MLP, attention) provide additional gains with cost–accuracy trade-offs; attention is the most expensive and benefits from the short two-phase schedule.

5) Reproducing the ablations

Use the --variant switch to toggle modules:

euclid — pure baseline

hyp-head — hyperbolic head only (t = 0.96)

hyp-pos — positional embeddings only (t = 0.98)

hyp-mlp — linear/MLP only (t = 0.985; often with accumulation)

hyp-attn — self-attention only (two-phase, short schedule)

hyp-all — all hyperbolic blocks enabled

Each variant keeps the same model size unless noted. The training script routes Euclidean vs. manifold parameters to the right optimizer and applies Euclidean/Riemannian clipping.

6) Checkpoints & logs

Weights: do not commit .pth files to git. Upload them as GitHub Release assets and link them here.

Logs: training logs (e.g., in experiments/) are git-ignored; optionally attach a zip to a Release.

10) Limitations & compute

Single-GPU runs (e.g., 16 GB) with AMP; accumulation used to reach effective batch 128.

Hyperbolic ops are heavier and more sensitive to numerics; safe clipping and per-module curvature help but may still need careful scheduling.

LayerNorm remains Euclidean.

7) Roadmap

Add figures (residual schematic, curvature tracking, accuracy vs. images/sec).

Publish pretrained checkpoints and link from README.

Minimal unit tests (Euclidean-limit checks as curvature approaches zero).

Optional: Colab or Hugging Face Space demo.
