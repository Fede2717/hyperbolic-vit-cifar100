import torch
import torch.nn as nn

# ============================================================
# 2) Utils
# ============================================================

# DropPath (Stochastic Depth)
def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic Depth (per sample)."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) #per-sample mask
    rand = x.new_empty(shape).bernoulli_(keep)
    return x.div(keep) * rand #x.div(keep) to conserve E[x]

class DropoutResidual(nn.Module):
    """Layer for apply stochastic depth to a residual branch."""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.p = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)

# ===== default factories (euclidean) =====
def default_head_factory(embed_dim: int, num_classes: int):
    return nn.Linear(embed_dim, num_classes)

def default_pos_factory(num_tokens: int, dim: int, dropout: float):
    return PositionalEmbedding(num_tokens, dim, dropout)

def default_attn_factory(dim: int, heads: int, attn_drop: float, proj_drop: float):
    return SelfAttention(dim, heads=heads, attn_drop=attn_drop, proj_drop=proj_drop)

def default_mlp_factory(dim: int, mlp_ratio: float, drop: float):
    return FeedForward(dim, mlp_ratio=mlp_ratio, drop=drop)

def default_residual_factory():
    return ResidualAdd(learnable_scale=False, init_scale=1.0)

# ----------------- Euclidean building blocks (default) -----------------

# ResidualAdd 
class ResidualAdd(nn.Module):
    """Skip connection (x + alpha*y)"""
    def __init__(self, learnable_scale: bool = False, init_scale: float = 1.0):
        super().__init__()
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)), persistent=True) # saved in state_dict (persistent=False), not trainable 
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + self.scale * y


# Norm (LayerNorm)
class Norm(nn.Module):
    """LayerNorm always on euclidean"""
    def __init__(self, dim: int, eps: float = 1e-5): #eps to avoid numerical instability (especially with hyperbolic geom)
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


# FeedForward (MLP with 2 layer)
class FeedForward(nn.Module):
    """x -> Linear(dim -> dim * mlp_ratio) -> GELU ->
       -> Dropout(drop_rate)-> Linear(dim * mlp_ratio -> dim) -> Dropout(drop_rate)"""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# PatchEmbed (img -> embedded tokens)
class PatchEmbed(nn.Module):
    """(B,C,H,W) -> (B,N,D) with learnable conv with patch_size=kernel_size=stride"""
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.side = img_size // patch_size
        self.num_patches = self.side * self.side
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        x = self.proj(x)                  # (B,D,H/p,W/p), p is patch_size
        x = x.flatten(2).transpose(1, 2)  # (B,N,D), N  = token img
        return x


# PositionalEmbedding (learnable + dropout)
class PositionalEmbedding(nn.Module):
    """PE learnable"""
    def __init__(self, num_tokens: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim)) # (1,N,D), N = num_tokens = tokens img + cls
        nn.init.trunc_normal_(self.pos, std=0.02) #like in a ViT
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos
        return self.drop(x)


# SelfAttention (MHA) 
class SelfAttention(nn.Module):
    """Multi-Head Self-Attention (scaled dot-product)."""
    def __init__(
        self, dim: int, heads: int = 3, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,Hd)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).reshape(B, N, H * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) * self.scale # (B,H,N,N)
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                  # (B,H,N,Hd)
        out = self._merge_heads(out)                    # (B,N,D)
        out = self.out(out)
        out = self.proj_drop(out)
        return out


# TransformerBlock
class TransformerBlock(nn.Module):
    """Norm -> Attn -> Drop -> HResidual -> Norm -> HMLP -> Drop -> HResidual."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_res: float, attn_drop: float, proj_drop: float,
                 attn_factory=default_attn_factory, mlp_factory=default_mlp_factory, residual_factory=default_residual_factory):
        super().__init__()
        self.norm1 = Norm(dim)
        self.attn  = attn_factory(dim, heads, attn_drop, proj_drop)
        self.res1  = residual_factory()
        self.drop_res1 = DropoutResidual(drop_res)

        self.norm2 = Norm(dim)
        self.mlp   = mlp_factory(dim, mlp_ratio, proj_drop)
        self.res2  = residual_factory()
        self.drop_res2 = DropoutResidual(drop_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.norm1(x))
        y = self.drop_res1(y)
        x = self.res1(x, y)

        y = self.mlp(self.norm2(x))
        y = self.drop_res2(y)
        x = self.res2(x, y)
        return x

# ============================================================
# 3) Vision modules and init
# ============================================================

def init_vit_weights(m: nn.Module):
    # Linear
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Conv (PatchEmbed)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # LayerNorm
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)



class CifarViT(nn.Module):
    """ViT for CIFAR-100"""
    def __init__(self, cfg,
                 head_factory      = default_head_factory,
                 pos_factory       = default_pos_factory,
                 attn_factory      = default_attn_factory,
                 mlp_factory       = default_mlp_factory,
                 residual_factory  = default_residual_factory):
                   
        super().__init__()
        self.cfg = cfg

        # Patch embedding
        self.patch = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2

        # Cls token and Positional embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = pos_factory(1 + num_patches, cfg.embed_dim, cfg.drop_rate)

        # Blocks
        dpr = torch.linspace(0, cfg.drop_res, steps=cfg.num_blocks).tolist() #stochastic depth
        blocks = []
        for i in range(cfg.num_blocks):
            blocks.append(
                TransformerBlock(
                    dim=cfg.embed_dim,
                    heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_res=dpr[i],
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.drop_rate,
                    attn_factory=attn_factory,
                    mlp_factory=mlp_factory,
                    residual_factory=residual_factory,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Norm and Head
        self.norm = Norm(cfg.embed_dim)
        self.head = head_factory(cfg.embed_dim, cfg.num_classes)

        # Initialize weights in ViT style
        self.apply(init_vit_weights)

    # forward
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch(x)                       # (B,N,D)
        cls = self.cls.expand(B, -1, -1)        # (B,1,D)
        x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
        x = self.pos(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0, :]                          # return CLS (B,D)

    def forward_head(self, cls_out: torch.Tensor) -> torch.Tensor:
        return self.head(cls_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

  
