import torch
import torch.nn as nn
from torch.amp import autocast
import geoopt

from hypervit.utils.manifold import inv_softplus, pre_clip, post_clip

# HyperbolicLinear
class HyperbolicLinear(nn.Module):
    """
    using y = p_out ⊕ mobius_matvec(W, (-p_in) ⊕ x) ⊕ exp0(b)
    """
    def __init__(self, hamp: bool, in_features: int, out_features: int, init_c: float = 1.0):
        super().__init__()
        self.hlball = geoopt.PoincareBall(c=init_c, learnable=False)   # to avoid a "false learnable"
        self.curv_hlin = nn.Parameter(inv_softplus(init_c))            # recorded parameter (Euclidean)
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.p_in  = geoopt.ManifoldParameter(torch.zeros(1, 1, in_features), manifold=self.hlball, requires_grad=True)   # domain's center 
        self.p_out = geoopt.ManifoldParameter(torch.zeros(1, 1, out_features), manifold=self.hlball, requires_grad=True)  # codomain's center 
        self.hamp = bool(hamp)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.float()
        with autocast("cuda", enabled=self.hamp):
            self.hlball.isp_c = self.curv_hlin
            c = self.hlball.c
            hx = self.hlball.expmap0(pre_clip(x, c))
            p_in  = self.hlball.projx(self.p_in).to(hx.dtype) # domain's center on manifold
            p_out = self.hlball.projx(self.p_out).to(hx.dtype) # codomain's center on manifold
            # y = p_out + mobius_matvec(W, (-p_in) + x) + exp0(b)
            x_p = self.hlball.mobius_add(-p_in, hx)                           # (-p_in) + x
            h   = post_clip(self.hlball.mobius_matvec(self.lin.weight, x_p),c)            # W * x_p
            b_h = self.hlball.expmap0(pre_clip(self.lin.bias, c))[None, None, :] # exp0(b), (1,1,D0) for broadcasting
            h = self.hlball.mobius_add(h, b_h)                           # (… ) + exp0(b)
            h = self.hlball.mobius_add(p_out, h)                              # p_out + (…)
            h = post_clip(h, c)
            e = self.hlball.logmap0(h)                                       # to euclidean
        return e.to(dtype=out_dtype)

# FeedForward (MLP with 2 layer)
class HyperbolicFeedForward(nn.Module):
    """x -> HyperbolicLinear(dim -> dim*mlp_ratio, centered) -> GELU (euclidean)
    -> Dropout(drop_rate) -> HyperbolicLinear(dim*mlp_ratio -> dim, centered) -> Dropout(drop_rate)"""
    def __init__(self, hamp: bool, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = HyperbolicLinear(hamp, dim, hidden)
        self.act = nn.GELU()
        self.fc2 = HyperbolicLinear(hamp, hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
