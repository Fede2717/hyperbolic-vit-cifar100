import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import geoopt

#MANIFOLD UTILS FUNCTIONS
def inv_softplus(y: float) -> torch.Tensor:
    y = torch.tensor(float(y))
    return torch.log(torch.expm1(y))  # softplus^{-1}(y) = log(e^y - 1)

def pre_clip(x: torch.Tensor, c: torch.Tensor, t: float = 0.985, eps: float = 1e-6) -> torch.Tensor:
    x_max = torch.atanh(torch.tensor(t, dtype=x.dtype, device=x.device)) / (c.sqrt() + eps) #this is vmax, from ∥expmap0​(v)∥<=t*R  
    x_norm = x.norm(dim=-1, keepdim=True)
    x = x * (x_max / (x_norm + eps)).clamp(max=1.0)   
    return x

def post_clip(y: torch.Tensor, c: torch.Tensor, t: float = 0.985, eps: float = 1e-6) -> torch.Tensor:
        # clip post-expmap: ||y|| <= t * (1/sqrt(c))
        R = 1.0 / (c.sqrt() + eps)
        r_max = R * t
        n = y.norm(dim=-1, keepdim=True)
        y = y * (r_max / (n + eps)).clamp(max=1.0) # x⋅rmax/||y|| if ||y|| > rmax
        return y 

# HyperbolicPositionalEmbedding (learnable + dropout)
class HyperbolicPositionalEmbedding(nn.Module):
    """HyperbolicPE learnable"""
    def __init__(self, num_tokens: int, dim: int, dropout: float = 0.0, init_c: float = 1.0, clip_t: float = 0.985):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim)) # (1,N,D), N = num_tokens = tokens img + cls
        nn.init.trunc_normal_(self.pos, std=0.02) #like in a ViT, near the origin
        self.drop = nn.Dropout(dropout)
        self.pball = geoopt.PoincareBall(c=init_c, learnable=False)
        self.curv_pe = nn.Parameter(inv_softplus(init_c))           # recorded parameter (Euclidean)
        self.clip_t = float(clip_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", enabled=False):
            self.pball.isp_c = self.curv_pe
            c = self.pball.c
            x = pre_clip(x, c, t=self.clip_t)
            pos = pre_clip(self.pos, c, t=self.clip_t)
            h_x = self.pball.expmap0(x)
            hpos = self.pball.expmap0(pos)
            h_x = post_clip(h_x, c, t=self.clip_t)
            hpos = post_clip(hpos, c, t=self.clip_t)
            h_sum = self.pball.mobius_add(h_x, hpos)
            h_sum = post_clip(h_sum, c, t=self.clip_t)
            e_x = self.pball.logmap0(h_sum)
        return self.drop(e_x.to(dtype=x.dtype))
