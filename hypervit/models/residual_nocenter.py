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

# HyperbolicResidualAdd
class HyperbolicResidualAdd(nn.Module):
    """
     x ⊕ (gamma ⊗_p y) without a learned center p and scale gamma.
    """
    def __init__(self, init_c: float = 1.0):
        super().__init__()
        self.hrball = geoopt.PoincareBall(c=init_c, learnable=False)  # to avoid a "false learnable"
        self.curv_hra = nn.Parameter(inv_softplus(init_c))           # recorded parameter (Euclidean)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0)) # x + gamma * y , but in the manifold
        self.gamma_scale = nn.Parameter(torch.tensor(0.0))  # g_max = 1 + softplus(...)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            self.hrball.isp_c = self.curv_hra
            c = self.hrball.c
            x = x.float(); y = y.float()
    
            gmax = 1.0 + F.softplus(self.gamma_scale)
            gamma = gmax * torch.tanh(self.gamma_raw)
    
            hx = self.hrball.expmap0(pre_clip(x, c))
            hy = self.hrball.expmap0(pre_clip(y, c))
    
            v  = self.hrball.logmap(hx, hy)              # log base hx
            h  = self.hrball.expmap(hx, gamma * v)       # exp base hx
    
            out = self.hrball.logmap0(self.hrball.projx(h))
        return out.to(out_dtype)
      
