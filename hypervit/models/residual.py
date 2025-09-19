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
    x ⊕ (gamma ⊗_p y) with a learned center p and scale gamma.
    """
    def __init__(self, init_c: float = 1.0):
        super().__init__()
        self.hrball = geoopt.PoincareBall(c=init_c, learnable=False)  # to avoid a "false learnable"
        self.curv_hra = nn.Parameter(inv_softplus(init_c))           # recorded parameter (Euclidean)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0)) # x + gamma * y , but in the manifold
        self.gamma_scale = nn.Parameter(torch.tensor(0.0))  # g_max = 1 + softplus(...)
        self.scale_center = nn.Parameter(torch.tensor(0.0)) # center of add and mul, s = sigmoid(scale_center) in (0,1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.float(); y = y.float() 
        with autocast("cuda", enabled=False):
            self.hrball.isp_c = self.curv_hra
            ra_c = self.hrball.c
            hr_x = post_clip(self.hrball.expmap0(pre_clip(x, ra_c)),ra_c)
            hr_y = post_clip(self.hrball.expmap0(pre_clip(y, ra_c)),ra_c)
            s = torch.sigmoid(self.scale_center)                # (0,1)          
            g_max = 1.0 + F.softplus(self.gamma_scale) #  to avoid problem with scalar_mul
            gamma = (g_max * torch.tanh(self.gamma_raw)).to(dtype=hr_x.dtype)  # (-g_max, g_max)
            scalar_center_hrx = post_clip(self.hrball.mobius_scalar_mul(s,hr_x), ra_c)
            scalar_center_hry = post_clip(self.hrball.mobius_scalar_mul(1 - s, hr_y), ra_c)
            p = post_clip(self.hrball.mobius_add(scalar_center_hrx,scalar_center_hry),ra_c) # center of add and mul         
            x_p =  post_clip(self.hrball.mobius_add(-p, hr_x),ra_c) # using p as center
            y_p =  post_clip(self.hrball.mobius_add(-p, hr_y),ra_c) # using p as center
            y_s =  post_clip(self.hrball.mobius_scalar_mul(gamma, y_p),ra_c)
            hres_p =  post_clip(self.hrball.mobius_add(x_p, y_s),ra_c)
            hres =  self.hrball.mobius_add(p, hres_p)
            r = post_clip(hres, ra_c)
            r_x = self.hrball.logmap0(r)
        return r_x.to(dtype=x.dtype)
