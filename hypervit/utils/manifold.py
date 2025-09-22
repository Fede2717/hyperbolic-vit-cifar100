import torch

# MANIFOLD UTILS FUNCTIONS

def inv_softplus(y: float | torch.Tensor) -> torch.Tensor:
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
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
