import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import geoopt
import geoopt.manifolds.stereographic.math as pmath

from hypervit.utils.manifold import inv_softplus, pre_clip, post_clip

class SharedHyperbolicCentroids(nn.Module):
    """
    Global centroids (shared between blocks/heads)
    - centroids: (H, M, Hd) in Euclidean (we map them to the ball in the forward)
    """
    def __init__(self, heads: int, M: int, head_dim: int, proto_std: float = 2e-2):
        super().__init__()
        self.centroids = nn.Parameter(torch.zeros(heads, M, head_dim)) 
        nn.init.normal_(self.centroids, std=proto_std)


# HyperbolicSelfAttention (HMHA) 
class HyperbolicSelfAttention(nn.Module):
    """
    Hyperbolic attention with shared global centroids:
    - Score: S = - d_H(q, K_aug)^2 / (2 * sigma_h^2)
    - Weights: softmax(S)
    - Weighted midpoint: Riemannian weighted midpoint, then we return to Euclidean.
    """
    def __init__(
        self, hamp: bool, dim: int, heads: int = 3, attn_drop: float = 0.0, proj_drop: float = 0.0, shared: SharedHyperbolicCentroids = None,
          eps : float = 1e-6, init_c : float = 1.0, init_sigma: float = 1.0
    ):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = HyperbolicLinear(hamp, dim, dim)
        self.k_proj = HyperbolicLinear(hamp, dim, dim)
        self.v_proj = HyperbolicLinear(hamp, dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = HyperbolicLinear(hamp, dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.eps = float(eps)
        self.shared = shared
        self.attnball = geoopt.PoincareBall(c=init_c, learnable=False)
        self.curv_attn = nn.Parameter(inv_softplus(init_c))  
        self.sigma_raw = nn.Parameter(inv_softplus(init_sigma) * torch.ones(heads))
        self.hamp = bool(hamp)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,Hd)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).reshape(B, N, H * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with autocast("cuda", enabled=self.hamp):
            q = self._split_heads(self.q_proj(x)).to(dtype=out_dtype)
            k = self._split_heads(self.k_proj(x)).to(dtype=q.dtype)
            v = self._split_heads(self.v_proj(x)).to(dtype=q.dtype)
            B, H, N, Hd = q.shape
            C = self.shared.centroids.to(dtype=q.dtype)  # (H, M, Hd)
            C_b = C.unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,M,Hd)
            K_aug = torch.cat([k, C_b], dim=2)  # (B,H,N+M,Hd)
            V_aug = torch.cat([v, C_b], dim=2)  # (B,H,N+M,Hd)
            self.attnball.isp_c = self.curv_attn
            c_attn = self.attnball.c.to(dtype=out_dtype)
            q_h = post_clip(self.attnball.expmap0(pre_clip(q,     c_attn)), c_attn).to(dtype=q.dtype)  # (B,H,N,Hd)
            K_h = post_clip(self.attnball.expmap0(pre_clip(K_aug, c_attn)), c_attn).to(dtype=q.dtype)  # (B,H,N+M,Hd), J =N+M
            V_h = post_clip(self.attnball.expmap0(pre_clip(V_aug, c_attn)), c_attn).to(dtype=q.dtype)  # (B,H,N+M,Hd)
            d_attn = self.attnball.dist(q_h.unsqueeze(3), K_h.unsqueeze(2)).to(dtype=q.dtype)  
            sigma = (F.softplus(self.sigma_raw) + self.eps).reshape(1, H, 1, 1).to(dtype=q.dtype)  # (1,H,1,1)
            S = (- (d_attn ** 2) / (2.0 * sigma ** 2)).to(dtype=q.dtype)  # (B,H,N,N+M)
            W = torch.softmax(S, dim=-1)                            # (B,H,N,N+M) = (B,H,N,J)
            W = self.attn_drop(W).to(dtype=q.dtype)
            # Weighted midpoint (Einstein midpoint) on the ball
            B, H, N, J = W.shape
            Hd = V_h.size(-1)
            xs = V_h[:, :, None, :, :].expand(-1, -1, N, -1, -1).to(dtype=q.dtype) # (B,H,N,J,Hd)
            O_h = pmath.weighted_midpoint(
                xs=xs,                                  # (B,H,N,J,Hd)
                weights=W,                              # (B,H,N,J)
                k=torch.as_tensor(-c_attn, dtype=xs.dtype, device=xs.device), # curvature
                reducedim=[-2],                         # weighted midpoint on the j axis
                dim=-1,                                 # the last dimension is the manifold dimension (Hd)
                keepdim=False,                          # (B, H, N, Hd)
                posweight=False                       
            )                                            
            O_euc = self.attnball.logmap0(O_h) # (B,H,N,Hd)
        out = self._merge_heads(O_euc).to(dtype=out_dtype)          # (B,N,D)
        out = self.out(out)
        out = self.proj_drop(out)
        return out
      
