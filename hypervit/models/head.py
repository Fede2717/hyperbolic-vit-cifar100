import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import geoopt

from hypervit.utils.manifold import inv_softplus, pre_clip, post_clip

# HyperbolicHead
class HyperbolicHead(nn.Module):
    """
    Hyperbolic Head using cls_scale and centroids for classification
    """
    def __init__(self, hamp: bool, dim: int, num_classes: int, init_c: float = 1.0, hyp_init_s: float = 1.0, init_cls_scale: float=0.1,
                  eps: float = 1e-6, proto_std: float = 2e-2, clip_t: float: float = 0.985):
        super().__init__()
        self.sigma = nn.Parameter(inv_softplus(hyp_init_s), requires_grad=True) # param > 0
        self.eps = float(eps)
        self.cls_scale = nn.Parameter(inv_softplus(init_cls_scale), requires_grad=True)
        self.hball = geoopt.PoincareBall(c=init_c, learnable=False)  # to avoid a "false learnable"
        self.curv_rho = nn.Parameter(inv_softplus(init_c))           # recorded parameter (Euclidean)
        proto = torch.zeros(num_classes, dim)
        nn.init.normal_(proto, std=proto_std) # centroids initialized near the origin
        self.proto = geoopt.ManifoldParameter(proto, manifold=self.hball, requires_grad=True)
        self.bias  = nn.Parameter(torch.zeros(num_classes))
        self.hamp = bool(hamp)
        self.clip_t = float(clip_t)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", enabled=self.hamp):
            out_dtype = cls.dtype
            self.hball.isp_c = self.curv_rho
            c = self.hball.c
            s = F.softplus(self.cls_scale) + self.eps              # > 0
            cls32 = (cls * s).to(dtype=out_dtype)                      # to avoid big norm
            v = pre_clip(cls32, c, t=self.clip_t)
            hcls = self.hball.expmap0(v)                # [B, D]
            proto_ball = self.hball.projx(self.proto.to(dtype=out_dtype))       # [C, D]
            hcls = post_clip(hcls, c, t=self.clip_t)
            d = self.hball.dist(hcls.unsqueeze(1), proto_ball.unsqueeze(0))  # [B, C]
            sigma = (F.softplus(self.sigma) + self.eps).clamp(min=1e-2, max=10)        # sigma > 0
            logits = - (d ** 2) / (2.0 * sigma ** 2)     # better than -T*d
        return (logits + self.bias.to(logits.dtype)).to(dtype=out_dtype)
