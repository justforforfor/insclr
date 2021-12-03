import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


def build_pool(cfg):
    output_size = cfg.MODEL.POOL.OUTPUT_SIZE
    if cfg.MODEL.POOL.NAME == "max":
        return torch.nn.AdaptiveMaxPool2d(output_size)
    elif cfg.MODEL.POOL.NAME == "avg":
        return torch.nn.AdaptiveAvgPool2d(output_size)
    elif cfg.MODEL.POOL.NAME == "gem":
        p =  cfg.MODEL.POOL.GMP.P
        if p > 0:
            return GeneralizedMeanPool(output_size, p)
        else:
            return GeneralizedMeanPoolP(output_size, 3.0)
    elif cfg.MODEL.POOL.NAME == "identity":
        return torch.nn.Identity()
    elif cfg.MODEL.POOL.NAME == "attention":
        return AttentionVisualizer()
    else:
        raise ValueError(f"unsupported pooling layer: {cfg.MODEL.POOL.NAME}")


class GeneralizedMeanPool(Module):
    def __init__(self, output_size=1, p=1.0, eps=1e-6):
        super(GeneralizedMeanPool, self).__init__()
        assert p > 0
        self.output_size = output_size
        self.p = float(p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __str__(self):
        return f"GeneralizedMeanPool(p={self.p})"
    
    def __repr__(self):
        return self.__str__()


class GeneralizedMeanPoolP(GeneralizedMeanPool):
    def __init__(self, output_size=1, p=3.0, eps=1e-6):
        super(GeneralizedMeanPoolP, self).__init__(output_size, p, eps)
        self.p = Parameter(torch.ones(1) * p)


class AttentionVisualizer(Module):
    def __init__(self):
        super(AttentionVisualizer, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=False)
