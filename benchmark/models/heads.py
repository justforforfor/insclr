import torch
import torch.nn as nn
import torch.nn.functional as F


def build_head(cfg):
    if cfg.MODEL.HEAD.NAME == "linear_norm":
        return LinearNorm(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.OUT_DIM)
    elif cfg.MODEL.HEAD.NAME == "norm_linear_norm":
        return NormLinearNorm(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.OUT_DIM)
    elif cfg.MODEL.HEAD.NAME == "norm":
        return Norm()
    elif cfg.MODEL.HEAD.NAME == "identity":
        return nn.Identity()
    else:
        raise ValueError()


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.normalize(self.fc(x))
        return x


class NormLinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinearNorm, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.normalize(self.fc(F.normalize(x)))
        return x


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return F.normalize(x, p=2, dim=1)
