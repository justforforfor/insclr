import torch
from torch import nn
import torch.nn.functional as F


def build_attention(cfg):
    if cfg.ATTENTION.NAME == "spatial_attention":
        return SpatialAttention(2048, cfg.ATTENTION.REDUCE_DIM, cfg.ATTENTION.REDUCE_RELU, cfg.ATTENTION.FUNC)
    elif cfg.ATTENTION.NAME == "feature_attention":
        return FeatureAttention(2048, cfg.ATTENTION.REDUCE_DIM)
    elif cfg.ATTENTION.NAME == "identity":
        return nn.Identity()
    else:
        raise ValueError()


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, reduce_dim, reduce_relu, func):
        super(SpatialAttention, self).__init__()
        self.mode = "train"
        self.reduce_relu = reduce_relu
        self.conv1 = nn.Conv2d(in_dim, 512, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        
        self.reduce = None
        if reduce_dim > 0:
            self.reduce = nn.Conv2d(in_dim, reduce_dim, 1, 1)

        if func == "softplus":
            self.fn = nn.Softplus(beta=1, threshold=20)
        elif func == "sigmoid":
            self.fn = nn.Sigmoid()
        else:
            raise ValueError()

    def forward(self, x):
        x = x.detach()
        scores = self.fn(self.conv2((self.relu(self.conv1(x)))))

        if self.reduce is not None:
            x = self.reduce(x)
        if self.reduce_relu:
            x = F.relu(x)

        if self.mode == "train":
            return x * scores
        else:
            return x, scores

    def train(self, bool):
        if bool:
            self.mode = "train"
        else:
            self.mode = "test"


class FeatureAttention(nn.Module):
    def __init__(self, in_dim, reduce_dim):
        super(FeatureAttention, self).__init__()
        self.mode = "train"
        self.reduce_dim = reduce_dim
        if self.reduce_dim > 0:
            self.reduce = nn.Sequential(
                nn.Conv2d(in_dim, reduce_dim, 1, 1, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU()
            )
        else:
            self.reduce = None

    def forward(self, x):
        if self.reduce is not None:
            x = self.reduce(x)

        if self.mode == "train":
            return x
        else:
            scores = torch.mean(torch.square(x), dim=1, keepdim=True)
            return x, scores

    def train(self, bool):
        if bool:
            self.mode = "train"
        else:
            self.mode = "test"
