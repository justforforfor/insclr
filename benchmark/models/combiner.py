import torch.nn as nn

from .attentions import SpatialAttention
from .heads import NormLinearNorm

from .pools import build_pool
from .heads import build_head


class Combiner(nn.Module):
    def __init__(self, cfg):
        super(Combiner, self).__init__()
        self.mode = "train"
        self.enable_head = cfg.MODEL.HEAD.ENABLE
        self.enable_attention = cfg.ATTENTION.ENABLE
        assert self.enable_head or self.enable_attention
        if self.enable_head:
            # for dim=2048 branch
            self.pool = build_pool(cfg)
            self.head = build_head(cfg)
        if self.enable_attention:
            # for dim=128 branch
            self.spatial_attention = SpatialAttention(2048, 128, False, "softplus")
            self.avg = nn.AdaptiveAvgPool2d(output_size=1)
            self.attention_head = NormLinearNorm(128, 128)

    def forward(self, x):
        if self.mode == "train":
            features1, features2 = None, None
            if self.enable_head:
                features1 = self.head(self.pool(x))
            if self.enable_attention:
                features2 = self.attention_head(self.avg(self.spatial_attention(x)))
            return features1, features2
        else:
            global_features, local_features, attentions = None, None, None
            if self.enable_head:
                global_features = self.head(self.pool(x))
            # test mode
            if self.enable_attention:
                local_features, attentions = self.spatial_attention(x)
            return global_features, local_features, attentions

    def train(self, bool):
        if bool:
            self.mode = "train"
        else:
            self.mode = "test"
        
        for child in self.children():
            child.train(bool)
