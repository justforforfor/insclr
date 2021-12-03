from .resnet import *


def build_backbone(cfg, **kwargs):
    return globals()[cfg.MODEL.BACKBONE.NAME](**kwargs)
