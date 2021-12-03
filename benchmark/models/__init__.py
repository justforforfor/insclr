from collections import OrderedDict

import torch
import torch.nn as nn

from .backbones import build_backbone
from .combiner import Combiner


MODEL_PATHS = {
    "resnet50": "model_zoo/resnet50-19c8e357.pth",
    "resnet101": "model_zoo/resnet101-5d3b4d8f.pth",
}


def build_model(cfg, logger, **kwargs):
    backbone = build_backbone(cfg, **kwargs)
    combiner = Combiner(cfg)
    model = nn.Sequential(OrderedDict([
            ("backbone", backbone),
            ("combiner", combiner)]))
    
    if cfg.MODEL.PRETRAIN == "imagenet":
        if logger is not None:
            logger.info(f"loading pretrained backbone from {MODEL_PATHS[cfg.MODEL.BACKBONE.NAME]}")
        state_dict = torch.load(MODEL_PATHS[cfg.MODEL.BACKBONE.NAME])
        result = model.backbone.load_state_dict(state_dict, strict=False)
        assert len(result.missing_keys) == 0 
        logger.warning(f"unexpected keys: {result.unexpected_keys}")
    else:
        if logger is not None:
            logger.info(f"loading pretrained backbone from {cfg.MODEL.PRETRAIN}")
        state_dict = torch.load(cfg.MODEL.PRETRAIN, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # remove module. prefix
        has_module_prefix = False
        for key in state_dict.keys():
            if "module." in key:
                has_module_prefix = True
            break
        if has_module_prefix:
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        result = model.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {result.missing_keys}")
        logger.info(f"unexpected keys: {result.unexpected_keys}")

    return model
