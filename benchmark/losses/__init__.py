from .contrastive_loss import ContrastiveLoss


def build_loss(cfg):
    if cfg.LOSS.NAME == "contrastive_loss":
        return ContrastiveLoss(cfg)
    else:
        raise ValueError()
