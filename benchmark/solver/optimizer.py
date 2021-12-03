import torch


def build_optimizer(cfg, model, loss_fn=None, logger=None, detail=False, exclude=None):
    params = []
    details = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if exclude is not None and key in exclude:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        details.append([key, f"lr: {lr}", f"weigh decay: {weight_decay:.0e}"])

    if loss_fn is not None:
        for key, value in loss_fn.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.LOSS_BASE_LR
            weight_decay = cfg.SOLVER.LOSS_WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.LOSS_BASE_LR * cfg.SOLVER.LOSS_BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.LOSS_WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            details.append([key, f"lr: {lr}", f"weight decay: {weight_decay:.0e}"])
    
    if detail:
        for line in details:
            logger.info(", ".join(line))
    if cfg.SOLVER.NAME == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.NAME == "adam":
        optimizer = torch.optim.Adam(params, lr)
    else:
        raise ValueError()
    
    return optimizer
