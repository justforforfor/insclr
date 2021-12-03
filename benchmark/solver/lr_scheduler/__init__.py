from .warmup_multi_step_lr import WarmupMultiStepLR


def build_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER_NAME == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_STEP,
            cfg.SOLVER.WARMUP_METHOD
        )
    else:
        raise ValueError(f"scheduler {cfg.SOLVER.SCHEDULER_NAME} is not defined")
