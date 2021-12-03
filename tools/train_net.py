import os
import random
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from benchmark.config import cfg
from benchmark.data import build_dataloader
from benchmark.models import build_model
from benchmark.losses import build_loss
from benchmark.solver import build_lr_scheduler, build_optimizer
from benchmark.evaluation import ROxford5kEvaluator, RParis6kEvaluator, InstreEvaluator
from benchmark.engine.trainer import do_train
from benchmark.utils.checkpointer import Checkpointer
from benchmark.utils.logger import setup_logger


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


def train():
    check(cfg)
    rename_output_dir(cfg)
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir)
    logger = setup_logger("insclr", output_dir)

    trainloader = build_dataloader(cfg, cfg.DATA.DATASET, mode="train")
    device = torch.device(cfg.DEVICE)
    model = build_model(cfg, logger)
    logger.info(model)
    model.to(device)
    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
        model = torch.nn.DataParallel(model)
    
    loss_fn = build_loss(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    logger.info(cfg)
    writer = SummaryWriter(output_dir)
    checkpointer = Checkpointer(model, output_dir, True, None, None, logger)

    # evaluator
    evaluators = []
    if cfg.DATA.DATASET == "gldv2":
        evaluators.append(ROxford5kEvaluator("datasets/roxford5k", 1024, cfg.TEST.SCALES))
        evaluators.append(RParis6kEvaluator("datasets/rparis6k", 1024, cfg.TEST.SCALES))
    elif cfg.DATA.DATASET == "instre":
        evaluators.append(InstreEvaluator("datasets/instre", 256, cfg.TEST.SCALES))
    else:
        raise ValueError(f"unsupported dataset: {cfg.DATA.DATASET}")

    for evaluator in evaluators:
        logger.info(evaluator)

    do_train(cfg, trainloader, model, loss_fn, optimizer, scheduler,
             evaluators, checkpointer, writer, device, logger)


def rename_output_dir(cfg):
    def _fint(number):
        if number % 1000 == 0:
            return f"{number // 1000}k"
        else:
            return f"{number}"

    # general
    cfg.OUTPUT_DIR += f"bs{cfg.SOLVER.IMS_PER_BATCH * (cfg.NEIGHBOR.K + 1)}_"
    cfg.OUTPUT_DIR += f"k{cfg.NEIGHBOR.K}_"
    cfg.OUTPUT_DIR += f"imgsize{cfg.INPUT.TRAIN_IMG_SIZE[0]}_"
    cfg.OUTPUT_DIR += f"sl{cfg.INPUT.SCALE_LOWER}_"
    cfg.OUTPUT_DIR += f"su{cfg.INPUT.SCALE_UPPER}_"
    cfg.OUTPUT_DIR += f"{cfg.SOLVER.NAME}_"
    cfg.OUTPUT_DIR += f"lr{cfg.SOLVER.BASE_LR:.0e}_"
    if cfg.SOLVER.GRAD_MAX_NORM < 99999:
        cfg.OUTPUT_DIR += f"gradn{cfg.SOLVER.GRAD_MAX_NORM}_"
    cfg.OUTPUT_DIR += f"maxstep{_fint(cfg.SOLVER.MAX_STEP)}_"

    if cfg.NOAUG.ENABLE:
        cfg.OUTPUT_DIR += f"noaug{len(cfg.NOAUG.SCALES)}s_"

    if cfg.SELECTION.ENABLE:
        cfg.OUTPUT_DIR += "select_"
        cfg.OUTPUT_DIR += f"noaug{cfg.SELECTION.USE_NOAUG_SIMILARITY}_"
        cfg.OUTPUT_DIR += f"th{cfg.SELECTION.THRESH}_"
        cfg.OUTPUT_DIR += f"at{cfg.SELECTION.ALWAYS_TRUE}_"
        cfg.OUTPUT_DIR += f"type_{cfg.SELECTION.TYPE}_"

    if cfg.XBM.ENABLE:
        cfg.OUTPUT_DIR += f"xbm{cfg.XBM.VERSION}_"
        cfg.OUTPUT_DIR += f"s{_fint(cfg.XBM.START_STEP)}_"
        cfg.OUTPUT_DIR += f"g{_fint(cfg.XBM.GET_STEP)}_"
        cfg.OUTPUT_DIR += f"gs{_fint(cfg.XBM.GET_SIZE)}_"
    
        if cfg.XBM.VERSION == "v2":
            cfg.OUTPUT_DIR += f"pos{cfg.XBM.POSITIVE_THRESH}_"
            cfg.OUTPUT_DIR += f"neg{cfg.XBM.NEGATIVE_THRESH}_"
            cfg.OUTPUT_DIR += f"mode_{cfg.XBM.CANDIDATE_MODE}_"

    if cfg.OUTPUT_DIR[-1] == "_":
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR[:-1]


def check(cfg):
    # check dependency
    if cfg.SELECTION.ENABLE and cfg.SELECTION.USE_NOAUG_SIMILARITY:
        assert cfg.NOAUG.ENABLE
    
    if cfg.XBM.ENABLE and cfg.XBM.VERSION == "v2":
        assert cfg.NOAUG.ENABLE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    train()
