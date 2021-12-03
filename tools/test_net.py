import os
import random
import argparse

import torch
import numpy as np

from benchmark.config import cfg
from benchmark.models import build_model
from benchmark.evaluation import ROxford5kEvaluator, RParis6kEvaluator, InstreEvaluator, GLDRetrievalEvaluator
from benchmark.utils.logger import setup_logger


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


def test():
    logger = setup_logger("insclr")
    device = torch.device(cfg.DEVICE)
    model = build_model(cfg, logger)
    logger.info(model)
    model.to(device)
    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
        model = torch.nn.DataParallel(model)

    logger.info(cfg)

    # evaluator
    evaluators = []
    # select evaluator according to your need
    # evaluators.append(GLDRetrievalEvaluator("datasets/google-landmark"))
    # evaluators.append(ROxford5kEvaluator("datasets/roxford5k", 1024, cfg.TEST.SCALES))
    # evaluators.append(RParis6kEvaluator("datasets/rparis6k", 1024, cfg.TEST.SCALES))
    # evaluators.append(InstreEvaluator("datasets/instre", 256, cfg.TEST.SCALES, pca=True))

    for evaluator in evaluators:
        logger.info(evaluator)

    for evaluator in evaluators:
        msg, _ = evaluator.evaluate(model, device)
        logger.info(msg)


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
    test()
