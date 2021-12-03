import os
import argparse
import copy

import torch
import numpy as np
from tqdm import tqdm

from benchmark.config import cfg
from benchmark.data import build_dataloader
from benchmark.models import build_model
from benchmark.utils.memory_bank import MemoryBank
from benchmark.utils.extract_fn import extract_features
from benchmark.utils.logger import setup_logger

OUTPUT_DIR = "training_dir/neighbor"
SCALES = [1, (1/2)**0.5, 1/2]


def run(task, dataset, name):
    logger = setup_logger("")
    os.makedirs(os.path.join(OUTPUT_DIR, dataset), exist_ok=True)
    if task == "feature":
        logger.info("begin to extract features")
        device = torch.device(cfg.DEVICE)
        model = build_model(cfg, logger)
        if cfg.MODEL.PRETRAIN == "imagenet":
            logger.info("convert head to identity")
            model.combiner.head = torch.nn.Identity()
        model.to(device)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            model = torch.nn.DataParallel(model)

        dataloader = build_dataloader(cfg, dataset, mode="cpool")
        targets = copy.deepcopy(dataloader.dataset.labels)
        targets = np.array(targets)
        features = extract_features(dataloader, model, device, scales=SCALES)
        features = features.squeeze()
        print(features.shape)
        output = os.path.join(OUTPUT_DIR, dataset, name + "_feature.npz")
        logger.info(f"save features to {output}")
        np.savez(output, feature=features, target=targets)
    else:
        feature_file = os.path.join(OUTPUT_DIR, dataset, name + "_feature.npz")
        logger.info(f"generate neighbors from {feature_file}")
        data = np.load(feature_file)
        features = data["feature"]
        targets = data["target"]
        memory_bank = MemoryBank(len(targets), features.shape[1], None, None)
        memory_bank.update(features, targets, np.arange(len(targets)))
        # avoid OOM
        num_chunks = 12
        ranges = np.linspace(0, len(targets), num=num_chunks + 1, dtype=np.int32)
        
        neighbors = []
        similarities = []

        for i in tqdm(range(num_chunks)):
            mined_features = features[ranges[i]:ranges[i + 1]]
            _neighbors = memory_bank.mine_nearest_neighbors(topk=500, features=mined_features)
            neighbors.append(_neighbors)

            for j in range(len(mined_features)):
                single_feature = mined_features[j]
                its_neighbors = features[_neighbors[j]]
                similarities.append(np.dot(single_feature, its_neighbors.T))

        neighbors = np.concatenate(neighbors, axis=0)
        similarities = np.stack(similarities, axis=0)

        logger.info(f"neighbors shape: {neighbors.shape}")
        logger.info(f"similarity shape: {similarities.shape}")
        output = os.path.join(OUTPUT_DIR, dataset, name + "_neighbor.npz")
        logger.info(f"save neighbors to {output}")
        np.savez(output, neighbor=neighbors, similarity=similarities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    if args.cfg is not None:
        print(f"merge config: {args.cfg}")
        cfg.merge_from_file(args.cfg)
    if args.pretrain is not None:
        cfg.MODEL.PRETRAIN = args.pretrain

    # check
    if args.task == "feature":
        assert not os.path.exists(os.path.join(OUTPUT_DIR, args.dataset.lower(), args.name + "_feature.npz"))
    elif args.task == "neighbor":
        assert os.path.exists(os.path.join(OUTPUT_DIR, args.dataset.lower(), args.name + "_feature.npz")) and \
            not os.path.exists(os.path.join(OUTPUT_DIR, args.dataset.lower(), args.name + "_neighbor.npz"))
    else:
        raise ValueError(f"undefined task: {args.task}")
    
    run(args.task, args.dataset.lower(), args.name)
