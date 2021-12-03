import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .datasets import GLDv2, DatasetWithNeighbors, BaseDataset
from .samplers import RandomSampler
from .collate_fn import collate_fn


def build_dataloader(cfg, dataset="gldv2", mode="train"):
    assert dataset in ["gldv2", "instre"]
    assert mode in ["train", "test", "cpool"]
    if mode == "train":
        train_transform = T.Compose([
            T.RandomResizedCrop(
                cfg.INPUT.TRAIN_IMG_SIZE,
                scale=(cfg.INPUT.SCALE_LOWER, cfg.INPUT.SCALE_UPPER), ratio=(0.75, 1.33)
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        noaug_transform = None
        if cfg.NOAUG.ENABLE:
            noaug_transform = T.Compose([
                T.Resize(size=cfg.INPUT.NOAUG_IMG_SIZE),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ])

        neighbors = np.load(cfg.NEIGHBOR.SRC)["neighbor"]
        
        if dataset == "gldv2":
            dataset = GLDv2(
                root="datasets/gldv2", fname="train_clean.csv",
                transform=train_transform, target_transform=None, noaug_transform=noaug_transform
            )
        else:
            dataset = BaseDataset(
                cfg.DATA.ROOT, fname=cfg.DATA.TRAIN_FILENAME,
                transform=train_transform, target_transform=None, noaug_transform=noaug_transform
            )
        train_dataset = DatasetWithNeighbors(dataset, neighbors, cfg.NEIGHBOR.K)

        batch_sampler = RandomSampler(
            len(train_dataset),
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.SOLVER.MAX_STEP
        )
        
        # check
        if cfg.XBM.ENABLE:
            assert cfg.XBM.SIZE == len(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader
    else:
        test_transform = T.Compose([
            T.Resize(size=cfg.INPUT.TEST_IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        if dataset == "gldv2":
            dataset = GLDv2(
                root="datasets/gldv2", fname="train_clean.csv",
                transform=test_transform, target_transform=None, noaug_transform=None
            )
        else:
            dataset = BaseDataset(
                cfg.DATA.ROOT, fname=cfg.DATA.TEST_FILENAME if mode == "test" else cfg.DATA.TRAIN_FILENAME,
                transform=test_transform, target_transform=None, noaug_transform=None
            )
        test_loader = DataLoader(dataset, batch_size=128, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=False)
        return test_loader


def build_xbm_dataloader(cfg):
    aug_transform = T.Compose([
        T.RandomResizedCrop(
            cfg.INPUT.TRAIN_IMG_SIZE,
            scale=(cfg.INPUT.SCALE_LOWER, cfg.INPUT.SCALE_UPPER), ratio=(0.75, 1.33)
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    noaug_transform = T.Compose([
        T.Resize(size=cfg.INPUT.NOAUG_IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    dataset = cfg.DATA.DATASET
    if dataset == "gldv2":
        dataset = GLDv2(
            root="datasets/gldv2", fname="train_clean.csv",
            transform=aug_transform, target_transform=None, noaug_transform=noaug_transform
        )
    else:
        dataset = BaseDataset(
            cfg.DATA.ROOT, fname=cfg.DATA.TRAIN_FILENAME,
            transform=aug_transform, target_transform=None, noaug_transform=noaug_transform
        )
    neighbors = np.load(cfg.NEIGHBOR.SRC)["neighbor"]
    dataset = DatasetWithNeighbors(dataset, neighbors, cfg.NEIGHBOR.K)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False)
    return dataloader
