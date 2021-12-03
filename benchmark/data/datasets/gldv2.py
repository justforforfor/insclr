import os
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np

from .utils import imread


class GLDv2(Dataset):
    def __init__(self, root, fname, transform=None, target_transform=None, noaug_transform=None):
        with open(os.path.join(root, fname), "r") as fr:
            lines = (line.strip() for line in fr.readlines())
        self.imgs = []
        self.landmarks = []
        self.paths = []
        self.labels = []
        self.landmark_to_label = None
        self.label_ind = None

        label = 0

        for idx, line in enumerate(lines):
            if idx == 0:
                # skip the first line
                continue
            landmark, imgs = line.split(",")
            landmark = int(landmark)
            imgs = imgs.split()
            if len(imgs) < 15:
                continue
            for img in imgs:
                self.imgs.append(img)
                self.landmarks.append(landmark)
                self.paths.append(os.path.join(root, img[0], img[1], img[2], img + ".jpg"))
                self.labels.append(label)
            label += 1

        self.build()
        self.transform = transform
        self.target_transform = target_transform
        self.noaug_transform = noaug_transform
        
    def __getitem__(self, idx):
        idx = int(idx)
        # read image
        path = self.paths[idx]
        img = imread(path)
        label = self.labels[idx]

        input = self.transform(img) if self.transform is not None else img
        target = self.target_transform(label) \
            if self.target_transform is not None else label

        res = {"input": input, "target": target, "idx": idx, "img": path.split("/")[-1]}
        if self.noaug_transform is not None:
            noaug_input = self.noaug_transform(img)
            res["noaug_input"] = noaug_input
        return res

    def __len__(self):
        return len(self.paths)

    def build(self):
        # build label to image index
        self.label_ind = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_ind[label].append(idx)
        self.landmark_to_label = {}
        for landmark, label in zip(self.landmarks, self.labels):
            self.landmark_to_label[landmark] = label
        print(f"found {len(self.label_ind)} labels, {len(self.paths)} images")

    @property
    def num_classes(self):
        return len(self.label_ind)
