import os
from collections import defaultdict

from torch.utils.data import Dataset
from .utils import imread


class BaseDataset(Dataset):
    def __init__(self, root, fname, transform=None, target_transform=None, noaug_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.noaug_transform = noaug_transform

        with open(os.path.join(root, fname), "r") as fr:
            lines = (line.strip() for line in fr.readlines())

        self.paths = []
        self.labels = []
        self.label_ind = None
        label = 0
        to_label = {}
        for idx, line in enumerate(lines):
            img, target = line.split(",")
            if target not in to_label:
                to_label[target] = label
                label += 1
            self.paths.append(os.path.join(root, img))
            self.labels.append(to_label[target])
            
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
        print(f"found {len(self.label_ind)} labels, {len(self.paths)} images")

    @property
    def num_classes(self):
        return len(self.label_ind)
