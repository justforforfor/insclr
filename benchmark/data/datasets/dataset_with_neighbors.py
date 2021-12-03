import torch
from .utils import imread


class DatasetWithNeighbors:
    def __init__(self, dataset, neighbors, k):
        self.dataset = dataset
        self.paths = dataset.paths
        self.labels = dataset.labels
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.noaug_transform = dataset.noaug_transform
        self.label_ind = dataset.label_ind
        
        self.neighbors = neighbors[:, :k]

    def __getitem__(self, idx):
        # read image
        path = self.paths[idx]
        label = self.labels[idx]
        neighbor_ind = self.neighbors[idx]
        neighbor_paths = [self.paths[i] for i in neighbor_ind]
        neighbor_labels = [self.labels[i] for i in neighbor_ind]
        neighbor_ind = torch.from_numpy(neighbor_ind).long()

        # fake label start from 1
        fake_label = idx + 1

        img = imread(path)
        neighbor_imgs = [imread(_) for _ in neighbor_paths]
        input = self.transform(img)
        neighbor_inputs = torch.stack([self.transform(_) for _ in neighbor_imgs], dim=0)

        if self.target_transform is not None:
            target = self.target_transform(label)
            neighbor_targets = [self.target_transform(_) for _ in neighbor_labels]
        else:
            target = label
            neighbor_targets = neighbor_labels
        neighbor_targets = torch.LongTensor(neighbor_targets)
        # we assume that neighbors have the same fake label as anchor
        neighbor_fake_labels = torch.ones_like(neighbor_targets) * fake_label
        
        noaug_input = None
        noaug_neighbor_inputs = None
        if self.noaug_transform is not None:
            noaug_input = self.noaug_transform(img)
            noaug_neighbor_inputs = torch.stack([self.noaug_transform(_) for _ in neighbor_imgs], dim=0)

        res = {}
        # input
        res.update({"input": input, "target": target, "label": fake_label, "idx": idx, 
                    "img": [path.split("/")[-1].split(".")[0]]})
        # neighbors
        res.update({"neighbor_inputs": neighbor_inputs, "neighbor_targets": neighbor_targets,
                    "neighbor_ind": neighbor_ind, "neighbor_labels": neighbor_fake_labels,
                    "neighbor_imgs": [_.split("/")[-1].split(".")[0] for _ in neighbor_paths]})
        # noaug
        res.update({"noaug_input": noaug_input, "noaug_neighbor_inputs": noaug_neighbor_inputs})
        return res

    def __len__(self):
        return len(self.dataset)

    @property
    def num_classes(self):
        return self.dataset.num_classes
