import torch
import numpy as np


class XBMv1:
    def __init__(self, cfg):
        # when getting features for a batch, we want to exclude all its neighbors
        # to avoid potential positives.
        self.cfg = cfg
        potential_positive = cfg.XBM.POTENTIAL_POSITIVE
        neighbors = np.load(cfg.NEIGHBOR.SRC)["neighbor"]
        potential_positive_neighbors = neighbors[:, :potential_positive]
        self.potential_positive_neighbors = torch.from_numpy(potential_positive_neighbors)
    
        self.features = torch.zeros((cfg.XBM.SIZE, cfg.XBM.FEATURE_DIM)).float()

        # when a feature has been updated, its counter plus one
        self.update_counter = torch.zeros(cfg.XBM.SIZE).int()
    
    def dequeue_and_enqueue(self, indices, features, **kwargs):
        indices = indices.cpu()
        self.update_counter[indices] += 1
        features = features.detach().cpu()
        self.features[indices] = features

    def get(self, size, indices, **kwargs):
        """
        Args:
            size: how many features we want to return
            indices: use to exclude potential positives
        """
        log = {}
        indices = indices.cpu()
        candidates = torch.nonzero(self.update_counter >= 1, as_tuple=True)[0].numpy()

        potential_positives = torch.unique(self.potential_positive_neighbors[indices]).numpy()
        # exclude potential_positives
        candidates = np.setdiff1d(candidates, potential_positives)
        negative_ind = np.random.choice(candidates, size=min(size, len(candidates)), replace=False)
        negative_ind = torch.from_numpy(negative_ind)

        features = self.features[negative_ind]
        # labels we have used are in the range [1, len(self.features)]
        # so here we set negative labels to len(self.features) + 1
        labels = torch.ones(len(features)).long() * (len(self.features) + 1)
        log["num_features"] = len(features)
        
        return features, labels, log
