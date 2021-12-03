"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import faiss


class MemoryBank:
    def __init__(self, dataset_size, feature_dim, num_classes, temperature):
        self.dataset_size = dataset_size
        self.feature_dim = feature_dim 
        self.features = np.zeros(shape=(self.dataset_size, self.feature_dim), dtype=np.float32)
        self.targets = np.zeros(shape=(self.dataset_size, ), dtype=np.int64)
        self.num_classes = num_classes
        self.temperature = temperature

    def mine_nearest_neighbors(self, topk, features=None):
        if features is None:
            features = self.features
        # mine the topk nearest neighbors for every sample
        index = faiss.IndexFlatIP(self.feature_dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(self.features)
        _, indices = index.search(features, topk + 1) # Sample itself is included
        return indices[:, 1:]
    
    def compute_nearest_neighbors_accuracy(self, topk):
        if isinstance(topk, int):
            topk = [topk]
        topk = sorted(topk)
        index = faiss.IndexFlatIP(self.feature_dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(self.features)
        _, indices = index.search(self.features, topk[-1] + 1) # Sample itself is included

        accuracy_msg = ""
        template = "--> neighbor={:3d}, accuracy={:.2f}\n"
        for _topk in topk:
            neighbor_targets = np.take(self.targets, indices[:, 1:_topk + 1], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(self.targets.reshape(-1, 1), _topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            accuracy_msg += template.format(_topk, accuracy)
        return accuracy_msg

    def update(self, features, targets, indices):
        if not isinstance(features, np.ndarray):
            features = features.detach().cpu().numpy()
        if not isinstance(targets, np.ndarray):
            targets = targets.detach().cpu().numpy()
        if not isinstance(indices, np.ndarray):
            indices = indices.cpu().numpy()
        
        self.features[indices] = features
        self.targets[indices] = targets
