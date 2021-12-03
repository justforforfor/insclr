import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F


class XBMv2:
    def __init__(self, cfg):
        self.cfg = cfg
        # equal or larger than this thresh, is positive
        positive_thresh = cfg.XBM.POSITIVE_THRESH
        # small than this thresh, is negative
        negative_thresh = cfg.XBM.NEGATIVE_THRESH
        # [[positive], [candidate], [negative]]
        self.candidate_mode = cfg.XBM.CANDIDATE_MODE
        
        data = np.load(cfg.NEIGHBOR.SRC)
        self.neighbors = data["neighbor"]
        similarities = data["similarity"]

        self.topk = len(self.neighbors[0])

        positive_boundary = -np.ones(len(self.neighbors), dtype=np.int32)
        row_ind, col_ind = np.nonzero(similarities >= positive_thresh)
        # positive_boundary[row_ind] = col_ind # work right when copy sequentially
        for idx, jdx in zip(row_ind, col_ind):
            positive_boundary[idx] = max(positive_boundary[idx], jdx)

        # boundary[i] means the the boundary[i]th neighbor for sample i
        # is the first whose similarity < positive_thresh
        positive_boundary += 1
        
        negative_boundary = -np.ones(len(self.neighbors), dtype=np.int32)
        row_ind, col_ind = np.nonzero(similarities >= negative_thresh)
        # negative_boundary[row_ind] = col_ind # work right when copy sequentially
        for idx, jdx in zip(row_ind, col_ind):
            negative_boundary[idx] = max(negative_boundary[idx], jdx)

        # boundary[i] means the the boundary[i]th neighbor for sample i
        # is the first whose similarity < positive_thresh
        negative_boundary += 1

        self.features = None
        self.noaug_features = None
        self.neighbors = torch.from_numpy(self.neighbors)
        self.positive_boundary = torch.from_numpy(positive_boundary)
        self.negative_boundary = torch.from_numpy(negative_boundary)
        
        # when a feature has been updated, its counter plus one
        self.update_counter = torch.zeros(cfg.XBM.SIZE).int()
        
    def init(self, model, dataloader, device):
        model = model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                indices = batch["idx"]
                inputs = batch["input"].to(device)
                noaug_inputs = batch["noaug_input"].to(device)
                features = model(inputs)[0].cpu()
                noaug_features = model(noaug_inputs)[0].cpu()
                self.dequeue_and_enqueue(indices, features, noaug_features)

    def dequeue_and_enqueue(self, indices, features, noaug_features, **kwargs):
        # lazy allocate
        if self.features is None:
            self.features = torch.zeros((self.cfg.XBM.SIZE, self.cfg.XBM.FEATURE_DIM)).float()
        if self.noaug_features is None:
            self.noaug_features = torch.zeros((self.cfg.XBM.SIZE, self.cfg.XBM.FEATURE_DIM)).float()

        indices = indices.cpu()
        features = features.detach().cpu()
        noaug_features = noaug_features.detach().cpu()
        
        self.update_counter[indices] += 1

        self.features[indices] = features
        self.noaug_features[indices] = noaug_features

    def get(self, size, indices, labels, **kwargs):
        """
        Args:
            size: how many features we want to return
            labels: label may be a negative value due to rerank
        Tips:
            1. name starting with '_' is a local variable.
        """
        batch_indices = indices.cpu()
        batch_labels = labels.cpu()
        mask = batch_labels > 0
        batch_indices = batch_indices[mask]
        batch_labels = batch_labels[mask]
        unique_batch_labels = torch.unique(batch_labels)

        log = {}
        log["batch/num_samples"] = len(batch_labels)

        # init pseudo positves
        init_positive_ind = []
        init_positive_labels = []
        
        init_anchor_positive_ind = []
        init_anchor_positive_labels = []

        for _label in unique_batch_labels:
            _ind = batch_indices[batch_labels == _label]
            
            _positive_ind = [self.neighbors[_idx, :self.positive_boundary[_idx]] for _idx in _ind]
            _positive_ind = torch.unique(torch.cat(_positive_ind, axis=0))
            _positive_ind = torch.from_numpy(
                np.setdiff1d(_positive_ind.numpy(), _ind.numpy())
            )
            init_positive_ind.append(_positive_ind)
            init_positive_labels.append(torch.ones(len(_positive_ind)).long() * _label)

            # do not change previous behavior
            _local_ind = torch.nonzero(batch_labels == _label, as_tuple=True)[0]
            _local_ind = torch.sort(_local_ind)[0]
            _anchor = batch_indices[_local_ind[0]]
            init_anchor_positive_ind.append(self.neighbors[_anchor, :self.positive_boundary[_anchor]])
            init_anchor_positive_labels.append(torch.ones(len(init_anchor_positive_ind[-1])).long() * _label)


        init_positive_ind = torch.cat(init_positive_ind, dim=0)
        init_positive_labels = torch.cat(init_positive_labels, dim=0)
        log["init/pred_p"] = len(init_positive_ind)
        init_positive_ind, init_positive_labels = \
            self.remove_duplicate(init_positive_ind, init_positive_labels)
        log["init/unique"] = len(init_positive_ind)
        log["init/avg"] = self.true_divide(log["init/unique"], len(unique_batch_labels))
        
        init_anchor_positive_ind = torch.cat(init_anchor_positive_ind, dim=0)
        init_anchor_positive_labels = torch.cat(init_anchor_positive_labels, dim=0)
        init_anchor_positive_ind, init_anchor_positive_labels = \
            self.remove_duplicate(init_anchor_positive_ind, init_anchor_positive_labels)

        # init negative positves
        init_negative_ind = []

        for _idx in batch_indices:
            _negative_ind = self.neighbors[_idx, self.negative_boundary[_idx]:]
            #TODO: this line can be removed
            _negative_ind = _negative_ind[self.update_counter[_negative_ind] >= 1]
            init_negative_ind.append(_negative_ind)
    
        init_negative_ind = torch.cat(init_negative_ind, dim=0)
        init_negative_ind = torch.unique(init_negative_ind)
        init_negative_ind = np.setdiff1d(init_negative_ind.numpy(), init_positive_ind.numpy())
        init_negative_ind = torch.from_numpy(init_negative_ind)

        # select pseudo positives in candidates
        already_selected = init_positive_ind.numpy()

        log["selected/total"] = 0
        selected_positive_ind = []
        selected_positive_labels = []

        for _label in unique_batch_labels:
            _ind = batch_indices[batch_labels == _label]
            _candidates = [self.neighbors[_idx, self.positive_boundary[_idx]:self.negative_boundary[_idx]] for _idx in _ind]
            _candidates = torch.unique(torch.cat(_candidates, axis=0))
            _candidates = torch.from_numpy(np.setdiff1d(_candidates.numpy(), already_selected))
            log["selected/total"] += len(_candidates)

            _noaug_batch_features = self.noaug_features[_ind]
            _noaug_candidate_features = self.noaug_features[_candidates]
            _positive_ind = init_positive_ind[init_positive_labels == _label]
            _noaug_init_positive_features = self.noaug_features[_positive_ind]
            _anchor_positive_ind = init_anchor_positive_ind[init_anchor_positive_labels == _label]
            _noaug_init_anchor_positive_features = self.noaug_features[_anchor_positive_ind]

            _selected_positive_ind, _, = self.select(
                self.candidate_mode, _noaug_batch_features,
                _noaug_candidate_features, _noaug_init_positive_features, _noaug_init_anchor_positive_features
            )
            # map local indices to global indices
            _selected_positive_ind = _candidates[_selected_positive_ind]
            selected_positive_ind.append(_selected_positive_ind)
            selected_positive_labels.append(torch.ones(len(_selected_positive_ind)).long() * _label)

        log["selected/avg_candidates"] = \
            self.true_divide(log["selected/total"], len(unique_batch_labels))
        
        selected_positive_ind = torch.cat(selected_positive_ind, dim=0)
        selected_positive_labels = torch.cat(selected_positive_labels, dim=0)
        log["selected/pred_p"] = len(selected_positive_ind)
        
        selected_positive_ind, selected_positive_labels = \
            self.remove_duplicate(selected_positive_ind, selected_positive_labels)

        log["selected/unique"] = len(selected_positive_ind)
        log["selected/unique_ratio"] = self.true_divide(log["selected/unique"], log["selected/pred_p"])
        log["selected/avg"] = self.true_divide(log["selected/unique"], len(unique_batch_labels))

        pseudo_positive_ind = torch.cat((init_positive_ind, selected_positive_ind), dim=0)
        pseudo_positive_labels = torch.cat((init_positive_labels, selected_positive_labels), dim=0)

        # pseudo_positive_ind and init_negative_ind may have intersection
        init_negative_ind = torch.from_numpy(
            np.setdiff1d(init_negative_ind.numpy(), pseudo_positive_ind.numpy())
        )

        other = torch.nonzero(self.update_counter >= 1, as_tuple=True)[0].numpy()
        # remove already selected
        other = np.setdiff1d(other, pseudo_positive_ind.numpy())
        other = np.setdiff1d(other, init_negative_ind.numpy())
        size = size - len(pseudo_positive_ind) - len(init_negative_ind)
        size = max(min(size, len(other)), 0)
        # size may be zero
        other_ind = torch.from_numpy(np.random.choice(other, size=size, replace=False))

        if self.cfg.XBM.USE_NEGATIVE:
            pseudo_negative_ind = torch.cat((init_negative_ind, other_ind), dim=0)
        else:
            pseudo_negative_ind = torch.tensor([], dtype=torch.int64)

        pseudo_negative_labels = torch.ones(len(pseudo_negative_ind)).long() * (len(self.features) + 1)

        ind = torch.cat((pseudo_positive_ind, pseudo_negative_ind), dim=0)
        features = self.features[ind]
        labels = torch.cat((pseudo_positive_labels, pseudo_negative_labels), dim=0)

        log["size"] = len(ind)
        log["pseudo_positive"] = len(pseudo_positive_ind)
        log["pseudo_negative"] = len(pseudo_negative_ind)

        return features, labels, log
    
    @staticmethod
    def remove_duplicate(indices, labels):
        # remove duplicate indices
        # the same index may have different labels, we randomly choose one
        unique_indices, mapping = torch.unique(indices, sorted=False, return_inverse=True)
        unique_labels = torch.zeros(len(unique_indices)).long()
        unique_labels[mapping] = labels
        return unique_indices, unique_labels

    @staticmethod
    def select(mode, batch_positive_features, candidate_features,
               init_positive_features, init_anchor_positive_features, sgt_mask=None):
        # parse mode
        mode = mode.split("_")
        assert len(mode) == 4, f"wrong mode string: {mode}" 
        init_positive_mode, traverse_mode, score_fn_mode, selection_mode = mode
        ntr = int(traverse_mode[2:])
        
        score_fn = []
        _score_fn_mode = score_fn_mode
        while len(score_fn_mode) > 0:
            score_fn.append((score_fn_mode[:3], float(score_fn_mode[3:7])))
            assert score_fn[-1][0] in ["avg", "max"], f"wrong score fn mode: {_score_fn_mode}"
            score_fn_mode = score_fn_mode[7:]
        while len(score_fn) < ntr:
            score_fn.append(score_fn[-1])
        
        _selection_mode = selection_mode
        selection_fn = []
        while len(selection_mode) > 0:
            if selection_mode.startswith("k"):
                selection_fn.append(("k", int(selection_mode[1:3])))
                selection_mode = selection_mode[3:]
            elif selection_mode.startswith("t"):
                selection_fn.append(("t", float(selection_mode[1:5])))
                selection_mode = selection_mode[5:]
            else:
                raise ValueError(f"undefined selection mode: {_selection_mode}")
        while len(selection_fn) < ntr:
            selection_fn.append(selection_fn[-1])

        if init_positive_mode == "a+i":
            positive_features = torch.cat((batch_positive_features[0].unsqueeze(dim=0), init_anchor_positive_features), axis=0)
        elif init_positive_mode == "b+0":
            positive_features = batch_positive_features
        elif init_positive_mode == "b+i":
            positive_features = torch.cat((batch_positive_features, init_positive_features), axis=0)
        else:
            raise ValueError(f"undefined init positive mode: {mode}")
        
        # do traverse
        selected_mask = torch.zeros(len(candidate_features), dtype=torch.bool)
        for i in range(ntr):
            similarities = torch.matmul(candidate_features, positive_features.t())

            thresh = score_fn[i][1]
            mask = (similarities >= thresh).float()
            num = mask.sum(dim=1, keepdims=False)
            # compute score for each candidate
            if score_fn[i][0] == "avg":
                similarities_sum = (similarities * mask).sum(dim=1, keepdims=False)
                # avoid division by zero
                similarities = similarities_sum / (num + 1e-6)
            else:
                similarities = similarities * mask
                if len(similarities) == 0:
                    similarities = torch.empty_like(similarities).reshape(-1)
                else:
                    similarities = similarities.max(dim=1, keepdim=False)[0]
            
            # for already selected, we set their similarites to be 0.0 to avoiding selecting them again 
            similarities[selected_mask] = 0.0
            num[selected_mask] = 0

            # select pseudo positves
            if selection_fn[i][0] == "k":
                k = min(selection_fn[i][1], int(sum(num > 0.01)))
                _, ind = torch.sort(similarities, descending=True)
                positive_ind = ind[:k]
            else:
                t = selection_fn[i][1]
                mask = similarities >= t
                positive_ind = torch.nonzero(mask, as_tuple=True)[0]
            
            # update positive features
            positive_features = torch.cat((positive_features, candidate_features[positive_ind]), axis=0)
            selected_mask[positive_ind] = True

        positive_ind = torch.nonzero(selected_mask, as_tuple=True)[0]
        negative_ind = torch.nonzero(~selected_mask, as_tuple=True)[0]

        return positive_ind, negative_ind            

    @staticmethod
    def true_divide(x, y):
        return torch.true_divide(x, y) if y > 0 else 0.0
