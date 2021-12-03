import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveLoss, self).__init__()
        self.margin = cfg.LOSS.CONTRASTIVE_LOSS.MARGIN
        self.pos_margin = cfg.LOSS.CONTRASTIVE_LOSS.POS_MARGIN

    def forward(self, inputs1, labels1, inputs2, labels2):
        if not (inputs1 is inputs2):
            # memory mode
            mask = labels1 > 0
            inputs1 = inputs1[mask]
            labels1 = labels1[mask]

        n = inputs1.size(0)
        # compute similarity matrix
        sim_mat = torch.matmul(inputs1, inputs2.t())

        epsilon = 1e-6
        loss = list()

        log = {}
        log["avg_neg"] = 0
        log["avg_pos"] = 0

        for i in range(n):
            _pos_pair = torch.masked_select(sim_mat[i], labels1[i] == labels2)
            pos_pair = torch.masked_select(_pos_pair, _pos_pair < 1 - epsilon - self.pos_margin)
            _neg_pair = torch.masked_select(sim_mat[i], labels1[i] != labels2)
            neg_pair = torch.masked_select(_neg_pair, _neg_pair > self.margin)
            
            pos_loss = torch.sum(-pos_pair + 1)
            log["avg_neg"] += len(neg_pair)
            log["avg_pos"] += len(pos_pair)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            else:
                neg_loss = 0
            loss.append(pos_loss + neg_loss)
        loss = sum(loss) / n  # / all_targets.shape[1]
        log["avg_neg"] /= n
        log["avg_pos"] = round(log["avg_pos"] / n, ndigits=2)
        return loss, log
