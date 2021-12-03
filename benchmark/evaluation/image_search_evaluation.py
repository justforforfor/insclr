import os
import pickle

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from benchmark.data.datasets import ImagesFromList
from benchmark.utils.extract_fn import extract_features


class ImageSearchEvaluator:
    def __init__(self, dataset, root, img_size, scales, revisitop1m_features=None):
        self.dataset = dataset
        self.img_size = img_size
        self.scales = scales

        with open(os.path.join(root, f"gnd_{dataset}.pkl"), "rb") as fr:
            gnd = pickle.load(fr)
            query_imgs = [os.path.join(root, "jpg", img + ".jpg") for img in gnd["qimlist"]]
            bboxes = [tuple(gnd["gnd"][i]["bbx"]) for i in range(len(query_imgs))]
            index_imgs = [os.path.join(root, "jpg", img + ".jpg") for img in gnd["imlist"]]
        
        self.revisitop1m_features = revisitop1m_features
        self.gnd = gnd["gnd"]

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # define dataloader
        self.query_dataloader = DataLoader(
            ImagesFromList(query_imgs, bboxes=bboxes, img_size=img_size, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.index_dataloader = DataLoader(
            ImagesFromList(index_imgs, img_size=img_size, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def evaluate(self, model, device):
        kappas = [1, 5, 10]
        query_features = extract_features(self.query_dataloader, model, device, self.scales)        
        index_features = extract_features(self.index_dataloader, model, device, self.scales)

        # update index features if exists
        if self.revisitop1m_features is not None:
            index_features = np.concatenate((index_features, self.revisitop1m_features), axis=0)
        
        scores = np.dot(query_features, index_features.T)
        ranks = np.argsort(-scores, axis=1).T

        gnd_t = []
        for i in range(len(self.gnd)):
            g = {}
            g["ok"] = np.concatenate([self.gnd[i]["easy"]])
            g["junk"] = np.concatenate([self.gnd[i]["junk"], self.gnd[i]["hard"]])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = self.compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(self.gnd)):
            g = {}
            g["ok"] = np.concatenate([self.gnd[i]["easy"], self.gnd[i]["hard"]])
            g["junk"] = np.concatenate([self.gnd[i]["junk"]])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = self.compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(self.gnd)):
            g = {}
            g["ok"] = np.concatenate([self.gnd[i]["hard"]])
            g["junk"] = np.concatenate([self.gnd[i]["junk"], self.gnd[i]["easy"]])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = self.compute_map(ranks, gnd_t, kappas)

        res = {}
        res = {
            f"{self.dataset}/mAP E": np.around(mapE * 100, decimals=2),
            f"{self.dataset}/mAP M": np.around(mapM * 100, decimals=2),
            f"{self.dataset}/mAP H": np.around(mapH * 100, decimals=2),
        }
        # res.update({f"mP@{k} E": v for k, v in zip(kappas, np.around(mprE * 100, decimals=2))})
        # res.update({f"mP@{k} M": v for k, v in zip(kappas, np.around(mprM * 100, decimals=2))})
        # res.update({f"mP@{k} H": v for k, v in zip(kappas, np.around(mprH * 100, decimals=2))})

        msg = ">> {}: mAP E: {}, M: {}, H: {}\n".format(
                self.dataset,
                np.around(mapE * 100, decimals=2),
                np.around(mapM * 100, decimals=2),
                np.around(mapH * 100, decimals=2),
            )
        # msg += ">> {}: mP@k{} E: {}, M: {}, H: {}".format(
        #         self.dataset,
        #         kappas,
        #         np.around(mprE * 100, decimals=2),
        #         np.around(mprM * 100, decimals=2),
        #         np.around(mprH * 100, decimals=2),
        #     )

        return "\n" + msg, res

    def compute_map(self, ranks, gnd, kappas=None):
        if kappas is None:
            kappas = []

        map = 0.0
        nq = len(gnd)  # number of queries
        aps = np.zeros(nq)
        pr = np.zeros(len(kappas))
        prs = np.zeros((nq, len(kappas)))
        nempty = 0

        for i in np.arange(nq):
            qgnd = np.array(gnd[i]["ok"])

            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                aps[i] = float("nan")
                prs[i, :] = float("nan")
                nempty += 1
                continue

            try:
                qgndj = np.array(gnd[i]["junk"])
            except:
                qgndj = np.empty(0)

            # sorted positions of positive and junk images (0 based)
            pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
            junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

            k = 0
            ij = 0
            if len(junk):
                # decrease positions of positives based on the number of
                # junk images appearing before them
                ip = 0
                while ip < len(pos):
                    while ij < len(junk) and pos[ip] > junk[ij]:
                        k += 1
                        ij += 1
                    pos[ip] = pos[ip] - k
                    ip += 1

            # compute ap
            ap = self.compute_ap(pos, len(qgnd))
            map = map + ap
            aps[i] = ap

            # compute precision @ k
            pos += 1  # get it to 1-based
            for j in np.arange(len(kappas)):
                kq = min(max(pos), kappas[j])
                prs[i, j] = (pos <= kq).sum() / kq
            pr = pr + prs[i, :]

        map = map / (nq - nempty)
        pr = pr / (nq - nempty)

        return map, aps, pr, prs
    
    @staticmethod
    def compute_ap(ranks, nres):

        # number of images ranked by the system
        nimgranks = len(ranks)

        # accumulate trapezoids in PR-plot
        ap = 0

        recall_step = 1.0 / nres

        for j in np.arange(nimgranks):
            rank = ranks[j]

            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = float(j) / rank

            precision_1 = float(j + 1) / (rank + 1)

            ap += (precision_0 + precision_1) * recall_step / 2.0

        return ap
    
    @property
    def name(self):
        return self.dataset

    def __str__(self):
        return f"evaluator: dataset={self.dataset} scales={self.scales} imsize={self.img_size}"
