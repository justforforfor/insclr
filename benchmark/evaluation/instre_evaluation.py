import os

from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from torchvision import transforms

from benchmark.data.datasets import ImagesFromList
from benchmark.utils.extract_fn import extract_features

from .image_search_evaluation import ImageSearchEvaluator


class InstreEvaluator(ImageSearchEvaluator):
    def __init__(self, root, img_size, scales, pca=False):
        self.dataset = "instre"
        self.img_size = img_size
        self.scales = scales
        self.pca = pca
        self.pca_dims = [128, 256, 512]

        mat = scipy.io.loadmat(os.path.join(root, "gnd_instre.mat"))
        qimlist = [str(_[0]) for _ in mat["qimlist"][0]]
        imlist = [str(_[0]) for _ in mat["imlist"][0]]
        gnd =  mat["gnd"][0]

        query_imgs = [os.path.join(root, img) for img in qimlist]
        bboxes = [tuple(_[1][0].astype(np.int64)) for _ in gnd]
        bboxes = [(x, y, x + w, y + h) for x, y, w, h in bboxes]

        for imgpath, bbox in zip(query_imgs, bboxes):
            img = Image.open(imgpath)
            img = img.crop(bbox)
            os.makedirs("crop", exist_ok=True)
            img.save(os.path.join("crop", os.path.basename(imgpath)))

        index_imgs = [os.path.join(root, img) for img in imlist]
        self.gnd = [{"ok": _[0][0]} for _ in gnd]

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
        query_features = extract_features(self.query_dataloader, model, device, self.scales)
        index_features = extract_features(self.index_dataloader, model, device, self.scales)
        
        gnd_t = []
        for i in range(len(self.gnd)):
            g = {}
            g["ok"] = self.gnd[i]["ok"]
            gnd_t.append(g)

        scores = np.dot(query_features, index_features.T)
        ranks = np.argsort(-scores, axis=1).T
        map, *_ = self.compute_map(ranks, gnd_t, [1, 5, 10])

        res = {}
        res = {
            f"{self.dataset}/mAP": np.around(map * 100, decimals=2),
        }

        msg = ">> {}: mAP: {}\n".format(
                self.dataset,
                np.around(map * 100, decimals=2),
            )
        
        if self.pca:
            for dim in self.pca_dims:
                pca = PCA(dim, whiten=True)
                pca.fit(index_features)
                _query_features = pca.transform(query_features)
                _index_features = pca.transform(index_features)
                scores = np.dot(_query_features, _index_features.T)
                ranks = np.argsort(-scores, axis=1).T
                map, *_ = self.compute_map(ranks, gnd_t, [1, 5, 10])
                res = {
                    f"{self.dataset}/mAP (pca {dim})": np.around(map * 100, decimals=2),
                }
                msg += f"pca {dim}: {np.around(map * 100, decimals=2)}\n"

        return "\n" + msg, res
