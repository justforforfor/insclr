import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from benchmark.data.datasets import ImagesFromList
from benchmark.utils.extract_fn import extract_features


class GLDRetrievalEvaluator:
    def __init__(self, root, img_size=1024, scales=(1.414, 1.0, 0.7), private_metric=True):
        self.img_size = img_size
        self.scales = scales
        self.private_metric = private_metric

        # load test images
        self.public_solution = {}
        self.private_solution = {}
        ignored_imgs = []
        
        with open(os.path.join(root, "test", "retrieval_solution_v2.1.csv"), "r") as fr:
            lines = [line.strip() for line in fr.readlines()]
            # skip the first line
            lines = lines[1:]
            for line in lines:
                img, imgs, usage = line.split(",")
                if usage == "Ignored":
                    ignored_imgs.append(img)
                    continue
                if usage == "Public":
                    self.public_solution[img] = imgs.split(" ")
                elif usage == "Private":
                    self.private_solution[img] = imgs.split(" ")
                else:
                    raise ValueError(f"Test image {img} has unrecognized usage tag {usage}.")
        
        query_imgs = list(self.public_solution.keys())
        if self.private_metric:
            query_imgs.extend(list(self.private_solution.keys()))

        # map image id to idx
        self.query_img_to_idx = {img: idx for idx, img in enumerate(query_imgs)}
        query_imgs = [os.path.join(root, "test", img[0], img[1], img[2], img + ".jpg") for img in query_imgs]
        
        # load index images
        index_imgs = []
        with open(os.path.join(root, "index", "index_image_to_landmark.csv"), "r") as fr:
            lines = [line.strip() for line in fr.readlines()]
            # skip the first line
            lines = lines[1:]
            for line in lines:
                img, _ = line.split(",")
                index_imgs.append(img)
    
        self.index_idx_to_img = {idx: img for idx, img in enumerate(index_imgs)}
        index_imgs = [os.path.join(root, "index", img[0], img[1], img[2], img + ".jpg") for img in index_imgs]

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # define dataloader
        self.query_dataloader = DataLoader(
            ImagesFromList(query_imgs, img_size=img_size, transform=transform),
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


    def evaluate(self, model, device, k=100, **kwargs):
        if isinstance(k, int):
            k = [k]

        public_map = []
        private_map = []
        query_features = kwargs.get("query_features", None)
        index_features = kwargs.get("index_features", None)
    
        # public and private are both in query 
        if query_features is None:
            query_features = extract_features(self.query_dataloader, model, device, self.scales)
        if index_features is None:
            index_features = extract_features(self.index_dataloader, model, device, self.scales)

        # print("Save index features to training_dir/cache/gldv2_index_features.npz")
        # os.makedirs("training_dir/cache", exist_ok=True)
        # np.savez("training_dir/cache/gldv2_index_features.npz", feature=index_features)

        scores = np.dot(query_features, index_features.T)
        # descending order
        ind = np.argsort(-scores, axis=1)

        for _k in k:
            public_predictions = self.generate_predictions(ind, self.public_solution, _k)

            # tf
            map = MeanAveragePrecision(public_predictions, self.public_solution, _k)

            # map = self.compute_map(public_predictions, self.public_solution, _k)
            public_map.append(round(map * 100, 2))
            if self.private_metric:       
                private_predictions = self.generate_predictions(ind, self.private_solution, _k)
                # tf
                map = MeanAveragePrecision(private_predictions, self.private_solution, _k)
                # map = self.compute_map(private_predictions, self.private_solution, _k)
                private_map.append(round(map * 100, 2))

        # log
        log = {f"GLD_retrieval_public/mAP@{_k}: {_map}" for _k, _map in zip(k, public_map)}
        msg = "\n>> GLD retrieval public: "
        for _k, _map in zip(k, public_map):
            msg += f"mAP@{_k}: {_map}  "
        msg = msg.strip()

        if self.private_metric:
            log = {f"GLD_retrieval_private/mAP@{_k}: {_map}" for _k, _map in zip(k, private_map)}
            msg += "\n>> GLD retrieval private: "
            for _k, _map in zip(k, private_map):
                msg += f"mAP@{_k}: {_map}  "
            msg = msg.strip()
        return msg, log
    
    def generate_predictions(self, indices, solution, k):
        """Generate required predictions format from indices.
        Args:
            indices: 2d np.array, each row of indices represents a prediction for a query.
            solution: dict mapping (test) image id to a list of (index) ground-truth image ids.
            k: maximum number of predictions per query to take into account.
        Return:
            predictions: dict mapping (test) image id to a list of (index) predicted image ids.
        """
        # image ids we consider
        img_ids = list(solution.keys())
        # map these images to their row number in indices
        imgs = np.array([self.query_img_to_idx[img] for img in img_ids])
        # get their topk predictions
        indices = indices[imgs, :k]

        # map index to image id
        predictions = {}
        for img_id, _ind in zip(img_ids, indices):
            predictions[img_id] = [self.index_idx_to_img[idx] for idx in _ind]
        return predictions

    def compute_map(self, predictions, solution, k):
        """Compute mean average precision for a given solution.
        Args:
            predicitions: dict mapping (test) image id to a list of (index) predicted image ids.
            solution: dict mapping (test) image id to a list of (index) ground-truth image ids.
            k: maximum number of predictions per query to take into account.
        Return:
            mean_ap: mean average precision
        """
        num_test_imgs = len(solution.keys())
        mean_ap = 0.0
        for img, gt in solution.items():
            mean_ap += self.compute_ap(predictions[img], gt, k)
        
        mean_ap /= num_test_imgs
        return mean_ap

    @staticmethod
    def compute_ap(prediction, gt, k):
        """Compute average precision for a query image.
        Args:
            prediction: list of (index) image ids.
            gt: list of (index) image ids.
            k: maximum number of predictions per query to take into account.
        """
        assert len(prediction) <= k
        ap = 0.0
        num_correct = 0
        num_expected = min(len(gt), k)
        already_predicted = set() # to remove duplicate image ids in predictions
        
        for i in range(len(prediction)):
            if prediction[i] not in already_predicted:
                if prediction[i] in gt:
                    num_correct += 1
                    ap += num_correct / (i + 1)
            already_predicted.add(prediction[i])

        ap /= num_expected
        return ap

    @property
    def name(self):
        return self.dataset

    def __str__(self):
        return f"evaluator: dataset={self.dataset} scales={self.scales} imsize={self.img_size}"


def MeanAveragePrecision(predictions, retrieval_solution, max_predictions=100):
    """Computes mean average precision for retrieval prediction.
    Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
        to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
        IDs.
        max_predictions: Maximum number of predictions per query to take into
        account. For the Google Landmark Retrieval challenge, this should be set
        to 100.
    Returns:
        mean_ap: Mean average precision score (float).
    Raises:
        ValueError: If a test image in `predictions` is not included in
        `retrieval_solutions`.
    """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)
        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
        num_correct = 0
        for i in range(min(len(prediction), max_predictions)):
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                    already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    mean_ap /= num_test_images

    return mean_ap
