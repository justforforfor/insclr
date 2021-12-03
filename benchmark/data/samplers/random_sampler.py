import random
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, dataset_size, batch_size, max_step):
        self.size = dataset_size
        self.batch_size = batch_size
        self.max_step = max_step

    def __len__(self):
        return self.max_step

    def _prepare(self):
        ind = list(range(self.size))
        random.shuffle(ind)
        num_steps = self.size // self.batch_size
        # drop last
        batch_ind = [ind[i * self.batch_size:(i + 1) * self.batch_size] for i in range(num_steps)]
        return batch_ind

    def __iter__(self):
        # init
        batch_ind = self._prepare()
        for _ in range(self.max_step):
            if len(batch_ind) < 1:
                batch_ind = self._prepare()
            yield batch_ind.pop(0)
