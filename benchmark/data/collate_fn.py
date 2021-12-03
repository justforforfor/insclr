import torch
import collections


def collate_fn(batch, cat=False):
    """
    batch = ((), (), ...)
    """
    if isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], torch.Tensor):
        if cat:
            return torch.cat(batch, dim=0)
        else:
            return torch.stack(batch, dim=0)
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([b[key] for b in batch], "neighbor" in key and "img" not in key) \
            for key in batch[0].keys()}
    elif isinstance(batch[0], list) and isinstance(batch[0][0], str):
        # (["xxx"], ["yyy", "zzz"], ...) -> ["xxx", "yyy", "zzz", ...]
        return sum(batch, list())
    elif isinstance(batch[0], str):
        # we do not allow this situation
        assert False
    elif isinstance(batch[0], collections.Sequence):
        return [collate_fn(samples) for samples in zip(*batch)]
    elif batch[0] is None:
        return None
    else:
        raise TypeError(f"unsupported type: {type(batch[0])}")
