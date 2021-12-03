# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch


class Checkpointer(object):
    def __init__(self, model, save_dir, save_to_disk=False, 
                 optimizer=None, scheduler=None, logger=None):
        self.model = model
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.optimizer = optimizer
        self.scheduler = scheduler
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_to_disk:
            return
        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True, module_prefix=False):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("no checkpoint found, initializing model from scratch")
            return {}
        self.logger.info("loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        state_dict = checkpoint.pop("model")
        if not module_prefix:
            # remove module prefix: module.backbone -> backbone
            _state_dict = {k[7:]: v for k, v in state_dict.items()}
        else:
            _state_dict = state_dict
        result = self.model.load_state_dict(_state_dict, strict=False)
        assert len(result.missing_keys) == 0, f"missing keys: {result.missing_keys}"
        if len(result.unexpected_keys) > 0:
            self.logger.warning(f"unexpected_keys: {result.unexpected_keys}")

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))
