"""Minimal checkpoint loading hook for vendored Chorus inference."""

from __future__ import annotations

import os
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from . import HookBase


class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_eval(self):
        self.before_train()

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if not self.trainer.cfg.weight or not os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
            return

        self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
        checkpoint = torch.load(
            self.trainer.cfg.weight,
            map_location=lambda storage, loc: storage.cuda(),
            weights_only=False,
        )

        model_state = self.trainer.model.state_dict()
        weight = OrderedDict()
        skipped_keys = {"shape_mismatch": [], "not_in_model": []}

        for key, value in checkpoint["state_dict"].items():
            processed_key = key
            if not key.startswith("module."):
                processed_key = "module." + key
            if self.keywords in processed_key:
                processed_key = processed_key.replace(self.keywords, self.replacement)
            if comm.get_world_size() == 1 and processed_key.startswith("module."):
                processed_key = processed_key[7:]

            if processed_key in model_state:
                if model_state[processed_key].shape == value.shape:
                    weight[processed_key] = value
                else:
                    skipped_keys["shape_mismatch"].append(processed_key)
            else:
                skipped_keys["not_in_model"].append(processed_key)

        load_state_info = self.trainer.model.load_state_dict(weight, strict=self.strict)
        self.trainer.logger.info(
            f"Successfully loaded {len(weight)}/{len(model_state)} model keys "
            f"from checkpoint with {len(checkpoint['state_dict'])} keys."
        )
        if skipped_keys["shape_mismatch"]:
            self.trainer.logger.warning(
                f"Skipped {len(skipped_keys['shape_mismatch'])} shape-mismatched keys."
            )
        if skipped_keys["not_in_model"]:
            self.trainer.logger.warning(
                f"Skipped {len(skipped_keys['not_in_model'])} keys not in model."
            )
        if self.strict:
            self.trainer.logger.info(f"Strict missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Strict unexpected keys: {load_state_info[1]}")

