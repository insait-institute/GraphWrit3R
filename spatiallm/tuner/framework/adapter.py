# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Set

import torch

from ..framework import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _layer_name_from_param(name: str) -> str:
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _module_trainability_report(label: str, module: torch.nn.Module) -> str:
    layers: dict[str, list[bool]] = {}
    total_params = 0
    trainable_params = 0
    for name, param in module.named_parameters():
        layer_name = _layer_name_from_param(name)
        layers.setdefault(layer_name, []).append(param.requires_grad)
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_layers = []
    frozen_layers = []
    mixed_layers = []
    for layer_name, states in sorted(layers.items()):
        if all(states):
            trainable_layers.append(layer_name)
        elif any(states):
            mixed_layers.append(layer_name)
        else:
            frozen_layers.append(layer_name)

    def format_layers(title: str, names: list[str]) -> str:
        if not names:
            return f"{title}: none"
        return f"{title} ({len(names)}):\n  - " + "\n  - ".join(names)

    return "\n".join(
        (
            f"{label}: trainable_params={trainable_params:,}/{total_params:,}",
            format_layers("TRAINABLE layers", trainable_layers),
            format_layers("MIXED layers", mixed_layers),
            format_layers("FROZEN layers", frozen_layers),
        )
    )


def _log_sonata_chorus_trainability(model: "PreTrainedModel") -> None:
    point_backbone = getattr(model, "point_backbone", None)
    if point_backbone is not None:
        logger.info_rank0(
            "Sonata trainability report:\n%s",
            _module_trainability_report("sonata.point_backbone", point_backbone),
        )

    chorus_fusion = getattr(model, "chorus_fusion_encoder", None)
    if chorus_fusion is None:
        return

    chorus_backbone = getattr(getattr(chorus_fusion, "chorus_model", None), "backbone", None)
    if chorus_backbone is not None:
        logger.info_rank0(
            "Chorus trainability report:\n%s",
            _module_trainability_report("chorus.chorus_model.backbone", chorus_backbone),
        )

    fourier_head = getattr(chorus_fusion, "fourier_head", None)
    if fourier_head is not None:
        logger.info_rank0(
            "Chorus projection-head trainability report:\n%s",
            _module_trainability_report("chorus.fourier_head", fourier_head),
        )

    pair_fusion_transformer = getattr(chorus_fusion, "pair_fusion_transformer", None)
    if pair_fusion_transformer is not None:
        logger.info_rank0(
            "Chorus pair-fusion transformer trainability report:\n%s",
            _module_trainability_report(
                "chorus.pair_fusion_transformer",
                pair_fusion_transformer,
            ),
        )


def get_forbidden_modules(
    finetuning_args: "FinetuningArguments", model: "PreTrainedModel"
) -> Set[str]:
    r"""
    Freezes network modules for tuning.
    """
    forbidden_modules = set()
    if finetuning_args.train_proj_only:
        forbidden_modules.update({"point_backbone", "model", "lm_head"})
    elif finetuning_args.freeze_point_tower:
        forbidden_modules.add("point_backbone")
    elif finetuning_args.freeze_language_tower:
        forbidden_modules.update({"model", "lm_head"})

    if not getattr(model.config, "use_3d_tokens", True):
        forbidden_modules.update({"point_backbone", "point_proj"})

    return forbidden_modules


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(finetuning_args, model)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)

    # force point_backbone to have float32
    model.set_point_backbone_dtype(torch.float32)


def init_adapter(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support only full-parameter training for now.

    Note that the trainable parameters must be cast to float32.
    """

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16:
        logger.info_rank0(
            "Pure bf16 detected, remaining trainable params in half precision."
        )
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    _setup_full_tuning(
        model, finetuning_args, is_trainable, cast_trainable_params_to_fp32
    )
    chorus_fusion = getattr(model, "chorus_fusion_encoder", None)
    if (
        is_trainable
        and chorus_fusion is not None
        and getattr(model.config, "chorus_mirror_sonata_trainability", False)
    ):
        counts = chorus_fusion.sync_trainability_from_sonata()
        logger.info_rank0(
            "Mirrored Sonata trainability to Chorus: %s/%s matched Chorus params trainable.",
            counts["trainable"],
            counts["matched"],
        )
    if is_trainable:
        _log_sonata_chorus_trainability(model)

    return model
