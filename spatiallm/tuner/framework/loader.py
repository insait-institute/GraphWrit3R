import os
from itertools import chain
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from torch import nn
import transformers.dynamic_module_utils
from transformers.dynamic_module_utils import get_relative_imports
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from . import logging
from .utils import check_version, count_parameters, is_env_enabled
from .patcher import patch_config, patch_model, patch_tokenizer, patch_processor
from .adapter import init_adapter
from spatiallm.model import mm_debug

if TYPE_CHECKING:
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        ProcessorMixin,
    )

    from ..hparams import DataArguments, ModelArguments, FinetuningArguments

logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _summarize_names(names: list[str], limit: int = 50) -> str:
    if len(names) <= limit:
        return "\n".join(f"  - {name}" for name in names)
    shown = "\n".join(f"  - {name}" for name in names[:limit])
    return f"{shown}\n  ... and {len(names) - limit} more"


def _maybe_attach_chorus_fusion(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    if not getattr(model_args, "chorus_fusion_enabled", False):
        return
    if getattr(model, "chorus_fusion_encoder", None) is not None:
        return
    if getattr(model, "point_backbone", None) is None:
        raise RuntimeError("Chorus fusion requires a SpatialLM point backbone.")
    from spatiallm.model.chorus_fusion import ChorusFusionPointEncoder

    model.chorus_fusion_encoder = ChorusFusionPointEncoder(
        sonata_backbone=model.point_backbone,
        chorus_repo_root=model_args.chorus_repo_root,
        chorus_config=model_args.chorus_config,
        chorus_checkpoint=model_args.chorus_checkpoint,
        aligned_root=model_args.chorus_aligned_root,
        input_mode=model_args.chorus_input_mode,
        native_match_radius=model_args.chorus_native_match_radius,
        use_sonata_lattice_for_coord_matched=model_args.chorus_use_sonata_lattice_for_coord_matched,
        min_valid_label_fraction=model_args.chorus_min_valid_label_fraction,
        trainable_name_patterns=model_args.chorus_trainable_name_patterns,
        mirror_sonata_trainability=model_args.chorus_mirror_sonata_trainability,
        fusion_mode=model_args.chorus_fusion_mode,
        missing_policy=model_args.chorus_missing_policy,
        modality_dropout_rate=model_args.chorus_modality_dropout_rate,
        drop_pcd_probability=model_args.chorus_drop_pcd_probability,
        match_grid_radius=model_args.chorus_match_grid_radius,
        exact_match_first=model_args.chorus_exact_match_first,
        discard_unmatched_tokens=model_args.chorus_discard_unmatched_tokens,
        fusion_chorus_weight=model_args.chorus_fusion_chorus_weight,
        fusion_gate_mode=model_args.chorus_fusion_gate_mode,
        fusion_gate_hidden_dim=model_args.chorus_fusion_gate_hidden_dim,
        fusion_transformer_layers=model_args.chorus_fusion_transformer_layers,
        fusion_transformer_heads=model_args.chorus_fusion_transformer_heads,
        fusion_transformer_ffn_dim=model_args.chorus_fusion_transformer_ffn_dim,
        fusion_transformer_dropout=model_args.chorus_fusion_transformer_dropout,
        contrastive_loss_weight=model_args.chorus_contrastive_loss_weight,
        contrastive_exact_only=model_args.chorus_contrastive_exact_only,
        contrastive_backprop_sonata=model_args.chorus_contrastive_backprop_sonata,
        contrastive_batch_wide_enabled=model_args.chorus_contrastive_batch_wide_enabled,
        contrastive_cosine_enabled=model_args.chorus_contrastive_cosine_enabled,
        contrastive_mse_enabled=model_args.chorus_contrastive_mse_enabled,
        contrastive_info_nce_enabled=model_args.chorus_contrastive_info_nce_enabled,
        contrastive_cosine_weight=model_args.chorus_contrastive_cosine_weight,
        contrastive_mse_weight=model_args.chorus_contrastive_mse_weight,
        contrastive_info_nce_weight=model_args.chorus_contrastive_info_nce_weight,
        contrastive_temperature=model_args.chorus_contrastive_temperature,
        contrastive_min_matched_tokens=model_args.chorus_contrastive_min_matched_tokens,
        use_layer_norm=model_args.chorus_fourier_head_layer_norm,
        init_fourier_head_from_sonata=model_args.chorus_init_fourier_head_from_sonata,
        share_fourier_head_with_sonata=model_args.chorus_share_fourier_head_with_sonata,
        disable_drop_path=model_args.chorus_disable_drop_path,
    )
    logger.info_rank0(
        "Attached Chorus fusion encoder: mode=%s input=%s aligned_root=%s trainable_patterns=%s dropout=%.3f drop_pcd=%.3f radius=%d discard_unmatched=%s native_radius=%d valid_label_min=%.3f gate=%s transformer=(layers=%d heads=%d ffn=%d dropout=%.3f)",
        model_args.chorus_fusion_mode,
        model_args.chorus_input_mode,
        model_args.chorus_aligned_root,
        model_args.chorus_trainable_name_patterns,
        model_args.chorus_modality_dropout_rate,
        model_args.chorus_drop_pcd_probability,
        model_args.chorus_match_grid_radius,
        model_args.chorus_discard_unmatched_tokens,
        model_args.chorus_native_match_radius,
        model_args.chorus_min_valid_label_fraction,
        model_args.chorus_fusion_gate_mode,
        model_args.chorus_fusion_transformer_layers,
        model_args.chorus_fusion_transformer_heads,
        model_args.chorus_fusion_transformer_ffn_dim,
        model_args.chorus_fusion_transformer_dropout,
    )


def _maybe_tie_chorus_fourier_head(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    if not getattr(model_args, "chorus_share_fourier_head_with_sonata", False):
        return
    chorus_fusion = getattr(model, "chorus_fusion_encoder", None)
    if chorus_fusion is None:
        return
    input_proj = getattr(getattr(chorus_fusion, "sonata_backbone", None), "input_proj", None)
    if input_proj is not None and getattr(chorus_fusion.fourier_head, "proj", None) is input_proj:
        return
    chorus_fusion.tie_fourier_head_to_sonata(copy_from_sonata=True)
    logger.info_rank0(
        "Tied Chorus fourier_head.proj to Sonata input_proj using shared weights."
    )


def _init_materialized_parameter(
    module: nn.Module,
    name: str,
    param: nn.Parameter,
    initializer_range: float,
) -> nn.Parameter:
    new_param = nn.Parameter(
        torch.empty(
            param.shape,
            dtype=param.dtype,
            device="cpu",
            requires_grad=param.requires_grad,
        ),
        requires_grad=param.requires_grad,
    )

    if isinstance(module, nn.Linear):
        if name == "weight":
            nn.init.normal_(new_param, mean=0.0, std=initializer_range)
        elif name == "bias":
            nn.init.zeros_(new_param)
        else:
            nn.init.normal_(new_param, mean=0.0, std=initializer_range)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(new_param, mean=0.0, std=initializer_range)
        padding_idx = getattr(module, "padding_idx", None)
        if padding_idx is not None and name == "weight":
            with torch.no_grad():
                new_param[padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        if name == "weight":
            nn.init.ones_(new_param)
        elif name == "bias":
            nn.init.zeros_(new_param)
        else:
            nn.init.normal_(new_param, mean=0.0, std=initializer_range)
    elif "bias" in name:
        nn.init.zeros_(new_param)
    else:
        nn.init.normal_(new_param, mean=0.0, std=initializer_range)

    return new_param


def _materialize_meta_tensors(model: "PreTrainedModel") -> list[str]:
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))
    initialized_names = []

    for module_name, module in model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if not getattr(param, "is_meta", False):
                continue

            setattr(
                module,
                param_name,
                _init_materialized_parameter(module, param_name, param, initializer_range),
            )
            initialized_names.append(full_name)

        for buffer_name, buffer in list(module.named_buffers(recurse=False)):
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if not getattr(buffer, "is_meta", False):
                continue

            setattr(module, buffer_name, torch.zeros(buffer.shape, dtype=buffer.dtype, device="cpu"))
            initialized_names.append(full_name)

    return initialized_names


def skip_check_imports() -> None:
    r"""Avoid flash attention import error in custom model files."""
    if not is_env_enabled("FORCE_CHECK_IMPORTS"):
        transformers.dynamic_module_utils.check_imports = get_relative_imports


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    if use_modelscope():
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import snapshot_download  # type: ignore

        revision = (
            "master"
            if model_args.model_revision == "main"
            else model_args.model_revision
        )
        return snapshot_download(
            model_args.model_name_or_path,
            revision=revision,
            cache_dir=model_args.cache_dir,
        )


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    return {"tokenizer": tokenizer}


def _log_multimodal_trainability(model: "PreTrainedModel") -> None:
    if not mm_debug.enabled():
        return

    language_params = list(
        chain(
            getattr(model, "model", nn.Module()).named_parameters(),
            getattr(model, "lm_head", nn.Module()).named_parameters(),
        )
    )

    mm_debug.log(
        "trainability",
        (
            f"point_backbone={mm_debug.format_param_summary(mm_debug.module_param_summary(getattr(model, 'point_backbone', None)))} "
            f"point_proj={mm_debug.format_param_summary(mm_debug.module_param_summary(getattr(model, 'point_proj', None)))} "
            f"language={mm_debug.format_param_summary(mm_debug.named_param_summary(language_params))}"
        ),
    )


def register_autoclass(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()


def load_model(
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    config.point_config["num_bins"] = data_args.num_bins
    for attr in (
        "use_3d_tokens",
        "chorus_fusion_enabled",
        "chorus_repo_root",
        "chorus_config",
        "chorus_checkpoint",
        "chorus_input_mode",
        "chorus_native_match_radius",
        "chorus_use_sonata_lattice_for_coord_matched",
        "chorus_min_valid_label_fraction",
        "chorus_trainable_name_patterns",
        "chorus_mirror_sonata_trainability",
        "chorus_fusion_mode",
        "chorus_missing_policy",
        "chorus_aligned_root",
        "chorus_modality_dropout_rate",
        "chorus_drop_pcd_probability",
        "chorus_match_grid_radius",
        "chorus_exact_match_first",
        "chorus_discard_unmatched_tokens",
        "chorus_fusion_chorus_weight",
        "chorus_fusion_gate_mode",
        "chorus_fusion_gate_hidden_dim",
        "chorus_fusion_transformer_layers",
        "chorus_fusion_transformer_heads",
        "chorus_fusion_transformer_ffn_dim",
        "chorus_fusion_transformer_dropout",
        "chorus_contrastive_loss_weight",
        "chorus_contrastive_loss_final_weight",
        "chorus_contrastive_warmup_ratio",
        "chorus_contrastive_decay_start_ratio",
        "chorus_contrastive_exact_only",
        "chorus_contrastive_backprop_sonata",
        "chorus_contrastive_batch_wide_enabled",
        "chorus_contrastive_cosine_enabled",
        "chorus_contrastive_mse_enabled",
        "chorus_contrastive_info_nce_enabled",
        "chorus_contrastive_cosine_weight",
        "chorus_contrastive_mse_weight",
        "chorus_contrastive_info_nce_weight",
        "chorus_contrastive_temperature",
        "chorus_contrastive_min_matched_tokens",
        "chorus_fourier_head_layer_norm",
        "chorus_init_fourier_head_from_sonata",
        "chorus_share_fourier_head_with_sonata",
        "chorus_disable_drop_path",
    ):
        setattr(config, attr, getattr(model_args, attr))
    patch_config(config, model_args, init_kwargs, is_trainable)
    if getattr(model_args, "chorus_fusion_enabled", False):
        init_kwargs["low_cpu_mem_usage"] = False
        logger.info_rank0(
            "Disabled low_cpu_mem_usage for Chorus fusion because Pointcept/spconv "
            "cannot be constructed under a meta-tensor init context."
        )

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

    if model_args.train_from_scratch:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=model_args.trust_remote_code
        )
        loading_info = None
    else:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            output_loading_info=True,
            **init_kwargs,
        )

    if loading_info is not None:
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        mismatched_keys = loading_info.get("mismatched_keys", [])
        if missing_keys:
            logger.warning_rank0(
                "Weights not found in checkpoint and left for initialization:\n%s",
                _summarize_names(missing_keys),
            )
        if unexpected_keys:
            logger.warning_rank0(
                "Checkpoint weights not used by this model:\n%s",
                _summarize_names(unexpected_keys),
            )
        if mismatched_keys:
            logger.warning_rank0(
                "Checkpoint weights with mismatched shapes:\n%s",
                _summarize_names([str(item) for item in mismatched_keys]),
            )
        if (
            getattr(model_args, "chorus_fusion_enabled", False)
            and getattr(model_args, "chorus_checkpoint", None) is None
        ):
            missing_chorus_keys = [
                key
                for key in missing_keys
                if key.startswith("chorus_fusion_encoder.")
            ]
            if missing_chorus_keys:
                logger.warning_rank0(
                    "`chorus_checkpoint` is null and the loaded model checkpoint "
                    "is missing Chorus weights. These Chorus parameters were left "
                    "randomly initialized:\n%s",
                    _summarize_names(missing_chorus_keys),
                )

    initialized_meta_tensors = _materialize_meta_tensors(model)
    if initialized_meta_tensors:
        logger.warning_rank0(
            "Initialized parameters/buffers that were still on the meta device:\n%s",
            _summarize_names(initialized_meta_tensors),
        )
    _maybe_tie_chorus_fourier_head(model, model_args)

    if model_args.resize_vocab:
        current_vocab_size = model.get_input_embeddings().num_embeddings
        if current_vocab_size != len(tokenizer):
            logger.info_rank0(
                "Resizing token embeddings from %d to %d.",
                current_vocab_size,
                len(tokenizer),
            )
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
            if hasattr(model, "vocab_size"):
                model.vocab_size = len(tokenizer)

    _maybe_attach_chorus_fusion(model, model_args)
    _maybe_tie_chorus_fourier_head(model, model_args)
    patch_model(model, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(model, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if (
                param.data.dtype == torch.float32
                and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)
    _log_multimodal_trainability(model)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(
                f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
            )

    return model
