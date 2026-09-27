# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional, Union

import torch
from typing_extensions import Self

from ..network import AttentionFunction, RopeScaling


@dataclass
class BaseModelArguments:
    r"""Arguments pertaining to the model."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."
        },
    )
    resize_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to resize the tokenizer vocab and the embedding layers."
        },
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."
        },
    )
    add_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: Optional[RopeScaling] = field(
        default=None,
        metadata={
            "help": "Which scaling strategy should be adopted for the RoPE embeddings."
        },
    )
    flash_attn: AttentionFunction = field(
        default=AttentionFunction.AUTO,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(
        default=True,
        metadata={"help": "Whether or not to use reentrant gradient checkpointing."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."
        },
    )
    use_3d_tokens: bool = field(
        default=True,
        metadata={"help": "Whether to insert projected 3D point tokens into the LLM prompt."},
    )
    chorus_fusion_enabled: bool = field(
        default=False,
        metadata={"help": "Use aligned Chorus 3DGS tokens as an optional second 3D modality."},
    )
    chorus_repo_root: str = field(
        default="third_party/chorus",
        metadata={"help": "Vendored Chorus/Pointcept inference root."},
    )
    chorus_config: str = field(
        default="chorus_3dgs",
        metadata={"help": "Chorus config path or alias, e.g. chorus_3dgs."},
    )
    chorus_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional checkpoint for the pretrained Chorus 3DGS encoder. "
                "Set to null when resuming from a GraphWrit3R checkpoint that "
                "already contains Chorus weights."
            )
        },
    )
    chorus_aligned_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root containing aligned split dirs, usually .../train/<scene_id> and .../val/<scene_id>."},
    )
    chorus_input_mode: Literal["prepared", "native"] = field(
        default="prepared",
        metadata={
            "help": "Chorus input source: prepared uses aligned .npy bridge rows, native reads raw 3DGS PLYs online from summary.json."
        },
    )
    chorus_native_match_radius: int = field(
        default=-1,
        metadata={
            "help": "Raw bridge voxel radius used to assign native 3DGS splats to Sonata voxels; -1 uses each scene summary."
        },
    )
    chorus_use_sonata_lattice_for_coord_matched: bool = field(
        default=False,
        metadata={
            "help": (
                "When chorus_fusion_mode=chorus_coord_matched, override Chorus sparse grid_coord "
                "with a physical Sonata/SceneVerse lattice derived from the raw bridge cache. "
                "This is intended for paired PCD+3DGS alignment experiments and is disabled by default."
            )
        },
    )
    chorus_min_valid_label_fraction: float = field(
        default=0.5,
        metadata={
            "help": "Minimum fraction of rows in a pooled Chorus token that must share a valid Sonata label before fusion matching."
        },
    )
    chorus_trainable_name_patterns: str = field(
        default="backbone.enc.enc4",
        metadata={"help": "Comma-separated Chorus parameter-name substrings to keep trainable."},
    )
    chorus_mirror_sonata_trainability: bool = field(
        default=False,
        metadata={"help": "Mirror Sonata point-backbone trainability onto matching Chorus backbone parameters."},
    )
    chorus_fusion_mode: Literal[
        "avg",
        "append",
        "pcd",
        "chorus",
        "chorus_matched",
        "chorus_coord_matched",
        "transformer",
    ] = field(
        default="avg",
        metadata={
            "help": (
                "3D token source: pcd, raw chorus, sidecar-matched chorus, "
                "coordinate-intersection matched chorus, appended pcd+chorus tokens, "
                "voxel-aligned average, or pair-local transformer fusion where both are available."
            )
        },
    )
    chorus_missing_policy: Literal["pcd", "error"] = field(
        default="pcd",
        metadata={"help": "What to do when an aligned Chorus split is missing for a scene."},
    )
    chorus_modality_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Training-only probability of dropping either PCD or Chorus for scenes with both modalities."},
    )
    chorus_drop_pcd_probability: float = field(
        default=0.5,
        metadata={"help": "Conditional probability of dropping PCD, given Chorus modality dropout is activated."},
    )
    chorus_match_grid_radius: int = field(
        default=0,
        metadata={"help": "Chebyshev radius in pooled grid cells for Chorus-to-Sonata token matching; 0 keeps exact matching."},
    )
    chorus_exact_match_first: bool = field(
        default=False,
        metadata={"help": "When radius matching is enabled, prefer exact Chorus/Sonata grid matches and use radius neighbors only as fallback."},
    )
    chorus_discard_unmatched_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "When Chorus/Sonata alignment has matches, discard tokens that did not match either modality. "
                "In avg mode this drops unmatched PCD/Sonata tokens instead of keeping them as PCD-only tokens."
            )
        },
    )
    chorus_fusion_chorus_weight: float = field(
        default=0.5,
        metadata={"help": "Weight of Chorus tokens in avg fusion; Sonata weight is 1 - this value."},
    )
    chorus_fusion_gate_mode: Literal["fixed", "global", "token"] = field(
        default="fixed",
        metadata={
            "help": "Fusion gate mode for avg fusion: fixed uses chorus_fusion_chorus_weight, global learns one shared gate, token learns a per-token gate."
        },
    )
    chorus_fusion_gate_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden width for the per-token Chorus fusion gate MLP."},
    )
    chorus_fusion_transformer_layers: int = field(
        default=2,
        metadata={"help": "Number of pair-local transformer layers for Chorus transformer fusion."},
    )
    chorus_fusion_transformer_heads: int = field(
        default=8,
        metadata={"help": "Attention heads for pair-local Chorus transformer fusion."},
    )
    chorus_fusion_transformer_ffn_dim: int = field(
        default=1024,
        metadata={"help": "Feed-forward width for pair-local Chorus transformer fusion."},
    )
    chorus_fusion_transformer_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout for pair-local Chorus transformer fusion."},
    )
    chorus_contrastive_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Peak/static weight for auxiliary Chorus-to-Sonata token alignment loss during SFT. "
                "When chorus_contrastive_loss_final_weight is set, this is the peak weight."
            )
        },
    )
    chorus_contrastive_loss_final_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional final auxiliary Chorus contrastive loss weight. If set, the trainer ramps "
                "from 0 to chorus_contrastive_loss_weight, holds until "
                "chorus_contrastive_decay_start_ratio, then cosine-decays to this value."
            )
        },
    )
    chorus_contrastive_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of total training steps used to ramp Chorus contrastive weight from 0 to peak."},
    )
    chorus_contrastive_decay_start_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of total training steps at which Chorus contrastive weight starts cosine decay."},
    )
    chorus_contrastive_exact_only: bool = field(
        default=False,
        metadata={"help": "Use only exact Chorus/Sonata token matches for the auxiliary contrastive loss, even if fusion uses radius fallback."},
    )
    chorus_contrastive_backprop_sonata: bool = field(
        default=False,
        metadata={"help": "Allow the auxiliary Chorus contrastive loss to backpropagate through Sonata target tokens."},
    )
    chorus_contrastive_batch_wide_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Pool matched Chorus/Sonata token pairs across the current per-device "
                "microbatch before computing the auxiliary alignment loss."
            )
        },
    )
    chorus_contrastive_cosine_enabled: bool = field(
        default=True,
        metadata={"help": "Include the cosine-distance component in the auxiliary Chorus alignment loss."},
    )
    chorus_contrastive_mse_enabled: bool = field(
        default=True,
        metadata={"help": "Include the MSE component in the auxiliary Chorus alignment loss."},
    )
    chorus_contrastive_info_nce_enabled: bool = field(
        default=True,
        metadata={"help": "Include the bidirectional InfoNCE component in the auxiliary Chorus alignment loss."},
    )
    chorus_contrastive_cosine_weight: float = field(
        default=1.0,
        metadata={"help": "Cosine component weight inside the auxiliary Chorus contrastive loss."},
    )
    chorus_contrastive_mse_weight: float = field(
        default=0.25,
        metadata={"help": "MSE component weight inside the auxiliary Chorus contrastive loss."},
    )
    chorus_contrastive_info_nce_weight: float = field(
        default=0.1,
        metadata={"help": "Bidirectional InfoNCE component weight inside the auxiliary Chorus contrastive loss."},
    )
    chorus_contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for auxiliary Chorus/Sonata token InfoNCE."},
    )
    chorus_contrastive_min_matched_tokens: int = field(
        default=32,
        metadata={"help": "Minimum matched token count required before applying auxiliary Chorus contrastive loss."},
    )
    chorus_fourier_head_layer_norm: bool = field(
        default=False,
        metadata={"help": "Apply LayerNorm after mapping Chorus+Fourier features into Sonata token space."},
    )
    chorus_init_fourier_head_from_sonata: bool = field(
        default=True,
        metadata={"help": "Initialize the Chorus Fourier head from Sonata input_proj."},
    )
    chorus_share_fourier_head_with_sonata: bool = field(
        default=False,
        metadata={
            "help": (
                "Tie Chorus fourier_head.proj to Sonata input_proj as the exact same nn.Linear. "
                "When enabled, the Chorus projection is initialized from Sonata and then shares weights."
            )
        },
    )
    chorus_disable_drop_path: bool = field(
        default=True,
        metadata={"help": "Disable drop-path inside Chorus during token-fusion training."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError(
                "`split_special_tokens` is only supported for slow tokenizers."
            )

        if self.add_tokens is not None:  # support multiple tokens
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",")]

        if self.add_special_tokens is not None:  # support multiple special tokens
            self.add_special_tokens = [
                token.strip() for token in self.add_special_tokens.split(",")
            ]

        if self.chorus_fusion_enabled:
            if self.chorus_aligned_root is None:
                raise ValueError("`chorus_aligned_root` is required when `chorus_fusion_enabled` is true.")
            if self.chorus_native_match_radius < -1:
                raise ValueError("`chorus_native_match_radius` must be -1 or non-negative.")
            if not 0.0 <= self.chorus_min_valid_label_fraction <= 1.0:
                raise ValueError("`chorus_min_valid_label_fraction` must be between 0 and 1.")
            if not 0.0 <= self.chorus_modality_dropout_rate <= 1.0:
                raise ValueError("`chorus_modality_dropout_rate` must be between 0.0 and 1.0.")
            if not 0.0 <= self.chorus_drop_pcd_probability <= 1.0:
                raise ValueError("`chorus_drop_pcd_probability` must be between 0.0 and 1.0.")
            if self.chorus_match_grid_radius < 0:
                raise ValueError("`chorus_match_grid_radius` must be non-negative.")
            if not 0.0 <= self.chorus_fusion_chorus_weight <= 1.0:
                raise ValueError("`chorus_fusion_chorus_weight` must be between 0 and 1.")
            if self.chorus_fusion_gate_hidden_dim <= 0:
                raise ValueError("`chorus_fusion_gate_hidden_dim` must be positive.")
            if self.chorus_contrastive_loss_weight < 0.0:
                raise ValueError("`chorus_contrastive_loss_weight` must be non-negative.")
            if (
                self.chorus_contrastive_loss_final_weight is not None
                and self.chorus_contrastive_loss_final_weight < 0.0
            ):
                raise ValueError("`chorus_contrastive_loss_final_weight` must be non-negative.")
            if not 0.0 <= self.chorus_contrastive_warmup_ratio <= 1.0:
                raise ValueError("`chorus_contrastive_warmup_ratio` must be between 0.0 and 1.0.")
            if not 0.0 <= self.chorus_contrastive_decay_start_ratio <= 1.0:
                raise ValueError("`chorus_contrastive_decay_start_ratio` must be between 0.0 and 1.0.")
            if (
                self.chorus_contrastive_loss_final_weight is not None
                and self.chorus_contrastive_decay_start_ratio < self.chorus_contrastive_warmup_ratio
            ):
                raise ValueError(
                    "`chorus_contrastive_decay_start_ratio` must be >= "
                    "`chorus_contrastive_warmup_ratio` when scheduling is enabled."
                )
            if self.chorus_contrastive_temperature <= 0.0:
                raise ValueError("`chorus_contrastive_temperature` must be positive.")
            if self.chorus_contrastive_min_matched_tokens < 1:
                raise ValueError("`chorus_contrastive_min_matched_tokens` must be positive.")


@dataclass
class ExportArguments:
    r"""Arguments pertaining to the model export."""

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=5,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={
            "help": "The device used in model export, use `auto` to accelerate exporting."
        },
    )


@dataclass
class ModelArguments(
    ExportArguments,
    BaseModelArguments,
):
    r"""Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.

    The class on the most right will be displayed first.
    """

    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={
            "help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."
        },
    )
    device_map: Optional[Union[str, dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={
            "help": "Device map for model placement, derived from training stage. Do not specify it."
        },
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={
            "help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."
        },
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={
            "help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."
        },
    )

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {
            k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()
        }
        return args
