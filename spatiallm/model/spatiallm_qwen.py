# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    Qwen2Model,
    Qwen2ForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    pass  # Ignore the import error if torchsparse is not installed for SpatialLM1.1

from spatiallm.model import PointBackboneType, ProjectorType, mm_debug

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)


class SpatialLMQwenConfig(Qwen2Config):
    model_type = "spatiallm_qwen"


class SpatialLMQwenForCausalLM(Qwen2ForCausalLM):
    config_class = SpatialLMQwenConfig
    _tied_weights_keys = [
        "lm_head.weight",
        "chorus_fusion_encoder.fourier_head.proj.weight",
        "chorus_fusion_encoder.fourier_head.proj.bias",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.point_backbone_type = PointBackboneType(config.point_backbone)
        self.point_backbone = None
        point_config = config.point_config
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            from spatiallm.model.scenescript_encoder import PointCloudEncoder

            self.point_backbone = PointCloudEncoder(
                input_channels=point_config["input_channels"],
                d_model=point_config["embed_channels"],
                conv_layers=point_config["conv_layers"],
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["embed_channels"]
        elif self.point_backbone_type == PointBackboneType.SONATA:
            from spatiallm.model.sonata_encoder import Sonata

            self.point_backbone = Sonata(
                in_channels=point_config["in_channels"],
                order=point_config["order"],
                stride=point_config["stride"],
                enc_depths=point_config["enc_depths"],
                enc_channels=point_config["enc_channels"],
                enc_num_head=point_config["enc_num_head"],
                enc_patch_size=point_config["enc_patch_size"],
                mlp_ratio=point_config["mlp_ratio"],
                mask_token=point_config["mask_token"],
                enc_mode=point_config["enc_mode"],
                enable_fourier_encode=True,
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["enc_channels"][-1]
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

        self.projector_type = ProjectorType(getattr(config, "projector", "linear"))
        if self.projector_type == ProjectorType.LINEAR:
            self.point_proj = nn.Linear(embed_channels, config.hidden_size)
        elif self.projector_type == ProjectorType.MLP:
            self.point_proj = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.GELU(),
                nn.Linear(embed_channels, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")

        self.use_3d_tokens = getattr(config, "use_3d_tokens", True)

        self.point_start_token_id = self.config.point_start_token_id
        self.point_end_token_id = self.config.point_end_token_id
        self.point_token_id = self.config.point_token_id

        # Initialize weights and apply final processing
        self.post_init()
        self.chorus_fusion_encoder = None
        if getattr(config, "chorus_fusion_enabled", False):
            self.initialize_chorus_fusion()

    def initialize_chorus_fusion(self):
        if getattr(self, "chorus_fusion_encoder", None) is not None:
            return
        from spatiallm.model.chorus_fusion import ChorusFusionPointEncoder

        self.chorus_fusion_encoder = ChorusFusionPointEncoder(
            sonata_backbone=self.point_backbone,
            chorus_repo_root=getattr(self.config, "chorus_repo_root"),
            chorus_config=getattr(self.config, "chorus_config", "chorus_3dgs"),
            chorus_checkpoint=getattr(self.config, "chorus_checkpoint"),
            aligned_root=getattr(self.config, "chorus_aligned_root"),
            input_mode=getattr(self.config, "chorus_input_mode", "prepared"),
            native_match_radius=getattr(self.config, "chorus_native_match_radius", -1),
            use_sonata_lattice_for_coord_matched=getattr(
                self.config, "chorus_use_sonata_lattice_for_coord_matched", False
            ),
            min_valid_label_fraction=getattr(self.config, "chorus_min_valid_label_fraction", 0.5),
            trainable_name_patterns=getattr(
                self.config, "chorus_trainable_name_patterns", "backbone.enc.enc4"
            ),
            mirror_sonata_trainability=getattr(
                self.config, "chorus_mirror_sonata_trainability", False
            ),
            fusion_mode=getattr(self.config, "chorus_fusion_mode", "avg"),
            missing_policy=getattr(self.config, "chorus_missing_policy", "pcd"),
            modality_dropout_rate=getattr(self.config, "chorus_modality_dropout_rate", 0.0),
            drop_pcd_probability=getattr(self.config, "chorus_drop_pcd_probability", 0.5),
            match_grid_radius=getattr(self.config, "chorus_match_grid_radius", 0),
            exact_match_first=getattr(self.config, "chorus_exact_match_first", False),
            discard_unmatched_tokens=getattr(
                self.config, "chorus_discard_unmatched_tokens", False
            ),
            fusion_chorus_weight=getattr(self.config, "chorus_fusion_chorus_weight", 0.5),
            fusion_gate_mode=getattr(self.config, "chorus_fusion_gate_mode", "fixed"),
            fusion_gate_hidden_dim=getattr(self.config, "chorus_fusion_gate_hidden_dim", 256),
            fusion_transformer_layers=getattr(
                self.config, "chorus_fusion_transformer_layers", 2
            ),
            fusion_transformer_heads=getattr(
                self.config, "chorus_fusion_transformer_heads", 8
            ),
            fusion_transformer_ffn_dim=getattr(
                self.config, "chorus_fusion_transformer_ffn_dim", 1024
            ),
            fusion_transformer_dropout=getattr(
                self.config, "chorus_fusion_transformer_dropout", 0.0
            ),
            contrastive_loss_weight=getattr(self.config, "chorus_contrastive_loss_weight", 0.0),
            contrastive_exact_only=getattr(self.config, "chorus_contrastive_exact_only", False),
            contrastive_backprop_sonata=getattr(
                self.config, "chorus_contrastive_backprop_sonata", False
            ),
            contrastive_batch_wide_enabled=getattr(
                self.config, "chorus_contrastive_batch_wide_enabled", False
            ),
            contrastive_cosine_enabled=getattr(
                self.config, "chorus_contrastive_cosine_enabled", True
            ),
            contrastive_mse_enabled=getattr(
                self.config, "chorus_contrastive_mse_enabled", True
            ),
            contrastive_info_nce_enabled=getattr(
                self.config, "chorus_contrastive_info_nce_enabled", True
            ),
            contrastive_cosine_weight=getattr(self.config, "chorus_contrastive_cosine_weight", 1.0),
            contrastive_mse_weight=getattr(self.config, "chorus_contrastive_mse_weight", 0.25),
            contrastive_info_nce_weight=getattr(self.config, "chorus_contrastive_info_nce_weight", 0.1),
            contrastive_temperature=getattr(self.config, "chorus_contrastive_temperature", 0.07),
            contrastive_min_matched_tokens=getattr(
                self.config, "chorus_contrastive_min_matched_tokens", 32
            ),
            use_layer_norm=getattr(self.config, "chorus_fourier_head_layer_norm", False),
            init_fourier_head_from_sonata=getattr(
                self.config, "chorus_init_fourier_head_from_sonata", True
            ),
            share_fourier_head_with_sonata=getattr(
                self.config, "chorus_share_fourier_head_with_sonata", False
            ),
            disable_drop_path=getattr(self.config, "chorus_disable_drop_path", True),
        )

    def forward_point_cloud(
        self,
        point_cloud,
        device,
        dtype,
        scene_id: Optional[str] = None,
        chorus_forced_mode: Optional[str] = None,
    ):
        # point cloud has shape (n_points, n_features)
        if getattr(self, "chorus_fusion_encoder", None) is not None:
            encoded_features = self.chorus_fusion_encoder(
                point_cloud,
                scene_id or "",
                device,
                forced_mode=chorus_forced_mode,
            ).unsqueeze(0)
            mm_debug.log(
                "point_cloud_encoded",
                "encoded_features_shape="
                f"{mm_debug.tensor_shape(encoded_features)} "
                "before_projection=chorus_pcd_fusion",
            )
            return self.point_proj(encoded_features.to(dtype))

        # find the points that have nan values
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        mm_debug.log(
            "point_cloud_input",
            "point_cloud_shape="
            f"{mm_debug.tensor_shape(point_cloud)} "
            f"coords_shape={mm_debug.tensor_shape(coords)} "
            f"feats_shape={mm_debug.tensor_shape(feats)} "
            f"point_backbone={self.point_backbone_type.value}",
        )
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_features = self.point_backbone(pc_sparse_tensor)
            mm_debug.log(
                "point_cloud_encoded",
                "encoded_features_shape="
                f"{mm_debug.tensor_shape(encoded_features['context'])} "
                "before_projection=scenescript_context",
            )
            return self.point_proj(encoded_features["context"].to(dtype))
        elif self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(device),
            }
            encoded_features = self.point_backbone(input_dict)
            mm_debug.log(
                "point_cloud_encoded",
                "encoded_features_shape="
                f"{mm_debug.tensor_shape(encoded_features)} "
                "before_projection=sonata_context",
            )
            # add the batch dimension
            encoded_features = encoded_features.unsqueeze(0)
            mm_debug.log(
                "point_cloud_projector_input",
                "projector_input_shape="
                f"{mm_debug.tensor_shape(encoded_features)}",
            )
            return self.point_proj(encoded_features.to(dtype))
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    @staticmethod
    def _decode_scene_id(scene_id_bytes: Optional[torch.Tensor], index: int) -> Optional[str]:
        if scene_id_bytes is None or index >= scene_id_bytes.shape[0]:
            return None
        row = scene_id_bytes[index].detach().cpu().tolist()
        row = [int(value) for value in row if int(value) != 0]
        if not row:
            return None
        return bytes(row).decode("utf-8")

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        if self.point_backbone is None:
            return
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        point_clouds: Optional[torch.Tensor] = None,
        scene_ids: Optional[List[str]] = None,
        scene_id_bytes: Optional[torch.Tensor] = None,
        chorus_forced_modes: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            point_clouds (`torch.Tensor` of shape `(batch_size, n_points, n_features)`, *optional*):
                Point clouds to be used for the point cloud encoder.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("manycore-research/SpatialLM-Qwen-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("manycore-research/SpatialLM-Qwen-0.5B")

        >>> prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"
        >>> conversation = [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}]
        >>> input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(input_ids, point_clouds=point_clouds, max_length=4096)
        >>> tokenizer.batch_decode(generate_ids, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        multimodal_span_edits = None
        self._last_mm_debug = {
            "processed": False,
            "used_3d": False,
            "samples": [],
        }
        collect_mm_debug = mm_debug.enabled() or mm_debug.grads_enabled()
        chorus_fusion = getattr(self, "chorus_fusion_encoder", None)
        batch_wide_contrastive_enabled = bool(
            chorus_fusion is not None
            and getattr(chorus_fusion, "contrastive_batch_wide_enabled", False)
        )
        batch_contrastive_pairs = []
        if batch_wide_contrastive_enabled:
            chorus_fusion.reset_batch_contrastive_loss()

        # compute multimodal embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        should_process_multimodal = input_ids.shape[1] != 1 or self.training
        if should_process_multimodal:
            multimodal_features = []
            mm_debug_samples = []
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                has_3d = (
                    self.use_3d_tokens
                    and self.point_backbone is not None
                    and point_clouds is not None
                    and i < point_clouds.shape[0]
                )
                use_3d = has_3d

                sample_debug = {
                    "sample": i,
                    "has_3d": bool(has_3d),
                    "use_3d": bool(use_3d),
                    "n_3d_tokens": 0,
                    "total_feature_tokens": 0,
                }
                cur_features = []
                if use_3d:
                    cur_scene_id = scene_ids[i] if scene_ids is not None and i < len(scene_ids) else None
                    if cur_scene_id is None:
                        cur_scene_id = self._decode_scene_id(scene_id_bytes, i)
                    cur_chorus_forced_mode = (
                        chorus_forced_modes[i]
                        if chorus_forced_modes is not None and i < len(chorus_forced_modes)
                        else None
                    )
                    if cur_chorus_forced_mode == "":
                        cur_chorus_forced_mode = None
                    cur_point_features = self.forward_point_cloud(
                        point_clouds[i],
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                        scene_id=cur_scene_id,
                        chorus_forced_mode=cur_chorus_forced_mode,
                    ).to(device=inputs_embeds.device).squeeze(0)
                    if batch_wide_contrastive_enabled:
                        contrastive_pair = chorus_fusion.last_contrastive_pair
                        if contrastive_pair is not None:
                            batch_contrastive_pairs.append(contrastive_pair)
                    sample_debug["chorus_forced_mode"] = cur_chorus_forced_mode
                    sample_debug["n_3d_tokens"] = int(cur_point_features.shape[0])
                    cur_features.append(cur_point_features)

                if cur_features:
                    cur_multimodal_features = torch.cat(cur_features, dim=0)
                    sample_debug["total_feature_tokens"] = int(cur_multimodal_features.shape[0])
                    multimodal_features.append(cur_multimodal_features)
                else:
                    multimodal_features.append(None)
                mm_debug_samples.append(sample_debug)

            if batch_wide_contrastive_enabled:
                chorus_fusion.finalize_batch_contrastive_loss(batch_contrastive_pairs)

            # Insert multimodal features into the input ids.
            multimodal_span_edits = []
            new_input_embeds = []
            new_attention_mask = []
            max_num_tokens = 0
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                cur_multimodal_features = multimodal_features[len(new_input_embeds)]
                num_point_start_tokens = (
                    (cur_input_ids == self.config.point_start_token_id).sum().item()
                )
                num_point_end_tokens = (
                    (cur_input_ids == self.config.point_end_token_id).sum().item()
                )
                if cur_multimodal_features is not None:
                    assert num_point_start_tokens == num_point_end_tokens == 1, (
                        "The number of point start tokens and point end tokens should be 1, "
                        f"but got {num_point_start_tokens} and {num_point_end_tokens}."
                    )
                    point_start_token_pos = torch.where(
                        cur_input_ids == self.config.point_start_token_id
                    )[0][0]
                    point_end_token_pos = torch.where(
                        cur_input_ids == self.config.point_end_token_id
                    )[0][0]
                    num_patches = cur_multimodal_features.shape[0]
                    cur_new_input_embeds = torch.cat(
                        (
                            cur_input_embeds[: point_start_token_pos + 1],
                            cur_multimodal_features,
                            cur_input_embeds[point_end_token_pos:],
                        ),
                        dim=0,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            cur_attention_mask[: point_start_token_pos + 1],
                            torch.ones(
                                num_patches, device=cur_attention_mask.device
                            ),
                            cur_attention_mask[point_end_token_pos:],
                        ),
                        dim=0,
                    )
                    if collect_mm_debug:
                        mm_debug_samples[len(new_input_embeds)].update(
                            {
                                "point_start_pos": int(point_start_token_pos.item()),
                                "point_end_pos": int(point_end_token_pos.item()),
                                "original_seq_len": int(cur_input_embeds.shape[0]),
                                "final_seq_len": int(cur_new_input_embeds.shape[0]),
                                "inserted_tokens": int(num_patches),
                                "kept_markers": True,
                            }
                        )
                    multimodal_span_edits.append(
                        (point_start_token_pos, num_patches, point_end_token_pos, True)
                    )
                elif num_point_start_tokens == num_point_end_tokens == 1:
                    point_start_token_pos = torch.where(
                        cur_input_ids == self.config.point_start_token_id
                    )[0][0]
                    point_end_token_pos = torch.where(
                        cur_input_ids == self.config.point_end_token_id
                    )[0][0]
                    cur_new_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:point_start_token_pos],
                            cur_input_embeds[point_end_token_pos + 1 :],
                        ),
                        dim=0,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            cur_attention_mask[:point_start_token_pos],
                            cur_attention_mask[point_end_token_pos + 1 :],
                        ),
                        dim=0,
                    )
                    if collect_mm_debug:
                        mm_debug_samples[len(new_input_embeds)].update(
                            {
                                "point_start_pos": int(point_start_token_pos.item()),
                                "point_end_pos": int(point_end_token_pos.item()),
                                "original_seq_len": int(cur_input_embeds.shape[0]),
                                "final_seq_len": int(cur_new_input_embeds.shape[0]),
                                "inserted_tokens": 0,
                                "kept_markers": False,
                            }
                        )
                    multimodal_span_edits.append(
                        (point_start_token_pos, 0, point_end_token_pos, False)
                    )
                else:
                    cur_new_input_embeds = cur_input_embeds
                    cur_new_attention_mask = cur_attention_mask
                    if collect_mm_debug:
                        mm_debug_samples[len(new_input_embeds)].update(
                            {
                                "original_seq_len": int(cur_input_embeds.shape[0]),
                                "final_seq_len": int(cur_new_input_embeds.shape[0]),
                                "inserted_tokens": 0,
                                "kept_markers": None,
                            }
                        )
                    multimodal_span_edits.append(None)

                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            # pad the new input embeds and attention mask to the max dimension
            for i in range(len(new_input_embeds)):
                cur_input_embeds = new_input_embeds[i]
                last_row = cur_input_embeds[-1]
                padding = last_row.repeat(max_num_tokens - cur_input_embeds.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_input_embeds, padding], dim=0)

                cur_attention_mask = new_attention_mask[i]
                new_attention_mask[i] = F.pad(
                    cur_attention_mask,
                    (0, max_num_tokens - cur_attention_mask.shape[0]),
                    value=0,
                )
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"
            self._last_mm_debug = {
                "processed": True,
                "used_3d": any(sample["n_3d_tokens"] > 0 for sample in mm_debug_samples),
                "samples": mm_debug_samples,
                "input_ids_shape": mm_debug.tensor_shape(input_ids),
                "inputs_embeds_shape": mm_debug.tensor_shape(inputs_embeds),
                "attention_mask_shape": mm_debug.tensor_shape(attention_mask),
            }
            mm_debug.log(
                "model_forward",
                (
                    f"model=spatiallm_qwen training={self.training} "
                    f"input_ids_shape={self._last_mm_debug['input_ids_shape']} "
                    f"inputs_embeds_shape={self._last_mm_debug['inputs_embeds_shape']} "
                    f"attention_mask_shape={self._last_mm_debug['attention_mask_shape']} "
                    f"used_3d={self._last_mm_debug['used_3d']} "
                    f"samples={mm_debug_samples}"
                ),
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        self._last_next_token_loss = None
        self._last_chorus_contrastive_loss = None
        self._last_chorus_contrastive_metrics = {}
        self._last_chorus_modality_usage = []
        if labels is not None:
            if multimodal_span_edits is not None:
                new_labels = []
                max_num_tokens = logits.shape[1]
                for i, span_edit in enumerate(multimodal_span_edits):
                    cur_labels = labels[i]
                    if span_edit is None:
                        cur_new_labels = cur_labels
                    else:
                        start_token_pos, num_patches, end_token_pos, keep_markers = (
                            span_edit
                        )
                        if keep_markers:
                            cur_new_labels = torch.cat(
                                (
                                    cur_labels[: start_token_pos + 1],
                                    torch.full(
                                        (num_patches,),
                                        IGNORE_INDEX,
                                        device=cur_labels.device,
                                    ),
                                    cur_labels[end_token_pos:],
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_labels = torch.cat(
                                (
                                    cur_labels[:start_token_pos],
                                    cur_labels[end_token_pos + 1 :],
                                ),
                                dim=0,
                            )

                    cur_new_labels = F.pad(
                        cur_new_labels,
                        (0, max_num_tokens - cur_new_labels.shape[0]),
                        value=IGNORE_INDEX,
                    )
                    new_labels.append(cur_new_labels)
                labels = torch.stack(new_labels, dim=0)

            assert (
                labels.shape[1] == logits.shape[1]
            ), "The length of labels and logits should be the same"

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )
            self._last_next_token_loss = loss.detach()

            chorus_fusion = getattr(self, "chorus_fusion_encoder", None)
            chorus_aux_loss = getattr(chorus_fusion, "last_aux_loss", None)
            chorus_aux_weight = float(
                getattr(
                    self.config,
                    "chorus_contrastive_loss_current_weight",
                    getattr(self.config, "chorus_contrastive_loss_weight", 0.0),
                )
            )
            chorus_usage = getattr(chorus_fusion, "last_modality_usage", None)
            self._last_chorus_modality_usage = [chorus_usage] if chorus_usage else []
            if chorus_aux_loss is not None and chorus_aux_weight > 0.0:
                self._last_chorus_contrastive_loss = chorus_aux_loss.detach()
                self._last_chorus_contrastive_metrics = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in getattr(chorus_fusion, "last_aux_metrics", {}).items()
                }
                loss = loss + chorus_aux_weight * chorus_aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
                "scene_ids": kwargs.get("scene_ids", None),
                "scene_id_bytes": kwargs.get("scene_id_bytes", None),
                "chorus_forced_modes": kwargs.get("chorus_forced_modes", None),
            }
        )
        return model_inputs


AutoConfig.register("spatiallm_qwen", SpatialLMQwenConfig)
AutoModelForCausalLM.register(SpatialLMQwenConfig, SpatialLMQwenForCausalLM)
