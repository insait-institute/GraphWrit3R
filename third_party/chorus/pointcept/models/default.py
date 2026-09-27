import copy
import logging
from collections import defaultdict
import torch
import torch.nn as nn
import torch_scatter

from typing import Dict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointSequential
from pointcept.models.utils.teacher_norm import TeacherNorm
from .builder import MODELS, build_model
import pointcept.utils.comm as comm

DEFAULT_AUTO_MASK_MIN_NORM = 0.01


def _emit_repo_log(level: int, message: str, *args) -> None:
    chorus_logger = logging.getLogger("chorus.inference")
    if chorus_logger.handlers:
        chorus_logger.log(level, message, *args)
        return
    pointcept_logger = logging.getLogger("pointcept")
    if pointcept_logger.handlers:
        pointcept_logger.log(level, message, *args)
        return
    if args:
        message = message % args
    print(message)

def _clone_point_for_branch(point: Point) -> Point:
    """Create a lightweight copy of the point for projector branches."""
    point_copy = Point(copy.copy(point))
    if "feat" in point_copy:
        point_copy.feat = point_copy.feat.clone()
    if "sparse_conv_feat" in point_copy:
        point_copy.sparse_conv_feat = point_copy.sparse_conv_feat.replace_feature(
            point_copy.feat
        )
    point_copy.pop("pooling_parent", None)
    point_copy.pop("pooling_inverse", None)
    return point_copy


def _resolve_ptv3_block(block_type: str):
    block_type = block_type.lower()
    if block_type in {"pt-v3m1", "ptv3m1", "pt-v3m1-base"}:
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
            Block,
        )

        return Block
    if block_type in {"pt-v3m2", "ptv3m2", "pt-v3m2-sonata"}:
        from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
            Block,
        )

        return Block
    raise ValueError(f"Unsupported projector block type: {block_type}")


class TeacherProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        projector_cfg: Dict,
        *,
        clone_inputs: bool = True,
    ) -> None:
        super().__init__()
        cfg = projector_cfg.copy()
        self.clone_inputs = clone_inputs
        block_type = cfg.pop("block_type", "pt-v3m2").lower()
        self.block_type = block_type

        input_norm = cfg.pop("input_norm", True)
        output_norm = cfg.pop("output_norm", False)

        head_module = None
        mlp_module = None

        if block_type == "mlp":
            mlp_ratio = cfg.pop("mlp_ratio", 4.0)
            hidden_channels = max(1, int(round(in_channels * mlp_ratio)))
            self.blocks = None
            mlp_module = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
            )
        else:
            block_cls = _resolve_ptv3_block(block_type)
            depth = cfg.pop("depth", 1)
            drop_path_rate = cfg.pop("drop_path_rate", 0.0)
            drop_path_list = cfg.pop("drop_path_list", None)
            if drop_path_list is None:
                if depth > 0:
                    drop_path_list = torch.linspace(0, drop_path_rate, depth).tolist()
                else:
                    drop_path_list = []
            if len(drop_path_list) != depth:
                raise ValueError("drop_path_list length must equal projector depth")

            num_heads = cfg.pop("num_heads", None)
            if num_heads is None:
                raise ValueError("Projector config must specify num_heads")

            patch_size = cfg.pop("patch_size", 1024)
            mlp_ratio = cfg.pop("mlp_ratio", 4.0)
            qkv_bias = cfg.pop("qkv_bias", True)
            qk_scale = cfg.pop("qk_scale", None)
            attn_drop = cfg.pop("attn_drop", 0.0)
            proj_drop = cfg.pop("proj_drop", 0.0)
            layer_scale = cfg.pop("layer_scale", None)
            pre_norm = cfg.pop("pre_norm", True)
            enable_rpe = cfg.pop("enable_rpe", False)
            enable_flash = cfg.pop("enable_flash", True)
            upcast_attention = cfg.pop("upcast_attention", False)
            upcast_softmax = cfg.pop("upcast_softmax", False)

            blocks = []
            for idx in range(depth):
                blocks.append(
                    block_cls(
                        channels=in_channels,
                        num_heads=num_heads,
                        patch_size=patch_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path_list[idx],
                        layer_scale=layer_scale,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                        pre_norm=pre_norm,
                        order_index=idx,
                        cpe_indice_key=None,
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    )
                )

            self.blocks = PointSequential(*blocks) if blocks else None
            head_module = nn.Linear(in_channels, out_channels)

        if cfg:
            _emit_repo_log(
                logging.INFO,
                "Unused projector config keys: %s",
                list(cfg.keys()),
            )

        self.input_norm = nn.LayerNorm(in_channels) if input_norm else nn.Identity()
        if head_module is not None:
            self.head = head_module
            self.mlp_head = None
        else:
            self.head = None
            self.mlp_head = mlp_module
        self.output_norm = nn.LayerNorm(out_channels) if output_norm else nn.Identity()

    def forward(self, point: Point) -> torch.Tensor:
        point_branch = _clone_point_for_branch(point) if self.clone_inputs else point
        point_branch.feat = self.input_norm(point_branch.feat)
        if self.blocks is not None:
            point_branch = self.blocks(point_branch)
            feat = point_branch.feat
        else:
            feat = point_branch.feat
        if self.mlp_head is not None:
            feat = self.mlp_head(feat)
        else:
            feat = self.head(feat)
        feat = self.output_norm(feat)
        return feat


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        freeze_backbone_embedding=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if not freeze_backbone_embedding:
                for p in self.backbone.embedding.parameters():
                    p.requires_grad = True
        # Log backbone key status
        named_params = list(self.backbone.named_parameters())
        t = sum(p.requires_grad for _, p in named_params)
        _emit_repo_log(
            logging.INFO,
            "[Backbone] [keys]: %d/%d trainable, %d/%d frozen",
            t,
            len(named_params),
            len(named_params) - t,
            len(named_params),
        )

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class LangPretrainer(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict, chunk_size=None):
        if (
            chunk_size is not None
            and chunk_size > 0
            and input_dict["coord"].shape[0] > chunk_size
        ):
            return self._chunked_forward(input_dict, chunk_size)
        point = Point(input_dict)
        point_feat = self.backbone(point)
        # normalize the feature
        point_feat["feat"] = nn.functional.normalize(point_feat["feat"], p=2, dim=1)

        # train
        if self.training:
            segment = input_dict["segment"] if "segment" in input_dict.keys() else None
            loss = self.criteria(
                point_feat["feat"],
                input_dict["lang_feat"],
                valid_feat_mask=input_dict["valid_feat_mask"],
                segment=segment,
                epoch_progress=input_dict["epoch_progress"],
            )
            return dict(loss=loss)
        # test
        else:
            return dict(point_feat=point_feat)

    def _chunked_forward(self, input_dict, chunk_size):
        """
        Break the large point set into smaller chunks, pass each chunk through backbone,
        and concat the output features.
        NOTE: This only works if your model's global context isn't critical across chunks.
        """

        # We'll assume "coord" (Nx3 or NxD) is the main key to figure out total #points N.
        # Modify if your data structure is different.
        coords = input_dict["coord"]
        N = coords.shape[0]

        # Prepare a list to store chunk outputs
        chunk_outputs = []

        # We'll do the same logic as normal forward, but inside a loop
        # that processes chunk by chunk.
        is_training = self.training  # track if we are in training or eval

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            # split input_dict into chunks
            chunk_input_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == N:
                    chunk_input_dict[k] = v[start_idx:end_idx]
            if "condition" in input_dict.keys():
                chunk_input_dict["condition"] = input_dict["condition"][0]
            # need to address the 'offset' key separately, which is the same as N
            chunk_input_dict["offset"] = torch.tensor(
                [end_idx - start_idx], device=coords.device
            )
            chunk_point = Point(chunk_input_dict)

            chunk_point_feat = self.backbone(chunk_point)
            chunk_point_feat["feat"] = nn.functional.normalize(
                chunk_point_feat["feat"], p=2, dim=1
            )

            if is_training:
                segment = chunk_input_dict.get("segment", None)
                loss = self.criteria(
                    chunk_point_feat["feat"],
                    chunk_input_dict["lang_feat"],
                    valid_feat_mask=chunk_input_dict["valid_feat_mask"],
                    segment=segment,
                    epoch_progress=chunk_input_dict.get("epoch_progress", None),
                )
                chunk_outputs.append(loss)
            else:
                # If eval, store chunk feats to concat
                chunk_outputs.append(chunk_point_feat["feat"])

        if is_training:
            # sum or average the chunk losses
            # e.g., total_loss = sum(chunk_outputs) / len(chunk_outputs)
            total_loss = torch.stack(chunk_outputs).mean()
            return dict(loss=total_loss)
        else:
            full_feat = torch.cat(chunk_outputs, dim=0)  # shape [N, C]
            del chunk_outputs, chunk_input_dict, chunk_point, chunk_point_feat
            return dict(point_feat={"feat": full_feat})


@MODELS.register_module()
class LangPretrainerMultiTeacher(nn.Module):
    def __init__(
        self,
        backbone=None,
        teachers=None,
        projector_in_channels=None,
        training_mode="joint",
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.use_lora = use_lora
        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.projector_in_channels = projector_in_channels
        if teachers is None or len(teachers) == 0:
            raise ValueError("LangPretrainerMultiTeacher requires at least one teacher")

        self.training_mode = training_mode
        if self.training_mode not in {"joint", "alternating"}:
            raise ValueError(
                "training mode must be one of {'joint', 'alternating'}"
            )

        self.projectors = nn.ModuleDict()
        self.teacher_norms = nn.ModuleDict()
        self.criteria = {}
        self.teacher_meta = {}
        self.teacher_order = []

        multi_teacher = len(teachers) > 1

        for teacher_cfg in teachers:
            teacher_name = teacher_cfg.pop("name")
            if teacher_name is None:
                raise ValueError("Each teacher must define a unique 'name'")
            if teacher_name in self.projectors:
                raise ValueError(f"Duplicated teacher name: {teacher_name}")
            self.teacher_order.append(teacher_name)

            target_key = teacher_cfg.pop("target_key", teacher_name)
            mask_key = teacher_cfg.pop("mask_key", None)
            segment_key = teacher_cfg.pop("segment_key", None)
            loss_weight = teacher_cfg.pop("loss_weight", 1.0)
            mask_min_norm = teacher_cfg.pop("mask_min_norm", DEFAULT_AUTO_MASK_MIN_NORM)

            projector_cfg = teacher_cfg.pop("projector", {})
            out_channels = projector_cfg.pop("out_channels", None)
            if out_channels is None:
                raise ValueError(
                    f"Teacher {teacher_name} projector config must include 'out_channels'"
                )
            teacher_in_channels = projector_cfg.pop(
                "in_channels", self.projector_in_channels
            )
            if teacher_in_channels is None:
                raise ValueError(
                    f"Teacher {teacher_name} projector requires 'in_channels' or a global 'projector_in_channels'"
                )
            clone_inputs = projector_cfg.pop("clone_inputs", None)
            if clone_inputs is None:
                clone_inputs = multi_teacher
            elif not isinstance(clone_inputs, bool):
                raise TypeError(
                    f"Teacher {teacher_name} projector 'clone_inputs' must be a bool when provided"
                )

            projector = TeacherProjector(
                in_channels=teacher_in_channels,
                out_channels=out_channels,
                projector_cfg=projector_cfg,
                clone_inputs=clone_inputs,
            )
            self.projectors[teacher_name] = projector

            teacher_norm_cfg = teacher_cfg.get("teacher_norm") or {}
            if teacher_norm_cfg.get("enabled", False):
                momentum = teacher_norm_cfg.get("momentum", 0.9)
                agg_dims = tuple(teacher_norm_cfg.get("agg_dims", (0,)))
                eps = teacher_norm_cfg.get("eps", 1e-6)
                teacher_norm = TeacherNorm(
                    feature_dim=out_channels,
                    agg_dims=agg_dims,
                    momentum=momentum,
                    eps=eps,
                )
            else:
                teacher_norm = nn.Identity()
            self.teacher_norms[teacher_name] = teacher_norm

            criteria_cfg = teacher_cfg.pop("criteria", [])
            self.criteria[teacher_name] = build_criteria(criteria_cfg)

            self.teacher_meta[teacher_name] = dict(
                target_key=target_key,
                mask_key=mask_key,
                segment_key=segment_key,
                loss_weight=loss_weight,
                mask_min_norm=mask_min_norm,
                **teacher_cfg, # any remaining
            )

        self.register_buffer("_teacher_pointer", torch.zeros(1, dtype=torch.long))
        if comm.is_main_process():
            self.count_params()


    def count_params(self):
        # total number of parameters of backbone enc, dec, embeddings, and each projector
        def _count_parameters(module: nn.Module) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        backbone_enc = getattr(self.backbone, "enc", None)
        backbone_dec = getattr(self.backbone, "dec", None)
        backbone_emb = getattr(self.backbone, "embedding", None)

        enc_params = _count_parameters(backbone_enc)
        dec_params = _count_parameters(backbone_dec)
        emb_params = _count_parameters(backbone_emb)
        backbone_total = enc_params + dec_params + emb_params

        if backbone_total > 0:
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][Backbone] Encoder: %s (%.2f%%)",
                format(enc_params, ","),
                enc_params / backbone_total * 100 if backbone_total else 0,
            )
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][Backbone] Decoder: %s (%.2f%%)",
                format(dec_params, ","),
                dec_params / backbone_total * 100 if backbone_total else 0,
            )
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][Backbone] Embedding: %s (%.2f%%)",
                format(emb_params, ","),
                emb_params / backbone_total * 100 if backbone_total else 0,
            )
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][Backbone] Total: %s",
                format(backbone_total, ","),
            )

        # Per-projector parameter counts and totals
        projectors_total = 0
        for _name, _proj in self.projectors.items():
            n_params = _count_parameters(_proj)
            projectors_total += n_params
            _emit_repo_log(
                logging.INFO,
                "[ParamCount][Projector:%s] Total: %s",
                _name,
                format(n_params, ","),
            )

    def _select_mask(self, meta, input_dict):
        mask_key = meta.get("mask_key")
        if mask_key is None:
            return None
        mask = input_dict.get(mask_key)
        if mask is None:
            return None
        return mask.bool()

    def _infer_mask_from_target(self, teacher_name, target, pred_length):
        meta = self.teacher_meta[teacher_name] # dict_keys(['target_key', 'mask_key', 'segment_key', 'loss_weight', 'mask_min_norm'])
        threshold = meta.get("mask_min_norm")

        if target.ndim == 0:
            return torch.ones(pred_length, dtype=torch.bool, device=target.device)
        if target.shape[0] != pred_length:
            raise ValueError(
                f"Teacher {teacher_name} target length {target.shape[0]} does not match predictions {pred_length}"
            )
        if target.numel() == 0:
            return torch.zeros(pred_length, dtype=torch.bool, device=target.device)

        target_flat = target.reshape(target.shape[0], -1).float()
        norms = torch.linalg.vector_norm(target_flat, ord=2, dim=1)
        return norms > float(threshold)

    def _prepare_target(self, teacher_name, target, mask):
        norm_module = self.teacher_norms[teacher_name]
        if isinstance(norm_module, nn.Identity) or target is None:
            return target
        target = target.clone() # potential mem issue here
        if mask is None:
            return norm_module(target)
        valid_count = mask.sum()
        if valid_count == 0:
            return target
        target[mask] = norm_module(target[mask])
        return target

    def _compute_teacher_loss(self, teacher_name, point_feat, input_dict):
        projector = self.projectors[teacher_name]
        meta = self.teacher_meta[teacher_name]
        pred = projector(point_feat)
        l2_norm_pred = meta.get("l2_norm_pred", True)
        if l2_norm_pred:
            pred = nn.functional.normalize(pred, p=2, dim=1)
        device = pred.device

        target_key = meta["target_key"]
        target = input_dict.get(target_key, None)
        assert target is not None, f"Teacher {teacher_name} requires target key '{target_key}', current input keys: {list(input_dict.keys())}"
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, device=device, dtype=pred.dtype)
        else:
            target = target.to(device=device, dtype=pred.dtype)

        mask = self._select_mask(meta, input_dict)
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=device).bool()

        if mask is None and target is not None:
            mask = self._infer_mask_from_target(teacher_name, target, pred.shape[0])
        # lastly check on the target itself
        # row_mask = target.norm(p=1, dim=1) != 0 
        # mask = mask & row_mask
        if mask.shape[0] != pred.shape[0]:
            raise ValueError(
                f"Teacher {teacher_name} mask length {mask.shape[0]} does not match predictions {pred.shape[0]}"
            )
        target = self._prepare_target(teacher_name, target, mask)

        extra_kwargs = {}
        if mask is not None:
            extra_kwargs["valid_feat_mask"] = mask
        segment_key = meta.get("segment_key")
        if segment_key is not None and segment_key in input_dict:
            segment_value = input_dict[segment_key]
            if not isinstance(segment_value, torch.Tensor):
                segment_value = torch.as_tensor(segment_value, device=device)
            else:
                segment_value = segment_value.to(device)
            extra_kwargs["segment"] = segment_value
        else:
            extra_kwargs["segment"] = None
        epoch_progress = input_dict.get("epoch_progress", None)
        if epoch_progress is not None:
            extra_kwargs["epoch_progress"] = epoch_progress

        criteria = self.criteria[teacher_name]
        if target is None:
            loss = torch.tensor(0.0, device=device)
        else:
            loss = criteria(pred, target, **extra_kwargs)

        weighted_loss = loss * meta.get("loss_weight", 1.0)
        return dict(
            loss=weighted_loss,
            raw_loss=loss,
            pred=pred,
            mask=mask,
        )

    def _advance_pointer(self):
        self._teacher_pointer.add_(1)
        if self._teacher_pointer.item() >= len(self.teacher_order):
            self._teacher_pointer.zero_()

    def _resolve_eval_teacher_names(self, teacher_names=None):
        if teacher_names is None:
            return list(self.teacher_order)
        unknown = [name for name in teacher_names if name not in self.projectors]
        if unknown:
            raise KeyError(
                "Unknown teacher names requested for inference: "
                + ", ".join(sorted(set(unknown)))
            )
        return list(dict.fromkeys(teacher_names))

    def forward(
        self, input_dict, chunk_size=None, return_backbone=False, teacher_names=None
    ):
        if (
            chunk_size is not None
            and chunk_size > 0
            and input_dict["coord"].shape[0] > chunk_size
        ):
            return self._chunked_forward(
                input_dict,
                chunk_size,
                return_backbone=return_backbone,
                teacher_names=teacher_names,
            )

        point = Point(input_dict)
        point_feat = self.backbone(point)
        if isinstance(point_feat, Point):
            _emit_repo_log(
                logging.INFO,
                "[CHORUS_DEBUG] deepest_encoder_feat_shape=%s",
                tuple(point_feat.feat.shape),
            )

        if isinstance(point_feat, Point) and self.backbone.enc_mode:
            while "pooling_parent" in point_feat.keys(): # GridUnpooling
                assert "pooling_inverse" in point_feat.keys()
                parent = point_feat.pop("pooling_parent")
                inverse = point_feat.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_feat.feat[inverse]], dim=-1) # point.feat[inverse] does indexing to align with parent.feat shape
                point_feat = parent
        if isinstance(point_feat, Point):
            _emit_repo_log(
                logging.INFO,
                "[CHORUS_DEBUG] encoder_feat_shape=%s",
                tuple(point_feat.feat.shape),
            )
        if return_backbone:
            backbone_feat = point_feat.feat if isinstance(point_feat, Point) else point_feat

        if self.training:
            if self.training_mode == "alternating":
                index = int(self._teacher_pointer.item())
                teacher_name = self.teacher_order[index]
                result = self._compute_teacher_loss(teacher_name, point_feat, input_dict)
                self._advance_pointer()
                return dict(
                    loss=result["loss"],
                    active_teacher=teacher_name,
                    per_teacher_loss={teacher_name: result["raw_loss"].detach()},
                )

            total_loss = 0.0
            per_teacher_loss = {}
            for teacher_name in self.teacher_order:
                result = self._compute_teacher_loss(teacher_name, point_feat, input_dict)
                total_loss = total_loss + result["loss"]
                per_teacher_loss[teacher_name] = result["raw_loss"].detach()
            return dict(loss=total_loss, per_teacher_loss=per_teacher_loss)

        # evaluation / testing: return normalized features per teacher
        eval_teacher_names = self._resolve_eval_teacher_names(teacher_names)
        projected = {}
        for teacher_name in eval_teacher_names:
            projector = self.projectors[teacher_name]
            feat = projector(point_feat)
            l2_norm_proj = self.teacher_meta[teacher_name].get("l2_norm_pred", True)
            if l2_norm_proj:
                projected[teacher_name] = nn.functional.normalize(feat, p=2, dim=1)
        result = dict(point_feat=projected)
        if return_backbone:
            result["backbone_feat"] = backbone_feat
        return result

    def _chunked_forward(
        self, input_dict, chunk_size, return_backbone=False, teacher_names=None
    ):
        coords = input_dict["coord"]
        N = coords.shape[0]
        device = coords.device

        is_training = self.training
        chunk_losses = []
        per_teacher_raw = defaultdict(list)

        if self.training_mode == "alternating" and is_training:
            teacher_index = int(self._teacher_pointer.item())
            active_teacher = self.teacher_order[teacher_index]
        else:
            active_teacher = None

        eval_teacher_names = None
        if not is_training:
            eval_teacher_names = self._resolve_eval_teacher_names(teacher_names)
        projected_chunks = defaultdict(list) if not is_training else None
        backbone_chunks = [] if (return_backbone and not is_training) else None
        backbone_feat_dim = self.projector_in_channels
        backbone_device = device

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            chunk_input_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == N:
                    chunk_input_dict[key] = value[start_idx:end_idx]
            if "condition" in input_dict:
                chunk_input_dict["condition"] = input_dict["condition"][0]
            chunk_input_dict["offset"] = torch.tensor(
                [end_idx - start_idx], device=device
            )
            if "epoch_progress" in input_dict:
                chunk_input_dict["epoch_progress"] = input_dict["epoch_progress"]

            chunk_point = Point(chunk_input_dict)
            chunk_point_feat = self.backbone(chunk_point)
            if isinstance(chunk_point_feat, Point):
                _emit_repo_log(
                    logging.INFO,
                    "[CHORUS_DEBUG] deepest_encoder_feat_shape=%s chunk=[%d:%d]",
                    tuple(chunk_point_feat.feat.shape),
                    start_idx,
                    end_idx,
                )

            if isinstance(chunk_point_feat, Point) and self.backbone.enc_mode:
                while "pooling_parent" in chunk_point_feat.keys(): # GridUnpooling
                    assert "pooling_inverse" in chunk_point_feat.keys()
                    parent = chunk_point_feat.pop("pooling_parent")
                    inverse = chunk_point_feat.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, chunk_point_feat.feat[inverse]], dim=-1) # point.feat[inverse] does indexing to align with parent.feat shape
                    chunk_point_feat = parent
            if isinstance(chunk_point_feat, Point):
                _emit_repo_log(
                    logging.INFO,
                    "[CHORUS_DEBUG] encoder_feat_shape=%s chunk=[%d:%d]",
                    tuple(chunk_point_feat.feat.shape),
                    start_idx,
                    end_idx,
                )

            if is_training:
                if self.training_mode == "alternating":
                    result = self._compute_teacher_loss(
                        active_teacher, chunk_point_feat, chunk_input_dict
                    )
                    chunk_losses.append(result["loss"])
                    per_teacher_raw[active_teacher].append(result["raw_loss"].detach())
                else:
                    chunk_total = None
                    for teacher_name in self.teacher_order:
                        result = self._compute_teacher_loss(
                            teacher_name, chunk_point_feat, chunk_input_dict
                        )
                        per_teacher_raw[teacher_name].append(
                            result["raw_loss"].detach()
                        )
                        if chunk_total is None:
                            chunk_total = result["loss"]
                        else:
                            chunk_total = chunk_total + result["loss"]
                    if chunk_total is None:
                        chunk_total = torch.tensor(0.0, device=device)
                    chunk_losses.append(chunk_total)
            else:
                for teacher_name in eval_teacher_names:
                    projector = self.projectors[teacher_name]
                    feat = projector(chunk_point_feat)
                    feat = nn.functional.normalize(feat, p=2, dim=1)
                    projected_chunks[teacher_name].append(feat)

                if return_backbone:
                    chunk_feat_tensor = (
                        chunk_point_feat.feat
                        if isinstance(chunk_point_feat, Point)
                        else chunk_point_feat
                    )
                    backbone_chunks.append(chunk_feat_tensor)
                    if backbone_feat_dim is None and chunk_feat_tensor.ndim == 2:
                        backbone_feat_dim = chunk_feat_tensor.size(1)
                    backbone_device = chunk_feat_tensor.device

        if is_training:
            if chunk_losses:
                total_loss = torch.stack(chunk_losses).mean()
            else:
                total_loss = torch.tensor(0.0, device=device)

            if self.training_mode == "alternating":
                raw_list = per_teacher_raw.get(active_teacher, [])
                if raw_list:
                    per_teacher_mean = torch.stack(raw_list).mean().detach()
                else:
                    per_teacher_mean = torch.tensor(0.0, device=device)
                self._advance_pointer()
                return dict(
                    loss=total_loss,
                    active_teacher=active_teacher,
                    per_teacher_loss={active_teacher: per_teacher_mean},
                )

            per_teacher_loss = {}
            for teacher_name in self.teacher_order:
                values = per_teacher_raw.get(teacher_name, [])
                if values:
                    per_teacher_loss[teacher_name] = (
                        torch.stack(values).mean().detach()
                    )
                else:
                    per_teacher_loss[teacher_name] = torch.tensor(0.0, device=device)
            return dict(loss=total_loss, per_teacher_loss=per_teacher_loss)

        projected = {}
        for teacher_name in eval_teacher_names:
            chunks = projected_chunks.get(teacher_name, [])
            if chunks:
                projected[teacher_name] = torch.cat(chunks, dim=0)
            else:
                out_dim = self.projectors[teacher_name].head.out_features
                projected[teacher_name] = torch.empty(0, out_dim, device=device)
        result = dict(point_feat=projected)
        if return_backbone:
            if backbone_chunks and len(backbone_chunks) > 0:
                result["backbone_feat"] = torch.cat(backbone_chunks, dim=0)
            else:
                feat_dim = backbone_feat_dim if backbone_feat_dim is not None else 0
                fill_device = backbone_device if backbone_device is not None else device
                result["backbone_feat"] = torch.empty(0, feat_dim, device=fill_device)
        return result

@MODELS.register_module()
class DefaultSegmentorSkip(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        # (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class DefaultPretrainer(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        # seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss)
        # eval
        elif "clip_feat" in input_dict.keys():
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss, seg_logits=feat)
        # test
        else:
            return dict(seg_logits=feat)
