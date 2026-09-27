import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import collate_fn
from pointcept.engines.hooks.misc import CheckpointLoader
from pointcept.models import build_model
from pointcept.utils.config import Config

from .checkpoint_utils import resolve_checkpoint_reference


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Config):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


class LangPretrainerInference:
    """Standalone inference pipeline for LangPretrainerMultiTeacher."""

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(cfg, (str, os.PathLike)):
            self.cfg = Config.fromfile(cfg)
        elif isinstance(cfg, Config):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be a path or Config instance")
        if not hasattr(self.cfg, "inference"):
            raise ValueError("Inference config must define an `inference` section.")
        self.inference_cfg = self.cfg.inference
        feat_keys_cfg = self.cfg.get("feat_keys", ())
        if isinstance(feat_keys_cfg, (list, tuple)):
            self.feat_keys = tuple(feat_keys_cfg)
        elif feat_keys_cfg:
            self.feat_keys = (feat_keys_cfg,)
        else:
            self.feat_keys = ()

        self.logger = logging.getLogger("chorus.inference")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        requested_device = device or "cuda"
        if not str(requested_device).startswith("cuda"):
            raise RuntimeError(
                "Standalone Chorus inference currently requires CUDA. "
                "CPU execution is not supported by the current PT-v3m2 + spconv setup."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Standalone Chorus inference currently requires CUDA, but no CUDA device is available."
            )
        self.device = torch.device(requested_device)

        self.chunk_size = self.inference_cfg.get("chunk_size", None)
        self.default_scene_name = self.inference_cfg.get(
            "default_scene_name", "inference_sample"
        )
        self.return_numpy = bool(self.inference_cfg.get("return_numpy", True))

        self.transform = Compose(self.inference_cfg.get("transform", []))
        test_cfg = self.inference_cfg.get("test_cfg", {})
        self.test_voxelize = (
            TRANSFORMS.build(test_cfg["voxelize"])
            if test_cfg and test_cfg.get("voxelize")
            else None
        )
        self.test_crop = (
            TRANSFORMS.build(test_cfg["crop"])
            if test_cfg and test_cfg.get("crop")
            else None
        )
        post_transform_cfg = test_cfg.get("post_transform", []) if test_cfg else []
        self.post_transform = Compose(post_transform_cfg)
        aug_transform_cfg = test_cfg.get("aug_transform", [[]]) if test_cfg else [[]]
        self.aug_transform = [Compose(cfg_) for cfg_ in aug_transform_cfg]

        self.save_cfg = self.inference_cfg.get("save_features", {}) or {}
        self.output_dir = self.save_cfg.get("output_dir")
        backbone_cfg = self.save_cfg.get("backbone", {}) or {}
        self.save_backbone = bool(backbone_cfg.get("enabled", False))
        teacher_save_cfg = self.save_cfg.get("teachers", {}) or {}
        self.teacher_save = {
            name: cfg_
            for name, cfg_ in teacher_save_cfg.items()
            if cfg_.get("enabled", False)
        }

        checkpoint_hub_cfg = self.inference_cfg.get("checkpoint_hub", None)
        if checkpoint_path:
            self.checkpoint_info = resolve_checkpoint_reference(
                checkpoint_path,
                hub_cfg=checkpoint_hub_cfg,
                logger=self.logger,
            )
        else:
            self.checkpoint_info = dict(
                source="none",
                requested=None,
                resolved_name=None,
                local_path=None,
            )
            self.logger.warning(
                "No standalone Chorus checkpoint was provided; constructing "
                "Chorus architecture with random initial weights. This is only "
                "safe when the parent GraphWrit3R checkpoint contains the "
                "chorus_fusion_encoder weights."
            )

        self.model = build_model(self.cfg.model)
        self.model.eval()
        self.model.to(self.device)
        self.teacher_order = list(getattr(self.model, "teacher_order", []))
        unknown_teachers = sorted(set(self.teacher_save) - set(self.teacher_order))
        if unknown_teachers:
            raise KeyError(
                "save_features.teachers contains unknown teacher names: "
                + ", ".join(unknown_teachers)
            )
        self.active_teachers = [
            name for name in self.teacher_order if name in self.teacher_save
        ]
        if not self.active_teachers and not self.save_backbone:
            raise ValueError(
                "Standalone inference requires at least one enabled teacher in "
                "inference.save_features.teachers or backbone saving enabled."
            )

        if self.checkpoint_info["local_path"] is not None:
            self._load_checkpoint(self.checkpoint_info["local_path"])

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        loader = CheckpointLoader(strict=False)
        trainer = SimpleNamespace(
            cfg=SimpleNamespace(weight=checkpoint_path, resume=False),
            model=self.model,
            logger=self.logger,
        )
        loader.trainer = trainer
        loader.before_train()

    def describe_runtime(self) -> Dict[str, Any]:
        teacher_targets = {}
        for name, cfg in self.teacher_save.items():
            teacher_targets[name] = dict(
                file_name=cfg.get("file_name", "feat.pt"),
            )
        return dict(
            checkpoint=self.checkpoint_info,
            chunk_size=self.chunk_size,
            output_dir=self.output_dir,
            return_numpy=self.return_numpy,
            save_features=dict(
                backbone=dict(enabled=self.save_backbone, file_name="feat.pt"),
                teachers=teacher_targets,
            ),
            test_cfg=_to_builtin(self.inference_cfg.get("test_cfg", {})),
        )

    def __call__(
        self,
        data: Dict[str, np.ndarray],
        *,
        scene_name: Optional[str] = None,
        save: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prepared = self._prepare_input_dict(data, scene_name)
        fragments = prepared.pop("fragment_list")
        scene = prepared.get("name", self.default_scene_name)

        num_points = (
            prepared["segment"].shape[0]
            if prepared.get("segment") is not None
            else fragments[0]["coord"].shape[0]
        )
        accumulators: Dict[str, torch.Tensor] = {}
        counts: Dict[str, torch.Tensor] = {}

        backbone_acc = None
        backbone_count = None

        for fragment in fragments:
            input_dict = collate_fn([fragment])
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device, non_blocking=True)

            with torch.no_grad():
                forward_kwargs = {}
                if self.save_backbone:
                    forward_kwargs["return_backbone"] = True
                if self.chunk_size:
                    forward_kwargs["chunk_size"] = self.chunk_size
                forward_kwargs["teacher_names"] = self.active_teachers
                out_dict = self.model(input_dict, **forward_kwargs)

            teacher_feats = out_dict["point_feat"]
            backbone_feat = out_dict.get("backbone_feat")

            idx_part = input_dict["index"]
            offset_list = input_dict["offset"]

            start = 0
            for end in offset_list:
                slice_idx = idx_part[start:end]
                for name, feat in teacher_feats.items():
                    acc = accumulators.get(name)
                    cnt = counts.get(name)
                    if acc is None:
                        accumulators[name] = torch.zeros(
                            (num_points, feat.size(1)),
                            device=feat.device,
                            dtype=feat.dtype,
                        )
                        counts[name] = torch.zeros(
                            num_points, device=feat.device, dtype=feat.dtype
                        )
                        acc = accumulators[name]
                        cnt = counts[name]
                    acc[slice_idx] += feat[start:end]
                    cnt[slice_idx] += 1

                if backbone_feat is not None:
                    if backbone_acc is None:
                        backbone_acc = torch.zeros(
                            (num_points, backbone_feat.size(1)),
                            device=backbone_feat.device,
                            dtype=backbone_feat.dtype,
                        )
                        backbone_count = torch.zeros(
                            num_points,
                            device=backbone_feat.device,
                            dtype=backbone_feat.dtype,
                        )
                    backbone_acc[slice_idx] += backbone_feat[start:end]
                    backbone_count[slice_idx] += 1
                start = end

        inverse_map = prepared.get("inverse")
        outputs_metadata = {
            "origin_coord": prepared.get("origin_coord"),
            "origin_segment": prepared.get("origin_segment"),
            "origin_feat_mask": prepared.get("origin_feat_mask"),
            "inverse": inverse_map,
        }
        if metadata:
            outputs_metadata.update(metadata)
        outputs = {
            "name": scene,
            "teacher_features": {},
            "backbone_features": None,
            "metadata": outputs_metadata,
        }

        teacher_tensors: Dict[str, torch.Tensor] = {}
        for name, acc in accumulators.items():
            cnt = counts[name]
            valid_mask = cnt > 0
            if torch.any(valid_mask):
                acc[valid_mask] = acc[valid_mask] / cnt[valid_mask].unsqueeze(1)
            if inverse_map is not None:
                inverse_tensor = torch.as_tensor(
                    inverse_map, device=acc.device, dtype=torch.long
                )
                acc = acc[inverse_tensor]
            acc = F.normalize(acc, p=2, dim=1)
            acc_cpu = acc.detach().cpu()
            teacher_tensors[name] = acc_cpu
            outputs["teacher_features"][name] = (
                acc_cpu.numpy() if self.return_numpy else acc_cpu
            )

        if backbone_acc is not None and backbone_count is not None:
            valid_mask = backbone_count > 0
            if torch.any(valid_mask):
                backbone_acc[valid_mask] = backbone_acc[valid_mask] / backbone_count[
                    valid_mask
                ].unsqueeze(1)
            if inverse_map is not None:
                inverse_tensor = torch.as_tensor(
                    inverse_map, device=backbone_acc.device, dtype=torch.long
                )
                backbone_acc = backbone_acc[inverse_tensor]
            backbone_cpu = backbone_acc.detach().cpu()
            teacher_tensors["_backbone"] = backbone_cpu
            outputs["backbone_features"] = (
                backbone_cpu.numpy() if self.return_numpy else backbone_cpu
            )

        if save:
            self._save_outputs(
                scene,
                teacher_tensors,
                source_keep_index=outputs_metadata.get("source_keep_index"),
            )

        return outputs

    def _prepare_input_dict(
        self, data: Dict[str, np.ndarray], scene_name: Optional[str]
    ) -> Dict[str, Any]:
        base_dict = self._format_numpy_inputs(data, scene_name)
        data_dict = self.transform(base_dict)

        result = dict(
            segment=data_dict.pop("segment", None),
            name=data_dict.pop("name", scene_name or self.default_scene_name),
        )
        if "coord" in data_dict:
            result["coord"] = data_dict["coord"]
        if "pc_coord" in data_dict:
            result["pc_coord"] = data_dict["pc_coord"]
        if "pc_segment" in data_dict:
            result["pc_segment"] = data_dict["pc_segment"]
        if "origin_coord" in data_dict:
            result["origin_coord"] = data_dict.pop("origin_coord")
        if "origin_feat_mask" in data_dict:
            result["origin_feat_mask"] = data_dict.pop("origin_feat_mask")
        if "origin_instance" in data_dict:
            result["origin_instance"] = data_dict.pop("origin_instance")
        if "origin_dino_feat" in data_dict:
            result["origin_dino_feat"] = data_dict.pop("origin_dino_feat")
        if "origin_segment" in data_dict:
            result["origin_segment"] = data_dict.pop("origin_segment")
        if "inverse" in data_dict:
            result["inverse"] = data_dict.pop("inverse")

        fragments = [aug(copy.deepcopy(data_dict)) for aug in self.aug_transform]

        fragment_list = []
        for fragment in fragments:
            if self.test_voxelize is not None:
                data_slices = self.test_voxelize(fragment)
            else:
                fragment["index"] = np.arange(fragment["coord"].shape[0])
                data_slices = [fragment]
            for slice_data in data_slices:
                if self.test_crop is not None:
                    crop_list = self.test_crop(slice_data)
                else:
                    crop_list = [slice_data]
                fragment_list.extend(crop_list)

        fragment_list = [self.post_transform(frag) for frag in fragment_list]
        result["fragment_list"] = fragment_list
        return result

    def _format_numpy_inputs(
        self, data: Dict[str, np.ndarray], scene_name: Optional[str]
    ) -> Dict[str, Any]:
        if self.feat_keys:
            missing = [key for key in self.feat_keys if key not in data]
            if missing:
                raise KeyError(
                    "Missing required features: " + ", ".join(sorted(set(missing)))
                )

        formatted: Dict[str, Any] = {}

        num_points = None
        if "coord" in data:
            coord = np.asarray(data["coord"], dtype=np.float32)
            formatted["coord"] = coord
            num_points = coord.shape[0]
        else:
            for key in self.feat_keys:
                if key in data:
                    reference = np.asarray(data[key])
                    num_points = reference.shape[0]
                    break
            if num_points is None:
                raise KeyError(
                    "Unable to determine point count; provide 'coord' or a feature listed in feat_keys."
                )

        def _ensure_num_points(name: str, array: np.ndarray) -> np.ndarray:
            if array.shape[0] != num_points:
                raise ValueError(f"{name} shape mismatch: expected {num_points} rows.")
            return array

        def _assign_float(name: str):
            array = np.asarray(data[name], dtype=np.float32)
            formatted[name] = _ensure_num_points(name, array)

        for key in ["color", "quat", "scale"]:
            if key in data:
                _assign_float(key)

        if "opacity" in data:
            opacity = np.asarray(data["opacity"], dtype=np.float32)
            if opacity.ndim == 1:
                opacity = opacity.reshape(-1, 1)
            formatted["opacity"] = _ensure_num_points("opacity", opacity)

        if "normal" in data:
            _assign_float("normal")

        if "segment" in data:
            segment = np.asarray(data["segment"])
            if segment.ndim == 2:
                segment = segment[:, 0]
            segment = segment.reshape(-1).astype(np.int32)
            formatted["segment"] = _ensure_num_points("segment", segment)
        else:
            formatted["segment"] = np.full((num_points,), -1, dtype=np.int32)

        if "instance" in data:
            instance = np.asarray(data["instance"])
            if instance.ndim == 2:
                instance = instance[:, 0]
            instance = instance.reshape(-1).astype(np.int32)
            formatted["instance"] = _ensure_num_points("instance", instance)
        else:
            formatted["instance"] = np.full((num_points,), -1, dtype=np.int32)

        if "valid_feat_mask" in data:
            mask = np.asarray(data["valid_feat_mask"]).astype(bool)
            formatted["valid_feat_mask"] = _ensure_num_points("valid_feat_mask", mask)

        if "dino_feat" in data:
            dino = np.asarray(data["dino_feat"], dtype=np.float32)
            formatted["dino_feat"] = _ensure_num_points("dino_feat", dino)

        formatted["name"] = scene_name or data.get("name", self.default_scene_name)
        return formatted

    def _save_outputs(
        self,
        scene: str,
        tensors: Dict[str, torch.Tensor],
        *,
        source_keep_index: Optional[np.ndarray] = None,
    ) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        for name, cfg in self.teacher_save.items():
            tensor = tensors.get(name)
            if tensor is None:
                self.logger.warning(
                    "Requested to save teacher '%s' but no features were produced.",
                    name,
                )
                continue
            tensor = tensor.half()
            file_name = cfg.get("file_name", "feat.pt")
            path = self._resolve_output_path(scene, file_name, teacher=name)
            self._dump_tensor(tensor, path, source_keep_index=source_keep_index)

        if self.save_backbone:
            backbone_tensor = tensors.get("_backbone")
            if backbone_tensor is None:
                self.logger.warning("Backbone saving enabled but tensor is missing.")
            else:
                backbone_tensor = backbone_tensor.half()
                path = self._resolve_output_path(scene, "feat.pt", teacher="backbone")
                self._dump_tensor(
                    backbone_tensor,
                    path,
                    source_keep_index=source_keep_index,
                )

    def _resolve_output_path(
        self, scene: str, file_name: str, *, teacher: Optional[str] = None
    ) -> str:
        if "{" in file_name:
            file_name = file_name.format(scene=scene, teacher=teacher or "backbone")
        elif teacher is not None:
            file_name = f"{scene}_{teacher}_{file_name}"
        else:
            file_name = f"{scene}_{file_name}"
        return os.path.join(self.output_dir, file_name)

    def _dump_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        *,
        source_keep_index: Optional[np.ndarray] = None,
    ) -> None:
        if path.endswith(".npy"):
            np.save(path, tensor.numpy())
        else:
            torch.save(tensor, path)
        self.logger.info("Saved features to %s", path)

        if source_keep_index is None:
            return

        index_path = self._index_sidecar_path(path)
        np.save(index_path, np.asarray(source_keep_index, dtype=np.int64))
        self.logger.info("Saved kept-splat index sidecar to %s", index_path)

    @staticmethod
    def _index_sidecar_path(path: str) -> str:
        path_obj = Path(path)
        return str(path_obj.with_name(f"{path_obj.stem}_index.npy"))


__all__ = ["LangPretrainerInference"]
