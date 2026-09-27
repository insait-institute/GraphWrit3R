import os
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union, Sequence, Optional, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import NORMALIZATION_PRESET
from spatiallm.layout.scene_graph_layout import SceneGraphLayout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors
from spatiallm.pcd.transform import Compose

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer

    PointCloudInput = Union[str, dict, NDArray]

LAYOUT_S_PLACEHOLDER = os.environ.get("LAYOUT_S_PLACEHOLDER", "<|layout_s|>")
LAYOUT_E_PLACEHOLDER = os.environ.get("LAYOUT_E_PLACEHOLDER", "<|layout_e|>")
POINT_S_TOKEN = os.environ.get("POINT_S_TOKEN", "<|point_start|>")
POINT_E_TOKEN = os.environ.get("POINT_E_TOKEN", "<|point_end|>")
POINT_CLOUD_PLACEHOLDER = os.environ.get("POINT_CLOUD_PLACEHOLDER", "<point_cloud>")


class SpatialLMPlugin:
    def __init__(
        self,
        point_token: str = "<|point_pad|>",
        num_bins: int = 1280,
        do_augmentation: bool = False,
        random_scaling: bool = False,
        random_rotation: bool = False,
        scene_graph_mode: str = "full",
        pcd_only_random_scaling: bool = False,
        pcd_only_scaling_min: float = 0.75,
        pcd_only_scaling_max: float = 1.25,
        chorus_fusion_enabled: bool = False,
        chorus_aligned_root: Optional[str] = None,
        chorus_fusion_mode: str = "avg",
        chorus_modality_dropout_rate: float = 0.0,
        chorus_drop_pcd_probability: float = 0.5,
    ):
        self.point_token = point_token
        valid_modes = {"full", "relationships_only", "auto"}
        if scene_graph_mode not in valid_modes:
            raise ValueError(
                f"scene_graph_mode must be one of {sorted(valid_modes)}, got: {scene_graph_mode}"
            )
        self.scene_graph_mode = scene_graph_mode

        global_extent = NORMALIZATION_PRESET["world"]
        self.num_bins = num_bins
        self.grid_size = (global_extent[1] - global_extent[0]) / self.num_bins
        self.do_augmentation = do_augmentation
        self.random_scaling = random_scaling
        self.random_rotation = random_rotation
        self.pcd_only_random_scaling = pcd_only_random_scaling
        self.pcd_only_scaling_min = float(pcd_only_scaling_min)
        self.pcd_only_scaling_max = float(pcd_only_scaling_max)
        self.chorus_fusion_enabled = chorus_fusion_enabled
        self.chorus_aligned_root = chorus_aligned_root
        self.chorus_fusion_mode = "chorus" if chorus_fusion_mode == "3dgs" else chorus_fusion_mode
        self.chorus_modality_dropout_rate = float(chorus_modality_dropout_rate)
        self.chorus_drop_pcd_probability = float(chorus_drop_pcd_probability)
        self.runtime_split = "eval"
        self._chorus_scene_ids_cache: Optional[set[str]] = None
        self.augmentation = Compose(
            [
                dict(type="RandomColorGrayScale", p=0.05),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.75, ratio=0.1),
                dict(type="ChromaticJitter", p=0.8, std=0.05),
                dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                dict(type="RandomColorDrop", p=0.1, color_augment=0.0),
                dict(type="RandomJitter", sigma=0.025, clip=0.05, ratio=0.8, p=0.9),
                dict(type="RandomJitter", sigma=0.2, clip=0.2, ratio=0.05, p=0.85),
                dict(type="RandomJitter", sigma=0.4, clip=1.0, ratio=0.001, p=0.75),
                dict(type="RandomJitter", sigma=0.5, clip=4.0, ratio=0.0005, p=0.7),
                dict(
                    type="ElasticDistortion",
                    distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    p=[0.85, 0.5],
                ),
            ]
        )

        self.transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=self.grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=self.num_bins,
                ),
            ]
        )

    def set_runtime_split(self, split: str) -> None:
        self.runtime_split = str(split or "eval").lower()

    @staticmethod
    def _scene_id_from_point_cloud_input(point_cloud: "PointCloudInput") -> Optional[str]:
        if isinstance(point_cloud, str):
            return os.path.splitext(os.path.basename(point_cloud))[0]
        if isinstance(point_cloud, dict):
            for key in ("scene_id", "name", "path"):
                value = point_cloud.get(key)
                if isinstance(value, str) and value:
                    return os.path.splitext(os.path.basename(value))[0]
        return None

    def _load_chorus_scene_ids(self) -> set[str]:
        if self._chorus_scene_ids_cache is not None:
            return self._chorus_scene_ids_cache
        scene_ids: set[str] = set()
        if self.chorus_aligned_root:
            root = Path(self.chorus_aligned_root)
            if root.exists():
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name in {"train", "val", "test", "validation"}:
                        scene_ids.update(grandchild.name for grandchild in child.iterdir() if grandchild.is_dir())
                    else:
                        scene_ids.add(child.name)
        self._chorus_scene_ids_cache = scene_ids
        return scene_ids

    def _has_chorus_cache(self, scene_id: Optional[str]) -> bool:
        if not scene_id:
            return False
        return scene_id in self._load_chorus_scene_ids()

    def _choose_forced_chorus_mode(self, scene_id: Optional[str]) -> Optional[str]:
        if (
            not self.pcd_only_random_scaling
            or self.runtime_split != "train"
            or not self.chorus_fusion_enabled
        ):
            return None

        mode = self.chorus_fusion_mode
        if mode == "pcd":
            return "pcd"
        if mode == "chorus":
            return "chorus" if self._has_chorus_cache(scene_id) else "pcd"

        has_chorus = self._has_chorus_cache(scene_id)
        if not has_chorus:
            return "pcd"

        if mode not in {"avg", "chorus_coord_matched", "transformer"}:
            return None
        if (
            self.chorus_modality_dropout_rate > 0.0
            and np.random.random() < self.chorus_modality_dropout_rate
        ):
            if np.random.random() < self.chorus_drop_pcd_probability:
                return "chorus"
            return "pcd"
        return "both"

    def _is_relationships_only_json(self, layout_content: str) -> bool:
        """Return True when JSON has relationships but no object geometry payload."""
        try:
            data = json.loads(layout_content)
        except json.JSONDecodeError:
            return False

        if not isinstance(data, dict):
            return False

        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            return False

        objects = data.get("objects", [])
        return not isinstance(objects, list) or len(objects) == 0

    def _preprocess_point_cloud(self, point_cloud: dict) -> np.ndarray:
        r"""
        Pre-processes a single point cloud.
        """
        point_cloud = self.transform(point_cloud)
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        color = point_cloud["color"]
        assert len(coord) == len(xyz) == len(color)
        return np.concatenate([coord, xyz, color], axis=1)

    def _regularize_point_clouds(
        self, point_clouds: Sequence["PointCloudInput"], **kwargs
    ) -> torch.Tensor:
        points_list = []
        max_len = 0
        for point_cloud in point_clouds:
            if not isinstance(point_cloud, dict):
                raise ValueError(
                    "Point cloud input must be a dictionary with 'name' and 'coord' keys."
                )
            point_feats = self._preprocess_point_cloud(point_cloud, **kwargs)
            max_len = max(max_len, len(point_feats))
            points_list.append(point_feats)

        for i in range(len(points_list)):
            points_list[i] = np.pad(
                points_list[i],
                ((0, max_len - len(points_list[i])), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

        # convert list of point clouds to batch with shape (batch_size, max_len, 3)
        return torch.as_tensor(np.stack(points_list, axis=0))

    def _get_mm_inputs(
        self,
        batched_messages: Sequence[Dict[str, str]],
        point_clouds: Sequence["PointCloudInput"],
    ) -> dict:
        input_dict = {"point_clouds": None}  # default key

        point_clouds_data = []
        transformations = []
        forced_chorus_modes = []
        for pcd_path in point_clouds:
            scene_id = self._scene_id_from_point_cloud_input(pcd_path)
            forced_mode = self._choose_forced_chorus_mode(scene_id)
            forced_chorus_modes.append(forced_mode or "")
            pcd = load_o3d_pcd(pcd_path)
            points, colors = get_points_and_colors(pcd)

            if self.do_augmentation:
                data_aug = {"name": "pcd", "coord": points, "color": colors}
                data_aug = self.augmentation(data_aug)
                points = data_aug["coord"]
                colors = data_aug["color"]

            # randomly apply scale and rotation transformation to the point cloud
            # TODO: rotation is disabled because directional relationship predicates
            # (e.g. "behind", "3 o'clock far") are not rotated together with the
            # point cloud and object positions, causing a train/label mismatch.
            # To re-enable, implement predicate rotation in SceneGraphLayout.rotate().
            angle_z = 0.0

            conditional_pcd_scaling = (
                forced_mode == "pcd"
                and self.pcd_only_random_scaling
                and self.runtime_split == "train"
            )
            if self.random_scaling or conditional_pcd_scaling:
                scaling = np.random.uniform(
                    self.pcd_only_scaling_min,
                    self.pcd_only_scaling_max,
                )
            else:
                scaling = 1.0
            #scaling = np.random.uniform(1.0, 1.0) # disable for 2d alignment
            rotmat = R.from_rotvec(np.array([0, 0, angle_z])).as_matrix()
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            center_pt = (min_bound + max_bound) / 2
            scaled_points = (points - center_pt) * scaling
            transformed_points = (rotmat @ scaled_points.T).T + center_pt
            # store transformation parameters for sync the augmentation to the layout
            transformations.append(
                {
                    "angle_z": angle_z,
                    "center_pt": center_pt,
                    "scaling": scaling,
                    "min_bound": np.min(transformed_points, axis=0),
                    "transformed_points": transformed_points,
                }
            )

            point_cloud = {"name": "pcd", "coord": transformed_points, "color": colors}
            point_clouds_data.append(point_cloud)

        # Here we assume each conversation has exactly one point cloud
        assert len(batched_messages) == len(point_clouds_data)
        processed_messages = []
        processed_object_source_ids = []
        for mi, messages in enumerate(batched_messages):
            processed, object_source_ids = self.process_messages(
                messages, [transformations[mi]]
            )
            processed_messages.append(processed)
            processed_object_source_ids.append(object_source_ids)

        if len(processed_messages) != 0:
            input_dict["messages"] = processed_messages
            input_dict["object_source_ids"] = processed_object_source_ids
        if len(point_clouds_data) != 0:
            # convert point clouds to batched tensors with shape (batch_size, max_len, 9)
            input_dict["point_clouds"] = self._regularize_point_clouds(
                point_clouds_data
            )
        if any(forced_chorus_modes):
            input_dict["chorus_forced_modes"] = forced_chorus_modes
        return input_dict

    def _validate_input(
        self,
        point_clouds: Sequence["PointCloudInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(point_clouds) != 0 and self.point_token is None:
            raise ValueError(
                "This model does not support point cloud input. Please check whether the correct `template` is used."
            )

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        point_clouds: Sequence["PointCloudInput"],
        tokenizer: "PreTrainedTokenizer",
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(point_clouds)
        return input_ids, labels

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        transformations: Sequence[dict],
    ) -> Tuple[List[Dict[str, str]], Optional[List[int]]]:
        r"""
        Pre-processes input messages to sync the transformation between point cloud and layout.
        """
        self._validate_input(transformations)
        messages = deepcopy(messages)
        num_point_tokens = 0
        object_source_ids: Optional[List[int]] = None

        for message in messages:
            content = message["content"]
            if LAYOUT_S_PLACEHOLDER in content and LAYOUT_E_PLACEHOLDER in content:
                layout_start_pos = content.index(LAYOUT_S_PLACEHOLDER)
                layout_end_pos = content.index(LAYOUT_E_PLACEHOLDER)
                layout_content = content[
                    layout_start_pos + len(LAYOUT_S_PLACEHOLDER) : layout_end_pos
                ]
                # Detect format: JSON scene graph vs legacy layout
                stripped_content = layout_content.strip()
                is_json_scene_graph = stripped_content.startswith('{')
                is_relationships_only = (
                    is_json_scene_graph
                    and self._is_relationships_only_json(stripped_content)
                )
                should_skip_layout_transforms = (
                    is_json_scene_graph
                    and (
                        self.scene_graph_mode == "relationships_only"
                        or (
                            self.scene_graph_mode == "auto"
                            and is_relationships_only
                        )
                    )
                )

                if should_skip_layout_transforms:
                    if not hasattr(self, "_debug_passthrough_count"):
                        self._debug_passthrough_count = 0
                    if self._debug_passthrough_count < 3:
                        print("DEBUG passthrough layout:", stripped_content[:300])
                        self._debug_passthrough_count += 1
                else:
                    transformation = transformations[num_point_tokens - 1]
                    min_bound = transformation["min_bound"]
                    center_pt = transformation["center_pt"]
                    scaling = transformation["scaling"]
                    transformed_points = transformation["transformed_points"]
                    if is_json_scene_graph:
                        layout = SceneGraphLayout(layout_content)
                    else:
                        layout = Layout(layout_content)
                    # transformation augmentation
                    layout.translate(-center_pt)
                    layout.scale(scaling)
                    layout.rotate(transformation["angle_z"])
                    layout.translate(center_pt)
                    layout.filter_empty_bboxes(transformed_points, num_points=0) # OLD NUM_POINTS=100 IN THE ORIGINAL SETUP
                    layout.reorder_entities()
                    if isinstance(layout, SceneGraphLayout):
                        object_source_ids = layout.source_object_ids()
                    layout.translate(-min_bound)
                    layout.normalize_and_discretize(self.num_bins)
                    new_layout_content = layout.to_language_string()
                    content = content.replace(
                        f"{LAYOUT_S_PLACEHOLDER}{layout_content}{LAYOUT_E_PLACEHOLDER}",
                        new_layout_content,
                    )
                    if not hasattr(self, '_debug_count'):
                        self._debug_count = 0
                    if self._debug_count < 3:
                        print("DEBUG quantized layout:", new_layout_content[:300])
                        self._debug_count += 1
                    message["content"] = content

            if POINT_CLOUD_PLACEHOLDER in content:
                content = content.replace(
                    POINT_CLOUD_PLACEHOLDER,
                    f"{POINT_S_TOKEN}{self.point_token}{POINT_E_TOKEN}",
                    1,
                )
                num_point_tokens += 1
                message["content"] = content

        if len(transformations) != num_point_tokens:
            raise ValueError(
                f"The number of point clouds does not match the number of {POINT_CLOUD_PLACEHOLDER} tokens."
            )
        return messages, object_source_ids

    def get_mm_inputs(
        self,
        point_clouds: Sequence["PointCloudInput"],
        batch_prompts: Sequence[List[int]],
    ) -> Dict[str, Union[List[dict]]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            point_clouds: a list of point cloud inputs, shape (num_point_clouds,)
            pointlens: number of point clouds in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(point_clouds)
        return self._get_mm_inputs(batch_prompts, point_clouds)


def get_mm_plugin(
    point_token: str = "<|point_pad|>",
    **kwargs,
) -> "SpatialLMPlugin":
    return SpatialLMPlugin(point_token, **kwargs)
