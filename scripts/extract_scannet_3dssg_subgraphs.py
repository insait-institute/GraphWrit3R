#!/usr/bin/env python3
"""Create deterministic, object-exact ScanNet subgraphs for SpatialLM.

The generated subgraphs follow the released 3DSSG subset convention:

* the largest floor instance is repeated in every split;
* every other object is assigned to exactly one spatially local split; and
* each split contains a bounded number of objects.

ScanNet scene-graph JSONs use compact object IDs after filtering labels such as
``ceiling`` and ``object``. SceneVerse point clouds retain the original instance
IDs. This script recovers that mapping by aligning the pre-taxonomy object
sequence with ``instance_id_to_label/<scene_id>.pth``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LAYOUT_START = "<|layout_s|>"
LAYOUT_END = "<|layout_e|>"


@dataclass(frozen=True)
class GenerationConfig:
    mapped_json: Path
    source_json: Path
    sceneverse_root: Path
    output_dir: Path
    scene_ids: tuple[str, ...]
    max_objects: int = 9
    min_objects: int = 5
    neighbor_margin: float = 0.5
    seed: int = 2020


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    instance_labels: np.ndarray
    indices_by_instance: dict[int, np.ndarray]


def normalize_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def scene_id_from_row(row: Mapping[str, Any]) -> str:
    point_clouds = row.get("point_clouds")
    if not isinstance(point_clouds, list) or len(point_clouds) != 1:
        raise ValueError("Each dataset row must reference exactly one point cloud.")
    return Path(str(point_clouds[0])).stem


def parse_scene_graph_row(row: Mapping[str, Any]) -> dict[str, Any]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Dataset row is missing a conversations list.")

    response = None
    for message in conversations:
        if message.get("from") == "gpt":
            response = message.get("value")
            break
    if not isinstance(response, str):
        raise ValueError("Dataset row is missing a GPT scene-graph response.")

    start = response.find(LAYOUT_START)
    end = response.rfind(LAYOUT_END)
    if start >= 0 and end > start:
        response = response[start + len(LAYOUT_START) : end]

    try:
        graph = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Could not parse scene graph from GPT response.") from exc
    if not isinstance(graph, dict):
        raise ValueError("Scene graph response must be a JSON object.")
    if not isinstance(graph.get("objects"), list):
        raise ValueError("Scene graph must contain an objects list.")
    if not isinstance(graph.get("relationships"), list):
        raise ValueError("Scene graph must contain a relationships list.")
    return graph


def load_dataset_rows(path: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in dataset JSON: {path}")

    indexed: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        scene_id = scene_id_from_row(row)
        if scene_id in indexed:
            raise ValueError(f"Duplicate scene ID {scene_id!r} in {path}")
        indexed[scene_id] = (row, parse_scene_graph_row(row))
    return indexed


def _load_legacy_numpy_pth(path: Path) -> Any:
    """Load SceneVerse's zip-wrapped NumPy pickle without requiring torch."""

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError(f"Expected one data.pkl member in {path}")
        payload = archive.read(members[0])
    return pickle.loads(payload, encoding="latin1")


def load_pth(path: Path) -> Any:
    """Load NumPy- or torch-backed PTH files used by SceneVerse."""

    legacy_error: Exception | None = None
    if zipfile.is_zipfile(path):
        try:
            return _load_legacy_numpy_pth(path)
        except Exception as exc:  # torch tensor archives need torch.load
            legacy_error = exc

    try:
        import torch
    except ImportError as exc:
        detail = f" Legacy loader error: {legacy_error}" if legacy_error else ""
        raise RuntimeError(f"Could not load {path} without PyTorch.{detail}") from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        detail = f" Legacy loader error: {legacy_error}" if legacy_error else ""
        raise RuntimeError(f"Could not load {path}.{detail}") from exc


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def index_instances(instance_labels: np.ndarray) -> dict[int, np.ndarray]:
    labels = np.asarray(instance_labels).reshape(-1)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    values, starts, counts = np.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return {
        int(value): order[start : start + count]
        for value, start, count in zip(values, starts, counts)
    }


def load_point_cloud(path: Path) -> PointCloudData:
    data = load_pth(path)
    if not isinstance(data, (tuple, list)) or len(data) < 3:
        raise ValueError(f"Unexpected point-cloud payload in {path}")

    points = np.asarray(to_numpy(data[0]), dtype=np.float64)
    colors = np.asarray(to_numpy(data[1]))
    instance_labels = np.asarray(to_numpy(data[-1])).reshape(-1)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3 points in {path}, got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(f"Expected Nx3 colors in {path}, got {colors.shape}")
    if len(points) != len(colors) or len(points) != len(instance_labels):
        raise ValueError(f"Point/color/instance lengths do not match in {path}")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError(f"Point cloud contains non-finite coordinates: {path}")

    return PointCloudData(
        points=points[:, :3],
        colors=colors[:, :3],
        instance_labels=instance_labels,
        indices_by_instance=index_instances(instance_labels),
    )


def load_instance_labels(path: Path) -> dict[int, str]:
    data = load_pth(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an instance-to-label dictionary in {path}")
    return {int(instance_id): str(label) for instance_id, label in data.items()}


def validate_graph_alignment(
    source_graph: Mapping[str, Any],
    mapped_graph: Mapping[str, Any],
    *,
    scene_id: str,
    atol: float = 1e-6,
) -> None:
    source_objects = source_graph["objects"]
    mapped_objects = mapped_graph["objects"]
    if len(source_objects) != len(mapped_objects):
        raise ValueError(
            f"{scene_id}: source/mapped object count mismatch "
            f"({len(source_objects)} != {len(mapped_objects)})"
        )

    for source_obj, mapped_obj in zip(source_objects, mapped_objects):
        source_id = int(source_obj["id"])
        mapped_id = int(mapped_obj["id"])
        if source_id != mapped_id:
            raise ValueError(
                f"{scene_id}: source/mapped object order mismatch "
                f"({source_id} != {mapped_id})"
            )
        for field in ("position", "size"):
            if not np.allclose(
                np.asarray(source_obj[field], dtype=float),
                np.asarray(mapped_obj[field], dtype=float),
                rtol=0.0,
                atol=atol,
            ):
                raise ValueError(
                    f"{scene_id}: object {source_id} has mismatched {field}"
                )
        if not math.isclose(
            float(source_obj["angle_z"]),
            float(mapped_obj["angle_z"]),
            rel_tol=0.0,
            abs_tol=atol,
        ):
            raise ValueError(
                f"{scene_id}: object {source_id} has mismatched angle_z"
            )


def align_json_objects_to_instances(
    source_objects: Sequence[Mapping[str, Any]],
    instance_to_label: Mapping[int, str],
) -> tuple[dict[int, int], list[int]]:
    """Align a filtered JSON object sequence to the original instance sequence."""

    ordered_instances = sorted(
        (int(instance_id), normalize_label(label))
        for instance_id, label in instance_to_label.items()
    )
    cursor = 0
    object_to_instance: dict[int, int] = {}

    source_labels = {normalize_label(obj["label"]) for obj in source_objects}
    for obj in source_objects:
        object_id = int(obj["id"])
        if object_id in object_to_instance:
            raise ValueError(f"Duplicate source object ID: {object_id}")
        target_label = normalize_label(obj["label"])

        while (
            cursor < len(ordered_instances)
            and ordered_instances[cursor][1] != target_label
        ):
            cursor += 1
        if cursor >= len(ordered_instances):
            raise ValueError(
                f"Could not align object {object_id} ({obj['label']!r}) "
                "to the ordered SceneVerse instances."
            )

        object_to_instance[object_id] = ordered_instances[cursor][0]
        cursor += 1

    mapped_instances = list(object_to_instance.values())
    if len(set(mapped_instances)) != len(mapped_instances):
        raise ValueError("JSON objects mapped to duplicate SceneVerse instances.")

    skipped = sorted(set(instance_to_label) - set(mapped_instances))
    ambiguous_skips = [
        instance_id
        for instance_id in skipped
        if normalize_label(instance_to_label[instance_id]) in source_labels
    ]
    if ambiguous_skips:
        raise ValueError(
            "Ordered label alignment is ambiguous because skipped instances share "
            f"retained labels: {ambiguous_skips}"
        )
    return object_to_instance, skipped


def select_floor_anchor(
    objects: Sequence[Mapping[str, Any]],
    object_point_counts: Mapping[int, int],
) -> int:
    floor_ids = [
        int(obj["id"])
        for obj in objects
        if normalize_label(obj.get("label")) == "floor"
    ]
    if not floor_ids:
        raise ValueError("Scene graph has no floor object to use as an anchor.")
    return max(floor_ids, key=lambda object_id: (object_point_counts[object_id], -object_id))


def compute_object_geometry(
    object_ids: Iterable[int],
    object_to_instance: Mapping[int, int],
    point_cloud: PointCloudData,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    aabbs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    centers: dict[int, np.ndarray] = {}
    for object_id in object_ids:
        instance_id = object_to_instance[object_id]
        indices = point_cloud.indices_by_instance.get(instance_id)
        if indices is None or len(indices) == 0:
            raise ValueError(
                f"Object {object_id} maps to instance {instance_id}, "
                "which has no point-cloud points."
            )
        instance_points = point_cloud.points[indices]
        lower = instance_points.min(axis=0)
        upper = instance_points.max(axis=0)
        aabbs[object_id] = (lower, upper)
        centers[object_id] = (lower + upper) / 2.0
    return aabbs, centers


def build_expanded_aabb_neighbors(
    aabbs: Mapping[int, tuple[np.ndarray, np.ndarray]],
    margin: float,
) -> dict[int, set[int]]:
    if margin < 0:
        raise ValueError("Neighbor margin must be non-negative.")

    object_ids = list(aabbs)
    neighbors = {object_id: set() for object_id in object_ids}
    for index, source_id in enumerate(object_ids):
        source_lower, source_upper = aabbs[source_id]
        source_lower = source_lower - margin
        source_upper = source_upper + margin
        for target_id in object_ids[index + 1 :]:
            target_lower, target_upper = aabbs[target_id]
            target_lower = target_lower - margin
            target_upper = target_upper + margin
            if np.all(source_lower <= target_upper) and np.all(
                target_lower <= source_upper
            ):
                neighbors[source_id].add(target_id)
                neighbors[target_id].add(source_id)
    return neighbors


def balanced_group_capacities(
    num_non_anchor_objects: int,
    *,
    max_objects: int,
    min_objects: int,
) -> list[int]:
    if max_objects < 2:
        raise ValueError("max_objects must leave room for an anchor and one object.")
    if min_objects < 1 or min_objects > max_objects:
        raise ValueError("min_objects must be between 1 and max_objects.")
    if num_non_anchor_objects < 0:
        raise ValueError("Object count cannot be negative.")

    capacity = max_objects - 1
    num_groups = max(1, math.ceil(num_non_anchor_objects / capacity))
    base, remainder = divmod(num_non_anchor_objects, num_groups)
    capacities = [
        base + (1 if index < remainder else 0) for index in range(num_groups)
    ]
    if min(capacities) + 1 < min_objects:
        raise ValueError(
            "Cannot satisfy min_objects/max_objects with one repeated anchor: "
            f"{num_non_anchor_objects} non-anchor objects."
        )
    return capacities


def spatially_partition_objects(
    non_anchor_ids: Sequence[int],
    *,
    centers: Mapping[int, np.ndarray],
    neighbors: Mapping[int, set[int]],
    capacities: Sequence[int],
    seed: int,
) -> list[list[int]]:
    if sum(capacities) != len(non_anchor_ids):
        raise ValueError("Group capacities do not cover all non-anchor objects.")
    if len(set(non_anchor_ids)) != len(non_anchor_ids):
        raise ValueError("Non-anchor object IDs must be unique.")
    if not non_anchor_ids:
        return [[]]

    rng = np.random.default_rng(seed)
    remaining = set(non_anchor_ids)
    chosen_seeds: list[int] = []
    groups: list[list[int]] = []

    for capacity in capacities:
        if capacity <= 0:
            groups.append([])
            continue

        if not chosen_seeds:
            first_index = int(rng.integers(len(non_anchor_ids)))
            group_seed = int(non_anchor_ids[first_index])
        else:
            group_seed = max(
                remaining,
                key=lambda object_id: (
                    min(
                        float(
                            np.linalg.norm(
                                centers[object_id][:2] - centers[seed_id][:2]
                            )
                        )
                        for seed_id in chosen_seeds
                    ),
                    -object_id,
                ),
            )

        chosen_seeds.append(group_seed)
        group = [group_seed]
        remaining.remove(group_seed)

        while len(group) < capacity:
            frontier: set[int] = set()
            for object_id in group:
                frontier.update(neighbors.get(object_id, set()))
            frontier.intersection_update(remaining)

            group_center = np.mean(
                [centers[object_id][:2] for object_id in group], axis=0
            )
            candidates = frontier if frontier else remaining
            next_object = min(
                candidates,
                key=lambda object_id: (
                    float(np.linalg.norm(centers[object_id][:2] - group_center)),
                    object_id,
                ),
            )
            group.append(next_object)
            remaining.remove(next_object)

        groups.append(group)

    if remaining:
        raise AssertionError(f"Partition left unassigned objects: {sorted(remaining)}")
    return groups


def filter_scene_graph(
    graph: Mapping[str, Any], selected_object_ids: Iterable[int]
) -> dict[str, Any]:
    selected = {int(object_id) for object_id in selected_object_ids}
    objects = [
        copy.deepcopy(obj)
        for obj in graph["objects"]
        if int(obj["id"]) in selected
    ]
    object_ids = {int(obj["id"]) for obj in objects}
    if object_ids != selected:
        raise ValueError(
            f"Selected object IDs are absent from graph: {sorted(selected - object_ids)}"
        )

    relationships = []
    for relationship in graph["relationships"]:
        if not isinstance(relationship, list) or len(relationship) < 3:
            raise ValueError(f"Malformed relationship: {relationship!r}")
        subject_id = int(relationship[0])
        object_id = int(relationship[1])
        if subject_id in selected and object_id in selected:
            relationships.append(copy.deepcopy(relationship))

    return {
        "num_objects": len(objects),
        "objects": objects,
        "relationships": relationships,
    }


def make_dataset_row(
    source_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    split_scene_id: str,
) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    row["point_clouds"] = [f"pcd/{split_scene_id}.ply"]

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Mapped dataset row has no conversations list.")
    response = (
        LAYOUT_START
        + json.dumps(graph, separators=(",", ":"), ensure_ascii=False)
        + LAYOUT_END
    )
    for message in conversations:
        if message.get("from") == "gpt":
            message["value"] = response
            break
    else:
        raise ValueError("Mapped dataset row has no GPT response to replace.")
    return row


def selected_point_indices(
    selected_object_ids: Iterable[int],
    object_to_instance: Mapping[int, int],
    indices_by_instance: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    instance_ids = [object_to_instance[int(object_id)] for object_id in selected_object_ids]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Selected objects resolve to duplicate point-cloud instances.")
    index_parts = []
    for instance_id in instance_ids:
        indices = indices_by_instance.get(instance_id)
        if indices is None or len(indices) == 0:
            raise ValueError(f"Selected instance {instance_id} has no points.")
        index_parts.append(indices)
    indices = np.sort(np.concatenate(index_parts), kind="stable")
    return indices, instance_ids


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if len(points) == 0:
        raise ValueError(f"Refusing to write an empty point cloud: {path}")
    if len(points) != len(colors):
        raise ValueError("Point and color counts do not match.")

    vertex_dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    colors_u8 = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    vertices["red"] = colors_u8[:, 0]
    vertices["green"] = colors_u8[:, 1]
    vertices["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by extract_scannet_3dssg_subgraphs\n"
        f"element vertex {len(vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_splits(
    parent_graph: Mapping[str, Any],
    *,
    anchor_object_id: int,
    split_graphs: Sequence[Mapping[str, Any]],
    min_objects: int,
    max_objects: int,
) -> None:
    parent_ids = {int(obj["id"]) for obj in parent_graph["objects"]}
    non_anchor_expected = parent_ids - {anchor_object_id}
    non_anchor_seen: list[int] = []

    for graph in split_graphs:
        objects = graph["objects"]
        object_ids = [int(obj["id"]) for obj in objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Generated split contains duplicate object IDs.")
        if not min_objects <= len(object_ids) <= max_objects:
            raise ValueError(
                f"Generated split has {len(object_ids)} objects; expected "
                f"{min_objects}..{max_objects}."
            )
        if anchor_object_id not in object_ids:
            raise ValueError("Generated split is missing the floor anchor.")
        non_anchor_seen.extend(
            object_id for object_id in object_ids if object_id != anchor_object_id
        )
        selected = set(object_ids)
        for relationship in graph["relationships"]:
            if int(relationship[0]) not in selected or int(relationship[1]) not in selected:
                raise ValueError("Generated relationship escapes its split.")

    if set(non_anchor_seen) != non_anchor_expected:
        missing = sorted(non_anchor_expected - set(non_anchor_seen))
        extra = sorted(set(non_anchor_seen) - non_anchor_expected)
        raise ValueError(f"Generated split coverage mismatch; missing={missing}, extra={extra}")
    duplicates = sorted(
        object_id
        for object_id in set(non_anchor_seen)
        if non_anchor_seen.count(object_id) != 1
    )
    if duplicates:
        raise ValueError(f"Non-anchor objects occur in multiple splits: {duplicates}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def generate_subgraphs(config: GenerationConfig) -> dict[str, Any]:
    if len(set(config.scene_ids)) != len(config.scene_ids):
        raise ValueError("scene_ids contains duplicates.")
    if config.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {config.output_dir}"
        )

    mapped_rows = load_dataset_rows(config.mapped_json)
    source_rows = load_dataset_rows(config.source_json)
    missing_mapped = sorted(set(config.scene_ids) - set(mapped_rows))
    missing_source = sorted(set(config.scene_ids) - set(source_rows))
    if missing_mapped or missing_source:
        raise ValueError(
            f"Missing requested scenes; mapped={missing_mapped}, source={missing_source}"
        )

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.tmp-",
            dir=config.output_dir.parent,
        )
    )
    dataset_rows: list[dict[str, Any]] = []
    manifest_scenes: list[dict[str, Any]] = []

    try:
        for scene_id in config.scene_ids:
            mapped_row, mapped_graph = mapped_rows[scene_id]
            _, source_graph = source_rows[scene_id]
            validate_graph_alignment(source_graph, mapped_graph, scene_id=scene_id)

            instance_labels_path = (
                config.sceneverse_root / "instance_id_to_label" / f"{scene_id}.pth"
            )
            point_cloud_path = (
                config.sceneverse_root
                / "pcd_with_global_alignment"
                / f"{scene_id}.pth"
            )
            if not instance_labels_path.is_file():
                raise FileNotFoundError(instance_labels_path)
            if not point_cloud_path.is_file():
                raise FileNotFoundError(point_cloud_path)

            instance_to_label = load_instance_labels(instance_labels_path)
            point_cloud = load_point_cloud(point_cloud_path)
            object_to_instance, skipped_instance_ids = align_json_objects_to_instances(
                source_graph["objects"], instance_to_label
            )

            object_ids = [int(obj["id"]) for obj in mapped_graph["objects"]]
            if set(object_ids) != set(object_to_instance):
                raise ValueError(
                    f"{scene_id}: mapped/source object IDs do not agree after alignment."
                )
            point_counts = {}
            for object_id, instance_id in object_to_instance.items():
                indices = point_cloud.indices_by_instance.get(instance_id)
                if indices is None or len(indices) == 0:
                    raise ValueError(
                        f"{scene_id}: object {object_id} maps to empty instance {instance_id}."
                    )
                point_counts[object_id] = int(len(indices))

            anchor_id = select_floor_anchor(mapped_graph["objects"], point_counts)
            non_anchor_ids = [
                object_id for object_id in object_ids if object_id != anchor_id
            ]
            aabbs, centers = compute_object_geometry(
                non_anchor_ids, object_to_instance, point_cloud
            )
            neighbors = build_expanded_aabb_neighbors(
                aabbs, config.neighbor_margin
            )
            capacities = balanced_group_capacities(
                len(non_anchor_ids),
                max_objects=config.max_objects,
                min_objects=config.min_objects,
            )
            partition = spatially_partition_objects(
                non_anchor_ids,
                centers=centers,
                neighbors=neighbors,
                capacities=capacities,
                seed=config.seed,
            )

            split_graphs: list[dict[str, Any]] = []
            split_manifests: list[dict[str, Any]] = []
            for split_index, growth_order in enumerate(partition, start=1):
                selected_set = {anchor_id, *growth_order}
                selected_ids = [
                    object_id for object_id in object_ids if object_id in selected_set
                ]
                split_graph = filter_scene_graph(mapped_graph, selected_ids)
                split_graphs.append(split_graph)
                split_scene_id = f"{scene_id}_split{split_index}"
                output_ply = staging_dir / "pcd" / f"{split_scene_id}.ply"

                point_indices, selected_instance_ids = selected_point_indices(
                    selected_ids,
                    object_to_instance,
                    point_cloud.indices_by_instance,
                )
                write_binary_ply(
                    output_ply,
                    point_cloud.points[point_indices],
                    point_cloud.colors[point_indices],
                )
                dataset_rows.append(
                    make_dataset_row(
                        mapped_row,
                        split_graph,
                        split_scene_id=split_scene_id,
                    )
                )
                split_manifests.append(
                    {
                        "split_index": split_index,
                        "scene_id": split_scene_id,
                        "point_cloud": f"pcd/{split_scene_id}.ply",
                        "object_count": len(split_graph["objects"]),
                        "relationship_count": len(split_graph["relationships"]),
                        "point_count": int(len(point_indices)),
                        "selected_object_ids": selected_ids,
                        "selected_instance_ids": selected_instance_ids,
                        "growth_order_without_anchor": growth_order,
                        "ply_sha256": file_sha256(output_ply),
                    }
                )

            validate_parent_splits(
                mapped_graph,
                anchor_object_id=anchor_id,
                split_graphs=split_graphs,
                min_objects=config.min_objects,
                max_objects=config.max_objects,
            )
            manifest_scenes.append(
                {
                    "parent_scene_id": scene_id,
                    "parent_object_count": len(mapped_graph["objects"]),
                    "parent_relationship_count": len(mapped_graph["relationships"]),
                    "parent_point_count": int(len(point_cloud.points)),
                    "anchor_object_id": anchor_id,
                    "anchor_instance_id": object_to_instance[anchor_id],
                    "anchor_point_count": point_counts[anchor_id],
                    "json_object_to_instance_id": {
                        str(object_id): object_to_instance[object_id]
                        for object_id in object_ids
                    },
                    "skipped_source_instance_ids": skipped_instance_ids,
                    "splits": split_manifests,
                }
            )

        dataset_info = {
            "scene_graph_val": {
                "file_name": "scene_graph_val.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "point_clouds": "point_clouds",
                },
            }
        }
        manifest = {
            "version": 1,
            "parameters": {
                "mapped_json": str(config.mapped_json),
                "source_json": str(config.source_json),
                "sceneverse_root": str(config.sceneverse_root),
                "scene_ids": list(config.scene_ids),
                "max_objects": config.max_objects,
                "min_objects": config.min_objects,
                "neighbor_margin": config.neighbor_margin,
                "seed": config.seed,
            },
            "summary": {
                "parent_scene_count": len(config.scene_ids),
                "subgraph_count": len(dataset_rows),
                "object_count_histogram": {
                    str(size): sum(
                        1
                        for scene in manifest_scenes
                        for split in scene["splits"]
                        if split["object_count"] == size
                    )
                    for size in range(config.min_objects, config.max_objects + 1)
                },
            },
            "scenes": manifest_scenes,
        }
        _write_json(staging_dir / "scene_graph_val.json", dataset_rows)
        _write_json(staging_dir / "dataset_info.json", dataset_info)
        _write_json(staging_dir / "subgraph_manifest.json", manifest)
        os.replace(staging_dir, config.output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, object-exact ScanNet subgraphs with 3DSSG labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapped-json", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument(
        "--sceneverse-root",
        type=Path,
        required=True,
        help=(
            "ScanNet scan_data root containing instance_id_to_label/ and "
            "pcd_with_global_alignment/."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    scene_selection = parser.add_mutually_exclusive_group(required=True)
    scene_selection.add_argument(
        "--scene-ids",
        nargs="+",
        help="Explicit parent scene IDs to process.",
    )
    scene_selection.add_argument(
        "--all-scenes",
        action="store_true",
        help="Process every scene in --mapped-json, preserving dataset order.",
    )
    parser.add_argument("--max-objects", type=int, default=9)
    parser.add_argument("--min-objects", type=int, default=5)
    parser.add_argument("--neighbor-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    mapped_json = args.mapped_json.resolve()
    if args.all_scenes:
        with mapped_json.open() as handle:
            mapped_rows = json.load(handle)
        if not isinstance(mapped_rows, list):
            raise ValueError(f"Expected a list in dataset JSON: {mapped_json}")
        scene_ids = tuple(scene_id_from_row(row) for row in mapped_rows)
    else:
        scene_ids = tuple(args.scene_ids)

    return GenerationConfig(
        mapped_json=mapped_json,
        source_json=args.source_json.resolve(),
        sceneverse_root=args.sceneverse_root.resolve(),
        output_dir=args.output_dir.resolve(),
        scene_ids=scene_ids,
        max_objects=args.max_objects,
        min_objects=args.min_objects,
        neighbor_margin=args.neighbor_margin,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    manifest = generate_subgraphs(config)
    print(
        f"Generated {manifest['summary']['subgraph_count']} subgraphs from "
        f"{manifest['summary']['parent_scene_count']} parent scenes."
    )
    for scene in manifest["scenes"]:
        sizes = [split_data["object_count"] for split_data in scene["splits"]]
        print(
            f"  {scene['parent_scene_id']}: {len(sizes)} splits, "
            f"object counts={sizes}"
        )
    print(f"Output: {config.output_dir}")


if __name__ == "__main__":
    main()
