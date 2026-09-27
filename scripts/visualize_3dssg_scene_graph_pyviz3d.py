"""Visualize 3DSSG/SpatialLM scene graphs with PyViz3D."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_R3SCAN_ROOT = Path("/work/luka_milivojevic/3RScan")
DEFAULT_SCENEVERSE_PCD_ROOT = Path("/work/luka_milivojevic/sceneverse_3rscan/3RScan/scan_data")
INSTALL_HINT = (
    "PyViz3D rendering requires pyviz3d. The live query box also requires "
    "the same object-matcher dependencies used by eval_scene_graph.py.\n"
    "Install: python -m pip install pyviz3d pillow torch transformers sentence-transformers"
)


@dataclass
class GraphObject:
    id: int
    label: str
    position: np.ndarray
    size: np.ndarray
    angle_z: float = 0.0
    color: tuple[int, int, int] = (180, 180, 180)
    source_scene_id: Optional[str] = None
    source_object_id: Optional[int] = None


@dataclass
class Relationship:
    subject_id: int
    object_id: int
    predicate: str
    source_scene_id: Optional[str] = None


@dataclass
class LoadedGraph:
    scene_id: str
    graph_json: Path
    source_graph_jsons: list[Path] = field(default_factory=list)
    objects: list[GraphObject] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PyViz3D scene-graph viewer with semantic object query.",
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=None,
        help=(
            "Scene-graph JSON to render. If this points to a directory, use it with --scene-id "
            "and the script will load <scene_id>_sg.json from that directory."
        ),
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Directory containing <scene_id>_sg.json. Use with --scene-id to avoid repeating the scene id.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help=(
            "Scene id to select from multi-scene inputs, or to resolve --graph-dir/<scene_id>_sg.json. "
            "For single-scene JSON files this is inferred from --graph-json and usually should be omitted."
        ),
    )
    parser.add_argument(
        "--allow-scene-id-mismatch",
        action="store_true",
        help="Allow --scene-id to differ from a single-scene --graph-json. Intended only for renamed/ad-hoc files.",
    )
    parser.add_argument(
        "--graph-kind",
        choices=["auto", "gt_3dssg", "gt_compact", "predicted"],
        default="auto",
        help="Accepted for compatibility with the previous visualizer CLI.",
    )
    parser.add_argument(
        "--all-parent-splits",
        action="store_true",
        help=(
            "Render every <parent>_split*_sg.json sibling of --graph-json as one parent-scene graph. "
            "Local split object ids are remapped so layers and relationships do not collide."
        ),
    )
    parser.add_argument("--box-json", type=Path, default=None, help="Optional JSON whose objects provide geometry by id.")
    parser.add_argument("--point-cloud", type=Path, default=None, help="Optional PLY point cloud to show behind boxes.")
    parser.add_argument("--r3scan-root", type=Path, default=DEFAULT_R3SCAN_ROOT)
    parser.add_argument("--sceneverse-pcd-root", type=Path, default=DEFAULT_SCENEVERSE_PCD_ROOT)
    parser.add_argument(
        "--base-mode",
        choices=["sceneverse_textured_mesh_points", "sceneverse_pcd", "raw_mesh", "raw_instance_ply", "none"],
        default="sceneverse_pcd",
        help=(
            "Base geometry mode. sceneverse_textured_mesh_points samples the textured 3RScan OBJ "
            "from --r3scan-root and colors points from mesh.refined_0.png."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to create the viewer.")
    parser.add_argument("--output-name", default=None, help="Output subdirectory name. Defaults to <scene_id>_pyviz3d.")
    parser.add_argument("--max-points", type=int, default=2000000, help="Maximum point-cloud vertices to render. 0 means all.")
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--mesh-point-samples", type=int, default=2200000)
    parser.add_argument("--mesh-point-size", type=float, default=9.0)
    parser.add_argument("--mesh-point-alpha", type=float, default=0.94)
    parser.add_argument("--scene-top-trim-percent", type=float, default=0.0)
    parser.add_argument(
        "--scene-top-trim-graph",
        action="store_true",
        help=(
            "When --scene-top-trim-percent is set, also drop graph objects whose centers are above "
            "the same height cutoff, and drop relationships touching those objects."
        ),
    )
    parser.add_argument("--box-alpha", type=float, default=0.34)
    parser.add_argument("--box-edge-width", type=float, default=0.006)
    parser.add_argument("--highlight-alpha", type=float, default=1.0)
    parser.add_argument("--highlight-edge-width", type=float, default=0.022)
    parser.add_argument("--box-angle-sign", type=float, default=-1.0, help="Multiplier applied to object angle_z.")
    parser.add_argument("--hide-object-labels", action="store_true")
    parser.add_argument("--hide-relationships", action="store_true")
    parser.add_argument("--edge-width", type=float, default=0.012)
    parser.add_argument("--edge-alpha", type=float, default=0.86)
    parser.add_argument("--center-sphere-radius", type=float, default=0.095, help="Radius of paper-demo object centroid spheres.")
    parser.add_argument("--ui-mode", choices=["paper_demo", "paper_phone", "paper", "pyviz3d"], default="paper_demo")
    parser.add_argument("--paper-point-size", type=float, default=None)
    parser.add_argument("--paper-camera-fov", type=float, default=58.0)
    parser.add_argument("--paper-camera-distance-scale", type=float, default=1.0)
    parser.add_argument("--tour-duration", type=float, default=28.0)
    parser.add_argument("--demo-title", default="Scene Graph Walkthrough")
    parser.add_argument("--demo-subtitle", default="3DSSG / SpatialLM-SG")
    parser.add_argument("--port", type=int, default=6008, help="Port for --serve-query.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/interface for --serve-query. Keep 127.0.0.1 when using SSH ProxyJump tunnels.",
    )
    parser.add_argument(
        "--serve-query",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The visualizer now only generates files; "
            "serve live query with scripts/serve_scene_graph_query.py."
        ),
    )
    parser.add_argument("--query-top-k", type=int, default=5, help="Number of top object matches returned by the query API.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Parse and summarize without importing PyViz3D.")
    args = parser.parse_args()
    resolve_graph_json_arg(args, parser)
    return args


def resolve_graph_json_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.graph_json is not None and args.graph_dir is not None:
        parser.error("Use either --graph-json or --graph-dir, not both.")

    if args.graph_json is not None and args.graph_json.is_dir():
        args.graph_dir = args.graph_json
        args.graph_json = None

    if args.graph_json is None:
        if args.graph_dir is None:
            parser.error("--graph-json is required unless --graph-dir is provided with --scene-id.")
        if not args.scene_id:
            parser.error("--scene-id is required when resolving a graph from --graph-dir.")
        args.graph_json = args.graph_dir / f"{args.scene_id}_sg.json"

    if not args.graph_json.is_file() and args.graph_dir is not None and args.all_parent_splits:
        paths = sorted(args.graph_dir.glob(f"{parent_scan_id(args.scene_id)}_split*_sg.json"), key=split_sort_key)
        if paths:
            args.graph_json = paths[0]

    if not args.graph_json.is_file():
        parser.error(f"Scene-graph JSON not found: {args.graph_json}")


def load_json_or_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_blob = extract_json_blob(text)
    if json_blob is None:
        raise ValueError(f"Could not parse JSON from {path}")
    return json.loads(json_blob)


def extract_json_blob(text: str) -> Optional[str]:
    match = re.search(r"<\|layout_s\|>(.*?)<\|layout_e\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


def infer_scene_id(path: Path, payload: Any, requested_scene_id: Optional[str]) -> str:
    if requested_scene_id:
        return requested_scene_id
    if isinstance(payload, dict):
        for key in ("scene_id", "scan_id", "scan", "id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return path.stem.removesuffix("_sg")


def single_scene_payload_scene_id(path: Path, payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    if "objects" in payload or isinstance(payload.get("scene_graph"), dict):
        return infer_scene_id(path, payload, requested_scene_id=None)
    return None


def scene_ids_compatible(actual_scene_id: str, requested_scene_id: str) -> bool:
    return requested_scene_id in {
        actual_scene_id,
        parent_scan_id(actual_scene_id),
    } or actual_scene_id == parent_scan_id(requested_scene_id)


def parent_scan_id(scene_id: str) -> str:
    match = re.match(r"^(?P<scan>.+)_split\d+$", scene_id)
    return match.group("scan") if match else scene_id


def split_index(scene_id: str) -> Optional[int]:
    match = re.search(r"_split(?P<split>\d+)$", scene_id)
    return int(match.group("split")) if match else None


def split_label(scene_id: Optional[str]) -> str:
    if not scene_id:
        return "scene"
    value = split_index(scene_id)
    return f"split{value}" if value is not None else scene_id


def scene_id_from_sample(sample: dict[str, Any]) -> Optional[str]:
    for key in ("scene_id", "id", "scan_id", "scan"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    point_clouds = sample.get("point_clouds")
    if isinstance(point_clouds, list) and point_clouds:
        return Path(str(point_clouds[0])).stem
    return None


def extract_graph_payload(payload: Any, requested_scene_id: Optional[str]) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []

    if isinstance(payload, list):
        for sample in payload:
            if not isinstance(sample, dict):
                continue
            sample_scene_id = scene_id_from_sample(sample)
            if requested_scene_id and sample_scene_id != requested_scene_id:
                continue
            graph = extract_graph_from_sharegpt_sample(sample)
            if graph is not None:
                return graph, sample_scene_id or requested_scene_id or "scene", warnings
        raise ValueError("Could not find a scene graph in the list payload.")

    if not isinstance(payload, dict):
        raise ValueError("Expected a dict or list JSON payload.")

    if "objects" in payload:
        return payload, infer_scene_id(Path("scene"), payload, requested_scene_id), warnings

    if isinstance(payload.get("scene_graph"), dict):
        graph = payload["scene_graph"]
        return graph, infer_scene_id(Path("scene"), payload, requested_scene_id), warnings

    scans = payload.get("scans")
    if isinstance(scans, list):
        for scan in scans:
            if not isinstance(scan, dict):
                continue
            scan_id = str(scan.get("scan") or scan.get("scan_id") or scan.get("id") or "")
            split = scan.get("split")
            split_scene_id = f"{scan_id}_split{split}" if split is not None else scan_id
            if requested_scene_id and requested_scene_id not in {scan_id, split_scene_id}:
                continue
            graph = {
                "objects": scan.get("objects", []),
                "relationships": scan.get("relationships", []),
            }
            warnings.append(
                "Loaded 3DSSG relationship JSON. If objects lack position/size, pass --box-json with OBB geometry."
            )
            return graph, split_scene_id or scan_id or requested_scene_id or "scene", warnings

    raise ValueError("Unsupported graph JSON format; expected objects/relationships or 3DSSG scans.")


def extract_graph_from_sharegpt_sample(sample: dict[str, Any]) -> Optional[dict[str, Any]]:
    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if not isinstance(message, dict):
                continue
            value = message.get("value")
            if not isinstance(value, str):
                continue
            blob = extract_json_blob(value)
            if blob is None:
                continue
            data = json.loads(blob)
            if isinstance(data, dict) and "objects" in data:
                return data

    for key in ("scene_graph", "graph", "response"):
        value = sample.get(key)
        if isinstance(value, dict) and "objects" in value:
            return value
        if isinstance(value, str):
            blob = extract_json_blob(value)
            if blob is not None:
                data = json.loads(blob)
                if isinstance(data, dict) and "objects" in data:
                    return data
    return None


def coerce_float_array(value: Any, length: int, field_name: str, object_id: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (length,):
        raise ValueError(f"Object {object_id} has invalid {field_name}: expected length {length}, got {value!r}")
    return arr


def parse_object(raw: Any, default_id: int) -> GraphObject:
    if isinstance(raw, (str, int)):
        raise ValueError(f"Object {default_id} has no geometry; pass --box-json with object boxes.")
    if not isinstance(raw, dict):
        raise ValueError(f"Object {default_id} must be a dict.")

    object_id = int(raw.get("id", raw.get("object_id", raw.get("instance_id", default_id))))
    label = str(raw.get("label", raw.get("class", raw.get("name", raw.get("rio27", "object"))))).lower().strip()
    position = raw.get("position", raw.get("center", raw.get("centroid")))
    size = raw.get("size", raw.get("scale", raw.get("extent", raw.get("dimensions"))))

    if position is None or size is None:
        bbox = raw.get("bbox") or raw.get("box") or {}
        if isinstance(bbox, dict):
            position = position if position is not None else bbox.get("position", bbox.get("center"))
            size = size if size is not None else bbox.get("size", bbox.get("scale", bbox.get("extent")))

    if position is None or size is None:
        raise ValueError(f"Object {object_id} ({label}) is missing position/size.")

    angle_z = raw.get("angle_z", raw.get("yaw", raw.get("rotation", 0.0)))
    if isinstance(angle_z, (list, tuple)):
        angle_z = angle_z[-1] if angle_z else 0.0

    return GraphObject(
        id=object_id,
        label=label,
        position=coerce_float_array(position, 3, "position", object_id),
        size=np.maximum(coerce_float_array(size, 3, "size", object_id), 1e-3),
        angle_z=float(angle_z),
        source_object_id=object_id,
    )


def parse_relationship(raw: Any) -> Optional[Relationship]:
    if isinstance(raw, dict):
        subject = raw.get("subject_id", raw.get("subject", raw.get("source")))
        target = raw.get("object_id", raw.get("object", raw.get("target")))
        predicate = raw.get("predicate", raw.get("relation", raw.get("relationship")))
        predicates = raw.get("predicates")
        if predicate is None and isinstance(predicates, list) and predicates:
            predicate = ", ".join(str(p) for p in predicates)
        if subject is None or target is None or predicate is None:
            return None
        return Relationship(int(subject), int(target), str(predicate).lower().strip())

    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        predicate = None
        for item in reversed(raw[2:]):
            if isinstance(item, str):
                predicate = item
                break
        if predicate is None:
            predicate = str(raw[2])
        return Relationship(int(raw[0]), int(raw[1]), str(predicate).lower().strip())

    return None


def merge_box_geometry(objects: list[GraphObject], box_json: Path) -> list[str]:
    payload = load_json_or_text(box_json)
    graph, _scene_id, warnings = extract_graph_payload(payload, requested_scene_id=None)
    box_objects = {}
    for idx, raw in enumerate(iter_raw_objects(graph.get("objects", []))):
        obj = parse_object(raw, idx)
        box_objects[obj.id] = obj

    for obj in objects:
        box_obj = box_objects.get(obj.id)
        if box_obj is None:
            continue
        obj.position = box_obj.position
        obj.size = box_obj.size
        obj.angle_z = box_obj.angle_z
    warnings.append(f"Applied OBB geometry from {box_json}")
    return warnings


def iter_raw_objects(raw_objects: Any) -> list[Any]:
    if isinstance(raw_objects, list):
        return raw_objects
    if isinstance(raw_objects, dict):
        rows = []
        for key, value in raw_objects.items():
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("id", key)
                rows.append(merged)
            else:
                rows.append({"id": key, "label": value})
        return rows
    return []


def load_graph(
    graph_json: Path,
    requested_scene_id: Optional[str],
    box_json: Optional[Path],
    allow_scene_id_mismatch: bool = False,
) -> LoadedGraph:
    payload = load_json_or_text(graph_json)
    actual_scene_id = single_scene_payload_scene_id(graph_json, payload)
    if (
        requested_scene_id
        and actual_scene_id
        and not allow_scene_id_mismatch
        and not scene_ids_compatible(actual_scene_id, requested_scene_id)
    ):
        raise ValueError(
            f"--scene-id {requested_scene_id!r} does not match --graph-json {graph_json} "
            f"(appears to contain scene {actual_scene_id!r}). Omit --scene-id for single-scene graph files, "
            "use --graph-dir with the desired scene id, or pass --allow-scene-id-mismatch for an ad-hoc renamed file."
        )
    graph_payload, scene_id, warnings = extract_graph_payload(payload, requested_scene_id)
    if scene_id == "scene" and requested_scene_id is None:
        scene_id = graph_json.stem.removesuffix("_sg")
    raw_objects = iter_raw_objects(graph_payload.get("objects", []))
    objects = [parse_object(raw, idx) for idx, raw in enumerate(raw_objects)]
    for obj in objects:
        obj.source_scene_id = scene_id
        obj.source_object_id = obj.id
    relationships = [
        relationship
        for relationship in (parse_relationship(raw) for raw in graph_payload.get("relationships", []))
        if relationship is not None
    ]
    for relationship in relationships:
        relationship.source_scene_id = scene_id

    graph = LoadedGraph(
        scene_id=scene_id,
        graph_json=graph_json,
        source_graph_jsons=[graph_json],
        objects=objects,
        relationships=relationships,
        warnings=warnings,
    )
    if box_json is not None:
        graph.warnings.extend(merge_box_geometry(graph.objects, box_json))
    assign_object_colors(graph.objects)
    return graph


def scene_id_from_graph_path(path: Path) -> str:
    return path.stem.removesuffix("_sg")


def split_sort_key(path: Path) -> tuple[int, str]:
    scene_id = scene_id_from_graph_path(path)
    index = split_index(scene_id)
    return (index if index is not None else 10**9, path.name)


def parent_split_graph_paths(graph_json: Path, requested_scene_id: Optional[str]) -> list[Path]:
    primary_scene_id = requested_scene_id or scene_id_from_graph_path(graph_json)
    parent_id = parent_scan_id(primary_scene_id)
    paths = sorted(graph_json.parent.glob(f"{parent_id}_split*_sg.json"), key=split_sort_key)
    if graph_json not in paths and graph_json.is_file():
        paths.append(graph_json)
        paths = sorted(set(paths), key=split_sort_key)
    return paths


def object_aabb_bounds(obj: GraphObject) -> tuple[np.ndarray, np.ndarray]:
    half_size = np.asarray(obj.size, dtype=np.float32) * 0.5
    offsets = np.asarray(
        [
            [sx * half_size[0], sy * half_size[1], sz * half_size[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )
    corners = obj.position.reshape(1, 3) + offsets @ rotation_z_matrix(obj.angle_z).T
    return corners.min(axis=0), corners.max(axis=0)


def merged_floor_object(floor_objects: list[GraphObject], object_id: int, parent_id: str) -> GraphObject:
    mins = []
    maxs = []
    for obj in floor_objects:
        min_corner, max_corner = object_aabb_bounds(obj)
        mins.append(min_corner)
        maxs.append(max_corner)
    union_min = np.min(np.asarray(mins, dtype=np.float32), axis=0)
    union_max = np.max(np.asarray(maxs, dtype=np.float32), axis=0)
    return GraphObject(
        id=object_id,
        label="floor",
        position=((union_min + union_max) * 0.5).astype(np.float32),
        size=np.maximum(union_max - union_min, 1e-3).astype(np.float32),
        angle_z=0.0,
        source_scene_id=parent_id,
        source_object_id=None,
    )


def load_parent_split_graphs(args: argparse.Namespace) -> LoadedGraph:
    primary_scene_id = args.scene_id or scene_id_from_graph_path(args.graph_json)
    parent_id = parent_scan_id(primary_scene_id)
    paths = parent_split_graph_paths(args.graph_json, primary_scene_id)
    if not paths:
        raise FileNotFoundError(f"Could not find split graph JSONs for parent scene {parent_id!r}.")

    if args.box_json is not None and len(paths) > 1:
        box_note = (
            f"Ignored --box-json while merging {len(paths)} split graphs; "
            "each split graph must carry its own object geometry to avoid local-id collisions."
        )
        per_split_box_json = None
    else:
        box_note = ""
        per_split_box_json = args.box_json

    warnings: list[str] = []
    split_graphs: list[LoadedGraph] = []

    for path in paths:
        split_scene_id = scene_id_from_graph_path(path)
        split_graphs.append(load_graph(path, split_scene_id, per_split_box_json, args.allow_scene_id_mismatch))

    floor_sources = [obj for split_graph in split_graphs for obj in split_graph.objects if is_floor_object(obj)]
    floor_merged_id = 0 if floor_sources else None
    next_object_id = 1 if floor_sources else 0
    combined_objects: list[GraphObject] = []
    combined_relationships: list[Relationship] = []
    id_maps: dict[str, dict[int, int]] = {}

    for split_graph in split_graphs:
        id_map: dict[int, int] = {}
        for obj in split_graph.objects:
            if is_floor_object(obj) and floor_merged_id is not None:
                id_map[obj.id] = floor_merged_id
                continue
            new_id = next_object_id
            next_object_id += 1
            id_map[obj.id] = new_id
            combined_objects.append(
                GraphObject(
                    id=new_id,
                    label=obj.label,
                    position=obj.position.copy(),
                    size=obj.size.copy(),
                    angle_z=obj.angle_z,
                    color=obj.color,
                    source_scene_id=split_graph.scene_id,
                    source_object_id=obj.source_object_id if obj.source_object_id is not None else obj.id,
                )
            )
        id_maps[split_graph.scene_id] = id_map

    if floor_sources and floor_merged_id is not None:
        combined_objects.insert(0, merged_floor_object(floor_sources, floor_merged_id, parent_id))

    for split_graph in split_graphs:
        id_map = id_maps[split_graph.scene_id]
        for rel in split_graph.relationships:
            subject_id = id_map.get(rel.subject_id)
            object_id = id_map.get(rel.object_id)
            if subject_id is None or object_id is None:
                warnings.append(
                    f"Skipped relationship in {split_graph.scene_id}: "
                    f"{rel.subject_id} --{rel.predicate}--> {rel.object_id} references a missing local object."
                )
                continue
            combined_relationships.append(
                Relationship(
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate=rel.predicate,
                    source_scene_id=split_graph.scene_id,
                )
            )

        warnings.extend(f"{split_graph.scene_id}: {warning}" for warning in split_graph.warnings)

    if box_note:
        warnings.append(box_note)
    if floor_sources:
        warnings.append(f"Merged {len(floor_sources)} predicted floor objects into one parent-scene floor.")
    warnings.append(f"Merged {len(paths)} split graph JSONs for parent scene {parent_id}.")
    assign_object_colors(combined_objects)
    return LoadedGraph(
        scene_id=parent_id,
        graph_json=args.graph_json,
        source_graph_jsons=paths,
        objects=combined_objects,
        relationships=combined_relationships,
        warnings=warnings,
    )


def stable_color(label: str) -> tuple[int, int, int]:
    palette = [
        (87, 144, 255),
        (255, 177, 66),
        (90, 197, 136),
        (229, 112, 126),
        (170, 130, 255),
        (65, 196, 207),
        (220, 214, 86),
        (236, 131, 222),
    ]
    index = sum((idx + 1) * ord(ch) for idx, ch in enumerate(label)) % len(palette)
    base = np.asarray(palette[index], dtype=np.int32)
    jitter = np.asarray([(ord(ch) * 17) % 29 for ch in (label + "xyz")[:3]], dtype=np.int32) - 14
    color = np.clip(base + jitter, 40, 245)
    return int(color[0]), int(color[1]), int(color[2])


def assign_object_colors(objects: list[GraphObject]) -> None:
    for obj in objects:
        obj.color = stable_color(obj.label)


def darken_rgb(color: tuple[int, int, int] | np.ndarray, factor: float = 0.72) -> tuple[int, int, int]:
    arr = np.asarray(color, dtype=np.float32) * float(factor)
    arr = np.clip(arr, 22, 220).astype(np.uint8)
    return int(arr[0]), int(arr[1]), int(arr[2])


def load_point_cloud_ply(path: Path, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")

        fmt = None
        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no end_header.")
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                break
            parts = text.split()
            if len(parts) >= 3 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
                properties.append((parts[2], parts[1]))

        if fmt is None or vertex_count is None:
            raise ValueError(f"{path} is missing PLY format or vertex count.")
        if fmt == "ascii":
            data = np.loadtxt(handle, max_rows=vertex_count)
            if data.ndim == 1:
                data = data[None, :]
            columns = {name: data[:, idx] for idx, (name, _type_name) in enumerate(properties)}
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + ply_numpy_dtype(type_name)) for name, type_name in properties])
            data = np.frombuffer(handle.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count)
            columns = {name: data[name] for name, _type_name in properties}
        else:
            raise ValueError(f"Unsupported PLY format {fmt!r}.")

    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(columns):
        colors = np.stack([columns["red"], columns["green"], columns["blue"]], axis=1)
    elif {"r", "g", "b"}.issubset(columns):
        colors = np.stack([columns["r"], columns["g"], columns["b"]], axis=1)
    else:
        colors = np.full((xyz.shape[0], 3), 185, dtype=np.uint8)
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
        xyz = xyz[indices]
        colors = colors[indices]
    return xyz, colors


def ply_numpy_dtype(type_name: str) -> str:
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported PLY property type {type_name!r}.")
    return mapping[type_name]


def normalize_colors(colors: Any, count: int) -> np.ndarray:
    if colors is None:
        return np.full((count, 3), 185, dtype=np.uint8)
    arr = np.asarray(colors)
    if arr.ndim != 2 or arr.shape[0] != count or arr.shape[1] < 3:
        return np.full((count, 3), 185, dtype=np.uint8)
    arr = arr[:, :3]
    if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.1:
        arr = arr * 255.0
    return np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def first_array_by_keys(payload: dict[str, Any], keys: list[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in payload:
            return tensor_to_numpy(payload[key])
    return None


def sceneverse_pcd_path(sceneverse_pcd_root: Path, scan_id: str) -> Path:
    return sceneverse_pcd_root / "pcd_with_global_alignment" / f"{scan_id}.pth"


def sceneverse_align_angle_path(sceneverse_pcd_root: Path, scan_id: str) -> Path:
    return sceneverse_pcd_root / "pcd_with_global_alignment" / f"{scan_id}_align_angle.npy"


def raw_instance_ply_path(r3scan_root: Path, scan_id: str) -> Path:
    return r3scan_root / scan_id / "labels.instances.annotated.v2.ply"


def textured_mesh_obj_path(r3scan_root: Path, scan_id: str) -> Path:
    return r3scan_root / scan_id / "mesh.refined.v2.obj"


def texture_path_from_obj(obj_path: Path) -> Optional[Path]:
    mtl_path: Optional[Path] = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("mtllib "):
                mtl_name = line.split(maxsplit=1)[1].strip()
                mtl_path = obj_path.parent / mtl_name
                break
            if line.startswith("v "):
                break

    if mtl_path is None or not mtl_path.is_file():
        fallback = obj_path.parent / "mesh.refined_0.png"
        return fallback if fallback.is_file() else None

    with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("map_Kd ") or stripped.startswith("map_Ka "):
                texture_name = stripped.split(maxsplit=1)[1].strip()
                texture_path = mtl_path.parent / texture_name
                return texture_path if texture_path.is_file() else None
    fallback = obj_path.parent / "mesh.refined_0.png"
    return fallback if fallback.is_file() else None


def obj_index(raw_index: str, count: int) -> int:
    value = int(raw_index)
    return value - 1 if value > 0 else count + value


def parse_obj_face_token(token: str, vertex_count: int, uv_count: int) -> tuple[int, int]:
    parts = token.split("/")
    vertex_index = obj_index(parts[0], vertex_count)
    uv_index = -1
    if len(parts) > 1 and parts[1]:
        uv_index = obj_index(parts[1], uv_count)
    return vertex_index, uv_index


def load_obj_vertices_uv_faces(obj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    face_vertices: list[list[int]] = []
    face_uvs: list[list[int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                parsed = [parse_obj_face_token(token, len(vertices), len(uvs)) for token in tokens]
                for idx in range(1, len(parsed) - 1):
                    tri = [parsed[0], parsed[idx], parsed[idx + 1]]
                    face_vertices.append([item[0] for item in tri])
                    face_uvs.append([item[1] for item in tri])

    if not vertices or not face_vertices:
        raise ValueError(f"Could not parse textured mesh geometry from {obj_path}")
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(face_vertices, dtype=np.int64),
        np.asarray(face_uvs, dtype=np.int64),
    )


def sample_texture_nearest(texture_path: Path, uv: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Sampling 3RScan textured mesh colors requires Pillow. Install: python -m pip install pillow"
        ) from exc

    image = Image.open(texture_path).convert("RGB")
    texture = np.asarray(image, dtype=np.uint8)
    height, width = texture.shape[:2]
    uv = np.asarray(uv, dtype=np.float32)
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.rint(u * float(width - 1)).astype(np.int64)
    y = np.rint((1.0 - v) * float(height - 1)).astype(np.int64)
    return texture[y, x, :3]


def rotate_z_axis_by_degrees(points: np.ndarray, theta: float, clockwise: bool = True) -> np.ndarray:
    theta_rad = np.deg2rad(float(theta))
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    rot_matrix = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    if not clockwise:
        rot_matrix = rot_matrix.T
    return np.asarray(points, dtype=np.float32).dot(rot_matrix)


def compute_box_3d(size: list[float], center: Any, rotmat: np.ndarray) -> np.ndarray:
    half_x, half_y, half_z = [float(value) / 2.0 for value in size]
    center_arr = np.asarray(center, dtype=np.float32).reshape(3)
    x_corners = [half_x, half_x, -half_x, -half_x, half_x, half_x, -half_x, -half_x]
    y_corners = [half_y, -half_y, -half_y, half_y, half_y, -half_y, -half_y, half_y]
    z_corners = [half_z, half_z, half_z, half_z, -half_z, -half_z, -half_z, -half_z]
    corners = np.dot(np.asarray(rotmat, dtype=np.float32).T, np.vstack([x_corners, y_corners, z_corners]))
    corners[0, :] += center_arr[0]
    corners[1, :] += center_arr[1]
    corners[2, :] += center_arr[2]
    return corners.T.astype(np.float32)


def is_axis_aligned_box(rotated_box: np.ndarray, threshold: float = 0.05) -> bool:
    x_diff = abs(float(rotated_box[0][0] - rotated_box[1][0]))
    y_diff = abs(float(rotated_box[0][1] - rotated_box[3][1]))
    return x_diff < threshold and y_diff < threshold


def calc_sceneverse_align_angle(bbox_list: list[np.ndarray]) -> float:
    for angle_range, num_bins, threshold in [((-45.0, 45.0), 90, 0.05), ((-90.0, 90.0), 180, 0.15)]:
        angle_counts: dict[float, int] = {}
        for angle in np.linspace(angle_range[0], angle_range[1], num_bins):
            bucket = round(float(angle), 3)
            for box in bbox_list:
                box_r = rotate_z_axis_by_degrees(np.asarray(box, dtype=np.float32), bucket)
                bottom = box_r[4:]
                if is_axis_aligned_box(bottom, threshold):
                    angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
        if angle_counts:
            return float(max(angle_counts, key=angle_counts.get))
    return 0.0


def compute_sceneverse_align_angle_from_semseg(r3scan_root: Path, scan_id: str) -> float:
    semseg_path = r3scan_root / scan_id / "semseg.v2.json"
    if not semseg_path.is_file():
        raise FileNotFoundError(f"Could not find {semseg_path} to recompute SceneVerse align angle.")
    payload = load_json_or_text(semseg_path)
    bbox_list: list[np.ndarray] = []
    for group in payload.get("segGroups", []):
        obb = group.get("obb", {}) if isinstance(group, dict) else {}
        if not {"normalizedAxes", "centroid", "axesLengths"}.issubset(obb):
            continue
        rotation = np.asarray(obb["normalizedAxes"], dtype=np.float32).reshape(3, 3)
        center = np.asarray(obb["centroid"], dtype=np.float32).reshape(3)
        size = np.asarray(obb["axesLengths"], dtype=np.float32).reshape(3).tolist()
        bbox_list.append(compute_box_3d(size, center, rotation))
    if not bbox_list:
        raise ValueError(f"No OBBs found in {semseg_path}; cannot compute SceneVerse align angle.")
    return calc_sceneverse_align_angle(bbox_list)


def sceneverse_alignment_center(rotated_vertices: np.ndarray) -> np.ndarray:
    center_points = np.mean(rotated_vertices, axis=0).astype(np.float32)
    center_points[2] = float(np.min(rotated_vertices[:, 2]))
    return center_points


def load_sceneverse_align_angle(sceneverse_pcd_root: Path, r3scan_root: Path, scan_id: str) -> tuple[float, str]:
    align_path = sceneverse_align_angle_path(sceneverse_pcd_root, scan_id)
    if align_path.is_file():
        value = np.load(align_path)
        return float(np.asarray(value).reshape(-1)[0]), str(align_path)
    return compute_sceneverse_align_angle_from_semseg(r3scan_root, scan_id), str(r3scan_root / scan_id / "semseg.v2.json")


def load_textured_mesh_points(
    obj_path: Path,
    sample_count: int,
    align_angle: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    texture_path = texture_path_from_obj(obj_path)
    if texture_path is None:
        raise ValueError(f"Could not find texture image referenced by {obj_path}")

    vertices, uvs, faces, face_uvs = load_obj_vertices_uv_faces(obj_path)
    triangles = vertices[faces]
    edge_ab = triangles[:, 1] - triangles[:, 0]
    edge_ac = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_ab, edge_ac), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-12)
    if not np.any(valid):
        raise ValueError(f"Mesh {obj_path} has no non-degenerate faces.")

    faces = faces[valid]
    face_uvs = face_uvs[valid]
    triangles = triangles[valid]
    areas = areas[valid]
    probabilities = areas / float(np.sum(areas))

    count = int(sample_count)
    if count <= 0:
        count = int(len(faces))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, replace=True, p=probabilities)
    chosen_triangles = triangles[chosen]

    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    weights = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1).astype(np.float32)
    points = np.sum(chosen_triangles * weights[:, :, None], axis=1).astype(np.float32)

    if uvs.size and np.all(face_uvs[chosen] >= 0):
        chosen_uvs = uvs[face_uvs[chosen]]
        sampled_uv = np.sum(chosen_uvs * weights[:, :, None], axis=1)
        colors = sample_texture_nearest(texture_path, sampled_uv)
    else:
        colors = np.full((count, 3), 185, dtype=np.uint8)

    rotated_vertices = rotate_z_axis_by_degrees(vertices, align_angle)
    center_points = sceneverse_alignment_center(rotated_vertices)
    aligned_points = rotate_z_axis_by_degrees(points, align_angle) - center_points
    meta = {
        "align_angle_degrees": float(align_angle),
        "sceneverse_center_translation": center_points.astype(float).tolist(),
        "raw_mesh_min": vertices.min(axis=0).astype(float).tolist(),
        "raw_mesh_max": vertices.max(axis=0).astype(float).tolist(),
        "aligned_mesh_min": (rotated_vertices - center_points).min(axis=0).astype(float).tolist(),
        "aligned_mesh_max": (rotated_vertices - center_points).max(axis=0).astype(float).tolist(),
    }
    return aligned_points.astype(np.float32), colors, meta


def load_sceneverse_pcd(path: Path, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Loading SceneVerse .pth point clouds requires torch. Install: python -m pip install torch"
        ) from exc

    try:
        payload = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # SceneVerse point-cloud caches can contain NumPy pickled objects.
        # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects
        # those caches. This fallback is limited to the user-supplied local
        # SceneVerse cache path used only for visualization base geometry.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    xyz = None
    colors = None

    if isinstance(payload, dict):
        xyz = first_array_by_keys(payload, ["coord", "coords", "points", "xyz", "point", "vertices"])
        colors = first_array_by_keys(payload, ["color", "colors", "rgb", "rgba"])
    elif isinstance(payload, (list, tuple)):
        arrays = [tensor_to_numpy(item) for item in payload if hasattr(item, "shape") or isinstance(item, np.ndarray)]
        xyz_candidates = [arr for arr in arrays if arr.ndim == 2 and arr.shape[1] >= 3]
        if xyz_candidates:
            xyz = xyz_candidates[0][:, :3]
            if len(xyz_candidates) > 1:
                colors = xyz_candidates[1][:, :3]

    if xyz is None:
        raise ValueError(f"Could not find Nx3 coordinates in SceneVerse point cloud {path}")

    xyz = np.asarray(xyz[:, :3], dtype=np.float32)
    point_colors = normalize_colors(colors, xyz.shape[0])
    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
        xyz = xyz[indices]
        point_colors = point_colors[indices]
    return xyz, point_colors


def scene_top_trim_threshold(points: np.ndarray, trim_percent: float) -> Optional[float]:
    trim_percent = float(trim_percent)
    if trim_percent <= 0.0 or points.size == 0:
        return None
    trim_percent = min(trim_percent, 99.0)
    return float(np.percentile(points[:, 2], 100.0 - trim_percent))


def trim_top_scene_points(points: np.ndarray, colors: np.ndarray, trim_percent: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = scene_top_trim_threshold(points, trim_percent)
    if threshold is None:
        return points, colors
    keep = points[:, 2] <= threshold
    if not np.any(keep):
        return points, colors
    return points[keep], colors[keep]


def scene_top_trim_metadata(points: Optional[np.ndarray], trim_percent: float) -> dict[str, Any]:
    if points is None:
        return {}
    threshold = scene_top_trim_threshold(points, trim_percent)
    if threshold is None:
        return {}
    removed = int(np.count_nonzero(points[:, 2] > threshold))
    return {
        "scene_top_trim_percent": float(min(max(float(trim_percent), 0.0), 99.0)),
        "scene_top_trim_height": threshold,
        "scene_top_trim_points_removed": removed,
    }


def load_base_points(graph: LoadedGraph, args: argparse.Namespace) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    if args.point_cloud is not None:
        xyz, colors = load_point_cloud_ply(args.point_cloud, args.max_points)
        trim_meta = scene_top_trim_metadata(xyz, args.scene_top_trim_percent)
        xyz, colors = trim_top_scene_points(xyz, colors, args.scene_top_trim_percent)
        return xyz, colors, {
            "base_mode": "point_cloud",
            "path": str(args.point_cloud),
            "num_points_rendered": int(xyz.shape[0]),
            **trim_meta,
        }

    scan_id = parent_scan_id(graph.scene_id)
    if args.base_mode == "sceneverse_textured_mesh_points":
        mesh_path = textured_mesh_obj_path(args.r3scan_root, scan_id)
        if not mesh_path.is_file():
            graph.warnings.append(f"3RScan textured mesh not found at {mesh_path}.")
            return None, None, {"base_mode": args.base_mode, "missing": str(mesh_path)}
        align_angle, align_source = load_sceneverse_align_angle(args.sceneverse_pcd_root, args.r3scan_root, scan_id)
        xyz, colors, align_meta = load_textured_mesh_points(mesh_path, args.mesh_point_samples, align_angle)
        trim_meta = scene_top_trim_metadata(xyz, args.scene_top_trim_percent)
        xyz, colors = trim_top_scene_points(xyz, colors, args.scene_top_trim_percent)
        return xyz, colors, {
            "base_mode": args.base_mode,
            "rendered_base_mode": "3rscan_textured_mesh_surface_points",
            "path": str(mesh_path),
            "texture": str(texture_path_from_obj(mesh_path) or ""),
            "num_points_rendered": int(xyz.shape[0]),
            "alignment": "sceneverse_rscan_rotate_z_then_subtract_mean_xy_min_z",
            "align_angle_source": align_source,
            **align_meta,
            **trim_meta,
        }

    if args.base_mode == "sceneverse_pcd":
        pcd_path = sceneverse_pcd_path(args.sceneverse_pcd_root, scan_id)
        if not pcd_path.is_file():
            graph.warnings.append(f"SceneVerse point cloud not found at {pcd_path}.")
            return None, None, {"base_mode": args.base_mode, "missing": str(pcd_path)}
        xyz, colors = load_sceneverse_pcd(pcd_path, args.max_points)
        trim_meta = scene_top_trim_metadata(xyz, args.scene_top_trim_percent)
        xyz, colors = trim_top_scene_points(xyz, colors, args.scene_top_trim_percent)
        return xyz, colors, {
            "base_mode": args.base_mode,
            "rendered_base_mode": "sceneverse_pcd",
            "path": str(pcd_path),
            "num_points_rendered": int(xyz.shape[0]),
            **trim_meta,
        }

    if args.base_mode == "raw_instance_ply":
        ply_path = raw_instance_ply_path(args.r3scan_root, scan_id)
        if not ply_path.is_file():
            graph.warnings.append(f"Raw instance PLY not found at {ply_path}.")
            return None, None, {"base_mode": args.base_mode, "missing": str(ply_path)}
        xyz, colors = load_point_cloud_ply(ply_path, args.max_points)
        trim_meta = scene_top_trim_metadata(xyz, args.scene_top_trim_percent)
        xyz, colors = trim_top_scene_points(xyz, colors, args.scene_top_trim_percent)
        return xyz, colors, {
            "base_mode": args.base_mode,
            "path": str(ply_path),
            "num_points_rendered": int(xyz.shape[0]),
            **trim_meta,
        }

    if args.base_mode == "raw_mesh":
        mesh_path = textured_mesh_obj_path(args.r3scan_root, scan_id)
        if not mesh_path.is_file():
            graph.warnings.append(f"3RScan textured mesh not found at {mesh_path}.")
            return None, None, {"base_mode": args.base_mode, "missing": str(mesh_path)}
        align_angle, align_source = load_sceneverse_align_angle(args.sceneverse_pcd_root, args.r3scan_root, scan_id)
        xyz, colors, align_meta = load_textured_mesh_points(mesh_path, args.mesh_point_samples, align_angle)
        trim_meta = scene_top_trim_metadata(xyz, args.scene_top_trim_percent)
        xyz, colors = trim_top_scene_points(xyz, colors, args.scene_top_trim_percent)
        return xyz, colors, {
            "base_mode": args.base_mode,
            "rendered_base_mode": "3rscan_textured_mesh_surface_points",
            "path": str(mesh_path),
            "texture": str(texture_path_from_obj(mesh_path) or ""),
            "num_points_rendered": int(xyz.shape[0]),
            "alignment": "sceneverse_rscan_rotate_z_then_subtract_mean_xy_min_z",
            "align_angle_source": align_source,
            **align_meta,
            **trim_meta,
        }

    return None, None, {"base_mode": "none"}


def ensure_pyviz3d():
    try:
        import pyviz3d.visualizer as viz
    except ModuleNotFoundError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
    return viz


def load_clip_object_matcher():
    try:
        from sg_eval.matchers import CLIPObjectMatcher
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live query serving requires the sg_eval package on PYTHONPATH. "
            "Static viewer generation does not require it."
        ) from exc
    return CLIPObjectMatcher


def euler_z_quaternion(viz: Any, yaw: float) -> np.ndarray:
    if hasattr(viz, "euler_to_quaternion"):
        return np.asarray(viz.euler_to_quaternion(0.0, 0.0, yaw), dtype=np.float32)
    half = 0.5 * yaw
    return np.asarray([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)


def json_safe_pyviz_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe_pyviz_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_pyviz_value(item) for item in value]
    return value


def sanitize_pyviz_get_properties(visualizer: Any) -> None:
    """Make PyViz3D metadata JSON-safe without changing its binary writers."""
    elements = getattr(visualizer, "elements", None)
    if elements is None:
        elements = getattr(visualizer, "_elements", None)
    if not elements:
        return

    if isinstance(elements, dict):
        element_iterable = elements.values()
    else:
        element_iterable = elements

    for element in element_iterable:
        original = getattr(element, "get_properties", None)
        if original is None or getattr(element, "_sg_json_safe_properties", False):
            continue

        def safe_get_properties(filename: str, _original=original):
            return json_safe_pyviz_value(_original(filename))

        element.get_properties = safe_get_properties
        element._sg_json_safe_properties = True


def save_pyviz3d_json_safe(viz: Any, visualizer: Any, output_path: Path, port: int) -> None:
    """Save PyViz3D output while converting NumPy values in nodes.json metadata."""
    sanitize_pyviz_get_properties(visualizer)

    json_module = getattr(viz, "json", None)
    original_dump = getattr(json_module, "dump", None) if json_module is not None else None

    if original_dump is None:
        visualizer.save(str(output_path), port=int(port), verbose=False)
        return

    def safe_dump(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> Any:
        return original_dump(json_safe_pyviz_value(obj), fp, *args, **kwargs)

    json_module.dump = safe_dump
    try:
        visualizer.save(str(output_path), port=int(port), verbose=False)
    finally:
        json_module.dump = original_dump


def object_layer_name(obj: GraphObject) -> str:
    return f"Scene graph;Object boxes;object_{obj.id};{obj.label}"


def highlight_layer_name(obj: GraphObject) -> str:
    return f"Scene query;Highlight;object_{obj.id};{obj.label}"


def object_display_text(obj: GraphObject, graph_scene_id: str) -> str:
    return obj.label


def object_label_lower(obj: GraphObject) -> str:
    return obj.label.replace("_", " ").strip().lower()


def is_floor_object(obj: GraphObject) -> bool:
    return object_label_lower(obj) == "floor"


def is_ceiling_object(obj: GraphObject) -> bool:
    return object_label_lower(obj) == "ceiling"


def is_wall_object(obj: GraphObject) -> bool:
    return object_label_lower(obj) == "wall"


def is_generic_object_label(label: str) -> bool:
    return str(label).strip().lower() == "object"


def rotation_z_matrix(angle_z: float) -> np.ndarray:
    c = math.cos(angle_z)
    s = math.sin(angle_z)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def surface_anchor(source: GraphObject, target: GraphObject) -> np.ndarray:
    direction_world = np.asarray(target.position - source.position, dtype=np.float32)
    if float(np.linalg.norm(direction_world)) < 1e-6:
        return source.position.astype(np.float32)

    rotation = rotation_z_matrix(source.angle_z)
    direction_local = rotation.T @ direction_world
    half_size = np.maximum(source.size.astype(np.float32) * 0.5, 1e-4)
    scales = []
    for axis in range(3):
        value = float(direction_local[axis])
        if abs(value) > 1e-6:
            scales.append(float(half_size[axis]) / abs(value))
    if not scales:
        return source.position.astype(np.float32)
    local_hit = direction_local * min(scales)
    anchor = source.position + rotation @ local_hit
    outward = direction_world / max(float(np.linalg.norm(direction_world)), 1e-6)
    return (anchor + outward * 0.035).astype(np.float32)


def topdown_anchor(obj: GraphObject) -> np.ndarray:
    return obj.position.astype(np.float32)


def relationship_points(subject: GraphObject, target: GraphObject, style: str) -> Optional[np.ndarray]:
    if style == "center":
        start = subject.position.astype(np.float32)
        end = target.position.astype(np.float32)
    elif style == "surface":
        start = surface_anchor(subject, target)
        end = surface_anchor(target, subject)
    elif style == "raised":
        start = surface_anchor(subject, target)
        end = surface_anchor(target, subject)
        midpoint = (start + end) * 0.5
        lift = max(0.18, min(0.45, 0.18 + 0.08 * float(np.linalg.norm(end - start))))
        midpoint = midpoint + np.asarray([0.0, 0.0, lift], dtype=np.float32)
        return np.asarray([start, midpoint, end], dtype=np.float32)
    elif style == "topdown":
        if is_floor_object(subject) or is_floor_object(target) or is_ceiling_object(subject) or is_ceiling_object(target):
            return None
        start = topdown_anchor(subject)
        end = topdown_anchor(target)
    else:
        raise ValueError(f"Unknown relationship style: {style}")
    return np.asarray([start, end], dtype=np.float32)


def prune_graph_above_height(graph: LoadedGraph, height: Optional[float]) -> dict[str, Any]:
    if height is None:
        graph.warnings.append("--scene-top-trim-graph requested, but no scene top-trim height was available.")
        return {"scene_top_trim_graph_enabled": True, "scene_top_trim_graph_height": None}

    removed_ids = {obj.id for obj in graph.objects if float(obj.position[2]) > float(height)}
    if not removed_ids:
        graph.warnings.append(f"Scene top-trim graph pruning kept all objects at height <= {height:.4f}.")
        return {
            "scene_top_trim_graph_enabled": True,
            "scene_top_trim_graph_height": float(height),
            "scene_top_trim_graph_objects_removed": 0,
            "scene_top_trim_graph_relationships_removed": 0,
        }

    original_object_count = len(graph.objects)
    original_relationship_count = len(graph.relationships)
    graph.objects = [obj for obj in graph.objects if obj.id not in removed_ids]
    graph.relationships = [
        rel
        for rel in graph.relationships
        if rel.subject_id not in removed_ids and rel.object_id not in removed_ids
    ]
    removed_relationship_count = original_relationship_count - len(graph.relationships)
    graph.warnings.append(
        f"Scene top-trim graph pruning removed {original_object_count - len(graph.objects)} objects "
        f"and {removed_relationship_count} relationships above z={height:.4f}."
    )
    return {
        "scene_top_trim_graph_enabled": True,
        "scene_top_trim_graph_height": float(height),
        "scene_top_trim_graph_objects_removed": original_object_count - len(graph.objects),
        "scene_top_trim_graph_relationships_removed": removed_relationship_count,
        "scene_top_trim_graph_removed_object_ids": sorted(int(value) for value in removed_ids),
    }


def render_pyviz3d(graph: LoadedGraph, args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    viz = ensure_pyviz3d()
    visualizer = viz.Visualizer()

    xyz, colors, base_meta = load_base_points(graph, args)
    if args.scene_top_trim_graph:
        base_meta.update(prune_graph_above_height(graph, base_meta.get("scene_top_trim_height")))
    if xyz is not None and colors is not None:
        point_size = effective_scene_point_size(args)
        visualizer.add_points(
            "Scene;Point cloud",
            xyz,
            colors,
            point_size=float(point_size),
            visible=True,
            alpha=0.88,
        )

    label_positions = []
    label_texts = []
    label_colors = []
    center_positions = []
    center_colors = []
    for obj in graph.objects:
        rotation = euler_z_quaternion(viz, float(args.box_angle_sign) * obj.angle_z)
        color = np.asarray(obj.color, dtype=np.uint8)
        visualizer.add_bounding_box(
            object_layer_name(obj),
            position=obj.position.astype(np.float32),
            size=obj.size.astype(np.float32),
            rotation=rotation.astype(np.float32),
            color=color.astype(np.uint8),
            alpha=float(args.box_alpha),
            edge_width=float(args.box_edge_width),
            visible=True,
        )
        visualizer.add_bounding_box(
            highlight_layer_name(obj),
            position=obj.position.astype(np.float32),
            size=(obj.size * 1.03).astype(np.float32),
            rotation=rotation.astype(np.float32),
            color=np.asarray([255, 255, 32], dtype=np.uint8),
            alpha=float(args.highlight_alpha),
            edge_width=float(args.highlight_edge_width),
            visible=False,
        )
        label_positions.append(obj.position + np.asarray([0.0, 0.0, 0.55 * float(obj.size[2]) + 0.04]))
        label_texts.append(object_display_text(obj, graph.scene_id))
        label_colors.append(color)
        if not is_floor_object(obj) and not is_ceiling_object(obj):
            center_positions.append(topdown_anchor(obj))
            center_colors.append(np.asarray(darken_rgb(obj.color), dtype=np.uint8))

    if label_positions and not args.hide_object_labels:
        visualizer.add_labels(
            "Scene graph;Object labels",
            np.asarray(label_positions, dtype=np.float32),
            label_texts,
            np.asarray(label_colors, dtype=np.uint8),
            visible=True,
        )

    if center_positions:
        visualizer.add_points(
            "Scene graph;Object centers",
            np.asarray(center_positions, dtype=np.float32),
            np.asarray(center_colors, dtype=np.uint8),
            point_size=60.0,
            visible=args.ui_mode == "pyviz3d",
            alpha=0.96,
        )

    object_by_id = {obj.id: obj for obj in graph.objects}
    if not args.hide_relationships:
        edge_palette = [
            (214, 151, 42),
            (49, 159, 146),
            (67, 151, 218),
            (105, 176, 58),
            (204, 83, 72),
            (176, 82, 206),
        ]
        rel_label_positions_by_style: dict[str, list[np.ndarray]] = {style: [] for style in ("center", "surface", "raised", "topdown")}
        rel_label_texts_by_style: dict[str, list[str]] = {style: [] for style in ("center", "surface", "raised", "topdown")}
        rel_label_colors_by_style: dict[str, list[np.ndarray]] = {style: [] for style in ("center", "surface", "raised", "topdown")}
        for edge_index, edge in enumerate(graph.relationships):
            subject = object_by_id.get(edge.subject_id)
            target = object_by_id.get(edge.object_id)
            if subject is None or target is None:
                continue
            color = np.asarray(edge_palette[edge_index % len(edge_palette)], dtype=np.uint8)
            for style in ("center", "surface", "raised", "topdown"):
                edge_points = relationship_points(subject, target, style)
                if edge_points is None:
                    continue
                name = (
                    "Scene graph;Relationship edges;"
                    f"{style};{edge_index}_{edge.subject_id}_to_{edge.object_id};{edge.predicate}"
                )
                visualizer.add_polyline(
                    name,
                    edge_points,
                    color=color.astype(np.uint8),
                    alpha=float(args.edge_alpha),
                    edge_width=float(args.edge_width if style != "topdown" else max(args.edge_width, 0.018)),
                    visible=style == "raised",
                )
                rel_label_positions_by_style[style].append(np.mean(edge_points, axis=0))
                rel_label_texts_by_style[style].append(edge.predicate)
                rel_label_colors_by_style[style].append(color)

        for style, positions in rel_label_positions_by_style.items():
            if not positions:
                continue
            visualizer.add_labels(
                f"Scene graph;Relationship labels;{style}",
                np.asarray(positions, dtype=np.float32),
                rel_label_texts_by_style[style],
                np.asarray(rel_label_colors_by_style[style], dtype=np.uint8),
                visible=False,
            )

    save_pyviz3d_json_safe(viz, visualizer, output_path, int(args.port))
    write_query_payload(output_path, graph, args, base_meta)
    patch_scene_js_to_expose_api(output_path)
    patch_index_html_for_query(output_path, args)
    return base_meta


def effective_scene_point_size(args: argparse.Namespace) -> float:
    if args.paper_point_size is not None:
        return float(args.paper_point_size)
    if args.base_mode == "sceneverse_textured_mesh_points":
        return float(args.mesh_point_size)
    return float(args.point_size)


def query_object_payload(graph: LoadedGraph) -> list[dict[str, Any]]:
    return [
        {
            "id": int(obj.id),
            "label": obj.label,
            "display_label": object_display_text(obj, graph.scene_id),
            "source_scene_id": obj.source_scene_id,
            "source_object_id": obj.source_object_id,
            "position": obj.position.astype(float).tolist(),
            "size": obj.size.astype(float).tolist(),
            "color": [int(value) for value in obj.color],
            "layer": object_layer_name(obj),
            "highlight_layer": highlight_layer_name(obj),
        }
        for obj in graph.objects
    ]


def camera_trajectory_payload(graph: LoadedGraph, args: argparse.Namespace, base_meta: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not base_meta:
        return []
    align_angle = base_meta.get("align_angle_degrees")
    center_points = base_meta.get("sceneverse_center_translation")
    if align_angle is None or center_points is None:
        return []

    scan_id = parent_scan_id(graph.scene_id)
    transforms_path = args.r3scan_root / scan_id / "transforms_train.json"
    if not transforms_path.is_file():
        return []
    try:
        payload = load_json_or_text(transforms_path)
    except Exception:
        return []

    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    raw_positions = []
    for frame in frames:
        matrix = frame.get("transform_matrix") if isinstance(frame, dict) else None
        if matrix is None:
            continue
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape != (4, 4):
            continue
        raw_positions.append(arr[:3, 3])
    if len(raw_positions) < 2:
        return []

    positions = rotate_z_axis_by_degrees(np.asarray(raw_positions, dtype=np.float32), float(align_angle))
    positions = positions - np.asarray(center_points, dtype=np.float32).reshape(1, 3)

    # Keep the browser payload light while preserving the scan traversal shape.
    max_points = 90
    if len(positions) > max_points:
        indices = np.linspace(0, len(positions) - 1, max_points).round().astype(int)
        positions = positions[indices]

    return [
        {"position": np.asarray(position, dtype=float).tolist()}
        for position in positions
        if np.all(np.isfinite(position))
    ]


def write_query_payload(
    output_path: Path,
    graph: LoadedGraph,
    args: argparse.Namespace,
    base_meta: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "scene_id": graph.scene_id,
        "matcher": {
            "clip_model": args.clip_model,
        },
        "objects": query_object_payload(graph),
        "camera_trajectory": camera_trajectory_payload(graph, args, base_meta),
        "relationships": [
            {
                "index": int(idx),
                "subject_id": int(edge.subject_id),
                "object_id": int(edge.object_id),
                "predicate": edge.predicate,
                "source_scene_id": edge.source_scene_id if edge.source_scene_id != graph.scene_id else None,
            }
            for idx, edge in enumerate(graph.relationships)
        ],
    }
    (output_path / "scene_query_objects.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def patch_scene_js_to_expose_api(output_path: Path) -> None:
    scene_js_path = output_path / "js" / "scene.js"
    if not scene_js_path.is_file():
        return
    text = scene_js_path.read_text(encoding="utf-8")
    if "function get_viewer_size()" not in text and "function get_viewer_size(){" not in text:
        marker = "function render() {\n"
        helper = (
            "function get_viewer_size(){\n"
            "\tconst container = document.getElementById('render_container');\n"
            "\tconst width = Math.max(1, container ? container.clientWidth : window.innerWidth);\n"
            "\tconst height = Math.max(1, container ? container.clientHeight : window.innerHeight);\n"
            "\treturn {width, height};\n"
            "}\n\n"
        )
        if marker in text:
            text = text.replace(marker, helper + marker, 1)

    replacements = [
        (
            "scene.background = new THREE.Color(0xffffff);\n\trenderer.setSize(window.innerWidth, window.innerHeight);\n\tlabelRenderer.setSize(window.innerWidth, window.innerHeight);",
            "scene.background = new THREE.Color(0xffffff);\n\tconst viewerSize = get_viewer_size();\n\trenderer.setSize(viewerSize.width, viewerSize.height);\n\tlabelRenderer.setSize(viewerSize.width, viewerSize.height);",
        ),
        (
            "const renderer = new THREE.WebGLRenderer({antialias: true});",
            "const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});",
        ),
        (
            "const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});",
            "const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});",
        ),
        (
            "var camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.01, 1000);",
            "const initialViewerSize = get_viewer_size();\nvar camera = new THREE.PerspectiveCamera(58, initialViewerSize.width/initialViewerSize.height, 0.01, 1000);",
        ),
        (
            "labelRenderer.setSize( window.innerWidth, window.innerHeight );",
            "labelRenderer.setSize( initialViewerSize.width, initialViewerSize.height );",
        ),
        (
            "controls = new OrbitControls(camera, labelRenderer.domElement);",
            "controls = new OrbitControls(camera, renderer.domElement);",
        ),
        (
            "document.getElementById('render_container').appendChild(renderer.domElement)",
            "renderer.domElement.style.position = 'absolute';\nrenderer.domElement.style.inset = '0';\nrenderer.domElement.style.zIndex = '1';\ndocument.getElementById('render_container').appendChild(renderer.domElement)",
        ),
        (
            "labelRenderer.domElement.style.position = 'absolute';\nlabelRenderer.domElement.style.top = '0px';",
            "labelRenderer.domElement.style.position = 'absolute';\nlabelRenderer.domElement.style.top = '0px';\nlabelRenderer.domElement.style.left = '0px';\nlabelRenderer.domElement.style.zIndex = '8';\nlabelRenderer.domElement.style.pointerEvents = 'none';",
        ),
        (
            "const gui = new GUI({autoPlace: true, width: 120});",
            "const gui = {addFolder: () => ({add: () => ({name: () => ({onChange: () => {}}), onChange: () => {}}), open: () => {}}), add: () => ({name: () => ({onChange: () => {}}), onChange: () => {}})};",
        ),
        (
            "\t.then(() => init_gui(threejs_objects))",
            "\t.then(() => undefined)",
        ),
        (
            "function onWindowResize(){\n    const innerWidth = window.innerWidth\n    const innerHeight = window.innerHeight;\n    renderer.setSize(innerWidth, innerHeight);\n    labelRenderer.setSize(innerWidth, innerHeight);\n    camera.aspect = window.innerWidth / window.innerHeight;\n    camera.updateProjectionMatrix();\n    render();\n}",
            "function onWindowResize(){\n    const viewerSize = get_viewer_size();\n    renderer.setSize(viewerSize.width, viewerSize.height);\n    labelRenderer.setSize(viewerSize.width, viewerSize.height);\n    camera.aspect = viewerSize.width / viewerSize.height;\n    camera.updateProjectionMatrix();\n    render();\n}",
        ),
        (
            "document.body.appendChild(gProgressElement);",
            "document.getElementById('render_container').appendChild(gProgressElement);",
        ),
        (
            "\n\t// Add axis helper\n\tthreejs_objects['Axis'] = new THREE.AxesHelper(1);\n",
            "\n\t// Axis helper hidden for paper-demo presentation.\n",
        ),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)

    text = text.replace(
        "window.pyviz3dDemo = {THREE, scene, camera, controls, renderer, labelRenderer, objects: threejs_objects, render};",
        "window.pyviz3dDemo = {THREE, CSS2DObject, scene, camera, controls, renderer, labelRenderer, objects: threejs_objects, render};",
    )

    if "window.pyviz3dDemo" in text:
        scene_js_path.write_text(text, encoding="utf-8")
        return

    exposure = (
        ".then(() => {\n"
        "\t\twindow.pyviz3dDemo = {THREE, CSS2DObject, scene, camera, controls, renderer, labelRenderer, objects: threejs_objects, render};\n"
        "\t\twindow.dispatchEvent(new CustomEvent('pyviz3d-ready', {detail: window.pyviz3dDemo}));\n"
        "\t})\n"
        "\t.then(render);"
    )
    markers = [
        ".then(() => console.log('Done'))\n\t.then(render);",
        ".then(() => console.log('Done')).then(render);",
    ]
    for marker in markers:
        if marker in text:
            text = text.replace(marker, ".then(() => console.log('Done'))\n\t" + exposure, 1)
            scene_js_path.write_text(text, encoding="utf-8")
            return
    scene_js_path.write_text(text, encoding="utf-8")


def patch_index_html_for_query(output_path: Path, args: argparse.Namespace) -> None:
    index_path = output_path / "index.html"
    if not index_path.is_file():
        return
    text = index_path.read_text(encoding="utf-8")
    if "scene-query-panel" in text:
        return

    config = {
        "serveQuery": bool(args.serve_query),
        "topK": int(args.query_top_k),
        "clipModel": args.clip_model,
    }
    injection = f"""
<style>
  #scene-query-panel {{
    position: fixed;
    left: 16px;
    bottom: 18px;
    width: min(360px, calc(100vw - 32px));
    z-index: 10000;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    background: rgba(18, 22, 30, 0.88);
    box-shadow: 0 14px 40px rgba(0,0,0,0.28);
    color: #f7f7f8;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    backdrop-filter: blur(10px);
  }}
  #scene-query-panel label {{
    display: block;
    margin-bottom: 7px;
    font-weight: 650;
    letter-spacing: 0;
  }}
  #scene-query-form {{
    display: grid;
    grid-template-columns: 1fr;
  }}
  #scene-query-input {{
    min-width: 0;
    height: 34px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.26);
    background: rgba(255,255,255,0.09);
    color: #fff;
    padding: 0 10px;
    outline: none;
  }}
  #scene-query-input::placeholder {{
    color: rgba(255,255,255,0.62);
  }}
  #scene-query-status {{
    min-height: 18px;
    margin-top: 8px;
    color: rgba(255,255,255,0.76);
    line-height: 1.35;
  }}
  #scene-query-panel.is-docked {{
    position: static;
    width: auto;
    max-width: 360px;
    margin: 10px 12px;
  }}
</style>
<div id="scene-query-panel" aria-label="Scene object semantic query">
  <label for="scene-query-input">Scene query</label>
  <form id="scene-query-form">
    <input id="scene-query-input" type="text" autocomplete="off" placeholder="something to sit on" />
  </form>
  <div id="scene-query-status"></div>
</div>
<script>
window.sceneQueryConfig = {json.dumps(config)};
(function() {{
  const config = window.sceneQueryConfig || {{}};
  const form = document.getElementById('scene-query-form');
  const input = document.getElementById('scene-query-input');
  const panel = document.getElementById('scene-query-panel');
  const status = document.getElementById('scene-query-status');
  let lastMatchedIds = [];

  function dockPanelUnderOptions() {{
    const selectors = [
      '#control_panel',
      '#controls',
      '#options',
      '#tree',
      '#tree_container',
      '#object-tree',
      '.jstree',
      '.sidebar',
      '.dg.ac'
    ];
    for (const selector of selectors) {{
      const target = document.querySelector(selector);
      if (!target || target === panel || target.contains(panel) || !target.parentElement) continue;
      target.insertAdjacentElement('afterend', panel);
      panel.classList.add('is-docked');
      return;
    }}
  }}

  function setStatus(text) {{
    status.textContent = text || '';
  }}

  function getDemoApi() {{
    return window.pyviz3dDemo || null;
  }}

  function idFromHighlightName(name) {{
    const match = String(name || '').match(/Scene query;Highlight;object_(\\d+);/);
    return match ? match[1] : null;
  }}

  function setThreeObjectVisible(root, visible) {{
    if (!root) return;
    root.visible = visible;
    if (typeof root.traverse === 'function') {{
      root.traverse((child) => {{
        child.visible = visible;
        if (child.material) {{
          const materials = Array.isArray(child.material) ? child.material : [child.material];
          materials.forEach((mat) => {{
            if (!mat) return;
            mat.transparent = false;
            mat.opacity = 1.0;
            if (mat.uniforms && mat.uniforms.alpha) mat.uniforms.alpha.value = 1.0;
            mat.depthTest = true;
            mat.depthWrite = true;
            mat.needsUpdate = true;
          }});
          child.renderOrder = 0;
        }}
      }});
    }}
  }}

  function visitSceneHighlights(scene, activeIds, seen) {{
    if (!scene || typeof scene.traverse !== 'function') return seen;
    scene.traverse((node) => {{
      const id = idFromHighlightName(node.name);
      if (!id) return;
      setThreeObjectVisible(node, activeIds.has(id));
      seen.count += 1;
    }});
    return seen;
  }}

  function applyHighlight(ids) {{
    lastMatchedIds = (ids || []).map((id) => String(id));
    const activeIds = new Set(lastMatchedIds);
    const api = getDemoApi();
    const seen = {{count: 0}};

    if (api && api.objects) {{
      Object.entries(api.objects).forEach(([name, object3d]) => {{
        const id = idFromHighlightName(name);
        if (!id) return;
        setThreeObjectVisible(object3d, activeIds.has(id));
        seen.count += 1;
      }});
    }}
    if (api && api.scene) {{
      visitSceneHighlights(api.scene, activeIds, seen);
    }}
    if (api && typeof api.render === 'function') {{
      api.render();
    }}
    document.body.dataset.sceneQueryMatchedIds = lastMatchedIds.join(',');
    return seen.count;
  }}

  async function submitQuery(event) {{
    event.preventDefault();
    const query = input.value.trim();
    if (!query) {{
      applyHighlight([]);
      setStatus('');
      return;
    }}
    if (window.location.protocol === 'file:') {{
      setStatus('Open this through --serve-query so the eval matcher can encode the query.');
      return;
    }}
    setStatus('Encoding query with CLIP...');
    try {{
      const response = await fetch('/scene-query?text=' + encodeURIComponent(query) + '&top_k=' + encodeURIComponent(config.topK || 5));
      if (!response.ok) {{
        throw new Error(await response.text());
      }}
      const result = await response.json();
      const ids = result.matched_ids || (result.best ? [result.best.id] : []);
      const highlightCount = applyHighlight(ids);
      if (result.best) {{
        const suffix = ids.length > 1 ? ' (' + ids.length + ' tied instances)' : '';
        const highlightSuffix = highlightCount ? '' : ' Highlight layer not found yet; try again after the scene finishes loading.';
        setStatus('Matched ' + result.best.label + ' #' + result.best.id + ' · similarity ' + result.best.similarity.toFixed(3) + suffix + '.' + highlightSuffix);
      }} else {{
        setStatus('No object match.');
      }}
    }} catch (error) {{
      setStatus('Query backend unavailable: ' + error.message);
    }}
  }}

  form.addEventListener('submit', submitQuery);
  dockPanelUnderOptions();
  window.addEventListener('load', dockPanelUnderOptions);
  window.addEventListener('pyviz3d-ready', () => {{
    dockPanelUnderOptions();
    if (lastMatchedIds.length) applyHighlight(lastMatchedIds);
  }});
  if (!config.serveQuery) {{
    setStatus('Run this script with --serve-query to enable live CLIP matching.');
  }}
}})();
</script>
"""
    if "</body>" in text:
        text = text.replace("</body>", injection + "\n</body>", 1)
    else:
        text += injection
    index_path.write_text(text, encoding="utf-8")


def paper_demo_css() -> str:
    return r'''
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #0d1418; }
.paper-demo-shell { position: relative; width: 100vw; height: 100vh; display: grid; grid-template-columns: minmax(0, 1fr) 475px; gap: 18px; padding: 14px 14px 14px 18px; box-sizing: border-box; color: #f8fbfc; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.viewer-card { position: relative; min-width: 0; min-height: 0; overflow: hidden; border: 1px solid rgba(255,255,255,0.25); border-radius: 24px; background: #ffffff; }
#render_container { width: 100%; height: 100%; position: relative; }
.dg.ac, .lil-gui, .lil-gui.root { display: none !important; }
.label { position: relative; z-index: 20; pointer-events: none; }
.paper-label-anchor { position: relative; z-index: 20; width: 0; height: 0; overflow: visible; pointer-events: none; }
.paper-text-label { --paper-label-rotation: 0deg; position: absolute; left: 0; top: 0; display: inline-block; padding: 3px 6px; border-radius: 7px; color: #f6fbff; background: rgba(13, 18, 21, 0.68); border: 1px solid rgba(255,255,255,0.20); font-size: 12px; line-height: 1.15; text-shadow: 0 1px 2px rgba(0,0,0,0.75); white-space: nowrap; transform: translate(-50%, -100%) rotate(var(--paper-label-rotation)); transform-origin: center center; pointer-events: none; cursor: default; }
.paper-text-label.is-object { color: #edf7ff; font-weight: 700; }
.paper-text-label.is-relationship { color: #fff1c5; background: rgba(24, 18, 12, 0.70); }
.paper-demo-shell.is-top-mode .paper-text-label.is-relationship { transform: translate(-50%, -50%) rotate(var(--paper-label-rotation)); }
.load-pill { position: absolute; left: 18px; top: 16px; padding: 8px 12px; border-radius: 999px; background: rgba(10,16,20,0.68); color: #fff; font-size: 12px; pointer-events: none; }
.load-pill.is-hidden { display: none; }
.control-panel { min-width: 0; overflow-x: hidden; overflow-y: auto; border: 1px solid rgba(255,255,255,0.14); border-radius: 24px; padding: 22px; background: #111a1f; box-shadow: inset 0 1px 0 rgba(255,255,255,0.03); }
.control-panel::-webkit-scrollbar { width: 8px; }
.control-panel::-webkit-scrollbar-thumb { border-radius: 999px; background: rgba(165,198,206,0.28); }
.transport-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 12px; align-items: center; margin-bottom: 16px; }
button { font: inherit; }
.play-button, .icon-button, .segmented button { border: 1px solid rgba(255,255,255,0.10); color: #fff; background: #1b252a; cursor: pointer; }
.play-button { width: 52px; height: 52px; border-radius: 50%; color: #101810; background: #c6ff5d; font-weight: 800; }
.icon-button { width: 50px; height: 50px; border-radius: 50%; background: #172126; }
.icon-button.is-wide { width: 64px; border-radius: 999px; }
.icon-button.is-active { border-color: rgba(28, 220, 192, 0.9); background: #123b3b; }
.timeline, .range-control input { width: 100%; accent-color: #c6ff5d; }
.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }
.stat-card { padding: 14px 12px; border-radius: 16px; background: #1a2429; border: 1px solid rgba(255,255,255,0.08); }
.stat-card strong { display: block; font-size: 21px; line-height: 1; margin-bottom: 12px; }
.stat-card span { color: #a5c6ce; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.segmented { display: flex; gap: 10px; padding: 6px; margin-bottom: 12px; border-radius: 999px; background: #162025; border: 1px solid rgba(255,255,255,0.06); }
.segmented button { flex: 1; min-height: 44px; border-radius: 999px; padding: 0 12px; }
.segmented button.is-active { border-color: rgba(28, 220, 192, 0.9); background: #123b3b; }
.range-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
.range-control { display: grid; gap: 8px; padding: 14px; border-radius: 12px; color: #fff; background: #1b252a; border: 1px solid rgba(255,255,255,0.10); }
.scene-query-panel { margin: 10px 0 14px; padding: 12px; border-radius: 14px; background: #182329; border: 1px solid rgba(28,220,192,0.35); }
.scene-query-panel label { display: block; margin-bottom: 8px; color: #a5c6ce; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
#scene-query-form, #relationship-query-form { display: grid; grid-template-columns: 1fr; }
.relationship-query-label { margin-top: 10px !important; }
.triplet-query-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 7px; }
#scene-query-input, #relationship-query-input { min-width: 0; height: 38px; border: 1px solid rgba(255,255,255,0.14); border-radius: 9px; padding: 0 10px; color: #fff; background: #0f171b; }
#relationship-subject-input, #relationship-predicate-input { min-width: 0; height: 38px; border: 1px solid rgba(255,255,255,0.14); border-radius: 9px; padding: 0 10px; color: #fff; background: #0f171b; }
.triplet-submit-button { width: 54px; height: 38px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,0.14); border-radius: 9px; color: #071014; background: #c6ff5d; font: inherit; font-size: 12px; font-weight: 820; cursor: pointer; }
.triplet-submit-icon { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2.7; stroke-linecap: round; stroke-linejoin: round; pointer-events: none; }
.triplet-highlight-row { display: flex; align-items: center; gap: 9px; margin-top: 8px; color: #c9e8ee; font-size: 12px; cursor: default; user-select: text; touch-action: auto; }
.triplet-highlight-row span { color: #a5c6ce; }
.triplet-highlight-row label { display: inline-flex !important; align-items: center; gap: 5px; margin: 0 !important; color: #f7fbff; font-size: 12px; font-weight: 720; text-transform: none; letter-spacing: 0; cursor: pointer; user-select: none; }
.triplet-highlight-row input { accent-color: #c6ff5d; cursor: pointer; }
.query-drag-handle, .query-resize-handle { display: none; }
#scene-query-status { min-height: 18px; margin-top: 8px; color: #a5c6ce; font-size: 12px; }
.relationship-strip { display: flex; gap: 10px; overflow-x: auto; padding: 0 0 10px; }
.relationship-chip { flex: 0 0 185px; min-height: 78px; border-radius: 14px; border: 1px solid #c79622; background: #342d1a; color: #fff; text-align: left; padding: 12px; cursor: pointer; }
.relationship-chip strong { display: block; color: #69a7ff; margin-bottom: 7px; }
.relationship-chip.is-active { border-color: #20d9c2; background: #25352f; }
.relationship-chip.is-muted { opacity: 0.38; }
@media (max-width: 1050px) { .paper-demo-shell { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; } .control-panel { max-height: 48vh; overflow-y: auto; } }
'''


def paper_query_css() -> str:
    return r'''
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #ffffff; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.paper-query-shell { position: relative; width: 100vw; height: 100vh; overflow: hidden; background: #ffffff; }
#render_container { position: absolute; inset: 0; width: 100%; height: 100%; }
.dg.ac, .lil-gui, .lil-gui.root, .jstree, .sidebar { display: none !important; }
.label { pointer-events: none; }
.paper-export-button { position: absolute; top: 22px; right: 22px; z-index: 30; width: 62px; height: 42px; border: 1px solid rgba(255,255,255,0.22); border-radius: 999px; color: #071014; background: #c6ff5d; font: inherit; font-weight: 820; box-shadow: 0 14px 34px rgba(0,0,0,0.26); cursor: pointer; }
.scene-query-panel { position: absolute; left: 22px; top: 22px; width: min(430px, calc(100vw - 44px)); z-index: 20; padding: 13px; box-sizing: border-box; color: #f7fbff; background: rgba(13, 19, 22, 0.74); border: 1px solid rgba(255,255,255,0.22); border-radius: 10px; box-shadow: 0 18px 45px rgba(0,0,0,0.30); backdrop-filter: blur(10px); cursor: grab; user-select: none; touch-action: none; }
.scene-query-panel.is-dragging { cursor: grabbing; }
.scene-query-panel label { display: block; margin: 0 0 8px; color: #c9e8ee; font-size: 12px; font-weight: 760; text-transform: uppercase; letter-spacing: .06em; }
#scene-query-form, #relationship-query-form { display: grid; grid-template-columns: minmax(0, 1fr); }
.relationship-query-label { margin-top: 10px !important; }
.triplet-query-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 7px; }
#scene-query-input, #relationship-query-input { min-width: 0; height: 40px; border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; padding: 0 11px; color: #fff; background: rgba(5, 10, 12, 0.72); outline: none; cursor: text; user-select: text; touch-action: auto; }
#relationship-subject-input, #relationship-predicate-input { min-width: 0; height: 40px; border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; padding: 0 11px; color: #fff; background: rgba(5, 10, 12, 0.72); outline: none; cursor: text; user-select: text; touch-action: auto; }
.triplet-submit-button { width: 54px; height: 40px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; color: #071014; background: #c6ff5d; font: inherit; font-size: 12px; font-weight: 820; cursor: pointer; }
.triplet-submit-icon { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2.7; stroke-linecap: round; stroke-linejoin: round; pointer-events: none; }
.triplet-highlight-row { display: flex; align-items: center; gap: 9px; margin-top: 8px; color: #c9e8ee; font-size: 12px; cursor: default; user-select: text; touch-action: auto; }
.triplet-highlight-row span { color: #a5c6ce; }
.triplet-highlight-row label { display: inline-flex !important; align-items: center; gap: 5px; margin: 0 !important; color: #f7fbff; font-size: 12px; font-weight: 720; text-transform: none; letter-spacing: 0; cursor: pointer; user-select: none; }
.triplet-highlight-row input { accent-color: #c6ff5d; cursor: pointer; }
.paper-mode-row, #scene-query-status { cursor: default; user-select: text; touch-action: auto; }
#scene-query-input::placeholder { color: rgba(255,255,255,0.54); }
#relationship-query-input::placeholder { color: rgba(255,255,255,0.54); }
#relationship-subject-input::placeholder, #relationship-predicate-input::placeholder { color: rgba(255,255,255,0.54); }
.paper-mode-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 9px; }
.paper-mode-row button { min-width: 0; height: 32px; border: 1px solid rgba(255,255,255,0.16); border-radius: 7px; color: #dceff2; background: rgba(255,255,255,0.07); font: inherit; font-size: 12px; font-weight: 720; cursor: pointer; }
.paper-mode-row button.is-active { color: #071014; border-color: transparent; background: #c6ff5d; }
#scene-query-status { min-height: 18px; margin-top: 8px; color: #c9e8ee; font-size: 12px; line-height: 1.35; }
@media (max-width: 620px) { .scene-query-panel { left: 12px; right: 86px; top: 12px; width: auto; } .paper-export-button { top: 12px; right: 12px; } }
'''


def paper_query_js(config: dict[str, Any]) -> str:
    return r'''
const config = __CONFIG__;
const state = { matchedIds: [], matchedRelIds: [], mode: "scene" };
let summary = null;
function byId(id) { return document.getElementById(id); }
function api() { return window.pyviz3dDemo || null; }
function render() { const demo = api(); if (demo && typeof demo.render === "function") demo.render(); }
function clamp(value, minValue, maxValue) { return Math.max(minValue, Math.min(maxValue, value)); }
function installDraggableQueryPanel() {
  const panel = byId("scene-query-panel");
  if (!panel || panel.dataset.dragReady === "1") return;
  panel.dataset.dragReady = "1";
  const interactiveSelector = "input, button, textarea, select, a, label";
  function startResize(event) {
    const position = window.getComputedStyle(panel).position;
    if (position !== "absolute" && position !== "fixed") return;
    const rect = panel.getBoundingClientRect();
    const startWidth = Math.max(180, panel.offsetWidth || rect.width);
    const left = rect.left;
    const top = rect.top;
    panel.classList.add("is-resizing");
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.width = `${startWidth}px`;
    if (panel.setPointerCapture) panel.setPointerCapture(event.pointerId);
    event.preventDefault();

    function move(moveEvent) {
      const maxWidth = Math.max(180, window.innerWidth - left - 12);
      panel.style.width = `${clamp(moveEvent.clientX - left, 180, maxWidth)}px`;
    }
    function stop() {
      panel.classList.remove("is-resizing");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, {once: true});
    window.addEventListener("pointercancel", stop, {once: true});
  }
  panel.addEventListener("pointerdown", (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    if (event.target && event.target.closest && event.target.closest(".query-resize-handle")) {
      startResize(event);
      return;
    }
    if (event.target && event.target.closest && event.target.closest(interactiveSelector)) return;
    const position = window.getComputedStyle(panel).position;
    if (position !== "absolute" && position !== "fixed") return;
    const rect = panel.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;
    panel.classList.add("is-dragging");
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.width = `${rect.width}px`;
    if (panel.setPointerCapture) panel.setPointerCapture(event.pointerId);
    event.preventDefault();

    function move(moveEvent) {
      const maxLeft = Math.max(0, window.innerWidth - rect.width);
      const maxTop = Math.max(0, window.innerHeight - rect.height);
      const nextLeft = Math.min(Math.max(0, moveEvent.clientX - offsetX), maxLeft);
      const nextTop = Math.min(Math.max(0, moveEvent.clientY - offsetY), maxTop);
      panel.style.left = `${nextLeft}px`;
      panel.style.top = `${nextTop}px`;
    }
    function stop() {
      panel.classList.remove("is-dragging");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, {once: true});
    window.addEventListener("pointercancel", stop, {once: true});
  });
}
function eachObject(callback) { const demo = api(); if (!demo || !demo.objects) return; Object.entries(demo.objects).forEach(([name, object]) => callback(name, object)); }
function setVisible(root, visible) { if (!root) return; root.visible = visible; if (root.element) root.element.style.display = visible ? "" : "none"; if (typeof root.traverse === "function") root.traverse((child) => { child.visible = visible; if (child.element) child.element.style.display = visible ? "" : "none"; }); if (typeof root.updateMatrixWorld === "function") root.updateMatrixWorld(true); }
function objectIdFromHighlightName(name) { const match = String(name).match(/Scene query;Highlight;object_(\d+);/); return match ? match[1] : null; }
function relIndexFromName(name) { const match = String(name).match(/Relationship edges;(?:center|surface|raised|topdown);(\d+)_/); return match ? match[1] : null; }
function relStyleFromName(name) { const match = String(name).match(/Relationship edges;(center|surface|raised|topdown);/); return match ? match[1] : null; }
function objectLabelFromBoxName(name) { const parts = String(name).split(";"); return (parts[parts.length - 1] || "").toLowerCase(); }
function isCeilingLabel(label) { const value = String(label || "").toLowerCase(); return value === "ceiling" || value.includes("ceiling"); }
function isPaperQueryLayer(name, layer) {
  const value = String(name);
  if (layer === "scene") return value.startsWith("Scene;");
  if (layer === "boxes") return value.startsWith("Scene graph;Object boxes");
  if (layer === "labels") return value.startsWith("Scene graph;Object labels") || value.startsWith("Scene graph;Relationship labels");
  if (layer === "centers") return value.startsWith("Scene graph;Object centers") || value.startsWith("Scene graph;Object center spheres");
  if (layer === "edges") return value.startsWith("Scene graph;Relationship edges");
  if (layer === "query") return value.startsWith("Scene query;Highlight");
  return false;
}
function edgeStyleForMode() {
  return state.mode === "top" ? "topdown" : "raised";
}
function modeWantsBoxes() {
  return state.mode === "boxes" || state.mode === "graph" || state.mode === "top";
}
function modeWantsEdges() {
  return state.mode === "graph" || state.mode === "top";
}
function modeWantsCenters() {
  return state.mode === "top";
}
function styleQueryHighlight(root, visible) {
  if (!root || typeof root.traverse !== "function") return;
  root.traverse((child) => {
    child.renderOrder = 0;
    if (!child.material) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((mat) => {
      if (!mat) return;
      mat.transparent = false;
      mat.opacity = 1.0;
      if (mat.uniforms && mat.uniforms.alpha) mat.uniforms.alpha.value = 1.0;
      mat.depthTest = true;
      mat.depthWrite = true;
      mat.needsUpdate = true;
    });
  });
}
function applyPaperQueryView() {
  const edgeStyle = edgeStyleForMode();
  eachObject((name, object) => {
    let visible = true;
    if (isPaperQueryLayer(name, "scene")) visible = true;
    if (isPaperQueryLayer(name, "boxes")) {
      visible = modeWantsBoxes();
      if (state.mode === "top" && isCeilingLabel(objectLabelFromBoxName(name))) visible = false;
    }
    if (isPaperQueryLayer(name, "labels")) visible = false;
    if (isPaperQueryLayer(name, "centers")) visible = modeWantsCenters();
    if (isPaperQueryLayer(name, "edges")) {
      const relIndex = relIndexFromName(name);
      visible = (modeWantsEdges() || state.matchedRelIds.includes(String(relIndex))) && relStyleFromName(name) === edgeStyle;
    }
    if (isPaperQueryLayer(name, "query")) {
      const id = objectIdFromHighlightName(name);
      visible = state.matchedIds.includes(String(id));
      styleQueryHighlight(object, visible);
    }
    setVisible(object, visible);
  });
  render();
}
function syncModeButtons() { document.querySelectorAll("[data-paper-mode]").forEach((button) => button.classList.toggle("is-active", button.dataset.paperMode === state.mode)); }
function objectBounds() {
  const objects = (summary && summary.objects) ? summary.objects.filter((o) => !isCeilingLabel(o.label)) : [];
  if (!objects.length) return {min: [-1, -1, 0], max: [1, 1, 2], center: [0, 0, 1], span: [2, 2, 2]};
  const mins = objects.map((o) => [o.position[0] - o.size[0] * 0.5, o.position[1] - o.size[1] * 0.5, o.position[2] - o.size[2] * 0.5]);
  const maxs = objects.map((o) => [o.position[0] + o.size[0] * 0.5, o.position[1] + o.size[1] * 0.5, o.position[2] + o.size[2] * 0.5]);
  const min = [0, 1, 2].map((axis) => Math.min(...mins.map((p) => p[axis])));
  const max = [0, 1, 2].map((axis) => Math.max(...maxs.map((p) => p[axis])));
  const center = [0, 1, 2].map((axis) => (min[axis] + max[axis]) * 0.5);
  const span = [0, 1, 2].map((axis) => Math.max(0.1, max[axis] - min[axis]));
  return {min, max, center, span};
}
function setTopCamera() {
  const demo = api();
  if (!demo) return;
  const b = objectBounds();
  const span = Math.max(b.span[0], b.span[1], 1);
  demo.camera.position.set(b.center[0], b.center[1], b.max[2] + span * 1.12);
  demo.camera.up.set(0, 1, 0);
  demo.camera.lookAt(b.center[0], b.center[1], b.center[2]);
  if (demo.controls) {
    demo.controls.target.set(b.center[0], b.center[1], b.center[2]);
    demo.controls.update();
  }
}
function setPaperMode(mode) {
  state.mode = mode;
  syncModeButtons();
  if (mode === "top") setTopCamera();
  applyPaperQueryView();
}
function setStatus(text) { byId("scene-query-status").textContent = text || ""; }
function exportFileName() { const scene = String((summary && summary.scene_id) || "scene_graph").replace(/[^A-Za-z0-9_.-]+/g, "_"); return `${scene}_${state.mode}_paper_view.svg`; }
function drawRoundRect(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius || 0, width * 0.5, height * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
function isDomElementVisible(element) {
  if (!element) return false;
  const style = window.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity || 1) > 0.01 && rect.width > 0 && rect.height > 0;
}
function drawStyledDomBox(ctx, element, containerRect) {
  const style = window.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  const x = rect.left - containerRect.left;
  const y = rect.top - containerRect.top;
  const width = rect.width;
  const height = rect.height;
  const background = (style.backgroundColor === "rgba(0, 0, 0, 0)" || style.backgroundColor === "transparent") ? "none" : style.backgroundColor;
  const strokeWidth = parseFloat(style.borderTopWidth || "0") || 0;
  const stroke = (strokeWidth <= 0 || style.borderTopStyle === "none") ? "none" : style.borderTopColor;
  const radius = parseFloat(style.borderTopLeftRadius || "0") || 0;
  if (background !== "none") { ctx.fillStyle = background; drawRoundRect(ctx, x, y, width, height, radius); ctx.fill(); }
  if (stroke !== "none" && strokeWidth > 0) { ctx.strokeStyle = stroke; ctx.lineWidth = strokeWidth; drawRoundRect(ctx, x, y, width, height, radius); ctx.stroke(); }
  return {x, y, width, height, style};
}
function textForDomElement(element) {
  if (!element) return "";
  const tagName = String(element.tagName || "").toLowerCase();
  if (tagName === "input") return element.value || element.getAttribute("placeholder") || "";
  return element.textContent || "";
}
function drawDomElementText(ctx, element, box) {
  if (element && element.classList && element.classList.contains("triplet-submit-button")) {
    ctx.strokeStyle = window.getComputedStyle(element).color || "#5f96df";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const cx = box.x + box.width * 0.47;
    const cy = box.y + box.height * 0.45;
    const radius = Math.min(box.width, box.height) * 0.16;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx + radius * 0.72, cy + radius * 0.72);
    ctx.lineTo(cx + radius * 1.75, cy + radius * 1.75);
    ctx.stroke();
    return;
  }
  const text = textForDomElement(element).trim();
  if (!text) return;
  const style = box.style || window.getComputedStyle(element);
  const fontSize = parseFloat(style.fontSize || "12") || 12;
  const paddingLeft = parseFloat(style.paddingLeft || "0") || 0;
  const paddingRight = parseFloat(style.paddingRight || "0") || 0;
  let drawText = text;
  if (style.textTransform === "uppercase") drawText = drawText.toUpperCase();
  ctx.fillStyle = style.color || "#ffffff";
  ctx.font = `${style.fontWeight || "400"} ${fontSize}px ${style.fontFamily || "sans-serif"}`;
  ctx.textBaseline = "middle";
  ctx.fillText(drawText, box.x + paddingLeft + 1, box.y + box.height * 0.5, Math.max(1, box.width - paddingLeft - paddingRight - 2));
}
function drawVisibleDomOverlaysToCanvas(ctx, container) {
  const containerRect = container.getBoundingClientRect();
  const panel = byId("scene-query-panel");
  if (isDomElementVisible(panel)) {
    drawStyledDomBox(ctx, panel, containerRect);
    panel.querySelectorAll(".triplet-query-grid, .paper-highlight-toggle, label, input, button, #scene-query-status").forEach((element) => {
      if (!isDomElementVisible(element)) return;
      const box = drawStyledDomBox(ctx, element, containerRect);
      drawDomElementText(ctx, element, box);
    });
  }
  const button = byId("paper-query-export-pdf");
  if (isDomElementVisible(button)) {
    const box = drawStyledDomBox(ctx, button, containerRect);
    drawDomElementText(ctx, button, box);
  }
}
function byteStringToUint8Array(value) {
  const bytes = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i += 1) bytes[i] = value.charCodeAt(i) & 255;
  return bytes;
}
function dataUrlToUint8Array(dataUrl) {
  const base64 = String(dataUrl).split(",", 2)[1] || "";
  const binary = window.atob(base64);
  return byteStringToUint8Array(binary);
}
function concatUint8Arrays(parts) {
  const total = parts.reduce((count, part) => count + part.length, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  parts.forEach((part) => { output.set(part, offset); offset += part.length; });
  return output;
}
function escapeXmlAttribute(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[char]));
}
function buildSvgBlobFromPng(pngDataUrl, width, height) {
  const svg = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<image href="${escapeXmlAttribute(pngDataUrl)}" width="${width}" height="${height}" />`,
    "</svg>",
  ].join("\n");
  return new Blob([svg], {type: "image/svg+xml;charset=utf-8"});
}
function renderTransparentSceneForExport(demo) {
  const previousBackground = demo.scene ? demo.scene.background : null;
  const previousClearAlpha = demo.renderer.getClearAlpha ? demo.renderer.getClearAlpha() : 1;
  const previousClearColor = demo.THREE && demo.renderer.getClearColor ? new demo.THREE.Color() : null;
  if (previousClearColor) demo.renderer.getClearColor(previousClearColor);
  try {
    if (demo.scene) demo.scene.background = null;
    if (demo.renderer.setClearColor) demo.renderer.setClearColor(0xffffff, 0);
    else if (demo.renderer.setClearAlpha) demo.renderer.setClearAlpha(0);
    render();
  } finally {
    if (demo.scene) demo.scene.background = previousBackground;
    if (previousClearColor && demo.renderer.setClearColor) demo.renderer.setClearColor(previousClearColor, previousClearAlpha);
    else if (demo.renderer.setClearAlpha) demo.renderer.setClearAlpha(previousClearAlpha);
  }
}
function exportCurrentViewSvg() {
  const demo = api();
  const container = byId("render_container");
  if (!demo || !demo.renderer || !container) return;
  const width = Math.max(1, Math.round(container.clientWidth));
  const height = Math.max(1, Math.round(container.clientHeight));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  try {
    renderTransparentSceneForExport(demo);
    ctx.drawImage(demo.renderer.domElement, 0, 0, width, height);
    drawVisibleDomOverlaysToCanvas(ctx, container);
  } catch (error) {
    window.alert("Could not export SVG snapshot: " + error.message);
    return;
  }
  const pngDataUrl = canvas.toDataURL("image/png");
  const blob = buildSvgBlobFromPng(pngDataUrl, width, height);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = exportFileName();
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
function applyQueryHighlight(ids) { state.matchedIds = (ids || []).map((id) => String(id)); state.matchedRelIds = []; applyPaperQueryView(); }
function applyRelationshipHighlight(ids, relationshipIds) {
  state.matchedIds = (ids || []).map((id) => String(id));
  state.matchedRelIds = (relationshipIds || []).map((id) => String(id));
  applyPaperQueryView();
}
async function submitQuery(event) {
  event.preventDefault();
  const input = byId("scene-query-input");
  const query = input.value.trim();
  if (!query) {
    applyQueryHighlight([]);
    setStatus("");
    return;
  }
  if (window.location.protocol === "file:") {
    setStatus("Serve this viewer with scripts/serve_scene_graph_query.py to enable matching.");
    return;
  }
  setStatus("Encoding with CLIP...");
  try {
    const response = await fetch(`/scene-query?text=${encodeURIComponent(query)}&top_k=${encodeURIComponent(config.topK || 5)}`);
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    const ids = result.matched_ids || (result.best ? [result.best.id] : []);
    applyQueryHighlight(ids);
    if (result.best) {
      const suffix = ids.length > 1 ? ` (${ids.length} tied instances)` : "";
      setStatus(`Matched ${result.best.label} #${result.best.id} (${result.best.similarity.toFixed(3)})${suffix}`);
    } else {
      setStatus("No match.");
    }
  } catch (error) {
    setStatus("Query backend unavailable: " + error.message);
  }
}
async function submitRelationshipQuery(event) {
  event.preventDefault();
  const query = byId("relationship-query-input").value.trim();
  const roleInput = document.querySelector('input[name="relationship-highlight-role"]:checked');
  const highlightRole = roleInput ? roleInput.value : "object";
  if (!query) {
    applyRelationshipHighlight([], []);
    setStatus("");
    return;
  }
  if (window.location.protocol === "file:") {
    setStatus("Serve this viewer with scripts/serve_scene_graph_query.py to enable matching.");
    return;
  }
  setStatus("Encoding triplet query...");
  try {
    const response = await fetch(`/scene-triplet-query?text=${encodeURIComponent(query)}&highlight=${encodeURIComponent(highlightRole)}&top_k=${encodeURIComponent(config.topK || 5)}`);
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    const ids = result.highlighted_ids || result.matched_ids || [];
    const relIds = result.matched_relationship_indices || [];
    applyRelationshipHighlight(ids, relIds);
    if (result.best) {
      const role = result.highlight_role || highlightRole;
      const highlightedId = role === "subject" ? result.best.subject_id : result.best.object_id;
      setStatus(`Matched ${result.best.subject_label} ${result.best.predicate} ${result.best.object_label}; highlighted ${role} #${highlightedId} (${result.best.similarity.toFixed(3)})`);
    } else {
      setStatus("No triplet match.");
    }
  } catch (error) {
    setStatus("Triplet backend unavailable: " + error.message);
  }
}
function installTripletEnterShortcut() {
  ["relationship-query-input"].forEach((id) => {
    const input = byId(id);
    if (!input || input.dataset.enterReady === "1") return;
    input.dataset.enterReady = "1";
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const form = byId("relationship-query-form");
      if (form && typeof form.requestSubmit === "function") form.requestSubmit();
      else submitRelationshipQuery(event);
    });
  });
}
async function initPaperQuery() {
  try {
    const response = await fetch("scene_query_objects.json");
    summary = await response.json();
  } catch (error) {
    summary = null;
  }
  byId("scene-query-form").addEventListener("submit", submitQuery);
  const relationshipForm = byId("relationship-query-form");
  if (relationshipForm) relationshipForm.addEventListener("submit", submitRelationshipQuery);
  installTripletEnterShortcut();
  byId("paper-query-export-pdf").addEventListener("click", exportCurrentViewSvg);
  installDraggableQueryPanel();
  document.querySelectorAll("[data-paper-mode]").forEach((button) => button.addEventListener("click", () => setPaperMode(button.dataset.paperMode)));
  syncModeButtons();
  applyPaperQueryView();
}
let paperQueryStarted = false;
function startPaperQueryOnce() {
  if (paperQueryStarted || !window.pyviz3dDemo) return;
  paperQueryStarted = true;
  initPaperQuery().catch((error) => {
    paperQueryStarted = false;
    setStatus("UI load failed: " + error.message);
    console.error(error);
  });
}
window.addEventListener("pyviz3d-ready", startPaperQueryOnce);
window.addEventListener("load", () => {
  startPaperQueryOnce();
  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    startPaperQueryOnce();
    if (paperQueryStarted || attempts > 100) window.clearInterval(timer);
  }, 100);
});
'''.replace("__CONFIG__", json.dumps(config))


def paper_demo_js(config: dict[str, Any]) -> str:
    return r'''
const config = __CONFIG__;
const state = { mode: "top", edgeStyle: "topdown", layers: {scene: true, boxes: true, labels: true, edges: true, relLabels: false}, relationshipVisible: new Set(), spotlight: false, selectedRelationship: "", matchedIds: [], matchedRelIds: [], tourReveal: false, revealedObjectIds: new Set(), revealFractions: new Map() };
let summary = null;
function byId(id) { return document.getElementById(id); }
function api() { return window.pyviz3dDemo || null; }
function render() { const demo = api(); if (demo && typeof demo.render === "function") demo.render(); }
function clamp(value, minValue, maxValue) { return Math.max(minValue, Math.min(maxValue, value)); }
function installDraggableQueryPanel() {
  const panel = byId("scene-query-panel");
  if (!panel || panel.dataset.dragReady === "1") return;
  panel.dataset.dragReady = "1";
  const interactiveSelector = "input, button, textarea, select, a, label";
  function startResize(event) {
    const position = window.getComputedStyle(panel).position;
    if (position !== "absolute" && position !== "fixed") return;
    const rect = panel.getBoundingClientRect();
    const startWidth = Math.max(180, panel.offsetWidth || rect.width);
    const left = rect.left;
    const top = rect.top;
    panel.classList.add("is-resizing");
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.width = `${startWidth}px`;
    if (panel.setPointerCapture) panel.setPointerCapture(event.pointerId);
    event.preventDefault();

    function move(moveEvent) {
      const maxWidth = Math.max(180, window.innerWidth - left - 12);
      panel.style.width = `${clamp(moveEvent.clientX - left, 180, maxWidth)}px`;
    }
    function stop() {
      panel.classList.remove("is-resizing");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, {once: true});
    window.addEventListener("pointercancel", stop, {once: true});
  }
  panel.addEventListener("pointerdown", (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    if (event.target && event.target.closest && event.target.closest(".query-resize-handle")) {
      startResize(event);
      return;
    }
    if (event.target && event.target.closest && event.target.closest(interactiveSelector)) return;
    const position = window.getComputedStyle(panel).position;
    if (position !== "absolute" && position !== "fixed") return;
    const rect = panel.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;
    panel.classList.add("is-dragging");
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.width = `${rect.width}px`;
    if (panel.setPointerCapture) panel.setPointerCapture(event.pointerId);
    event.preventDefault();

    function move(moveEvent) {
      const maxLeft = Math.max(0, window.innerWidth - rect.width);
      const maxTop = Math.max(0, window.innerHeight - rect.height);
      const nextLeft = Math.min(Math.max(0, moveEvent.clientX - offsetX), maxLeft);
      const nextTop = Math.min(Math.max(0, moveEvent.clientY - offsetY), maxTop);
      panel.style.left = `${nextLeft}px`;
      panel.style.top = `${nextTop}px`;
    }
    function stop() {
      panel.classList.remove("is-dragging");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, {once: true});
    window.addEventListener("pointercancel", stop, {once: true});
  });
}
function resizeViewerToContainer() {
  const demo = api();
  const container = byId("render_container");
  if (!demo || !container) return;
  const width = Math.max(1, Math.round(container.clientWidth));
  const height = Math.max(1, Math.round(container.clientHeight));
  if (demo.camera && "aspect" in demo.camera) {
    demo.camera.aspect = width / height;
    if (typeof demo.camera.updateProjectionMatrix === "function") demo.camera.updateProjectionMatrix();
  }
  if (demo.renderer && typeof demo.renderer.setSize === "function") demo.renderer.setSize(width, height);
  if (demo.labelRenderer && typeof demo.labelRenderer.setSize === "function") demo.labelRenderer.setSize(width, height);
  render();
}
function scheduleViewerResize() {
  if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(resizeViewerToContainer);
  window.setTimeout(resizeViewerToContainer, 80);
}
function eachObject(callback) { const demo = api(); if (!demo || !demo.objects) return; Object.entries(demo.objects).forEach(([name, object]) => callback(name, object)); }
function setLabelElementVisible(object, visible) { if (!object || !object.element) return; object.element.style.display = visible ? "" : "none"; object.element.style.visibility = visible ? "visible" : "hidden"; object.element.style.opacity = visible ? "1" : "0"; }
function setVisible(root, visible) { if (!root) return; root.visible = visible; setLabelElementVisible(root, visible); if (typeof root.traverse === "function") root.traverse((child) => { child.visible = visible; setLabelElementVisible(child, visible); }); if (typeof root.updateMatrixWorld === "function") root.updateMatrixWorld(true); }
function styleQueryHighlight(root, visible) { if (!root || typeof root.traverse !== "function") return; root.renderOrder = 0; root.traverse((child) => { child.renderOrder = 0; if (!child.material) return; const materials = Array.isArray(child.material) ? child.material : [child.material]; materials.forEach((mat) => { if (!mat) return; mat.transparent = false; mat.opacity = 1.0; if (mat.uniforms && mat.uniforms.alpha) mat.uniforms.alpha.value = 1.0; mat.depthTest = true; mat.depthWrite = true; mat.needsUpdate = true; }); }); }
function eachMaterial(root, callback) { if (!root || typeof root.traverse !== "function") return; root.traverse((child) => { if (!child.material) return; const materials = Array.isArray(child.material) ? child.material : [child.material]; materials.forEach((mat) => { if (mat) callback(mat); }); }); }
function setOpacity(root, opacity) { eachMaterial(root, (mat) => { mat.transparent = true; mat.opacity = opacity; if (mat.uniforms && mat.uniforms.alpha) mat.uniforms.alpha.value = opacity; mat.needsUpdate = true; }); }
function pointCloudFactor() { const input = byId("scene-cloudiness"); const cloudiness = input ? Number(input.value || 0) : 0; return Math.max(0.22, 1.0 - Math.max(0, Math.min(100, cloudiness)) * 0.0078); }
function setPointCloudiness(root) {
  const baseFallback = Math.max(1, Number(config.scenePointSize || 9));
  const factor = pointCloudFactor();
  eachMaterial(root, (mat) => {
    mat.userData = mat.userData || {};
    if (mat.userData.basePointSize === undefined) {
      if (mat.uniforms && mat.uniforms.pointSize) mat.userData.basePointSize = Number(mat.uniforms.pointSize.value || baseFallback);
      else if (mat.size !== undefined) mat.userData.basePointSize = Number(mat.size || baseFallback);
      else mat.userData.basePointSize = baseFallback;
    }
    const nextSize = Math.max(1, Number(mat.userData.basePointSize || baseFallback) * factor);
    if (mat.uniforms && mat.uniforms.pointSize) mat.uniforms.pointSize.value = nextSize;
    if (mat.size !== undefined) mat.size = nextSize;
    mat.needsUpdate = true;
  });
}
function applySceneCloudiness() { eachObject((name, object) => { if (isPaperLayer(name, "scene")) setPointCloudiness(object); }); render(); }
function styleGraphOverlay(name, root) { if (!root || !String(name).startsWith("Scene graph;Relationship edges")) return; root.renderOrder = 1100; if (typeof root.traverse !== "function") return; root.traverse((child) => { child.renderOrder = 1100; if (!child.material) return; const materials = Array.isArray(child.material) ? child.material : [child.material]; materials.forEach((mat) => { if (!mat) return; mat.depthTest = false; mat.depthWrite = false; mat.transparent = true; mat.needsUpdate = true; }); }); }
function relIndexFromName(name) { const match = String(name).match(/Relationship edges;(?:center|surface|raised|topdown);(\d+)_/); return match ? match[1] : null; }
function relStyleFromName(name) { const match = String(name).match(/Relationship edges;(center|surface|raised|topdown);/); return match ? match[1] : null; }
function relationshipLabelMatchesStyle(name, style) { const value = String(name); const match = value.match(/Relationship labels;(center|surface|raised|topdown)$/); return match ? match[1] === style : true; }
function objectIdFromHighlightName(name) { const match = String(name).match(/Scene query;Highlight;object_(\d+);/); return match ? match[1] : null; }
function objectIdFromBoxName(name) { const match = String(name).match(/Scene graph;Object boxes;object_(\d+);/); return match ? match[1] : null; }
function objectLabelFromBoxName(name) { const parts = String(name).split(";"); return (parts[parts.length - 1] || "").toLowerCase(); }
function shortSceneLabel(sceneId) { const match = String(sceneId || "").match(/_split(\d+)$/); return match ? `split${match[1]}` : String(sceneId || ""); }
function isCeilingLabel(label) { const value = String(label || "").toLowerCase(); return value === "ceiling" || value.includes("ceiling"); }
function isNativeObjectLabelLayer(name) { return String(name).startsWith("Scene graph;Object labels"); }
function isNativeRelationshipLabelLayer(name) { return String(name).startsWith("Scene graph;Relationship labels"); }
function isPaperObjectLabelLayer(name) { return String(name).startsWith("Scene graph;Paper object labels"); }
function isPaperRelationshipLabelLayer(name) { return String(name).startsWith("Scene graph;Paper relationship labels"); }
const tour = { running: false, frame: null, start: 0, progress: 0, durationMs: Math.max(5000, Number(config.tourDuration || 28) * 1000), pathCache: null, pathKey: "", routeCache: null, routeKey: "", timelineCache: null, timelineKey: "", revealOrder: [] };
function isPaperLayer(name, layer) {
  const value = String(name);
  if (layer === "scene") return value.startsWith("Scene;");
  if (layer === "boxes") return value.startsWith("Scene graph;Object boxes");
  if (layer === "labels") return isNativeObjectLabelLayer(value) || isPaperObjectLabelLayer(value);
  if (layer === "centers") return value.startsWith("Scene graph;Object center spheres") || value.startsWith("Scene graph;Object centers");
  if (layer === "query") return value.startsWith("Scene query;Highlight");
  if (layer === "edges") return value.startsWith("Scene graph;Relationship edges");
  if (layer === "relLabels") return isNativeRelationshipLabelLayer(value) || isPaperRelationshipLabelLayer(value);
  return false;
}
function syncTopModeUi() {
  const shell = document.querySelector(".paper-demo-shell");
  if (shell) shell.classList.toggle("is-top-mode", state.mode === "top");
}
function syncModeButtons() {
  document.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("is-active", button.dataset.mode === state.mode));
  syncTopModeUi();
}
function topLabelRotationDegrees() {
  return 0;
}
function syncPaperLabelRotation() {
  const demo = api();
  const degrees = topLabelRotationDegrees();
  if (!demo || !demo.objects) return;
  ["Scene graph;Paper object labels", "Scene graph;Paper relationship labels;topdown"].forEach((name) => {
    const group = demo.objects[name];
    if (!group || typeof group.traverse !== "function") return;
    group.traverse((child) => {
      if (child && child.element) child.element.style.setProperty("--paper-label-rotation", `${degrees}deg`);
      const box = labelTextElement(child);
      if (box) box.style.setProperty("--paper-label-rotation", `${degrees}deg`);
    });
  });
}
function labelTextElement(label) {
  return label && label.userData && label.userData.labelBox ? label.userData.labelBox : (label ? label.element : null);
}
function topAnchorForObject(obj) {
  const label = String(obj.label || "").toLowerCase();
  if (label === "wall" || label.includes("wall")) return obj.position;
  return [obj.position[0], obj.position[1], obj.position[2] + 0.5 * obj.size[2] + 0.07];
}
function objectLabelPosition(obj) {
  return [obj.position[0], obj.position[1], obj.position[2] + 0.5 * obj.size[2] + 0.16];
}
function moveCenterSpheres(useCentroids) {
  const demo = api();
  const group = demo && demo.objects ? demo.objects["Scene graph;Object center spheres"] : null;
  if (!group || !Array.isArray(group.children)) return;
  group.children.forEach((sphere) => {
    const pos = useCentroids ? sphere.userData.centroid : sphere.userData.topAnchor;
    if (Array.isArray(pos) && pos.length >= 3) sphere.position.set(pos[0], pos[1], pos[2]);
  });
  if (typeof group.updateMatrixWorld === "function") group.updateMatrixWorld(true);
}
function darkenColor(color, factor = 0.72) {
  return color.clone().multiplyScalar(factor);
}
function cssRgb(color, fallback = [240, 246, 255]) {
  const values = (Array.isArray(color) && color.length >= 3) ? color : fallback;
  return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
}
function objectMapById() {
  const objects = (summary && summary.objects) ? summary.objects : [];
  return new Map(objects.map((obj) => [String(obj.id), obj]));
}
function baseObjectLabel(obj) {
  return String((obj && obj.label) || "object").trim() || "object";
}
function objectLabelPrefix(obj) {
  const display = String((obj && obj.display_label) || "");
  const base = baseObjectLabel(obj);
  if (display.toLowerCase().endsWith(base.toLowerCase())) return display.slice(0, display.length - base.length);
  const idx = display.lastIndexOf(": ");
  return idx >= 0 ? display.slice(0, idx + 2) : `${obj.id}: `;
}
function objectDisplayText(obj) {
  return `${objectLabelPrefix(obj)}${baseObjectLabel(obj)}`;
}
function baseRelationshipText(rel) {
  return String((rel && rel.predicate) || "related").trim() || "related";
}
function isFloorOrCeiling(obj) {
  const label = String((obj && obj.label) || "").toLowerCase();
  return label === "floor" || label.includes("floor") || isCeilingLabel(label);
}
function relationshipLabelPosition(rel, style, objectsById) {
  const subject = objectsById.get(String(rel.subject_id));
  const target = objectsById.get(String(rel.object_id));
  if (!subject || !target) return null;
  const start = (style === "raised" || style === "surface") ? topAnchorForObject(subject) : subject.position;
  const end = (style === "raised" || style === "surface") ? topAnchorForObject(target) : target.position;
  const pos = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5, (start[2] + end[2]) * 0.5];
  if (style === "raised") pos[2] += 0.14;
  if (style === "surface") pos[2] += 0.07;
  return pos;
}
function createPaperLabel(text, className, position, color) {
  const demo = api();
  if (!demo || !demo.CSS2DObject) return null;
  const anchor = document.createElement("div");
  anchor.className = "paper-label-anchor";
  const box = document.createElement("div");
  box.className = className;
  box.textContent = text;
  if (color) box.style.color = color;
  anchor.appendChild(box);
  const label = new demo.CSS2DObject(anchor);
  label.userData.labelBox = box;
  label.position.set(position[0], position[1], position[2]);
  return label;
}
function ensurePaperTextLabels() {
  const demo = api();
  if (!demo || !demo.THREE || !demo.CSS2DObject || !summary) return;
  if (demo.objects["Scene graph;Paper object labels"]) {
    refreshPaperObjectLabels();
    syncPaperLabelRotation();
    return;
  }
  const THREE = demo.THREE;
  const objectGroup = new THREE.Group();
  objectGroup.name = "Scene graph;Paper object labels";
  (summary.objects || []).forEach((obj) => {
    const label = createPaperLabel(objectDisplayText(obj), "paper-text-label is-object", objectLabelPosition(obj), cssRgb(obj.color));
    if (label) {
      label.userData.objectId = String(obj.id);
      label.userData.isCeiling = isCeilingLabel(obj.label);
      if (label.element) label.element.dataset.objectId = String(obj.id);
      const box = labelTextElement(label);
      if (box) box.dataset.objectId = String(obj.id);
      objectGroup.add(label);
    }
  });
  objectGroup.visible = false;
  demo.scene.add(objectGroup);
  demo.objects[objectGroup.name] = objectGroup;

  const objectsById = objectMapById();
  ["center", "surface", "raised", "topdown"].forEach((style) => {
    const relGroup = new THREE.Group();
    relGroup.name = `Scene graph;Paper relationship labels;${style}`;
    (summary.relationships || []).forEach((rel) => {
      const pos = relationshipLabelPosition(rel, style, objectsById);
      if (!pos) return;
      const label = createPaperLabel(baseRelationshipText(rel), "paper-text-label is-relationship", pos, null);
      if (label) {
        label.userData.relIndex = String(rel.index);
        relGroup.add(label);
      }
    });
    relGroup.visible = false;
    demo.scene.add(relGroup);
    demo.objects[relGroup.name] = relGroup;
  });
  syncPaperLabelRotation();
}
function refreshPaperObjectLabels() {
  const demo = api();
  const group = demo && demo.objects ? demo.objects["Scene graph;Paper object labels"] : null;
  if (!group || !summary) return;
  const objectsById = objectMapById();
  group.traverse((child) => {
    const objectId = child && child.userData ? child.userData.objectId : null;
    const obj = objectsById.get(String(objectId));
    const box = labelTextElement(child);
    if (obj && box) box.textContent = objectDisplayText(obj);
  });
}
function setRelationshipLabelVisible(root, visible, activeRels, selected) {
  if (!root) return;
  root.visible = visible;
  if (typeof root.traverse !== "function") return;
  root.traverse((child) => {
    const relIndex = child.userData ? child.userData.relIndex : null;
    const childVisible = !!visible && (!relIndex || activeRels.has(String(relIndex))) && (!state.spotlight || !selected || String(relIndex) === selected);
    child.visible = childVisible;
    setLabelElementVisible(child, childVisible);
  });
  if (typeof root.updateMatrixWorld === "function") root.updateMatrixWorld(true);
}
function setObjectLabelVisible(root, visible, modeTop) {
  if (!root) return;
  root.visible = visible;
  if (typeof root.traverse !== "function") return;
  root.traverse((child) => {
    const childVisible = !!visible && !(modeTop && child.userData && child.userData.isCeiling);
    child.visible = childVisible;
    setLabelElementVisible(child, childVisible);
  });
  if (typeof root.updateMatrixWorld === "function") root.updateMatrixWorld(true);
}
function ensureCentroidSphereLights() {
  const demo = api();
  if (!demo || !demo.THREE || demo.objects["Scene graph;Centroid sphere lights"]) return;
  const THREE = demo.THREE;
  const group = new THREE.Group();
  group.name = "Scene graph;Centroid sphere lights";
  const ambient = new THREE.AmbientLight(0xffffff, 0.58);
  const key = new THREE.DirectionalLight(0xffffff, 0.95);
  key.position.set(0.35, -0.45, 0.9);
  const fill = new THREE.DirectionalLight(0xffffff, 0.36);
  fill.position.set(-0.75, 0.35, 0.65);
  group.add(ambient, key, fill);
  demo.scene.add(group);
  demo.objects[group.name] = group;
}
function ensureCenterSpheres() {
  const demo = api();
  if (!demo || !demo.THREE || !summary || demo.objects["Scene graph;Object center spheres"]) return;
  const THREE = demo.THREE;
  ensureCentroidSphereLights();
  const group = new THREE.Group();
  group.name = "Scene graph;Object center spheres";
  const radius = Math.max(0.025, Number(config.centerSphereRadius || 0.095));
  const geometry = new THREE.SphereGeometry(radius, 32, 20);
  (summary.objects || []).forEach((obj) => {
    const label = String(obj.label || "").toLowerCase();
    if (label === "floor" || label.includes("floor") || label === "ceiling" || label.includes("ceiling")) return;
    const color = darkenColor(new THREE.Color((obj.color && obj.color.length >= 3) ? `rgb(${obj.color[0]}, ${obj.color[1]}, ${obj.color[2]})` : "#69a7ff"));
    const material = new THREE.MeshLambertMaterial({
      color,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.96,
    });
    const sphere = new THREE.Mesh(geometry, material);
    const pos = topAnchorForObject(obj);
    sphere.userData.centroid = obj.position.slice(0, 3);
    sphere.userData.topAnchor = pos.slice(0, 3);
    sphere.position.set(pos[0], pos[1], pos[2]);
    sphere.userData.objectId = String(obj.id);
    sphere.renderOrder = 1200;
    group.add(sphere);
  });
  group.visible = false;
  group.renderOrder = 1200;
  demo.scene.add(group);
  demo.objects[group.name] = group;
}
function applyLayers() {
  const activeRels = state.relationshipVisible;
  const modeTop = state.edgeStyle === "topdown";
  const selected = state.selectedRelationship;
  const baseBoxOpacity = Number(byId("box-opacity").value) / 100;
  ensureCenterSpheres();
  ensurePaperTextLabels();
  syncPaperLabelRotation();
  moveCenterSpheres(modeTop);
  const paperTextReady = !!(api() && api().objects && api().objects["Scene graph;Paper object labels"]);
  eachObject((name, object) => {
    let visible = true;
    if (isPaperLayer(name, "scene")) visible = state.layers.scene;
    if (isPaperLayer(name, "boxes")) {
      const objectId = objectIdFromBoxName(name);
      const revealFraction = objectId ? Number(state.revealFractions.get(String(objectId)) || 0) : 0;
      visible = state.layers.boxes;
      if (state.tourReveal) visible = visible && !!objectId && revealFraction > 0.02;
      if (modeTop && isCeilingLabel(objectLabelFromBoxName(name))) visible = false;
      setOpacity(object, baseBoxOpacity * (state.tourReveal ? Math.max(0.02, Math.min(1, revealFraction)) : 1));
    }
    if (isNativeObjectLabelLayer(name)) visible = state.layers.labels && !paperTextReady;
    if (isPaperObjectLabelLayer(name)) { setObjectLabelVisible(object, state.layers.labels, modeTop); return; }
    if (isPaperLayer(name, "centers")) visible = state.layers.edges && name.startsWith("Scene graph;Object center spheres");
    if (isPaperLayer(name, "query")) { const id = objectIdFromHighlightName(name); visible = state.matchedIds.includes(String(id)); if (modeTop && isCeilingLabel(objectLabelFromBoxName(name))) visible = false; styleQueryHighlight(object, visible); }
    if (isPaperLayer(name, "edges")) {
      const relIndex = relIndexFromName(name);
      const style = relStyleFromName(name);
      const relationshipMatched = state.matchedRelIds.includes(String(relIndex));
      visible = ((state.layers.edges && activeRels.has(String(relIndex))) || relationshipMatched) && style === state.edgeStyle;
      if (state.spotlight && selected && !relationshipMatched) visible = visible && String(relIndex) === selected;
    }
    if (isNativeRelationshipLabelLayer(name)) visible = state.layers.relLabels && !paperTextReady && relationshipLabelMatchesStyle(name, state.edgeStyle);
    if (isPaperRelationshipLabelLayer(name)) { setRelationshipLabelVisible(object, state.layers.relLabels && relationshipLabelMatchesStyle(name, state.edgeStyle), activeRels, selected); return; }
    styleGraphOverlay(name, object);
    setVisible(object, visible);
  });
  render();
}
function syncLayerButtons() { document.querySelectorAll("[data-layer]").forEach((button) => button.classList.toggle("is-active", !!state.layers[button.dataset.layer])); }
function setLayerActive(layer, active) { state.layers[layer] = active; syncLayerButtons(); applyLayers(); }
function setMode(mode) {
  stopTour(true);
  state.mode = mode;
  state.tourReveal = false;
  state.revealedObjectIds = new Set();
  state.revealFractions = new Map();
  syncModeButtons();
  if (mode === "scene") state.layers = {scene: true, boxes: false, labels: false, edges: false, relLabels: false};
  else if (mode === "boxes") state.layers = {scene: false, boxes: true, labels: false, edges: false, relLabels: false};
  else if (mode === "graph") state.layers = {scene: false, boxes: true, labels: false, edges: true, relLabels: false};
  else if (mode === "top") { state.layers = {scene: true, boxes: true, labels: true, edges: true, relLabels: false}; setEdgeStyle("topdown", false); byId("box-opacity").value = "48"; byId("scene-opacity").value = "55"; applyOpacity(); setTopCamera(); }
  else state.layers = {scene: true, boxes: false, labels: false, edges: true, relLabels: false};
  syncLayerButtons();
  applyLayers();
}
function setEdgeStyle(style, updateButtons = true) { state.edgeStyle = style; if (updateButtons) document.querySelectorAll("[data-edge-style]").forEach((button) => button.classList.toggle("is-active", button.dataset.edgeStyle === style)); applyLayers(); }
function applyOpacity() { const boxOpacity = Number(byId("box-opacity").value) / 100; const sceneOpacity = Number(byId("scene-opacity").value) / 100; eachObject((name, object) => { if (isPaperLayer(name, "boxes")) { const objectId = objectIdFromBoxName(name); const revealFraction = objectId ? Number(state.revealFractions.get(String(objectId)) || 0) : 0; setOpacity(object, boxOpacity * (state.tourReveal ? Math.max(0.02, Math.min(1, revealFraction)) : 1)); } if (isPaperLayer(name, "scene")) setOpacity(object, sceneOpacity); }); render(); }
function objectBoundsFor(objects) {
  if (!objects.length) return {min: [-1, -1, 0], max: [1, 1, 2], center: [0, 0, 1], span: [2, 2, 2]};
  const mins = objects.map((o) => [o.position[0] - o.size[0] * 0.5, o.position[1] - o.size[1] * 0.5, o.position[2] - o.size[2] * 0.5]);
  const maxs = objects.map((o) => [o.position[0] + o.size[0] * 0.5, o.position[1] + o.size[1] * 0.5, o.position[2] + o.size[2] * 0.5]);
  const min = [0, 1, 2].map((axis) => Math.min(...mins.map((p) => p[axis])));
  const max = [0, 1, 2].map((axis) => Math.max(...maxs.map((p) => p[axis])));
  const center = [0, 1, 2].map((axis) => (min[axis] + max[axis]) * 0.5);
  const span = [0, 1, 2].map((axis) => Math.max(0.1, max[axis] - min[axis]));
  return {min, max, center, span};
}
function sceneObjects() { return (summary && summary.objects) ? summary.objects.filter((obj) => obj && Array.isArray(obj.position)) : []; }
function isWallLike(obj) {
  const label = String((obj && obj.label) || "").toLowerCase();
  return label === "wall" || label.includes("wall") || label === "ceiling" || label.includes("ceiling") || label === "floor" || label.includes("floor");
}
function isWindowLike(obj) {
  const label = String((obj && obj.label) || "").toLowerCase();
  return label === "window" || label.includes("window");
}
function isDoorLike(obj) {
  const label = String((obj && obj.label) || "").toLowerCase();
  return label === "door" || label.includes("door");
}
function isStructuralForWalkthrough(obj) {
  return isWallLike(obj) || isWindowLike(obj) || isDoorLike(obj);
}
function walkthroughObjects() {
  const objects = sceneObjects().filter((obj) => !isStructuralForWalkthrough(obj));
  if (objects.length) return objects;
  const nonLayoutObjects = sceneObjects().filter((obj) => !isWallLike(obj) && !isWindowLike(obj));
  if (nonLayoutObjects.length) return nonLayoutObjects;
  const nonCeilingObjects = sceneObjects().filter((obj) => !isFloorOrCeiling(obj));
  return nonCeilingObjects.length ? nonCeilingObjects : sceneObjects();
}
function focusObjects() { return walkthroughObjects(); }
function navigationObjects() { return walkthroughObjects(); }
function objectBounds() { return objectBoundsFor(sceneObjects()); }
function focusBounds() { return objectBoundsFor(focusObjects()); }
function lerp(a, b, t) { return a + (b - a) * t; }
function clamp01(t) { return Math.min(1, Math.max(0, Number(t) || 0)); }
function smoothstep(t) { t = clamp01(t); return t * t * (3 - 2 * t); }
function distance3(a, b) { const dx = a[0] - b[0]; const dy = a[1] - b[1]; const dz = a[2] - b[2]; return Math.sqrt(dx * dx + dy * dy + dz * dz); }
function normalize3(v) { const length = Math.max(1e-6, Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])); return [v[0] / length, v[1] / length, v[2] / length]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize2(v) { const length = Math.max(1e-6, Math.sqrt(v[0] * v[0] + v[1] * v[1])); return [v[0] / length, v[1] / length]; }
function dot2(a, b) { return a[0] * b[0] + a[1] * b[1]; }
function fallbackTourPath() {
  const b = objectBounds();
  const z = b.min[2] + Math.min(Math.max(b.span[2] * 0.45, 1.15), 1.75);
  return [
    [b.min[0] - b.span[0] * 0.22, b.center[1] - b.span[1] * 0.34, z],
    [b.center[0] - b.span[0] * 0.22, b.center[1] - b.span[1] * 0.10, z + 0.04],
    [b.center[0] + b.span[0] * 0.18, b.center[1] + b.span[1] * 0.08, z + 0.02],
    [b.max[0] + b.span[0] * 0.18, b.center[1] + b.span[1] * 0.26, z],
  ];
}
function smoothRawTourPoints(points) {
  if (points.length <= 2) return points;
  const radius = Math.min(4, Math.max(1, Math.floor(points.length / 24)));
  return points.map((_point, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(points.length - 1, index + radius);
    const count = end - start + 1;
    const sum = [0, 0, 0];
    for (let item = start; item <= end; item += 1) {
      sum[0] += points[item][0];
      sum[1] += points[item][1];
      sum[2] += points[item][2];
    }
    return [sum[0] / count, sum[1] / count, sum[2] / count];
  });
}
function calmTourWaypoints(points) {
  return smoothRawTourPoints(points);
}
function removeTurnarounds(points) {
  if (points.length <= 3) return points;
  const filtered = [points[0], points[1]];
  for (let index = 2; index < points.length; index += 1) {
    const prev = filtered[filtered.length - 2];
    const current = filtered[filtered.length - 1];
    const next = points[index];
    const incoming = normalize3([current[0] - prev[0], current[1] - prev[1], current[2] - prev[2]]);
    const outgoing = normalize3([next[0] - current[0], next[1] - current[1], next[2] - current[2]]);
    if (dot3(incoming, outgoing) < -0.18 && filtered.length > 1) {
      filtered[filtered.length - 1] = next;
    } else {
      filtered.push(next);
    }
  }
  return filtered.length >= 2 ? filtered : points;
}
function objectPlanarRadius(obj) {
  const size = (obj && Array.isArray(obj.size)) ? obj.size : [0.5, 0.5, 0.5];
  return Math.max(0.18, Math.max(Number(size[0]) || 0.5, Number(size[1]) || 0.5) * 0.5);
}
function objectTourAnchor(obj) {
  if (!obj || !Array.isArray(obj.position)) return [0, 0, 0];
  const size = Array.isArray(obj.size) ? obj.size : [0.5, 0.5, 0.5];
  const height = Number(size[2]) || 1;
  return [
    Number(obj.position[0]) || 0,
    Number(obj.position[1]) || 0,
    (Number(obj.position[2]) || 0) + Math.max(0.07, 0.18 * height),
  ];
}
function roomOrderedObjects(objects, cameraPose = null) {
  const rows = objects.slice().sort((a, b) => Number(a.id) - Number(b.id));
  if (!cameraPose) return rows;
  const forward = normalize2([
    cameraPose.target[0] - cameraPose.position[0],
    cameraPose.target[1] - cameraPose.position[1],
  ]);
  const side = [-forward[1], forward[0]];
  return rows
    .map((obj) => {
      const anchor = objectTourAnchor(obj);
      const rel = [anchor[0] - cameraPose.position[0], anchor[1] - cameraPose.position[1]];
      return {
        obj,
        forward: dot2(rel, forward),
        side: dot2(rel, side),
        height: anchor[2],
      };
    })
    .sort((a, b) => a.forward - b.forward || a.side - b.side || a.height - b.height || Number(a.obj.id) - Number(b.obj.id))
    .map((row) => row.obj);
}
function parentSourceKey(obj) {
  return String((obj && obj.source_scene_id) || "scene");
}
function balancedRoomObjectOrder(objects, cameraPose = null) {
  const ordered = roomOrderedObjects(objects, cameraPose);
  const groups = [];
  const bySource = new Map();
  ordered.forEach((obj) => {
    const source = parentSourceKey(obj);
    if (!bySource.has(source)) {
      bySource.set(source, []);
      groups.push(source);
    }
    bySource.get(source).push(obj);
  });
  if (groups.length <= 1) return ordered;
  const balanced = [];
  let cursor = 0;
  while (balanced.length < ordered.length) {
    const source = groups[cursor % groups.length];
    const bucket = bySource.get(source) || [];
    if (bucket.length) balanced.push(bucket.shift());
    cursor += 1;
    if (cursor > ordered.length * groups.length * 2) break;
  }
  return balanced.length === ordered.length ? balanced : ordered;
}
function deterministicDirection(seedIndex) {
  const angle = (Number(seedIndex) || 0) * 2.399963229728653;
  return [Math.cos(angle), Math.sin(angle)];
}
function clampInteriorCameraPosition(position, scene) {
  const marginX = Math.min(Math.max(scene.span[0] * 0.055, 0.12), 0.34);
  const marginY = Math.min(Math.max(scene.span[1] * 0.055, 0.12), 0.34);
  const minX = scene.min[0] + marginX;
  const maxX = scene.max[0] - marginX;
  const minY = scene.min[1] + marginY;
  const maxY = scene.max[1] - marginY;
  return [
    minX <= maxX ? Math.min(Math.max(position[0], minX), maxX) : scene.center[0],
    minY <= maxY ? Math.min(Math.max(position[1], minY), maxY) : scene.center[1],
    position[2],
  ];
}
function roomViewCandidateDirections() {
  return [
    [-0.72, -1.0],
    [1.0, 0.72],
    [-1.0, 0.72],
    [0.72, -1.0],
  ].map(normalize2);
}
function routeKeyframeForRoomDirection(objects, scene, focus, direction, index) {
  const squareSide = Math.max(1.0, Math.min(scene.span[0], scene.span[1]));
  const cameraRadius = Math.min(Math.max(squareSide * 0.24, 0.36), 0.76);
  const lookAhead = Math.min(Math.max(squareSide * 0.14, 0.18), 0.42);
  const roomCenter = [
    scene.center[0] * 0.68 + focus.center[0] * 0.32,
    scene.center[1] * 0.68 + focus.center[1] * 0.32,
  ];
  const targetZValues = objects.map((obj) => objectTourAnchor(obj)[2]);
  const targetZ = targetZValues.length
    ? targetZValues.reduce((sum, value) => sum + value, 0) / targetZValues.length
    : focus.center[2];
  const target = [
    roomCenter[0] - direction[0] * lookAhead,
    roomCenter[1] - direction[1] * lookAhead,
    Math.min(Math.max(targetZ, scene.min[2] + 0.82), scene.min[2] + Math.min(Math.max(scene.span[2] * 0.56, 1.08), 1.62)),
  ];
  const eyeHeight = Math.min(Math.max(scene.span[2] * 0.52, 1.34), 1.90);
  const position = [
    roomCenter[0] + direction[0] * cameraRadius,
    roomCenter[1] + direction[1] * cameraRadius,
    scene.min[2] + eyeHeight,
  ];
  return {
    position: clampInteriorCameraPosition(position, scene),
    target,
    objectIds: [],
    viewKey: `room_view_${index}`,
  };
}
function objectVisibilityScore(obj, pose) {
  const anchor = objectTourAnchor(obj);
  const forward = normalize3([
    pose.target[0] - pose.position[0],
    pose.target[1] - pose.position[1],
    pose.target[2] - pose.position[2],
  ]);
  const toObjectRaw = [
    anchor[0] - pose.position[0],
    anchor[1] - pose.position[1],
    anchor[2] - pose.position[2],
  ];
  const toObject = normalize3(toObjectRaw);
  const alignment = dot3(forward, toObject);
  const distance = Math.max(1e-6, distance3(anchor, pose.position));
  const viewSpan = Math.max(focusBounds().span[0], focusBounds().span[1], 1);
  return alignment * 2.4 - distance / viewSpan;
}
function objectVisibleFromPose(obj, pose) {
  const anchor = objectTourAnchor(obj);
  const forward = normalize3([
    pose.target[0] - pose.position[0],
    pose.target[1] - pose.position[1],
    pose.target[2] - pose.position[2],
  ]);
  const toObject = normalize3([
    anchor[0] - pose.position[0],
    anchor[1] - pose.position[1],
    anchor[2] - pose.position[2],
  ]);
  const alignment = dot3(forward, toObject);
  const distance = distance3(anchor, pose.position);
  const viewSpan = Math.max(focusBounds().span[0], focusBounds().span[1], 1);
  return alignment >= Math.cos(THREE_DEG_TO_RAD * 68) || distance <= viewSpan * 0.42;
}
const THREE_DEG_TO_RAD = Math.PI / 180;
function assignObjectsToRoomViews(frames, objects) {
  if (!frames.length) return [];
  const ordered = balancedRoomObjectOrder(objects, frames[0]);
  const frameCount = Math.min(frames.length, Math.max(2, Math.ceil(ordered.length / 3)));
  const activeFrames = frames.slice(0, frameCount);
  const chunkSize = Math.ceil(ordered.length / activeFrames.length);
  activeFrames.forEach((frame, index) => {
    const chunk = ordered.slice(index * chunkSize, (index + 1) * chunkSize);
    frame.objectIds = chunk.map((obj) => String(obj.id));
    frame.viewKey = `room_all_objects_${index}`;
    if (chunk.length) {
      const chunkFocus = objectBoundsFor(chunk);
      const targetZValues = chunk.map((obj) => objectTourAnchor(obj)[2]);
      const targetZ = targetZValues.reduce((sum, value) => sum + value, 0) / targetZValues.length;
      frame.target = [
        lerp(frame.target[0], chunkFocus.center[0], 0.62),
        lerp(frame.target[1], chunkFocus.center[1], 0.62),
        lerp(frame.target[2], targetZ, 0.42),
      ];
    }
  });
  return activeFrames.filter((frame) => frame.objectIds.length);
}
function objectDrivenTourKeyframes() {
  const objects = navigationObjects().filter((obj) => obj && Array.isArray(obj.position) && obj.position.length >= 3);
  if (!objects.length) return [];
  const scene = objectBounds();
  const focus = focusBounds();
  const candidateFrames = roomViewCandidateDirections().map((direction, index) => (
    routeKeyframeForRoomDirection(objects, scene, focus, direction, index)
  ));
  return assignObjectsToRoomViews(candidateFrames, objects);
}
function cameraTrajectoryKeyframes() {
  const trajectory = (summary && summary.camera_trajectory) ? summary.camera_trajectory : [];
  const positions = trajectory.map((item) => item.position).filter((p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite));
  if (positions.length < 2) return [];
  const focus = focusBounds();
  const points = calmTourWaypoints(positions);
  return points.map((position, index) => {
    const next = points[Math.min(points.length - 1, index + 1)];
    const forward = normalize3([next[0] - position[0], next[1] - position[1], next[2] - position[2]]);
    const target = [
      position[0] + forward[0] * 1.6 + focus.center[0] * 0.18,
      position[1] + forward[1] * 1.6 + focus.center[1] * 0.18,
      Math.max(focus.center[2], position[2] + 0.02),
    ];
    return {position, target};
  });
}
function fallbackTourKeyframes() {
  const focus = focusBounds();
  return fallbackTourPath().map((position) => ({position, target: focus.center.slice(0, 3)}));
}
function tourKeyframes() {
  const objectRoute = objectDrivenTourKeyframes();
  if (objectRoute.length) return objectRoute;
  const trajectoryRoute = cameraTrajectoryKeyframes();
  if (trajectoryRoute.length >= 2) return trajectoryRoute;
  return fallbackTourKeyframes();
}
function tourPath() {
  return tourKeyframes().map((frame) => frame.position);
}
function tourPathKey(points) {
  return points.map((p) => p.map((value) => Number(value).toFixed(3)).join(",")).join("|");
}
function tourRouteKey(frames) {
  return frames.map((frame) => {
    const pos = frame.position.map((value) => Number(value).toFixed(3)).join(",");
    const target = frame.target.map((value) => Number(value).toFixed(3)).join(",");
    const objectIds = Array.isArray(frame.objectIds) ? frame.objectIds.join(",") : (frame.objectId || "");
    return `${pos}->${target}:${frame.viewKey || ""}:${objectIds}`;
  }).join("|");
}
function catmullRomPoint(points, index, t) {
  const p0 = points[Math.max(0, index - 1)];
  const p1 = points[index];
  const p2 = points[Math.min(points.length - 1, index + 1)];
  const p3 = points[Math.min(points.length - 1, index + 2)];
  const t2 = t * t;
  const t3 = t2 * t;
  return [0, 1, 2].map((axis) => 0.5 * (
    (2 * p1[axis]) +
    (-p0[axis] + p2[axis]) * t +
    (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t2 +
    (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * t3
  ));
}
function buildSmoothPath(points) {
  if (!points.length) return {points: [], cumulative: [], total: 0};
  if (points.length === 1) return {points: [points[0]], cumulative: [0], total: 0};
  const samples = [];
  const segmentCount = Math.max(1, points.length - 1);
  const samplesPerSegment = Math.max(6, Math.min(18, Math.round(180 / segmentCount)));
  for (let index = 0; index < points.length - 1; index += 1) {
    for (let step = 0; step < samplesPerSegment; step += 1) {
      samples.push(catmullRomPoint(points, index, step / samplesPerSegment));
    }
  }
  samples.push(points[points.length - 1]);
  const cumulative = [0];
  let total = 0;
  for (let index = 1; index < samples.length; index += 1) {
    total += distance3(samples[index - 1], samples[index]);
    cumulative.push(total);
  }
  return {points: samples, cumulative, total};
}
function buildSmoothRoute(frames) {
  if (!frames.length) return {poses: [], cumulative: [], total: 0};
  if (frames.length === 1) {
    return {
      poses: [{
        position: frames[0].position,
        target: frames[0].target,
        objectId: frames[0].objectId || "",
        objectIds: Array.isArray(frames[0].objectIds) ? frames[0].objectIds.slice() : [],
        viewKey: frames[0].viewKey || "",
      }],
      cumulative: [0],
      total: 0,
    };
  }
  const positions = frames.map((frame) => frame.position);
  const targets = frames.map((frame) => frame.target);
  const poses = [];
  const segmentCount = Math.max(1, frames.length - 1);
  const samplesPerSegment = Math.max(10, Math.min(30, Math.round(240 / segmentCount)));
  for (let index = 0; index < frames.length - 1; index += 1) {
    for (let step = 0; step < samplesPerSegment; step += 1) {
      poses.push({
        position: catmullRomPoint(positions, index, step / samplesPerSegment),
        target: catmullRomPoint(targets, index, step / samplesPerSegment),
        objectId: frames[index].objectId || "",
        objectIds: Array.isArray(frames[index].objectIds) ? frames[index].objectIds.slice() : [],
        viewKey: frames[index].viewKey || "",
      });
    }
  }
  poses.push({
    position: frames[frames.length - 1].position,
    target: frames[frames.length - 1].target,
    objectId: frames[frames.length - 1].objectId || "",
    objectIds: Array.isArray(frames[frames.length - 1].objectIds) ? frames[frames.length - 1].objectIds.slice() : [],
    viewKey: frames[frames.length - 1].viewKey || "",
  });
  const cumulative = [0];
  let total = 0;
  for (let index = 1; index < poses.length; index += 1) {
    total += distance3(poses[index - 1].position, poses[index].position);
    cumulative.push(total);
  }
  return {poses, points: poses.map((pose) => pose.position), cumulative, total};
}
function smoothTourRoute() {
  const frames = tourKeyframes();
  const key = tourRouteKey(frames);
  if (tour.routeCache && tour.routeKey === key) return tour.routeCache;
  tour.routeKey = key;
  tour.routeCache = buildSmoothRoute(frames);
  tour.pathKey = key;
  tour.pathCache = {points: tour.routeCache.points, cumulative: tour.routeCache.cumulative, total: tour.routeCache.total};
  tour.revealOrder = [];
  return tour.routeCache;
}
function smoothTourPath() {
  if (tour.pathCache && tour.routeCache && tour.pathKey === tour.routeKey) return tour.pathCache;
  const route = smoothTourRoute();
  tour.pathCache = {points: route.points || [], cumulative: route.cumulative || [], total: route.total || 0};
  tour.pathKey = tour.routeKey;
  return tour.pathCache;
}
function interpolatePoseFromRoute(rawT) {
  const cache = smoothTourRoute();
  const poses = cache.poses || [];
  if (!poses.length) return {position: [0, 0, 0], target: [0, 0, 1]};
  if (poses.length === 1 || cache.total <= 1e-6) return poses[0];
  const targetDistance = clamp01(rawT) * cache.total;
  let index = 1;
  while (index < cache.cumulative.length - 1 && cache.cumulative[index] < targetDistance) index += 1;
  const prevDistance = cache.cumulative[index - 1];
  const nextDistance = cache.cumulative[index];
  const local = (targetDistance - prevDistance) / Math.max(1e-6, nextDistance - prevDistance);
  const a = poses[index - 1];
  const b = poses[index];
  return {
    position: [
      lerp(a.position[0], b.position[0], local),
      lerp(a.position[1], b.position[1], local),
      lerp(a.position[2], b.position[2], local),
    ],
    target: [
      lerp(a.target[0], b.target[0], local),
      lerp(a.target[1], b.target[1], local),
      lerp(a.target[2], b.target[2], local),
    ],
  };
}
function interpolateSmoothPath(rawT) {
  return interpolatePoseFromRoute(rawT).position;
}
function buildTourTimeline(frames) {
  if (!frames.length) return {segments: [], frames: [], totalWeight: 0};
  const segments = [];
  let totalWeight = 0;
  frames.forEach((frame, index) => {
    const objectCount = Array.isArray(frame.objectIds) ? frame.objectIds.length : 0;
    const holdWeight = Math.max(1.35, objectCount * 0.72);
    segments.push({type: "hold", from: frame, to: frame, viewIndex: index, weight: holdWeight});
    totalWeight += holdWeight;
    if (index < frames.length - 1) {
      const next = frames[index + 1];
      const moveDistance = distance3(frame.position, next.position);
      const moveWeight = Math.min(Math.max(moveDistance * 1.65, 2.25), 4.20);
      segments.push({type: "move", from: frame, to: next, viewIndex: index, weight: moveWeight});
      totalWeight += moveWeight;
    }
  });
  let cursor = 0;
  segments.forEach((segment) => {
    segment.start = totalWeight > 0 ? cursor / totalWeight : 0;
    cursor += segment.weight;
    segment.end = totalWeight > 0 ? cursor / totalWeight : 1;
  });
  return {segments, frames, totalWeight};
}
function tourTimeline() {
  const frames = tourKeyframes();
  const key = tourRouteKey(frames);
  if (tour.timelineCache && tour.timelineKey === key) return tour.timelineCache;
  tour.timelineKey = key;
  tour.timelineCache = buildTourTimeline(frames);
  tour.revealOrder = [];
  return tour.timelineCache;
}
function holdObjectAnchors(frame) {
  const objectsById = objectMapById();
  const ids = Array.isArray(frame.objectIds) ? frame.objectIds : [];
  return ids
    .map((id) => objectsById.get(String(id)))
    .filter((obj) => obj && Array.isArray(obj.position))
    .map(objectTourAnchor);
}
function holdFocusAt(frame, local) {
  const anchors = holdObjectAnchors(frame);
  if (!anchors.length) return frame.target.slice(0, 3);
  if (anchors.length === 1) {
    return lerpVec(frame.target, anchors[0], 0.36);
  }
  const scaled = clamp01(local) * Math.max(1, anchors.length - 1);
  const index = Math.min(anchors.length - 2, Math.floor(scaled));
  const segmentT = smoothstep(scaled - index);
  const objectFocus = lerpVec(anchors[index], anchors[index + 1], segmentT);
  return lerpVec(frame.target, objectFocus, 0.40);
}
function poseWithinHoldSegment(segment, rawLocal) {
  const frame = segment.from;
  const scene = objectBounds();
  const local = clamp01(rawLocal);
  const eased = smoothstep(local);
  const envelope = Math.sin(local * Math.PI);
  const basePosition = frame.position.slice(0, 3);
  const baseTarget = frame.target.slice(0, 3);
  if (local <= 1e-6 || local >= 1 - 1e-6) {
    return {position: basePosition, target: baseTarget};
  }
  const focus = holdFocusAt(frame, eased);
  const direction = normalize2([baseTarget[0] - basePosition[0], baseTarget[1] - basePosition[1]]);
  const side = [-direction[1], direction[0]];
  const drift = Math.min(Math.max(Math.min(scene.span[0], scene.span[1]) * 0.10, 0.14), 0.34);
  const sway = Math.sin((eased - 0.15) * Math.PI * 1.15);
  const forwardSway = Math.sin((eased + 0.10) * Math.PI * 2.0);
  const minTargetZ = scene.min[2] + 0.70;
  const maxTargetZ = Math.max(minTargetZ, scene.max[2] - 0.22);
  const target = [
    lerp(baseTarget[0], focus[0], envelope),
    lerp(baseTarget[1], focus[1], envelope),
    Math.min(Math.max(lerp(baseTarget[2], focus[2], envelope), minTargetZ), maxTargetZ),
  ];
  const targetDelta = [target[0] - baseTarget[0], target[1] - baseTarget[1], target[2] - baseTarget[2]];
  const unclampedPosition = [
    basePosition[0] + targetDelta[0] * 0.28 + side[0] * sway * drift * envelope + direction[0] * forwardSway * drift * 0.28 * envelope,
    basePosition[1] + targetDelta[1] * 0.28 + side[1] * sway * drift * envelope + direction[1] * forwardSway * drift * 0.28 * envelope,
    basePosition[2] + targetDelta[2] * 0.08 + Math.sin(eased * Math.PI) * 0.035 * envelope,
  ];
  return {
    position: clampInteriorCameraPosition(unclampedPosition, scene),
    target: [target[0], target[1], Math.min(Math.max(target[2], minTargetZ), maxTargetZ)],
  };
}
function segmentEndpointPose(segment, endLocal) {
  if (!segment || !segment.from) return null;
  if (segment.type === "hold") return poseWithinHoldSegment(segment, endLocal);
  return endLocal <= 0
    ? {position: segment.from.position.slice(0, 3), target: segment.from.target.slice(0, 3)}
    : {position: segment.to.position.slice(0, 3), target: segment.to.target.slice(0, 3)};
}
function timelinePoseAt(rawT) {
  const timeline = tourTimeline();
  const segments = timeline.segments || [];
  if (!segments.length) return interpolatePoseFromRoute(rawT);
  const t = clamp01(rawT);
  let segment = segments[segments.length - 1];
  let segmentIndex = segments.length - 1;
  for (let candidateIndex = 0; candidateIndex < segments.length; candidateIndex += 1) {
    const candidate = segments[candidateIndex];
    if (t <= candidate.end || candidate === segments[segments.length - 1]) {
      segment = candidate;
      segmentIndex = candidateIndex;
      break;
    }
  }
  if (segment.type === "hold" || segment.start === segment.end) {
    const local = segment.start === segment.end ? 0 : (t - segment.start) / Math.max(1e-6, segment.end - segment.start);
    return poseWithinHoldSegment(segment, local);
  }
  const local = smoothstep((t - segment.start) / Math.max(1e-6, segment.end - segment.start));
  const startPose = segmentEndpointPose(segments[Math.max(0, segmentIndex - 1)], 1) || {
    position: segment.from.position.slice(0, 3),
    target: segment.from.target.slice(0, 3),
  };
  const endPose = segmentEndpointPose(segments[Math.min(segments.length - 1, segmentIndex + 1)], 0) || {
    position: segment.to.position.slice(0, 3),
    target: segment.to.target.slice(0, 3),
  };
  return {
    position: lerpVec(startPose.position, endPose.position, local),
    target: lerpVec(startPose.target, endPose.target, local),
  };
}
function tourRevealSchedule() {
  if (tour.revealOrder.length) return tour.revealOrder;
  const timeline = tourTimeline();
  const scheduled = [];
  const seen = new Set();
  (timeline.segments || []).forEach((segment) => {
    if (segment.type !== "hold") return;
    const ids = (Array.isArray(segment.from.objectIds) ? segment.from.objectIds : []).filter((id) => !seen.has(String(id)));
    if (!ids.length) return;
    const revealStart = segment.start + (segment.end - segment.start) * 0.12;
    const revealEnd = segment.end - (segment.end - segment.start) * 0.12;
    ids.forEach((id, index) => {
      const local = ids.length === 1 ? 0.45 : index / Math.max(1, ids.length - 1);
      const t = revealStart + (revealEnd - revealStart) * local;
      scheduled.push({id: String(id), t, viewIndex: segment.viewIndex, order: scheduled.length});
      seen.add(String(id));
    });
  });
  walkthroughObjects().forEach((obj) => {
    const id = String(obj.id);
    if (seen.has(id)) return;
    scheduled.push({id, t: 0.94, viewIndex: Number.MAX_SAFE_INTEGER, order: scheduled.length});
  });
  tour.revealOrder = scheduled.sort((a, b) => a.t - b.t || a.viewIndex - b.viewIndex || a.order - b.order || Number(a.id) - Number(b.id));
  return tour.revealOrder;
}
function nearestPathT(position, cache) {
  const points = cache.points || [];
  if (!points.length || cache.total <= 1e-6) return 0;
  let bestIndex = 0;
  let bestDistance = Infinity;
  points.forEach((point, index) => {
    const distance = distance3(point, position);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return cache.cumulative[bestIndex] / cache.total;
}
function tourForward2() {
  const cache = smoothTourPath();
  const points = cache.points || [];
  if (points.length < 2) return [1, 0];
  const first = points[0];
  const last = points[points.length - 1];
  const global = normalize2([last[0] - first[0], last[1] - first[1]]);
  if (Math.abs(global[0]) + Math.abs(global[1]) < 1e-5) return [1, 0];
  return global;
}
function stableWalkForward2(position, ahead) {
  const global = tourForward2();
  const travel = normalize2([ahead[0] - position[0], ahead[1] - position[1]]);
  if (dot2(travel, global) < 0.10) return global;
  return normalize2([
    travel[0] * 0.64 + global[0] * 0.36,
    travel[1] * 0.64 + global[1] * 0.36,
  ]);
}
function tourPoseAt(rawT) {
  return timelinePoseAt(rawT);
}
function objectRevealOrder() {
  return tourRevealSchedule();
}
function updateTourReveal(rawT) {
  if (!state.tourReveal) return false;
  const order = objectRevealOrder();
  const next = new Set();
  const fractions = new Map();
  if (order.length) {
    order.forEach((item) => {
      const fade = clamp01((rawT - item.t) / 0.045);
      if (fade > 0.01) {
        next.add(String(item.id));
        fractions.set(String(item.id), smoothstep(fade));
      }
    });
    if (rawT >= 0.985) order.forEach((item) => {
      next.add(String(item.id));
      fractions.set(String(item.id), 1);
    });
  }
  let changed = next.size !== state.revealedObjectIds.size;
  if (!changed) {
    for (const id of next) {
      if (!state.revealedObjectIds.has(id)) {
        changed = true;
        break;
      }
    }
  }
  if (!changed) {
    for (const [id, fraction] of fractions.entries()) {
      if (Math.abs(Number(state.revealFractions.get(id) || 0) - fraction) > 0.02) {
        changed = true;
        break;
      }
    }
  }
  state.revealedObjectIds = next;
  state.revealFractions = fractions;
  return changed;
}
function setAllRelationshipsVisible() {
  state.relationshipVisible.clear();
  (summary && summary.relationships ? summary.relationships : []).forEach((rel) => state.relationshipVisible.add(String(rel.index)));
  document.querySelectorAll("#relationship-strip .relationship-chip").forEach((button) => {
    button.classList.add("is-active");
    button.classList.remove("is-muted");
  });
  state.selectedRelationship = "";
  state.spotlight = false;
  const spotlight = byId("spotlight-toggle");
  if (spotlight) spotlight.classList.remove("is-active");
}
function setCameraPose(position, target) {
  const demo = api();
  if (!demo) return;
  demo.camera.up.set(0, 0, 1);
  if ("fov" in demo.camera && Number.isFinite(Number(config.cameraFov))) {
    const baseFov = Number(config.cameraFov);
    demo.camera.fov = state.tourReveal ? Math.min(72, baseFov + 8) : baseFov;
    if (typeof demo.camera.updateProjectionMatrix === "function") demo.camera.updateProjectionMatrix();
  }
  demo.camera.position.set(position[0], position[1], position[2]);
  demo.camera.lookAt(target[0], target[1], target[2]);
  if (demo.controls) {
    demo.controls.target.set(target[0], target[1], target[2]);
    demo.controls.update();
  }
  render();
}
function vectorToArray(vector, fallback) {
  if (!vector) return fallback.slice(0, 3);
  const values = [Number(vector.x), Number(vector.y), Number(vector.z)];
  return values.every(Number.isFinite) ? values : fallback.slice(0, 3);
}
function currentCameraPose() {
  const demo = api();
  const focus = focusBounds();
  if (!demo || !demo.camera) {
    return {position: [focus.center[0], focus.center[1], focus.max[2] + 1.2], target: focus.center.slice(0, 3), up: [0, 0, 1]};
  }
  return {
    position: vectorToArray(demo.camera.position, [focus.center[0], focus.center[1], focus.max[2] + 1.2]),
    target: vectorToArray(demo.controls ? demo.controls.target : null, focus.center.slice(0, 3)),
    up: vectorToArray(demo.camera.up, [0, 0, 1]),
  };
}
function topCameraPose() {
  const b = objectBounds();
  const focus = focusBounds();
  const span = Math.max(b.span[0], b.span[1], 1);
  const target = [b.center[0], b.center[1], focus.center[2]];
  return {
    position: [b.center[0], b.center[1], b.max[2] + span * 1.12],
    target,
    up: [0, 1, 0],
  };
}
function applyCameraPose(pose) {
  const demo = api();
  if (!demo || !pose) return;
  const up = normalize3(pose.up || [0, 0, 1]);
  demo.camera.up.set(up[0], up[1], up[2]);
  demo.camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  demo.camera.lookAt(pose.target[0], pose.target[1], pose.target[2]);
  if (demo.controls) {
    demo.controls.target.set(pose.target[0], pose.target[1], pose.target[2]);
    demo.controls.update();
  }
  render();
}
function lerpVec(a, b, t) { return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]; }
function prepareTopGraphView() {
  state.tourReveal = false;
  state.revealedObjectIds = new Set();
  state.revealFractions = new Map();
  setAllRelationshipsVisible();
  state.mode = "top";
  state.edgeStyle = "topdown";
  state.layers = {scene: true, boxes: true, labels: true, edges: true, relLabels: false};
  syncModeButtons();
  document.querySelectorAll("[data-edge-style]").forEach((button) => button.classList.toggle("is-active", button.dataset.edgeStyle === "topdown"));
  syncLayerButtons();
  byId("box-opacity").value = "48";
  byId("scene-opacity").value = "55";
  applyOpacity();
  applyLayers();
}
function animateToTopView() {
  const startPose = currentCameraPose();
  const endPose = topCameraPose();
  const startTime = performance.now();
  const duration = 2200;
  function step(now) {
    const t = smoothstep((now - startTime) / duration);
    applyCameraPose({
      position: lerpVec(startPose.position, endPose.position, t),
      target: lerpVec(startPose.target, endPose.target, t),
      up: normalize3(lerpVec(startPose.up, endPose.up, t)),
    });
    if (t < 1) {
      tour.frame = window.requestAnimationFrame(step);
    } else {
      tour.frame = null;
      applyCameraPose(endPose);
    }
  }
  tour.frame = window.requestAnimationFrame(step);
}
function setTourProgress(rawT) {
  rawT = clamp01(rawT);
  tour.progress = rawT;
  const pose = tourPoseAt(rawT);
  if (updateTourReveal(rawT)) applyLayers();
  setCameraPose(pose.position, pose.target);
  byId("demo-progress").value = String(Math.round(rawT * 1000));
}
function stopTour(resetLabel = true) {
  tour.running = false;
  if (tour.frame !== null) window.cancelAnimationFrame(tour.frame);
  tour.frame = null;
  if (resetLabel) byId("demo-play").textContent = "Play";
}
function enterTourRevealMode(resetProgress) {
  state.tourReveal = true;
  state.revealedObjectIds = new Set();
  state.revealFractions = new Map();
  tour.revealOrder = [];
  state.mode = "all";
  state.edgeStyle = "raised";
  state.layers = {scene: true, boxes: true, labels: false, edges: false, relLabels: false};
  state.selectedRelationship = "";
  state.spotlight = false;
  const spotlight = byId("spotlight-toggle");
  if (spotlight) spotlight.classList.remove("is-active");
  syncModeButtons();
  document.querySelectorAll("[data-edge-style]").forEach((button) => button.classList.toggle("is-active", button.dataset.edgeStyle === "raised"));
  syncLayerButtons();
  if (resetProgress) {
    tour.progress = 0;
    byId("demo-progress").value = "0";
  }
}
function finishTour() {
  stopTour(true);
  tour.progress = 1;
  byId("demo-progress").value = "1000";
  prepareTopGraphView();
  animateToTopView();
}
function animateTour(now) {
  if (!tour.running) return;
  const elapsed = now - tour.start;
  const rawT = Math.min(1, elapsed / tour.durationMs);
  setTourProgress(rawT);
  if (rawT >= 1) {
    finishTour();
    return;
  }
  tour.frame = window.requestAnimationFrame(animateTour);
}
function toggleTour() {
  if (tour.running) {
    stopTour(true);
    return;
  }
  const shouldRestart = tour.progress >= 0.999;
  enterTourRevealMode(shouldRestart);
  byId("box-opacity").value = "68";
  byId("scene-opacity").value = "100";
  applyOpacity();
  applyLayers();
  byId("demo-play").textContent = "Pause";
  tour.running = true;
  tour.start = performance.now() - tour.progress * tour.durationMs;
  setTourProgress(tour.progress);
  tour.frame = window.requestAnimationFrame(animateTour);
}
function setTopCamera() { stopTour(true); applyCameraPose(topCameraPose()); syncPaperLabelRotation(); }
function applyTourStartCamera(saveState = false) {
  const pose = tourPoseAt(0);
  setCameraPose(pose.position, pose.target);
  const demo = api();
  if (saveState && demo && demo.controls && typeof demo.controls.saveState === "function") demo.controls.saveState();
}
function resetCamera() {
  stopTour(true);
  state.tourReveal = false;
  state.revealedObjectIds = new Set();
  state.revealFractions = new Map();
  tour.progress = 0;
  byId("demo-progress").value = "0";
  applyTourStartCamera(true);
  applyLayers();
}
function exportFileName() { const scene = String((summary && summary.scene_id) || "scene_graph").replace(/[^A-Za-z0-9_.-]+/g, "_"); return `${scene}_${state.edgeStyle}_view.svg`; }
function drawRoundRect(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius || 0, width * 0.5, height * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
function cssAngleRadians(value) {
  const raw = String(value || "").trim();
  if (!raw || raw === "none") return 0;
  const amount = parseFloat(raw);
  if (!Number.isFinite(amount)) return 0;
  if (raw.endsWith("rad")) return amount;
  if (raw.endsWith("turn")) return amount * Math.PI * 2;
  return amount * Math.PI / 180;
}
function drawVisibleLabelsToCanvas(ctx, container) {
  const demo = api();
  if (!demo || !demo.labelRenderer || !demo.labelRenderer.domElement) return;
  const containerRect = container.getBoundingClientRect();
  const labels = Array.from(demo.labelRenderer.domElement.querySelectorAll(".label, .paper-text-label"));
  labels.forEach((element) => {
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity || 1) <= 0.01) return;
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const width = Math.ceil((element.offsetWidth || rect.width) + 2);
    const height = Math.ceil((element.offsetHeight || rect.height) + 2);
    const centerX = rect.left - containerRect.left + rect.width * 0.5;
    const centerY = rect.top - containerRect.top + rect.height * 0.5;
    const x = -width * 0.5;
    const y = -height * 0.5;
    const rotation = cssAngleRadians(style.rotate || style.getPropertyValue("--paper-label-rotation") || element.style.rotate);
    const background = (style.backgroundColor === "rgba(0, 0, 0, 0)" || style.backgroundColor === "transparent") ? "none" : style.backgroundColor;
    const strokeWidth = parseFloat(style.borderTopWidth || "0") || 0;
    const stroke = (strokeWidth <= 0 || style.borderTopStyle === "none") ? "none" : style.borderTopColor;
    const radius = parseFloat(style.borderTopLeftRadius || "0") || 0;
    const fontSize = parseFloat(style.fontSize || "12") || 12;
    const paddingLeft = parseFloat(style.paddingLeft || "0") || 0;
    const textX = x + paddingLeft + 1;
    const textY = y + height * 0.5;
    ctx.save();
    ctx.translate(centerX, centerY);
    if (rotation) ctx.rotate(rotation);
    if (background !== "none") {
      ctx.fillStyle = background;
      drawRoundRect(ctx, x, y, width, height, radius);
      ctx.fill();
    }
    if (stroke !== "none" && strokeWidth > 0) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = strokeWidth;
      drawRoundRect(ctx, x, y, width, height, radius);
      ctx.stroke();
    }
    ctx.fillStyle = style.color || "#ffffff";
    ctx.font = `${style.fontWeight || "400"} ${fontSize}px ${style.fontFamily || "sans-serif"}`;
    ctx.textBaseline = "middle";
    ctx.fillText(element.textContent || "", textX, textY);
    ctx.restore();
  });
}
function isDomElementVisible(element) {
  if (!element) return false;
  const style = window.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity || 1) > 0.01 && rect.width > 0 && rect.height > 0;
}
function drawStyledDomBox(ctx, element, containerRect) {
  const style = window.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  const x = rect.left - containerRect.left;
  const y = rect.top - containerRect.top;
  const width = rect.width;
  const height = rect.height;
  const background = (style.backgroundColor === "rgba(0, 0, 0, 0)" || style.backgroundColor === "transparent") ? "none" : style.backgroundColor;
  const strokeWidth = parseFloat(style.borderTopWidth || "0") || 0;
  const stroke = (strokeWidth <= 0 || style.borderTopStyle === "none") ? "none" : style.borderTopColor;
  const radius = parseFloat(style.borderTopLeftRadius || "0") || 0;
  if (background !== "none") {
    ctx.fillStyle = background;
    drawRoundRect(ctx, x, y, width, height, radius);
    ctx.fill();
  }
  if (stroke !== "none" && strokeWidth > 0) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = strokeWidth;
    drawRoundRect(ctx, x, y, width, height, radius);
    ctx.stroke();
  }
  return {x, y, width, height, style};
}
function textForDomElement(element) {
  if (!element) return "";
  const tagName = String(element.tagName || "").toLowerCase();
  if (tagName === "input") return element.value || element.getAttribute("placeholder") || "";
  return element.textContent || "";
}
function drawDomElementText(ctx, element, box) {
  if (element && element.classList && element.classList.contains("triplet-submit-button")) {
    ctx.strokeStyle = window.getComputedStyle(element).color || "#5f96df";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const cx = box.x + box.width * 0.47;
    const cy = box.y + box.height * 0.45;
    const radius = Math.min(box.width, box.height) * 0.16;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx + radius * 0.72, cy + radius * 0.72);
    ctx.lineTo(cx + radius * 1.75, cy + radius * 1.75);
    ctx.stroke();
    return;
  }
  const text = textForDomElement(element).trim();
  if (!text) return;
  const style = box.style || window.getComputedStyle(element);
  const fontSize = parseFloat(style.fontSize || "12") || 12;
  const paddingLeft = parseFloat(style.paddingLeft || "0") || 0;
  const paddingRight = parseFloat(style.paddingRight || "0") || 0;
  let drawText = text;
  if (style.textTransform === "uppercase") drawText = drawText.toUpperCase();
  const textX = box.x + paddingLeft + 1;
  const textY = box.y + box.height * 0.5;
  ctx.fillStyle = style.color || "#ffffff";
  ctx.font = `${style.fontWeight || "400"} ${fontSize}px ${style.fontFamily || "sans-serif"}`;
  ctx.textBaseline = "middle";
  const maxWidth = Math.max(1, box.width - paddingLeft - paddingRight - 2);
  ctx.fillText(drawText, textX, textY, maxWidth);
}
function drawVisibleDomOverlaysToCanvas(ctx, container) {
  const containerRect = container.getBoundingClientRect();
  const panel = byId("scene-query-panel");
  if (isDomElementVisible(panel)) {
    drawStyledDomBox(ctx, panel, containerRect);
    panel.querySelectorAll(".triplet-query-grid, .paper-highlight-toggle, label, input, button, #scene-query-status").forEach((element) => {
      if (!isDomElementVisible(element)) return;
      const box = drawStyledDomBox(ctx, element, containerRect);
      drawDomElementText(ctx, element, box);
    });
  }
  ["paper-query-export-pdf"].forEach((id) => {
    const button = byId(id);
    if (!isDomElementVisible(button)) return;
    const box = drawStyledDomBox(ctx, button, containerRect);
    drawDomElementText(ctx, button, box);
  });
}
function byteStringToUint8Array(value) {
  const bytes = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i += 1) bytes[i] = value.charCodeAt(i) & 255;
  return bytes;
}
function dataUrlToUint8Array(dataUrl) {
  const base64 = String(dataUrl).split(",", 2)[1] || "";
  const binary = window.atob(base64);
  return byteStringToUint8Array(binary);
}
function concatUint8Arrays(parts) {
  const total = parts.reduce((count, part) => count + part.length, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  parts.forEach((part) => {
    output.set(part, offset);
    offset += part.length;
  });
  return output;
}
function escapeXmlAttribute(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[char]));
}
function buildSvgBlobFromPng(pngDataUrl, width, height) {
  const svg = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<image href="${escapeXmlAttribute(pngDataUrl)}" width="${width}" height="${height}" />`,
    "</svg>",
  ].join("\n");
  return new Blob([svg], {type: "image/svg+xml;charset=utf-8"});
}
function renderTransparentSceneForExport(demo) {
  const previousBackground = demo.scene ? demo.scene.background : null;
  const previousClearAlpha = demo.renderer.getClearAlpha ? demo.renderer.getClearAlpha() : 1;
  const previousClearColor = demo.THREE && demo.renderer.getClearColor ? new demo.THREE.Color() : null;
  if (previousClearColor) demo.renderer.getClearColor(previousClearColor);
  try {
    if (demo.scene) demo.scene.background = null;
    if (demo.renderer.setClearColor) demo.renderer.setClearColor(0xffffff, 0);
    else if (demo.renderer.setClearAlpha) demo.renderer.setClearAlpha(0);
    render();
  } finally {
    if (demo.scene) demo.scene.background = previousBackground;
    if (previousClearColor && demo.renderer.setClearColor) demo.renderer.setClearColor(previousClearColor, previousClearAlpha);
    else if (demo.renderer.setClearAlpha) demo.renderer.setClearAlpha(previousClearAlpha);
  }
}
function exportCurrentViewSvg() {
  const demo = api();
  const container = byId("render_container");
  if (!demo || !demo.renderer || !container) return;
  stopTour(true);
  const width = Math.max(1, Math.round(container.clientWidth));
  const height = Math.max(1, Math.round(container.clientHeight));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  try {
    renderTransparentSceneForExport(demo);
    ctx.drawImage(demo.renderer.domElement, 0, 0, width, height);
    drawVisibleLabelsToCanvas(ctx, container);
    drawVisibleDomOverlaysToCanvas(ctx, container);
  } catch (error) {
    window.alert("Could not export SVG snapshot: " + error.message);
    return;
  }
  const pngDataUrl = canvas.toDataURL("image/png");
  const blob = buildSvgBlobFromPng(pngDataUrl, width, height);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = exportFileName();
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
function populateRelationshipStrip() {
  const strip = byId("relationship-strip");
  if (!strip || !summary) return;
  const initializeVisibility = state.relationshipVisible.size === 0;
  strip.innerHTML = "";
  (summary.relationships || []).forEach((rel) => {
    const key = String(rel.index);
    if (initializeVisibility) state.relationshipVisible.add(key);
    const button = document.createElement("button");
    const active = state.relationshipVisible.has(key);
    button.className = active ? "relationship-chip is-active" : "relationship-chip is-muted";
    button.type = "button";
    button.dataset.relIndex = key;
    const source = shortSceneLabel(rel.source_scene_id);
    const prefix = source ? `${source} · ` : "";
    button.innerHTML = `<strong>${prefix}${rel.subject_id} to ${rel.object_id}</strong><span>${baseRelationshipText(rel)}</span>`;
    button.addEventListener("click", () => {
      if (state.relationshipVisible.has(key)) {
        state.relationshipVisible.delete(key);
        button.classList.remove("is-active");
        button.classList.add("is-muted");
      } else {
        state.relationshipVisible.add(key);
        button.classList.add("is-active");
        button.classList.remove("is-muted");
      }
      state.selectedRelationship = key;
      applyLayers();
    });
    strip.appendChild(button);
  });
}
function applyQueryHighlight(ids) { state.matchedIds = (ids || []).map((id) => String(id)); state.matchedRelIds = []; if (state.matchedIds.length) state.layers.boxes = false; syncLayerButtons(); applyLayers(); }
function applyRelationshipHighlight(ids, relationshipIds) { state.matchedIds = (ids || []).map((id) => String(id)); state.matchedRelIds = (relationshipIds || []).map((id) => String(id)); if (state.matchedIds.length) state.layers.boxes = false; syncLayerButtons(); applyLayers(); }
async function submitQuery(event) { event.preventDefault(); const input = byId("scene-query-input"); const status = byId("scene-query-status"); const query = input.value.trim(); if (!query) { applyQueryHighlight([]); status.textContent = ""; return; } if (window.location.protocol === "file:") { status.textContent = "Serve with scripts/serve_scene_graph_query.py so the eval matcher can encode this."; return; } status.textContent = "Encoding with CLIP..."; try { const response = await fetch(`/scene-query?text=${encodeURIComponent(query)}&top_k=${encodeURIComponent(config.topK || 5)}`); if (!response.ok) throw new Error(await response.text()); const result = await response.json(); const ids = result.matched_ids || (result.best ? [result.best.id] : []); applyQueryHighlight(ids); if (result.best) status.textContent = `Matched ${result.best.label} #${result.best.id} (${result.best.similarity.toFixed(3)})`; else status.textContent = "No match."; } catch (error) { status.textContent = "Query backend unavailable: " + error.message; } }
async function submitRelationshipQuery(event) { event.preventDefault(); const query = byId("relationship-query-input").value.trim(); const roleInput = document.querySelector('input[name="relationship-highlight-role"]:checked'); const highlightRole = roleInput ? roleInput.value : "object"; const status = byId("scene-query-status"); if (!query) { applyRelationshipHighlight([], []); status.textContent = ""; return; } if (window.location.protocol === "file:") { status.textContent = "Serve with scripts/serve_scene_graph_query.py so the eval matcher can encode this."; return; } status.textContent = "Encoding triplet query..."; try { const response = await fetch(`/scene-triplet-query?text=${encodeURIComponent(query)}&highlight=${encodeURIComponent(highlightRole)}&top_k=${encodeURIComponent(config.topK || 5)}`); if (!response.ok) throw new Error(await response.text()); const result = await response.json(); const ids = result.highlighted_ids || result.matched_ids || []; const relIds = result.matched_relationship_indices || []; applyRelationshipHighlight(ids, relIds); if (result.best) { const role = result.highlight_role || highlightRole; const highlightedId = role === "subject" ? result.best.subject_id : result.best.object_id; status.textContent = `Matched ${result.best.subject_label} ${result.best.predicate} ${result.best.object_label}; highlighted ${role} #${highlightedId} (${result.best.similarity.toFixed(3)})`; } else status.textContent = "No triplet match."; } catch (error) { status.textContent = "Triplet backend unavailable: " + error.message; } }
function installTripletEnterShortcut() { ["relationship-query-input"].forEach((id) => { const input = byId(id); if (!input || input.dataset.enterReady === "1") return; input.dataset.enterReady = "1"; input.addEventListener("keydown", (event) => { if (event.key !== "Enter") return; event.preventDefault(); const form = byId("relationship-query-form"); if (form && typeof form.requestSubmit === "function") form.requestSubmit(); else submitRelationshipQuery(event); }); }); }
async function initPaperDemo() {
  const response = await fetch("scene_query_objects.json");
  summary = await response.json();
  byId("stat-objects").textContent = String((summary.objects || []).length);
  byId("stat-edges").textContent = String((summary.relationships || []).length);
  byId("stat-points").textContent = config.pointCountLabel || "full";
  populateRelationshipStrip();
  ensureCenterSpheres();
  ensurePaperTextLabels();
  document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => setLayerActive(button.dataset.layer, !state.layers[button.dataset.layer])));
  document.querySelectorAll("[data-edge-style]").forEach((button) => button.addEventListener("click", () => setEdgeStyle(button.dataset.edgeStyle)));
  byId("spotlight-toggle").addEventListener("click", () => {
    state.spotlight = !state.spotlight;
    byId("spotlight-toggle").classList.toggle("is-active", state.spotlight);
    applyLayers();
  });
  byId("reset-view").addEventListener("click", resetCamera);
  byId("demo-play").addEventListener("click", toggleTour);
  byId("demo-progress").addEventListener("input", () => {
    stopTour(true);
    enterTourRevealMode(false);
    setTourProgress(Number(byId("demo-progress").value) / 1000);
  });
  byId("box-opacity").addEventListener("input", applyOpacity);
  byId("scene-opacity").addEventListener("input", applyOpacity);
  byId("scene-cloudiness").addEventListener("input", applySceneCloudiness);
  byId("scene-query-form").addEventListener("submit", submitQuery);
  byId("relationship-query-form").addEventListener("submit", submitRelationshipQuery);
  installTripletEnterShortcut();
  installDraggableQueryPanel();
  applyTourStartCamera(true);
  setMode("top");
  applyOpacity();
  applySceneCloudiness();
  byId("load-pill").classList.add("is-hidden");
  scheduleViewerResize();
}
let paperDemoStarted = false;
function startPaperDemoOnce() {
  if (paperDemoStarted) return;
  if (!window.pyviz3dDemo) return;
  paperDemoStarted = true;
  initPaperDemo().catch((error) => {
    paperDemoStarted = false;
    const pill = byId("load-pill");
    if (pill) pill.textContent = "UI load failed: " + error.message;
    console.error(error);
  });
}
window.addEventListener("pyviz3d-ready", startPaperDemoOnce);
window.addEventListener("load", () => {
  startPaperDemoOnce();
  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    startPaperDemoOnce();
    if (paperDemoStarted || attempts > 100) window.clearInterval(timer);
  }, 100);
});
'''.replace("__CONFIG__", json.dumps(config))


def patch_index_html_for_query(output_path: Path, args: argparse.Namespace) -> None:
    index_path = output_path / "index.html"
    if not index_path.is_file():
        return
    text = index_path.read_text(encoding="utf-8")
    if "paper-demo-shell" in text or "paper-query-shell" in text:
        return

    config = {
        "serveQuery": bool(args.serve_query),
        "topK": int(args.query_top_k),
        "clipModel": args.clip_model,
        "pointCountLabel": f"{float(args.mesh_point_samples) / 1_000_000.0:.1f}M",
        "scenePointSize": effective_scene_point_size(args),
        "tourDuration": float(args.tour_duration),
        "cameraFov": float(args.paper_camera_fov),
        "cameraDistanceScale": float(args.paper_camera_distance_scale),
        "centerSphereRadius": float(args.center_sphere_radius),
    }
    if args.ui_mode == "paper":
        app_shell = '''
<body class="paper-query">
  <main class="paper-query-shell">
    <div id="render_container"></div>
    <button id="paper-query-export-pdf" class="paper-export-button" type="button" title="Export current view as SVG">SVG</button>
    <section id="scene-query-panel" class="scene-query-panel" aria-label="Scene object semantic query"><label for="scene-query-input">Object query</label><form id="scene-query-form"><input id="scene-query-input" type="text" autocomplete="off" placeholder="something to sit on"></form><label class="relationship-query-label" for="relationship-query-input">Triplet query</label><form id="relationship-query-form"><div class="triplet-query-grid"><input id="relationship-query-input" type="text" autocomplete="off" placeholder="chair next to table"><button class="triplet-submit-button" type="submit" aria-label="Search"><svg class="triplet-submit-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.75" cy="10.75" r="6.25"></circle><path d="m15.5 15.5 4.25 4.25"></path></svg></button></div><div class="triplet-highlight-row" aria-label="Triplet highlight target"><span>Highlight</span><label><input type="radio" name="relationship-highlight-role" value="object" checked>Object</label><label><input type="radio" name="relationship-highlight-role" value="subject">Subject</label></div></form><div class="paper-mode-row" aria-label="View modes"><button data-paper-mode="scene" class="is-active" type="button">Scene</button><button data-paper-mode="boxes" type="button">Boxes</button><button data-paper-mode="graph" type="button">Graph</button><button data-paper-mode="top" type="button">Top</button></div><div id="scene-query-status"></div><div class="paper-highlight-toggle" aria-label="Highlight target"><button data-relationship-highlight-role="object" class="is-active" type="button">Object</button><button data-relationship-highlight-role="subject" type="button">Subject</button></div><span class="query-drag-handle" aria-hidden="true"></span><span class="query-resize-handle" aria-hidden="true"></span></section>
  </main>
'''
        if "<title>PyViz3D</title>" in text:
            text = text.replace("<title>PyViz3D</title>", "<title>Scene Query</title>", 1)
        text = text.replace(
            '<link rel="stylesheet" href="css/bootstrap.min.css">',
            '<link rel="stylesheet" href="css/bootstrap.min.css">\n\t\t<link rel="stylesheet" href="paper_query.css">',
            1,
        )
        text = text.replace(
            '<script type="module" src="js/scene.js"></script>',
            '<script type="module" src="js/scene.js"></script>\n\t\t<script type="module" src="paper_query.js"></script>',
            1,
        )
        if '<body>\n\t<div id="render_container"></div>' in text:
            text = text.replace('<body>\n\t<div id="render_container"></div>', app_shell, 1)
        elif "<body>" in text:
            text = text.replace("<body>", app_shell.removesuffix("</body>\n"), 1)
        (output_path / "paper_query.css").write_text(paper_query_css(), encoding="utf-8")
        (output_path / "paper_query.js").write_text(paper_query_js(config), encoding="utf-8")
        index_path.write_text(text, encoding="utf-8")
        return

    app_shell = '''
<body class="paper-demo">
  <div class="paper-demo-shell">
    <main class="viewer-card">
      <div id="render_container"></div>
      <div id="load-pill" class="load-pill">Loading 3D scene</div>
    </main>
    <aside class="control-panel" aria-label="Scene graph controls">
      <div class="transport-row"><button id="demo-play" class="play-button" type="button">Play</button><input id="demo-progress" class="timeline" type="range" min="0" max="1000" value="0" aria-label="Walkthrough progress"><button id="reset-view" class="icon-button" type="button">Reset</button></div>
      <div class="stats-row"><div class="stat-card"><strong id="stat-objects">0</strong><span>Objects</span></div><div class="stat-card"><strong id="stat-edges">0</strong><span>Edges</span></div><div class="stat-card"><strong id="stat-points">0</strong><span>Points</span></div></div>
      <div class="segmented mode-row"><button data-mode="all" type="button">All</button><button data-mode="scene" type="button">Scene</button><button data-mode="boxes" type="button">3D BBox</button><button data-mode="graph" type="button">Graph</button><button data-mode="top" class="is-active" type="button">Top</button></div>
      <div class="segmented layer-row"><button data-layer="scene" class="is-active" type="button">Scene</button><button data-layer="boxes" type="button">Boxes</button><button data-layer="labels" type="button">Labels</button><button data-layer="edges" class="is-active" type="button">Edges</button><button data-layer="relLabels" type="button">Rel text</button></div>
      <div class="segmented edge-row"><button data-edge-style="center" type="button">Center</button><button data-edge-style="surface" type="button">Surface</button><button data-edge-style="raised" class="is-active" type="button">Raised</button><button id="spotlight-toggle" type="button">Spotlight</button></div>
      <div class="range-grid"><label class="range-control">Boxes<input id="box-opacity" type="range" min="2" max="100" value="36"></label><label class="range-control">Scene<input id="scene-opacity" type="range" min="2" max="100" value="88"></label><label class="range-control">Cloud<input id="scene-cloudiness" type="range" min="0" max="100" value="0"></label></div>
      <section id="scene-query-panel" class="scene-query-panel" aria-label="Scene object semantic query"><label for="scene-query-input">Object query</label><form id="scene-query-form"><input id="scene-query-input" type="text" autocomplete="off" placeholder="something to sit on"></form><label class="relationship-query-label" for="relationship-query-input">Triplet query</label><form id="relationship-query-form"><div class="triplet-query-grid"><input id="relationship-query-input" type="text" autocomplete="off" placeholder="chair next to table"><button class="triplet-submit-button" type="submit" aria-label="Search"><svg class="triplet-submit-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.75" cy="10.75" r="6.25"></circle><path d="m15.5 15.5 4.25 4.25"></path></svg></button></div><div class="triplet-highlight-row" aria-label="Triplet highlight target"><span>Highlight</span><label><input type="radio" name="relationship-highlight-role" value="object" checked>Object</label><label><input type="radio" name="relationship-highlight-role" value="subject">Subject</label></div></form><div id="scene-query-status"></div></section>
      <div id="relationship-strip" class="relationship-strip" aria-label="Scene graph relationships"></div>
    </aside>
  </div>
'''
    if "<title>PyViz3D</title>" in text:
        text = text.replace("<title>PyViz3D</title>", "<title>Scene Graph Demo</title>", 1)
    text = text.replace(
        '<link rel="stylesheet" href="css/bootstrap.min.css">',
        '<link rel="stylesheet" href="css/bootstrap.min.css">\n\t\t<link rel="stylesheet" href="paper_demo.css">',
        1,
    )
    text = text.replace(
        '<script type="module" src="js/scene.js"></script>',
        '<script type="module" src="js/scene.js"></script>\n\t\t<script type="module" src="paper_demo.js"></script>',
        1,
    )
    if '<body>\n\t<div id="render_container"></div>' in text:
        text = text.replace('<body>\n\t<div id="render_container"></div>', app_shell, 1)
    elif "<body>" in text:
        text = text.replace("<body>", app_shell.removesuffix("</body>\n"), 1)
    (output_path / "paper_demo.css").write_text(paper_demo_css(), encoding="utf-8")
    (output_path / "paper_demo.js").write_text(paper_demo_js(config), encoding="utf-8")
    index_path.write_text(text, encoding="utf-8")


class SceneQueryMatcher:
    def __init__(self, objects: list[GraphObject], args: argparse.Namespace):
        if not objects:
            raise ValueError("Cannot build a query matcher for an empty scene.")
        self.objects = list(objects)
        self.labels = [obj.label for obj in self.objects]
        CLIPObjectMatcher = load_clip_object_matcher()
        self.matcher = CLIPObjectMatcher(args.clip_model)
        self.matcher.precompute(sorted(set(self.labels)))
        self.object_embeddings = self.matcher.get_embeddings(self.labels)

    def query(self, text: str, top_k: int) -> dict[str, Any]:
        query_text = text.strip()
        if not query_text:
            return {"query": text, "matches": [], "matched_ids": []}
        query_embedding = self.matcher.encode(query_text)
        similarities = np.asarray(self.object_embeddings @ query_embedding, dtype=np.float32)
        order = np.argsort(-similarities)
        specific_order = np.asarray(
            [idx for idx in order if not is_generic_object_label(self.objects[int(idx)].label)],
            dtype=order.dtype,
        )
        if len(specific_order):
            order = specific_order
        top_k = max(1, min(int(top_k), len(order)))

        matches = []
        for idx in order[:top_k]:
            obj = self.objects[int(idx)]
            matches.append(
                {
                    "id": int(obj.id),
                    "label": obj.label,
                    "similarity": float(similarities[int(idx)]),
                }
            )

        best = matches[0]
        best_label = best["label"]
        matched_ids = [
            int(obj.id)
            for obj, score in zip(self.objects, similarities)
            if obj.label == best_label and abs(float(score) - float(best["similarity"])) < 1e-6
        ]
        return {
            "query": query_text,
            "best": best,
            "matches": matches,
            "matched_ids": matched_ids or [int(best["id"])],
        }


def serve_viewer(output_path: Path, graph: LoadedGraph, args: argparse.Namespace) -> None:
    matcher = SceneQueryMatcher(graph.objects, args)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_path), **handler_kwargs)

        def log_message(self, format: str, *values: Any) -> None:
            if not getattr(args, "quiet", False):
                super().log_message(format, *values)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/scene-query":
                super().do_GET()
                return

            params = urllib.parse.parse_qs(parsed.query)
            text = params.get("text", [""])[0]
            top_k = int(params.get("top_k", [args.query_top_k])[0])
            try:
                payload = matcher.query(text, top_k)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:  # Keep browser feedback useful.
                body = str(exc).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    display_host = "127.0.0.1" if str(args.host) in {"0.0.0.0", "::"} else str(args.host)
    url = f"http://{display_host}:{int(args.port)}/index.html"
    print(f"Serving query-enabled viewer at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped viewer server.")
    finally:
        server.server_close()


def build_output_path(args: argparse.Namespace, graph: LoadedGraph) -> Path:
    if args.output_name:
        name = args.output_name
    elif args.all_parent_splits:
        name = f"{parent_scan_id(graph.scene_id)}_all_splits_pyviz3d"
    else:
        name = f"{graph.scene_id}_pyviz3d"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return args.output_dir / safe_name


def summary_payload(graph: LoadedGraph, output_path: Path, base_meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scene_id": graph.scene_id,
        "source_graph_json": str(graph.graph_json),
        "source_graph_jsons": [str(path) for path in graph.source_graph_jsons],
        "output_path": str(output_path),
        "base": base_meta,
        "query": {
            "enabled_in_html": True,
            "served_live": False,
            "server_script": "scripts/serve_scene_graph_query.py",
            "clip_model": args.clip_model,
        },
        "counts": {
            "objects": len(graph.objects),
            "relationships": len(graph.relationships),
            "source_graphs": len(graph.source_graph_jsons),
        },
        "objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "display_label": object_display_text(obj, graph.scene_id),
                "source_scene_id": obj.source_scene_id,
                "source_object_id": obj.source_object_id,
                "position": obj.position.astype(float).tolist(),
                "size": obj.size.astype(float).tolist(),
            }
            for obj in graph.objects
        ],
        "warnings": graph.warnings,
    }


def main() -> None:
    args = parse_args()
    try:
        graph = (
            load_parent_split_graphs(args)
            if args.all_parent_splits
            else load_graph(args.graph_json, args.scene_id, args.box_json, args.allow_scene_id_mismatch)
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
    output_path = build_output_path(args, graph)

    if args.dry_run:
        print(json.dumps(summary_payload(graph, output_path, {"base_mode": "dry_run"}, args), indent=2))
        return

    output_path.mkdir(parents=True, exist_ok=True)
    base_meta = render_pyviz3d(graph, args, output_path)
    summary = summary_payload(graph, output_path, base_meta, args)
    summary_path = output_path / "scene_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote PyViz3D viewer to {output_path}")
    print(f"Wrote summary to {summary_path}")
    if args.serve_query:
        print("--serve-query is now split out. Start the query-enabled server with:")
        print(
            "  python3 scripts/serve_scene_graph_query.py "
            f"--viewer-dir {output_path} --host {args.host} --port {args.port}"
        )
    else:
        print("For static viewing, serve it like before:")
        print(f"  cd {output_path}")
        print(f"  python -m http.server {args.port}")
        print("For live text query, use:")
        print(
            "  python3 scripts/serve_scene_graph_query.py "
            f"--viewer-dir {output_path} --host 0.0.0.0 --port {args.port}"
        )


if __name__ == "__main__":
    main()
