from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import visualize_3dssg_scene_graph_pyviz3d as sg_viz


DEFAULT_OUTPUT_SUFFIX = "room_walkthrough_pyviz3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a continuous PyViz3D room walkthrough with a new smooth camera path.",
    )
    parser.add_argument("--graph-json", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--allow-scene-id-mismatch", action="store_true")
    parser.add_argument(
        "--graph-kind",
        choices=["auto", "gt_3dssg", "gt_compact", "predicted"],
        default="auto",
        help="Accepted for CLI compatibility; graph format is auto-detected.",
    )
    parser.add_argument("--all-parent-splits", action="store_true")
    parser.add_argument("--box-json", type=Path, default=None)
    parser.add_argument("--point-cloud", type=Path, default=None)
    parser.add_argument("--r3scan-root", type=Path, default=sg_viz.DEFAULT_R3SCAN_ROOT)
    parser.add_argument("--sceneverse-pcd-root", type=Path, default=sg_viz.DEFAULT_SCENEVERSE_PCD_ROOT)
    parser.add_argument(
        "--base-mode",
        choices=["sceneverse_textured_mesh_points", "sceneverse_pcd", "raw_mesh", "raw_instance_ply", "none"],
        default="sceneverse_pcd",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--mesh-point-samples", type=int, default=2_200_000)
    parser.add_argument("--mesh-point-size", type=float, default=9.0)
    parser.add_argument("--mesh-point-alpha", type=float, default=0.94)
    parser.add_argument("--scene-top-trim-percent", type=float, default=0.0)
    parser.add_argument("--scene-top-trim-graph", action="store_true")
    parser.add_argument("--box-alpha", type=float, default=0.22)
    parser.add_argument("--box-edge-width", type=float, default=0.006)
    parser.add_argument("--box-angle-sign", type=float, default=-1.0)
    parser.add_argument("--hide-object-labels", action="store_true")
    parser.add_argument("--hide-relationships", action="store_true")
    parser.add_argument("--edge-width", type=float, default=0.012)
    parser.add_argument("--edge-alpha", type=float, default=0.82)
    parser.add_argument("--paper-point-size", type=float, default=None)
    parser.add_argument("--paper-camera-fov", type=float, default=58.0)
    parser.add_argument(
        "--paper-camera-distance-scale",
        type=float,
        default=1.0,
        help="Accepted for compatibility. The walkthrough uses --lookahead-distance instead.",
    )
    parser.add_argument(
        "--ui-mode",
        default="walkthrough",
        help="Accepted for compatibility. This script always writes the walkthrough UI.",
    )
    parser.add_argument(
        "--tour-duration",
        type=float,
        default=24.0,
        help="Playback duration in seconds. Kept as an alias for the old CLI.",
    )
    parser.add_argument("--walkthrough-duration", type=float, default=None)
    parser.add_argument("--eye-height", type=float, default=1.46)
    parser.add_argument("--target-height", type=float, default=1.28)
    parser.add_argument("--path-margin", type=float, default=0.55)
    parser.add_argument("--lookahead-distance", type=float, default=1.15)
    parser.add_argument(
        "--discovery-radius",
        type=float,
        default=1.85,
        help="Meters from the camera path at which an object's OBB starts appearing.",
    )
    parser.add_argument(
        "--discovery-fade-duration",
        type=float,
        default=0.045,
        help="Fraction of the walkthrough used to fade in each newly discovered OBB.",
    )
    parser.add_argument("--path-samples", type=int, default=720)
    parser.add_argument(
        "--route-kind",
        choices=["auto", "racetrack", "rounded_rectangle", "figure8"],
        default="auto",
    )
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--dry-run", action="store_true")
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
        paths = sorted(
            args.graph_dir.glob(f"{sg_viz.parent_scan_id(args.scene_id)}_split*_sg.json"),
            key=sg_viz.split_sort_key,
        )
        if paths:
            args.graph_json = paths[0]
    if not args.graph_json.is_file():
        parser.error(f"Scene-graph JSON not found: {args.graph_json}")


def effective_scene_point_size(args: argparse.Namespace) -> float:
    if args.paper_point_size is not None:
        return float(args.paper_point_size)
    if args.base_mode in {"sceneverse_textured_mesh_points", "raw_mesh"}:
        return float(args.mesh_point_size)
    return float(args.point_size)


def effective_camera_fov(args: argparse.Namespace) -> float:
    base_fov = float(args.paper_camera_fov)
    scale_bonus = max(0.0, float(args.paper_camera_distance_scale) - 1.0) * 6.0
    return float(min(82.0, max(24.0, base_fov + 6.0 + min(scale_bonus, 6.0))))


def finite_rows(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return arr[np.all(np.isfinite(arr[:, :3]), axis=1), :3]


def object_bounds(graph: sg_viz.LoadedGraph) -> Optional[tuple[np.ndarray, np.ndarray]]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for obj in graph.objects:
        obj_min, obj_max = sg_viz.object_aabb_bounds(obj)
        if np.all(np.isfinite(obj_min)) and np.all(np.isfinite(obj_max)):
            mins.append(obj_min.astype(np.float32))
            maxs.append(obj_max.astype(np.float32))
    if not mins:
        return None
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def robust_scene_bounds(
    graph: sg_viz.LoadedGraph,
    xyz: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    points = finite_rows(xyz) if xyz is not None else np.empty((0, 3), dtype=np.float32)
    if points.shape[0] >= 32:
        low = np.percentile(points, 1.0, axis=0).astype(np.float32)
        high = np.percentile(points, 99.0, axis=0).astype(np.float32)
        obj = object_bounds(graph)
        if obj is not None:
            obj_min, obj_max = obj
            low = np.minimum(low, obj_min)
            high = np.maximum(high, obj_max)
        return low, high, "scene_points_p01_p99_plus_objects"
    obj = object_bounds(graph)
    if obj is not None:
        return obj[0], obj[1], "objects"
    return (
        np.asarray([-1.0, -1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 2.4], dtype=np.float32),
        "fallback_unit_room",
    )


def estimate_floor_and_ceiling(
    graph: sg_viz.LoadedGraph,
    xyz: Optional[np.ndarray],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[float, float, str]:
    points = finite_rows(xyz) if xyz is not None else np.empty((0, 3), dtype=np.float32)
    if points.shape[0] >= 32:
        return float(np.percentile(points[:, 2], 2.0)), float(np.percentile(points[:, 2], 97.5)), "scene_points"

    floor_zs = []
    ceiling_zs = []
    for obj in graph.objects:
        obj_min, obj_max = sg_viz.object_aabb_bounds(obj)
        label = sg_viz.object_label_lower(obj)
        if label == "floor":
            floor_zs.append(float(obj_max[2]))
        elif label == "ceiling":
            ceiling_zs.append(float(obj_min[2]))
    floor_z = float(np.median(floor_zs)) if floor_zs else float(bounds_min[2])
    ceiling_z = float(np.median(ceiling_zs)) if ceiling_zs else float(bounds_max[2])
    return floor_z, ceiling_z, "objects"


def cardinal_closed_spline(points: np.ndarray, samples_per_segment: int = 32, tension: float = 0.45) -> np.ndarray:
    anchors = np.asarray(points, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[0] < 3:
        raise ValueError("Need at least 3 anchor points for a closed camera path.")
    out: list[np.ndarray] = []
    count = anchors.shape[0]
    tangent_scale = 0.5 * (1.0 - float(tension))
    for idx in range(count):
        p0 = anchors[(idx - 1) % count]
        p1 = anchors[idx]
        p2 = anchors[(idx + 1) % count]
        p3 = anchors[(idx + 2) % count]
        m1 = (p2 - p0) * tangent_scale
        m2 = (p3 - p1) * tangent_scale
        for step in range(samples_per_segment):
            t = step / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            out.append(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)
    out.append(out[0].copy())
    return np.asarray(out, dtype=np.float32)


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(distances, dtype=np.float64)])


def resample_closed_path(points: np.ndarray, sample_count: int) -> tuple[np.ndarray, float, float]:
    path = np.asarray(points, dtype=np.float32)
    if float(np.linalg.norm(path[0] - path[-1])) > 1e-5:
        path = np.concatenate([path, path[:1]], axis=0)
    cumulative = cumulative_lengths(path)
    total = float(cumulative[-1])
    if total <= 1e-6:
        repeated = np.repeat(path[:1], max(2, sample_count) + 1, axis=0)
        return repeated.astype(np.float32), 0.0, 0.0
    sample_count = max(24, int(sample_count))
    distances = np.linspace(0.0, total, sample_count + 1, dtype=np.float64)
    out = np.empty((sample_count + 1, path.shape[1]), dtype=np.float32)
    for axis in range(path.shape[1]):
        out[:, axis] = np.interp(distances, cumulative, path[:, axis]).astype(np.float32)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    return out, total, float(np.max(steps)) if len(steps) else 0.0


def room_route_anchors(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    margin: float,
    route_kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    center = ((bounds_min[:2] + bounds_max[:2]) * 0.5).astype(np.float32)
    dims = np.maximum(bounds_max[:2] - bounds_min[:2], 0.5).astype(np.float32)
    major_axis = int(np.argmax(dims))
    minor_axis = 1 - major_axis
    half_major = max(0.35, float(dims[major_axis]) * 0.5 - float(margin))
    half_minor = max(0.25, float(dims[minor_axis]) * 0.5 - min(float(margin), float(dims[minor_axis]) * 0.28))
    aspect = half_major / max(half_minor, 1e-6)
    selected_kind = route_kind
    if selected_kind == "auto":
        selected_kind = "rounded_rectangle" if aspect < 1.35 else "racetrack"

    if selected_kind == "figure8":
        theta = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        local = np.stack(
            [
                half_major * 0.86 * np.sin(theta),
                half_minor * 0.82 * np.sin(theta) * np.cos(theta),
            ],
            axis=1,
        )
    elif selected_kind == "rounded_rectangle":
        local = np.asarray(
            [
                [-0.72 * half_major, -0.72 * half_minor],
                [0.0, -0.92 * half_minor],
                [0.72 * half_major, -0.72 * half_minor],
                [0.92 * half_major, 0.0],
                [0.72 * half_major, 0.72 * half_minor],
                [0.0, 0.92 * half_minor],
                [-0.72 * half_major, 0.72 * half_minor],
                [-0.92 * half_major, 0.0],
            ],
            dtype=np.float32,
        )
    else:
        lateral = min(0.72 * half_minor, max(0.18, 0.42 * half_minor))
        local = np.asarray(
            [
                [-0.88 * half_major, -lateral],
                [-0.35 * half_major, -lateral],
                [0.60 * half_major, -lateral],
                [0.94 * half_major, 0.0],
                [0.60 * half_major, lateral],
                [-0.35 * half_major, lateral],
                [-0.88 * half_major, lateral],
                [-0.94 * half_major, 0.0],
            ],
            dtype=np.float32,
        )

    xy = np.empty_like(local)
    xy[:, major_axis] = center[major_axis] + local[:, 0]
    xy[:, minor_axis] = center[minor_axis] + local[:, 1]
    meta = {
        "selected_route_kind": selected_kind,
        "major_axis": "xy"[major_axis],
        "room_xy_center": center.astype(float).tolist(),
        "room_xy_size": dims.astype(float).tolist(),
        "path_margin": float(margin),
    }
    return xy.astype(np.float32), meta


def smooth_unit_vectors(vectors: np.ndarray, passes: int = 3) -> np.ndarray:
    dirs = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-6)
    for _ in range(max(0, int(passes))):
        dirs = 0.25 * np.roll(dirs, 1, axis=0) + 0.5 * dirs + 0.25 * np.roll(dirs, -1, axis=0)
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    return dirs.astype(np.float32)


def is_structural_object(obj: sg_viz.GraphObject) -> bool:
    return sg_viz.object_label_lower(obj) in {"floor", "ceiling", "wall"}


def rotate_closed_samples(samples: np.ndarray, shift: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.shape[0] <= 2:
        return arr
    core = arr[:-1]
    shift = int(shift) % core.shape[0]
    if shift == 0:
        rotated = core
    else:
        rotated = np.concatenate([core[shift:], core[:shift]], axis=0)
    return np.concatenate([rotated, rotated[:1]], axis=0).astype(np.float32)


def choose_walkthrough_start_shift(graph: sg_viz.LoadedGraph, positions: np.ndarray) -> int:
    if positions.shape[0] < 8:
        return 0
    closest_indices = []
    for obj in graph.objects:
        if is_structural_object(obj):
            continue
        distances = np.linalg.norm(obj.position.reshape(1, 3) - positions, axis=1)
        closest_indices.append(int(np.argmin(distances)))
    if not closest_indices:
        return 0
    lead_samples = max(8, int(round(0.12 * positions.shape[0])))
    return (min(closest_indices) - lead_samples) % positions.shape[0]


def build_walkthrough_poses(
    graph: sg_viz.LoadedGraph,
    xyz: Optional[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    bounds_min, bounds_max, bounds_source = robust_scene_bounds(graph, xyz)
    floor_z, ceiling_z, height_source = estimate_floor_and_ceiling(graph, xyz, bounds_min, bounds_max)
    room_height = max(0.75, ceiling_z - floor_z)
    camera_z = floor_z + float(args.eye_height)
    camera_z = min(camera_z, ceiling_z - 0.25) if ceiling_z > floor_z + 0.75 else camera_z
    camera_z = max(camera_z, floor_z + min(0.4, room_height * 0.45))
    target_z = floor_z + float(args.target_height)
    target_z = min(max(target_z, floor_z + 0.45), max(floor_z + 0.5, ceiling_z - 0.25))

    dims = np.maximum(bounds_max[:2] - bounds_min[:2], 0.5)
    margin = min(float(args.path_margin), float(np.min(dims)) * 0.34)
    margin = max(0.05, margin)
    anchors_xy, route_meta = room_route_anchors(bounds_min, bounds_max, margin, args.route_kind)
    anchors = np.column_stack([anchors_xy, np.full(anchors_xy.shape[0], camera_z, dtype=np.float32)])
    dense = cardinal_closed_spline(anchors, samples_per_segment=36, tension=0.42)
    positions, route_length, max_step = resample_closed_path(dense, int(args.path_samples))
    positions[-1] = positions[0]

    sample_count = positions.shape[0] - 1
    if route_length > 1e-6:
        lookahead_samples = max(3, int(round(sample_count * float(args.lookahead_distance) / route_length)))
    else:
        lookahead_samples = 3
    lookahead_samples = min(max(3, lookahead_samples), max(3, sample_count // 4))

    closed_positions = positions[:-1]
    future = np.roll(closed_positions, -lookahead_samples, axis=0)
    center = np.asarray(
        [0.5 * (bounds_min[0] + bounds_max[0]), 0.5 * (bounds_min[1] + bounds_max[1]), target_z],
        dtype=np.float32,
    )
    target_xy = future[:, :2] * 0.82 + center[:2].reshape(1, 2) * 0.18
    raw_targets = np.column_stack([target_xy, np.full(sample_count, target_z, dtype=np.float32)])
    directions = raw_targets - closed_positions
    directions[:, 2] = target_z - camera_z
    directions = smooth_unit_vectors(directions, passes=4)
    target_distance = max(0.65, min(2.4, float(args.lookahead_distance)))
    targets = closed_positions + directions * target_distance
    targets[:, 2] = target_z
    targets = np.concatenate([targets, targets[:1]], axis=0).astype(np.float32)
    start_shift = choose_walkthrough_start_shift(graph, closed_positions)
    positions = rotate_closed_samples(positions, start_shift)
    targets = rotate_closed_samples(targets, start_shift)

    poses = [
        {
            "position": positions[idx].astype(float).tolist(),
            "target": targets[idx].astype(float).tolist(),
            "up": [0.0, 0.0, 1.0],
        }
        for idx in range(positions.shape[0])
    ]
    closing_jump = float(np.linalg.norm(positions[-1] - positions[0]))
    return {
        "poses": poses,
        "anchors": anchors.astype(float).tolist(),
        "bounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
            "source": bounds_source,
            "floor_z": float(floor_z),
            "ceiling_z": float(ceiling_z),
            "height_source": height_source,
        },
        "route": {
            **route_meta,
            "length": float(route_length),
            "sample_count": int(sample_count),
            "max_step": float(max_step),
            "closing_jump": closing_jump,
            "lookahead_samples": int(lookahead_samples),
            "eye_height": float(args.eye_height),
            "target_height": float(args.target_height),
            "start_shift_samples": int(start_shift),
        },
    }


def object_discovery_schedule(
    graph: sg_viz.LoadedGraph,
    walkthrough: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    poses = walkthrough.get("poses", [])
    if len(poses) < 2:
        return {"objects": [], "radius": float(args.discovery_radius), "fade_duration": float(args.discovery_fade_duration)}

    positions = np.asarray([pose["position"] for pose in poses[:-1]], dtype=np.float32)
    targets = np.asarray([pose["target"] for pose in poses[:-1]], dtype=np.float32)
    forward = targets - positions
    forward = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-6)
    sample_count = max(1, positions.shape[0])
    base_radius = max(0.25, float(args.discovery_radius))
    fade_duration = max(0.005, min(0.25, float(args.discovery_fade_duration)))

    scheduled = []
    for obj in graph.objects:
        if is_structural_object(obj):
            discover_at = 0.0
            reason = "structural"
        else:
            center = obj.position.astype(np.float32).reshape(1, 3)
            to_object = center - positions
            distances = np.linalg.norm(to_object, axis=1)
            directions = to_object / np.maximum(distances.reshape(-1, 1), 1e-6)
            facing = np.sum(directions * forward, axis=1)
            object_radius = 0.5 * float(np.linalg.norm(obj.size[:2]))
            threshold = base_radius + min(0.75, object_radius)
            visible = (distances <= threshold) & (facing >= -0.08)
            close = distances <= threshold * 0.62
            candidates = np.flatnonzero(visible | close)
            if len(candidates):
                index = int(candidates[int(np.argmin(distances[candidates]))])
                reason = "nearest_visible_camera_view"
            else:
                index = int(np.argmin(distances))
                reason = "nearest_path_fallback"
            discover_at = float(index / sample_count)

        scheduled.append(
            {
                "id": int(obj.id),
                "label": obj.label,
                "layer": sg_viz.object_layer_name(obj),
                "discover_at": discover_at,
                "fade_duration": fade_duration,
                "reason": reason,
            }
        )

    scheduled.sort(key=lambda item: (float(item["discover_at"]), int(item["id"])))
    return {
        "objects": scheduled,
        "radius": base_radius,
        "fade_duration": fade_duration,
    }


def render_walkthrough_pyviz3d(
    graph: sg_viz.LoadedGraph,
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    viz = sg_viz.ensure_pyviz3d()
    visualizer = viz.Visualizer()
    xyz, colors, base_meta = sg_viz.load_base_points(graph, args)
    if args.scene_top_trim_graph:
        base_meta.update(sg_viz.prune_graph_above_height(graph, base_meta.get("scene_top_trim_height")))

    if xyz is not None and colors is not None:
        visualizer.add_points(
            "Scene;Point cloud",
            xyz,
            colors,
            point_size=float(effective_scene_point_size(args)),
            visible=True,
            alpha=1.0,
        )

    label_positions = []
    label_texts = []
    label_colors = []
    for obj in graph.objects:
        rotation = sg_viz.euler_z_quaternion(viz, float(args.box_angle_sign) * obj.angle_z)
        color = np.asarray(obj.color, dtype=np.uint8)
        visualizer.add_bounding_box(
            sg_viz.object_layer_name(obj),
            position=obj.position.astype(np.float32),
            size=obj.size.astype(np.float32),
            rotation=rotation.astype(np.float32),
            color=color.astype(np.uint8),
            alpha=float(args.box_alpha),
            edge_width=float(args.box_edge_width),
            visible=True,
        )
        label_positions.append(obj.position + np.asarray([0.0, 0.0, 0.55 * float(obj.size[2]) + 0.04]))
        label_texts.append(sg_viz.object_display_text(obj, graph.scene_id))
        label_colors.append(color)

    if label_positions and not args.hide_object_labels:
        visualizer.add_labels(
            "Scene graph;Object labels",
            np.asarray(label_positions, dtype=np.float32),
            label_texts,
            np.asarray(label_colors, dtype=np.uint8),
            visible=False,
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
        for edge_index, edge in enumerate(graph.relationships):
            subject = object_by_id.get(edge.subject_id)
            target = object_by_id.get(edge.object_id)
            if subject is None or target is None:
                continue
            points = sg_viz.relationship_points(subject, target, "raised")
            if points is None:
                continue
            color = np.asarray(edge_palette[edge_index % len(edge_palette)], dtype=np.uint8)
            visualizer.add_polyline(
                "Scene graph;Relationship edges;"
                f"raised;{edge_index}_{edge.subject_id}_to_{edge.object_id};{edge.predicate}",
                points,
                color=color.astype(np.uint8),
                alpha=float(args.edge_alpha),
                edge_width=float(args.edge_width),
                visible=True,
            )

    sg_viz.save_pyviz3d_json_safe(viz, visualizer, output_path, int(args.port))
    sg_viz.patch_scene_js_to_expose_api(output_path)
    walkthrough = build_walkthrough_poses(graph, xyz, args)
    write_walkthrough_manifest(output_path, graph, args, base_meta, walkthrough)
    patch_index_html_for_walkthrough(output_path, args)
    return base_meta, walkthrough


def write_walkthrough_manifest(
    output_path: Path,
    graph: sg_viz.LoadedGraph,
    args: argparse.Namespace,
    base_meta: dict[str, Any],
    walkthrough: dict[str, Any],
) -> None:
    duration = float(args.walkthrough_duration if args.walkthrough_duration is not None else args.tour_duration)
    payload = {
        "scene_id": graph.scene_id,
        "duration": max(3.0, duration),
        "camera_fov": effective_camera_fov(args),
        "requested_camera_fov": float(args.paper_camera_fov),
        "autoplay": bool(args.autoplay),
        "base": base_meta,
        "walkthrough": walkthrough,
        "discovery": object_discovery_schedule(graph, walkthrough, args),
        "objects": sg_viz.query_object_payload(graph),
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
    (output_path / "walkthrough_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def patch_index_html_for_walkthrough(output_path: Path, args: argparse.Namespace) -> None:
    index_path = output_path / "index.html"
    if not index_path.is_file():
        return
    text = index_path.read_text(encoding="utf-8")
    if "walkthrough-shell" in text:
        return

    app_shell = """
<body class="walkthrough-body">
  <main class="walkthrough-shell">
    <div id="render_container"></div>
    <div id="walkthrough-loading" class="walkthrough-loading">Loading scene</div>
    <section class="walkthrough-panel" aria-label="Walkthrough controls">
      <div class="walkthrough-row">
        <button id="walkthrough-play" type="button" title="Play or pause walkthrough">Play</button>
        <input id="walkthrough-progress" type="range" min="0" max="1000" value="0" aria-label="Walkthrough progress">
        <button id="walkthrough-reset" type="button" title="Return to the first camera pose">Reset</button>
      </div>
      <div class="walkthrough-row is-compact">
        <label><input id="walkthrough-loop" type="checkbox" checked> Loop</label>
        <label><input id="walkthrough-scene" type="checkbox" checked> Scene</label>
        <label><input id="walkthrough-boxes" type="checkbox" checked> Boxes</label>
        <label><input id="walkthrough-edges" type="checkbox" checked> Edges</label>
        <label><input id="walkthrough-labels" type="checkbox"> Labels</label>
      </div>
      <div id="walkthrough-status" class="walkthrough-status"></div>
    </section>
  </main>
"""
    if "<title>PyViz3D</title>" in text:
        text = text.replace("<title>PyViz3D</title>", "<title>Room Walkthrough</title>", 1)
    text = text.replace(
        '<link rel="stylesheet" href="css/bootstrap.min.css">',
        '<link rel="stylesheet" href="css/bootstrap.min.css">\n\t\t<link rel="stylesheet" href="walkthrough.css">',
        1,
    )
    text = text.replace(
        '<script type="module" src="js/scene.js"></script>',
        '<script type="module" src="js/scene.js"></script>\n\t\t<script type="module" src="walkthrough.js"></script>',
        1,
    )
    if '<body>\n\t<div id="render_container"></div>' in text:
        text = text.replace('<body>\n\t<div id="render_container"></div>', app_shell, 1)
    elif "<body>" in text:
        text = text.replace("<body>", app_shell.removesuffix("</body>\n"), 1)
    (output_path / "walkthrough.css").write_text(walkthrough_css(), encoding="utf-8")
    (output_path / "walkthrough.js").write_text(walkthrough_js(), encoding="utf-8")
    index_path.write_text(text, encoding="utf-8")


def walkthrough_css() -> str:
    return r"""
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #090d10; }
.walkthrough-body { color: #f7fafc; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.walkthrough-shell { position: relative; width: 100vw; height: 100vh; background: #090d10; }
#render_container { position: absolute; inset: 0; width: 100%; height: 100%; }
.walkthrough-loading { position: absolute; left: 18px; top: 16px; z-index: 20; padding: 8px 12px; border-radius: 7px; background: rgba(9, 13, 16, 0.72); color: #fff; font-size: 12px; pointer-events: none; }
.walkthrough-loading.is-hidden { display: none; }
.walkthrough-panel { position: absolute; left: 16px; right: 16px; bottom: 16px; z-index: 30; display: grid; gap: 8px; max-width: 880px; padding: 10px; border: 1px solid rgba(255,255,255,0.14); border-radius: 8px; background: rgba(10, 15, 18, 0.78); box-shadow: 0 16px 42px rgba(0,0,0,0.28); backdrop-filter: blur(12px); }
.walkthrough-row { display: grid; grid-template-columns: 72px minmax(160px, 1fr) 72px; gap: 8px; align-items: center; }
.walkthrough-row.is-compact { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; color: rgba(247,250,252,0.86); font-size: 12px; }
.walkthrough-row.is-compact label { display: inline-flex; gap: 5px; align-items: center; white-space: nowrap; }
.walkthrough-panel button { height: 34px; border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; background: #d6fb6f; color: #101510; font-weight: 750; cursor: pointer; }
.walkthrough-panel button:active { transform: translateY(1px); }
#walkthrough-progress { width: 100%; accent-color: #d6fb6f; }
.walkthrough-status { min-height: 16px; color: rgba(247,250,252,0.74); font-size: 12px; }
canvas { outline: none; }
@media (max-width: 680px) {
  .walkthrough-panel { left: 10px; right: 10px; bottom: 10px; }
  .walkthrough-row { grid-template-columns: 64px minmax(110px, 1fr) 64px; }
}
"""


def walkthrough_js() -> str:
    return r"""
let manifest = null;
let demo = null;
let poses = [];
let running = false;
let frameId = null;
let startedAt = 0;
let progress = 0;
let discoveredObjectIds = new Set();

const byId = (id) => document.getElementById(id);

function setStatus(text) {
  const node = byId("walkthrough-status");
  if (node) node.textContent = text || "";
}

function setLoading(active) {
  const node = byId("walkthrough-loading");
  if (node) node.classList.toggle("is-hidden", !active);
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpVec(a, b, t) {
  return [
    lerp(Number(a[0]), Number(b[0]), t),
    lerp(Number(a[1]), Number(b[1]), t),
    lerp(Number(a[2]), Number(b[2]), t),
  ];
}

function poseAt(rawProgress) {
  if (!poses.length) return null;
  const p = clamp01(rawProgress);
  const scaled = p * (poses.length - 1);
  const index = Math.min(poses.length - 2, Math.max(0, Math.floor(scaled)));
  const local = smoothstep(scaled - index);
  const a = poses[index];
  const b = poses[index + 1];
  return {
    position: lerpVec(a.position, b.position, local),
    target: lerpVec(a.target, b.target, local),
    up: lerpVec(a.up || [0, 0, 1], b.up || [0, 0, 1], local),
  };
}

function render() {
  if (demo && typeof demo.render === "function") demo.render();
}

function discoveryItems() {
  return (((manifest || {}).discovery || {}).objects || []);
}

function boxObjectIdFromName(name) {
  const match = String(name || "").match(/Scene graph;Object boxes;object_(\d+);/);
  return match ? match[1] : null;
}

function rememberMaterialBase(material) {
  if (!material) return;
  if (!material.userData) material.userData = {};
  if (material.userData.walkthroughBaseOpacity === undefined) {
    material.userData.walkthroughBaseOpacity = Number.isFinite(Number(material.opacity)) ? Number(material.opacity) : 1;
  }
  if (material.uniforms && material.uniforms.alpha && material.userData.walkthroughBaseAlpha === undefined) {
    material.userData.walkthroughBaseAlpha = Number(material.uniforms.alpha.value || 1);
  }
  if (material.uniforms && material.uniforms.opacity && material.userData.walkthroughBaseUniformOpacity === undefined) {
    material.userData.walkthroughBaseUniformOpacity = Number(material.uniforms.opacity.value || 1);
  }
}

function setRootOpacityScale(root, scale, enabled) {
  if (!root) return;
  const opacityScale = clamp01(scale);
  const visible = !!enabled && opacityScale > 0.015;
  root.visible = visible;
  if (typeof root.traverse !== "function") return;
  root.traverse((child) => {
    child.visible = visible;
    if (!child.material) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((mat) => {
      if (!mat) return;
      rememberMaterialBase(mat);
      mat.transparent = true;
      if ("opacity" in mat) mat.opacity = Number(mat.userData.walkthroughBaseOpacity || 1) * opacityScale;
      if (mat.uniforms && mat.uniforms.alpha) mat.uniforms.alpha.value = Number(mat.userData.walkthroughBaseAlpha || 1) * opacityScale;
      if (mat.uniforms && mat.uniforms.opacity) mat.uniforms.opacity.value = Number(mat.userData.walkthroughBaseUniformOpacity || 1) * opacityScale;
      mat.depthWrite = opacityScale > 0.98;
      mat.needsUpdate = true;
    });
  });
}

function resetDiscoveryState(rawProgress = 0) {
  const p = clamp01(rawProgress);
  discoveredObjectIds = new Set();
  discoveryItems().forEach((item) => {
    if (Number(item.discover_at || 0) <= p) discoveredObjectIds.add(String(item.id));
  });
}

function revealFraction(item, rawProgress) {
  if (!item) return 1;
  const id = String(item.id);
  if (discoveredObjectIds.has(id)) return 1;
  const start = Number(item.discover_at || 0);
  const fade = Math.max(0.005, Number(item.fade_duration || ((manifest.discovery || {}).fade_duration || 0.045)));
  if (rawProgress < start) return 0;
  return clamp01((rawProgress - start) / fade);
}

function applyObbReveal(rawProgress, stickyDiscovery = true) {
  if (!demo || !demo.objects) return;
  const boxesEnabled = byId("walkthrough-boxes") ? byId("walkthrough-boxes").checked : true;
  if (!stickyDiscovery) resetDiscoveryState(rawProgress);

  const itemsById = new Map(discoveryItems().map((item) => [String(item.id), item]));
  discoveryItems().forEach((item) => {
    const start = Number(item.discover_at || 0);
    const fade = Math.max(0.005, Number(item.fade_duration || ((manifest.discovery || {}).fade_duration || 0.045)));
    if (rawProgress >= start + fade || start <= 0) discoveredObjectIds.add(String(item.id));
  });

  Object.entries(demo.objects).forEach(([name, object]) => {
    const id = boxObjectIdFromName(name);
    if (!id) return;
    const fraction = boxesEnabled ? revealFraction(itemsById.get(id), rawProgress) : 0;
    setRootOpacityScale(object, fraction, boxesEnabled);
  });
}

function applyPose(rawProgress, stickyDiscovery = true) {
  if (!demo || !demo.camera) return;
  const pose = poseAt(rawProgress);
  if (!pose) return;
  const camera = demo.camera;
  camera.up.set(pose.up[0], pose.up[1], pose.up[2]);
  if ("fov" in camera && manifest && Number.isFinite(Number(manifest.camera_fov))) {
    camera.fov = Number(manifest.camera_fov);
    if (typeof camera.updateProjectionMatrix === "function") camera.updateProjectionMatrix();
  }
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  if (demo.controls && demo.controls.target) {
    demo.controls.target.set(pose.target[0], pose.target[1], pose.target[2]);
    if (typeof demo.controls.update === "function") demo.controls.update();
  } else {
    camera.lookAt(pose.target[0], pose.target[1], pose.target[2]);
  }
  applyObbReveal(rawProgress, stickyDiscovery);
  render();
}

function objectNameMatches(name, kind) {
  const value = String(name || "");
  if (kind === "scene") return value.startsWith("Scene;Point cloud");
  if (kind === "boxes") return value.startsWith("Scene graph;Object boxes");
  if (kind === "edges") return value.startsWith("Scene graph;Relationship edges");
  if (kind === "labels") return value.startsWith("Scene graph;Object labels");
  return false;
}

function setRootVisible(root, visible) {
  if (!root) return;
  root.visible = visible;
  if (typeof root.traverse === "function") {
    root.traverse((child) => {
      child.visible = visible;
      if (!child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((mat) => {
        if (!mat) return;
        mat.needsUpdate = true;
      });
    });
  }
}

function applyLayerVisibility() {
  if (!demo || !demo.objects) return;
  const visible = {
    scene: byId("walkthrough-scene") ? byId("walkthrough-scene").checked : true,
    boxes: byId("walkthrough-boxes") ? byId("walkthrough-boxes").checked : true,
    edges: byId("walkthrough-edges") ? byId("walkthrough-edges").checked : true,
    labels: byId("walkthrough-labels") ? byId("walkthrough-labels").checked : false,
  };
  Object.entries(demo.objects).forEach(([name, object]) => {
    for (const [kind, active] of Object.entries(visible)) {
      if (kind === "boxes") continue;
      if (objectNameMatches(name, kind)) setRootVisible(object, active);
    }
  });
  applyObbReveal(progress, true);
  render();
}

function setProgress(value, stickyDiscovery = true) {
  progress = clamp01(value);
  const slider = byId("walkthrough-progress");
  if (slider) slider.value = String(Math.round(progress * 1000));
  applyPose(progress, stickyDiscovery);
}

function stop() {
  running = false;
  if (frameId !== null) window.cancelAnimationFrame(frameId);
  frameId = null;
  const button = byId("walkthrough-play");
  if (button) button.textContent = "Play";
  if (demo && demo.controls) demo.controls.enabled = true;
}

function animate(now) {
  if (!running || !manifest) return;
  const durationMs = Math.max(3000, Number(manifest.duration || 24) * 1000);
  const elapsed = now - startedAt;
  let next = elapsed / durationMs;
  const loop = byId("walkthrough-loop") ? byId("walkthrough-loop").checked : true;
  if (loop) {
    next = next % 1;
  } else if (next >= 1) {
    setProgress(1, true);
    stop();
    return;
  }
  progress = next;
  const slider = byId("walkthrough-progress");
  if (slider) slider.value = String(Math.round(progress * 1000));
  applyPose(progress, true);
  frameId = window.requestAnimationFrame(animate);
}

function play() {
  if (!manifest || !poses.length) return;
  running = true;
  startedAt = performance.now() - progress * Math.max(3000, Number(manifest.duration || 24) * 1000);
  const button = byId("walkthrough-play");
  if (button) button.textContent = "Pause";
  if (demo && demo.controls) demo.controls.enabled = false;
  frameId = window.requestAnimationFrame(animate);
}

function togglePlay() {
  if (running) stop();
  else play();
}

function installControls() {
  byId("walkthrough-play").addEventListener("click", togglePlay);
  byId("walkthrough-reset").addEventListener("click", () => {
    stop();
    resetDiscoveryState(0);
    setProgress(0, false);
  });
  byId("walkthrough-progress").addEventListener("input", () => {
    stop();
    setProgress(Number(byId("walkthrough-progress").value) / 1000, false);
  });
  ["walkthrough-scene", "walkthrough-boxes", "walkthrough-edges", "walkthrough-labels"].forEach((id) => {
    const input = byId(id);
    if (input) input.addEventListener("change", applyLayerVisibility);
  });
}

function setupReady() {
  if (!manifest || !window.pyviz3dDemo) return;
  demo = window.pyviz3dDemo;
  poses = (((manifest.walkthrough || {}).poses) || []).filter((pose) => pose.position && pose.target);
  resetDiscoveryState(0);
  applyLayerVisibility();
  setProgress(0, false);
  const route = (manifest.walkthrough || {}).route || {};
  setStatus(`Route ${Number(route.length || 0).toFixed(2)}m - ${poses.length} continuous poses - max step ${Number(route.max_step || 0).toFixed(3)}m`);
  setLoading(false);
  if (manifest.autoplay) play();
}

async function init() {
  installControls();
  setLoading(true);
  try {
    const response = await fetch("walkthrough_manifest.json");
    manifest = await response.json();
    setupReady();
  } catch (error) {
    setLoading(false);
    setStatus("Could not load walkthrough manifest: " + error.message);
  }
}

window.addEventListener("pyviz3d-ready", setupReady);
window.addEventListener("load", init);
"""


def build_output_path(args: argparse.Namespace, graph: sg_viz.LoadedGraph) -> Path:
    if args.output_name:
        name = args.output_name
    elif args.all_parent_splits:
        name = f"{sg_viz.parent_scan_id(graph.scene_id)}_all_splits_{DEFAULT_OUTPUT_SUFFIX}"
    else:
        name = f"{graph.scene_id}_{DEFAULT_OUTPUT_SUFFIX}"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return args.output_dir / safe_name


def summary_payload(
    graph: sg_viz.LoadedGraph,
    output_path: Path,
    base_meta: dict[str, Any],
    walkthrough: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "scene_id": graph.scene_id,
        "source_graph_json": str(graph.graph_json),
        "source_graph_jsons": [str(path) for path in graph.source_graph_jsons],
        "output_path": str(output_path),
        "base": base_meta,
        "walkthrough": {
            "duration": float(args.walkthrough_duration if args.walkthrough_duration is not None else args.tour_duration),
            "camera_fov": effective_camera_fov(args),
            "requested_camera_fov": float(args.paper_camera_fov),
            "route": walkthrough.get("route", {}),
            "bounds": walkthrough.get("bounds", {}),
        },
        "discovery": object_discovery_schedule(graph, walkthrough, args),
        "counts": {
            "objects": len(graph.objects),
            "relationships": len(graph.relationships),
            "source_graphs": len(graph.source_graph_jsons),
        },
        "warnings": graph.warnings,
    }


def main() -> None:
    args = parse_args()
    try:
        graph = (
            sg_viz.load_parent_split_graphs(args)
            if args.all_parent_splits
            else sg_viz.load_graph(args.graph_json, args.scene_id, args.box_json, args.allow_scene_id_mismatch)
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    output_path = build_output_path(args, graph)
    if args.dry_run:
        walkthrough = build_walkthrough_poses(graph, None, args)
        print(json.dumps(summary_payload(graph, output_path, {"base_mode": "dry_run"}, walkthrough, args), indent=2))
        return

    output_path.mkdir(parents=True, exist_ok=True)
    base_meta, walkthrough = render_walkthrough_pyviz3d(graph, args, output_path)
    summary = summary_payload(graph, output_path, base_meta, walkthrough, args)
    summary_path = output_path / "room_walkthrough_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote room walkthrough viewer to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print("Serve it with:")
    print(f"  cd {output_path}")
    print(f"  python -m http.server {args.port}")


if __name__ == "__main__":
    main()
