from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gaussian_io import _read_gaussian_ply


PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw-frame Chorus npys matched one-to-one to Sonata voxels."
    )
    parser.add_argument("--scene-pcd", required=True, help="SceneVerse/Sonata split PCD .ply.")
    parser.add_argument("--chorus-ply", required=True, help="Raw 3DGS/Chorus Gaussian .ply.")
    parser.add_argument(
        "--scan-dir",
        required=True,
        help="Raw 3RScan scene directory containing mesh.refined.v2.obj and semseg.v2.json.",
    )
    parser.add_argument("--output", required=True, help="Output directory for npy files.")
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--cleanup-num-nb", type=int, default=15)
    parser.add_argument("--cleanup-std-ratio", type=float, default=2.0)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument(
        "--match-radius-voxels",
        type=int,
        default=2,
        help="Chebyshev radius in raw bridge voxel coordinates.",
    )
    parser.add_argument(
        "--missing-voxel-policy",
        choices=("nearest", "error", "drop"),
        default="nearest",
    )
    parser.add_argument(
        "--align-angle",
        type=float,
        default=None,
        help="Optional SceneVerse align angle in degrees. If omitted, recomputed from semseg OBBs.",
    )
    parser.add_argument(
        "--align-angle-path",
        type=Path,
        default=None,
        help="Optional .npy file containing the SceneVerse align angle.",
    )
    return parser.parse_args()


def rotate_z_axis_by_degrees(points: np.ndarray, theta: float, clockwise: bool = True) -> np.ndarray:
    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    rot = np.asarray(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=points.dtype,
    )
    if not clockwise:
        rot = rot.T
    return points @ rot


def z_rotation_matrix(theta: float, dtype=np.float32) -> np.ndarray:
    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.asarray(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def row_view(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    quat = quat / norm
    sign = np.sign(quat[:, :1])
    sign[sign == 0] = 1.0
    return quat * sign


def neighborhood_offsets(radius: int) -> list[tuple[int, int, int]]:
    return [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]


def build_grid_to_indices(grid: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    mapping: dict[tuple[int, int, int], list[int]] = {}
    for idx, row in enumerate(grid.tolist()):
        key = tuple(int(v) for v in row)
        mapping.setdefault(key, []).append(idx)
    return mapping


def read_ply_xyz(path: Path) -> np.ndarray:
    """Read vertex xyz from ASCII or binary_little_endian PLY without Open3D."""
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            text = line.decode("utf-8", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        data_offset = handle.tell()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    fmt = None
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError(f"List vertex properties are not supported for {path}")
            vertex_props.append((parts[2], parts[1]))

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format {fmt} in {path}")
    if vertex_count is None:
        raise ValueError(f"Missing vertex count in {path}")
    prop_names = [name for name, _ in vertex_props]
    for required in ("x", "y", "z"):
        if required not in prop_names:
            raise ValueError(f"Missing vertex property {required} in {path}")

    if fmt == "ascii":
        xyz = np.empty((vertex_count, 3), dtype=np.float32)
        x_i, y_i, z_i = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip() == "end_header":
                    break
            for idx in range(vertex_count):
                values = handle.readline().split()
                xyz[idx] = [float(values[x_i]), float(values[y_i]), float(values[z_i])]
        return xyz

    dtype = np.dtype([(name, PLY_TYPE_TO_DTYPE[prop_type]) for name, prop_type in vertex_props])
    with path.open("rb") as handle:
        handle.seek(data_offset)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def positive_shift_grid(coord: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    origin = coord.min(axis=0)
    grid = np.floor((coord - origin.reshape(1, 3)) / voxel_size).astype(np.int64)
    return origin.astype(np.float32), grid


def pointcept_center_shift_grid(coord: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror Pointcept CenterShift(apply_z=True) + GridSample grid normalization."""
    coord = coord.astype(np.float32, copy=False)
    coord_min = coord.min(axis=0)
    coord_max = coord.max(axis=0)
    shift = np.asarray(
        [
            (coord_min[0] + coord_max[0]) / 2.0,
            (coord_min[1] + coord_max[1]) / 2.0,
            coord_min[2],
        ],
        dtype=np.float32,
    )
    shifted = coord - shift.reshape(1, 3)
    grid_before_norm = np.floor(shifted / voxel_size).astype(np.int64)
    grid_min = grid_before_norm.min(axis=0)
    grid = grid_before_norm - grid_min.reshape(1, 3)
    return shift, grid_min.astype(np.int64), grid


def voxel_downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    _, grid = positive_shift_grid(points, voxel_size)
    _, first = np.unique(row_view(grid), return_index=True)
    return points[np.sort(first)]


def statistical_outlier_filter(
    points: np.ndarray,
    num_nb: int,
    std_ratio: float,
) -> tuple[np.ndarray, str]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return points, "skipped_no_scipy"

    if points.shape[0] <= max(num_nb, 1):
        return points, "skipped_too_few_points"
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(num_nb + 1, points.shape[0]), workers=-1)
    mean_dist = distances[:, 1:].mean(axis=1)
    threshold = float(mean_dist.mean() + std_ratio * mean_dist.std())
    keep = mean_dist <= threshold
    if not np.any(keep):
        return points, "skipped_filter_removed_all"
    return points[keep], "applied"


def build_sonata_support(
    scene_pcd_path: Path,
    voxel_size: float,
    do_cleanup: bool,
    cleanup_num_nb: int,
    cleanup_std_ratio: float,
):
    points = read_ply_xyz(scene_pcd_path)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError(f"No valid points in {scene_pcd_path}")

    cleanup_status = "disabled"
    points_before_cleanup = int(points.shape[0])
    if do_cleanup:
        points = voxel_downsample_points(points, voxel_size)
        points, cleanup_status = statistical_outlier_filter(
            points,
            num_nb=cleanup_num_nb,
            std_ratio=cleanup_std_ratio,
        )

    origin, grid = positive_shift_grid(points, voxel_size)
    unique_grid = np.unique(grid, axis=0)
    diagnostics = {
        "scene_points": int(points.shape[0]),
        "scene_points_before_cleanup": points_before_cleanup,
        "cleanup_status": cleanup_status,
        "scene_min": points.min(axis=0).tolist(),
        "scene_max": points.max(axis=0).tolist(),
        "positive_shift_origin": origin.tolist(),
        "sonata_grid_min": unique_grid.min(axis=0).tolist(),
        "sonata_grid_max": unique_grid.max(axis=0).tolist(),
    }
    return origin.astype(np.float32), unique_grid, diagnostics, points


def compute_box_3d(size: list[float], center: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    half_x, half_y, half_z = [float(v) / 2.0 for v in size]
    center = np.asarray(center, dtype=np.float64).reshape(3)
    x_corners = [half_x, half_x, -half_x, -half_x, half_x, half_x, -half_x, -half_x]
    y_corners = [half_y, -half_y, -half_y, half_y, half_y, -half_y, -half_y, half_y]
    z_corners = [half_z, half_z, half_z, half_z, -half_z, -half_z, -half_z, -half_z]
    corners = rotmat.T @ np.vstack([x_corners, y_corners, z_corners])
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    return corners.T


def is_axis_aligned(rotated_box: np.ndarray, threshold: float = 0.05) -> bool:
    x_diff = abs(rotated_box[0][0] - rotated_box[1][0])
    y_diff = abs(rotated_box[0][1] - rotated_box[3][1])
    return bool(x_diff < threshold and y_diff < threshold)


def calc_align_matrix(bbox_list: list[np.ndarray]) -> float:
    for lo, hi, num_bins, threshold in ((-45, 45, 90, 0.05), (-90, 90, 180, 0.15)):
        angle_counts: dict[float, int] = {}
        for angle in np.linspace(lo, hi, num_bins):
            bucket = round(float(angle), 3)
            for box in bbox_list:
                bottom = rotate_z_axis_by_degrees(box, bucket)[4:]
                if is_axis_aligned(bottom, threshold=threshold):
                    angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
        if angle_counts:
            return max(angle_counts, key=angle_counts.get)
    raise RuntimeError("Could not compute SceneVerse align angle from semseg OBBs.")


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32)


def load_semseg_boxes(path: Path) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        semseg = json.load(handle)
    boxes: list[np.ndarray] = []
    for group in semseg.get("segGroups", []):
        obb = group.get("obb") or {}
        if not {"normalizedAxes", "centroid", "axesLengths"}.issubset(obb):
            continue
        rotation = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
        center = np.asarray(obb["centroid"], dtype=np.float64).reshape(3)
        scale = np.asarray(obb["axesLengths"], dtype=np.float64).reshape(3)
        boxes.append(compute_box_3d(scale.tolist(), center, rotation))
    if not boxes:
        raise ValueError(f"No OBBs found in {path}")
    return boxes


def resolve_align_angle(scan_dir: Path, explicit_angle: float | None, angle_path: Path | None) -> tuple[float, str]:
    if explicit_angle is not None:
        return float(explicit_angle), "explicit"
    if angle_path is not None:
        return float(np.load(angle_path).reshape(-1)[0]), str(angle_path)
    semseg_path = scan_dir / "semseg.v2.json"
    return calc_align_matrix(load_semseg_boxes(semseg_path)), str(semseg_path)


def sceneverse_transform_from_raw(scan_dir: Path, align_angle: float) -> dict[str, Any]:
    raw_vertices = read_obj_vertices(scan_dir / "mesh.refined.v2.obj")
    rotated = rotate_z_axis_by_degrees(raw_vertices, align_angle)
    center = rotated.mean(axis=0)
    center[2] = rotated[:, 2].min()
    rot = z_rotation_matrix(align_angle, dtype=np.float32)
    return {
        "align_angle_degrees": float(align_angle),
        "rotation_row_major": rot.tolist(),
        "center_after_rotation": center.astype(np.float32).tolist(),
        "raw_mesh_min": raw_vertices.min(axis=0).tolist(),
        "raw_mesh_max": raw_vertices.max(axis=0).tolist(),
        "sceneverse_mesh_min": (rotated - center).min(axis=0).tolist(),
        "sceneverse_mesh_max": (rotated - center).max(axis=0).tolist(),
    }


def sceneverse_to_raw(points_sv: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rot = np.asarray(transform["rotation_row_major"], dtype=np.float32)
    center = np.asarray(transform["center_after_rotation"], dtype=np.float32)
    return (points_sv.astype(np.float32, copy=False) + center.reshape(1, 3)) @ rot.T


def grid_from_origin(coord: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((coord - origin.reshape(1, 3)) / voxel_size).astype(np.int64)


def unique_overlap_stats(reference_grid: np.ndarray, candidate_grid: np.ndarray) -> dict[str, Any]:
    ref_unique = np.unique(reference_grid, axis=0)
    cand_unique = np.unique(candidate_grid, axis=0)
    ref_view = row_view(ref_unique)
    cand_view = row_view(cand_unique)
    overlap = int(np.isin(ref_view, cand_view).sum())
    union = int(ref_unique.shape[0] + cand_unique.shape[0] - overlap)
    return {
        "reference_unique": int(ref_unique.shape[0]),
        "candidate_unique": int(cand_unique.shape[0]),
        "overlap_unique": overlap,
        "missing_reference": int(ref_unique.shape[0] - overlap),
        "coverage": float(overlap / max(ref_unique.shape[0], 1)),
        "jaccard": float(overlap / max(union, 1)),
        "reference_grid_min": ref_unique.min(axis=0).tolist() if ref_unique.size else None,
        "reference_grid_max": ref_unique.max(axis=0).tolist() if ref_unique.size else None,
        "candidate_grid_min": cand_unique.min(axis=0).tolist() if cand_unique.size else None,
        "candidate_grid_max": cand_unique.max(axis=0).tolist() if cand_unique.size else None,
    }


class NearestGridLookup:
    def __init__(self, unique_grid: np.ndarray):
        self.unique_grid = unique_grid.astype(np.int64, copy=False)
        self.grid_set = {tuple(int(v) for v in row) for row in self.unique_grid.tolist()}
        self.tree = None
        try:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.unique_grid.astype(np.float32))
        except ImportError:
            self.tree = None

    def query(self, key: tuple[int, int, int]) -> tuple[np.ndarray, float, float]:
        query = np.asarray(key, dtype=np.int64)
        if self.tree is not None:
            distance, index = self.tree.query(query.astype(np.float32), k=1, workers=-1)
            nearest = self.unique_grid[int(index)]
            return nearest, float(distance), float(np.max(np.abs(nearest - query)))

        # Pure-NumPy fallback: expand Chebyshev shells until any occupied voxel is found.
        # This is slower than scipy but avoids importing Open3D/IPython on restricted clusters.
        for radius in range(1, 1025):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                        if neighbor in self.grid_set:
                            candidates.append(neighbor)
            if candidates:
                arr = np.asarray(candidates, dtype=np.int64)
                distances = np.sum((arr - query.reshape(1, 3)) ** 2, axis=1)
                nearest = arr[int(np.argmin(distances))]
                return nearest, float(math.sqrt(float(distances.min()))), float(radius)
        raise RuntimeError("Could not find nearest Chorus voxel within fallback search radius.")


def make_output_scene(
    chorus_ply_path: Path,
    sonata_grid_sv: np.ndarray,
    sonata_centers_raw: np.ndarray,
    voxel_size: float,
    match_radius_voxels: int,
    missing_voxel_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    chorus_data, source_type = _read_gaussian_ply(chorus_ply_path)
    chorus_coord_raw = np.asarray(chorus_data["coord"], dtype=np.float32)
    raw_origin = sonata_centers_raw.min(axis=0).astype(np.float32)
    sonata_grid_raw = grid_from_origin(sonata_centers_raw, raw_origin, voxel_size)
    chorus_grid_raw = grid_from_origin(chorus_coord_raw, raw_origin, voxel_size)
    raw_chorus_shift, raw_chorus_grid_min, sonata_grid_raw_chorus = pointcept_center_shift_grid(
        sonata_centers_raw,
        voxel_size,
    )

    grid_to_indices = build_grid_to_indices(chorus_grid_raw)
    offsets = neighborhood_offsets(match_radius_voxels)
    nearest_unique_grid = None
    nearest_tree = None
    if missing_voxel_policy == "nearest":
        nearest_unique_grid = np.unique(chorus_grid_raw, axis=0)
        nearest_tree = NearestGridLookup(nearest_unique_grid)

    grouped_indices: list[np.ndarray] = []
    missing_mask = np.zeros((sonata_grid_raw.shape[0],), dtype=bool)
    nearest_l2: list[float] = []
    nearest_linf: list[float] = []
    for idx, key_row in enumerate(sonata_grid_raw.tolist()):
        key = tuple(int(v) for v in key_row)
        matches: list[int] = []
        for dx, dy, dz in offsets:
            neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
            if neighbor in grid_to_indices:
                matches.extend(grid_to_indices[neighbor])
        if matches:
            grouped_indices.append(np.asarray(matches, dtype=np.int64))
            continue

        missing_mask[idx] = True
        if missing_voxel_policy == "nearest":
            assert nearest_unique_grid is not None and nearest_tree is not None
            nearest_key_array, nearest_dist_l2, nearest_dist_linf = nearest_tree.query(key)
            nearest_key = tuple(int(v) for v in nearest_key_array.tolist())
            grouped_indices.append(np.asarray(grid_to_indices[nearest_key], dtype=np.int64))
            nearest_l2.append(nearest_dist_l2)
            nearest_linf.append(nearest_dist_linf)
        else:
            grouped_indices.append(np.empty((0,), dtype=np.int64))

    missing_count = int(missing_mask.sum())
    if missing_count and missing_voxel_policy == "error":
        stats = unique_overlap_stats(sonata_grid_raw, chorus_grid_raw)
        raise RuntimeError(
            f"{missing_count} Sonata voxels had no raw-frame Chorus splats. "
            f"Coverage={stats['coverage']:.4f}; rerun with --missing-voxel-policy nearest/drop."
        )

    dropped_mask = missing_mask if missing_voxel_policy == "drop" else np.zeros_like(missing_mask)
    target_indices = np.flatnonzero(~dropped_mask)
    target_grouped = [grouped_indices[int(i)] for i in target_indices]

    chorus_color = np.asarray(chorus_data["color"], dtype=np.float32)
    chorus_opacity = np.asarray(chorus_data["opacity"], dtype=np.float32).reshape(-1)
    chorus_scale = np.asarray(chorus_data["scale"], dtype=np.float32)
    chorus_quat = np.asarray(chorus_data["quat"], dtype=np.float32)

    output_coord = sonata_centers_raw[target_indices].astype(np.float32, copy=True)
    output_color = np.empty((target_indices.shape[0], 3), dtype=np.float32)
    output_opacity = np.empty((target_indices.shape[0],), dtype=np.float32)
    output_scale = np.empty((target_indices.shape[0], 3), dtype=np.float32)
    output_quat = np.empty((target_indices.shape[0], 4), dtype=np.float32)
    representative_indices = np.empty((target_indices.shape[0],), dtype=np.int64)

    for out_idx, (sonata_idx, splat_indices) in enumerate(zip(target_indices, target_grouped)):
        if splat_indices.size == 0:
            raise RuntimeError("Internal error: empty splat group survived output filtering.")
        target_xyz = sonata_centers_raw[int(sonata_idx)]
        splat_xyz = chorus_coord_raw[splat_indices]
        nearest_local = int(np.argmin(np.sum((splat_xyz - target_xyz.reshape(1, 3)) ** 2, axis=1)))
        nearest_idx = int(splat_indices[nearest_local])
        output_color[out_idx] = chorus_color[splat_indices].mean(axis=0)
        output_opacity[out_idx] = chorus_opacity[splat_indices].mean(axis=0)
        output_scale[out_idx] = chorus_scale[splat_indices].mean(axis=0)
        output_quat[out_idx] = chorus_quat[nearest_idx]
        representative_indices[out_idx] = nearest_idx

    output = {
        "coord": output_coord,
        "color": output_color,
        "opacity": output_opacity,
        "scale": output_scale,
        "quat": normalize_quat(output_quat),
        "sonata_grid": sonata_grid_sv[target_indices].astype(np.int64, copy=False),
        "sonata_grid_raw_bridge": sonata_grid_raw[target_indices].astype(np.int64, copy=False),
        "sonata_grid_raw_chorus": sonata_grid_raw_chorus[target_indices].astype(np.int64, copy=False),
        "representative_indices": representative_indices,
    }
    diagnostics = {
        "source_type": source_type,
        "chorus_points": int(chorus_coord_raw.shape[0]),
        "chorus_raw_min": chorus_coord_raw.min(axis=0).tolist(),
        "chorus_raw_max": chorus_coord_raw.max(axis=0).tolist(),
        "raw_bridge_origin": raw_origin.tolist(),
        "raw_bridge_overlap": unique_overlap_stats(sonata_grid_raw, chorus_grid_raw),
        "raw_chorus_center_shift": raw_chorus_shift.tolist(),
        "raw_chorus_grid_min_before_norm": raw_chorus_grid_min.tolist(),
        "raw_chorus_grid_min": sonata_grid_raw_chorus.min(axis=0).tolist(),
        "raw_chorus_grid_max": sonata_grid_raw_chorus.max(axis=0).tolist(),
        "match_radius_voxels": int(match_radius_voxels),
        "missing_voxel_policy": missing_voxel_policy,
        "missing_voxels": missing_count,
        "filled_missing_voxels": missing_count if missing_voxel_policy == "nearest" else 0,
        "dropped_missing_voxels": int(dropped_mask.sum()),
        "matched_chorus_splats": int(sum(int(indices.shape[0]) for indices in target_grouped)),
        "nearest_fill_grid_l2_mean": float(np.mean(nearest_l2)) if nearest_l2 else None,
        "nearest_fill_grid_l2_max": float(np.max(nearest_l2)) if nearest_l2 else None,
        "nearest_fill_grid_linf_mean": float(np.mean(nearest_linf)) if nearest_linf else None,
        "nearest_fill_grid_linf_max": float(np.max(nearest_linf)) if nearest_linf else None,
    }
    return output, diagnostics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    scene_pcd_path = Path(args.scene_pcd)
    chorus_ply_path = Path(args.chorus_ply)
    scan_dir = Path(args.scan_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    align_angle, align_angle_source = resolve_align_angle(
        scan_dir=scan_dir,
        explicit_angle=args.align_angle,
        angle_path=args.align_angle_path,
    )
    transform = sceneverse_transform_from_raw(scan_dir, align_angle)
    transform["align_angle_source"] = align_angle_source

    sonata_origin_sv, sonata_grid_sv, sonata_diag, _ = build_sonata_support(
        scene_pcd_path=scene_pcd_path,
        voxel_size=args.voxel_size,
        do_cleanup=not args.no_cleanup,
        cleanup_num_nb=args.cleanup_num_nb,
        cleanup_std_ratio=args.cleanup_std_ratio,
    )
    sonata_centers_sv = sonata_origin_sv.reshape(1, 3) + (
        sonata_grid_sv.astype(np.float32) + 0.5
    ) * np.float32(args.voxel_size)
    sonata_centers_raw = sceneverse_to_raw(sonata_centers_sv, transform)

    output, chorus_diag = make_output_scene(
        chorus_ply_path=chorus_ply_path,
        sonata_grid_sv=sonata_grid_sv,
        sonata_centers_raw=sonata_centers_raw,
        voxel_size=args.voxel_size,
        match_radius_voxels=args.match_radius_voxels,
        missing_voxel_policy=args.missing_voxel_policy,
    )

    np.save(output_dir / "coord.npy", output["coord"])
    np.save(output_dir / "color.npy", output["color"])
    np.save(output_dir / "opacity.npy", output["opacity"])
    np.save(output_dir / "scale.npy", output["scale"])
    np.save(output_dir / "quat.npy", output["quat"])
    np.save(output_dir / "sonata_grid.npy", output["sonata_grid"])
    np.save(output_dir / "sonata_grid_raw_bridge.npy", output["sonata_grid_raw_bridge"])
    np.save(output_dir / "sonata_grid_raw_chorus.npy", output["sonata_grid_raw_chorus"])
    np.save(output_dir / "representative_chorus_indices.npy", output["representative_indices"])
    np.save(output_dir / "sonata_origin.npy", sonata_origin_sv)
    np.save(output_dir / "sonata_centers_raw.npy", sonata_centers_raw)

    summary = {
        "scene_pcd": str(scene_pcd_path),
        "chorus_ply": str(chorus_ply_path),
        "scan_dir": str(scan_dir),
        "voxel_size": args.voxel_size,
        "cleanup_applied": not args.no_cleanup,
        "match_radius_voxels": args.match_radius_voxels,
        "missing_voxel_policy": args.missing_voxel_policy,
        "sonata_voxels": int(sonata_grid_sv.shape[0]),
        "output_voxels": int(output["coord"].shape[0]),
        "missing_voxels": chorus_diag["missing_voxels"],
        "filled_missing_voxels": chorus_diag["filled_missing_voxels"],
        "dropped_missing_voxels": chorus_diag["dropped_missing_voxels"],
        "matched_chorus_splats": chorus_diag["matched_chorus_splats"],
        "chorus_ply_type": chorus_diag["source_type"],
        "matching_frame": "raw_3rscan_bridge",
    }
    diagnostics = {
        "summary": summary,
        "sceneverse_to_raw": transform,
        "sonata_sceneverse": sonata_diag,
        "sonata_raw_centers": {
            "min": sonata_centers_raw.min(axis=0).tolist(),
            "max": sonata_centers_raw.max(axis=0).tolist(),
        },
        "chorus": chorus_diag,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "match_diagnostics.json", diagnostics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
