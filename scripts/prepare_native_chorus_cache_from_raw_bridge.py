from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


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
SH_C0 = 0.28209479177387814
EPS = 1e-9
INVALID_SONATA_GRID = (-1, -1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached native raw-3DGS Chorus inputs from raw-bridge sidecars."
    )
    parser.add_argument(
        "--raw-bridge-root",
        required=True,
        type=Path,
        help="Existing raw-bridge root with split/scene sidecar dirs.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root for cached native-Chorus prepared arrays.",
    )
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to process.")
    parser.add_argument(
        "--native-match-radius",
        type=int,
        default=-1,
        help="Raw-bridge voxel radius; -1 uses summary.json match_radius_voxels.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--crop-mode",
        choices=("none", "bbox", "radius"),
        default="none",
        help=(
            "Optional native-splat crop before writing. 'bbox' keeps splats inside "
            "the raw split Sonata center bounds expanded by --crop-margin-meters. "
            "'radius' keeps splats within --crop-margin-meters of a split Sonata "
            "raw voxel, then the prep step unions the crop with all labeled splats."
        ),
    )
    parser.add_argument(
        "--crop-margin-meters",
        type=float,
        default=0.0,
        help="Margin in meters for cropped modes. Ignored when crop-mode=none.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Optional scene id to process; can be repeated. Defaults to all scenes in selected splits.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def np_sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def sorted_prefixed_names(names: tuple[str, ...], prefix: str) -> list[str]:
    prefixed = []
    for name in names:
        if not name.startswith(prefix):
            continue
        try:
            suffix = int(name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        prefixed.append((suffix, name))
    return [name for _, name in sorted(prefixed)]


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + EPS)
    sign = np.sign(quat[:, :1])
    sign[sign == 0] = 1.0
    return quat * sign


def read_ply_vertex_data(path: Path) -> np.ndarray:
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
    active_element = None
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            active_element = parts[1]
            if active_element == "vertex":
                vertex_count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and active_element == "vertex":
            if parts[1] == "list":
                raise ValueError(f"List vertex properties are not supported for {path}")
            prop_type = parts[1]
            prop_name = parts[2]
            if prop_type not in PLY_TYPE_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type {prop_type!r} in {path}")
            vertex_props.append((prop_name, prop_type))

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format {fmt!r} in {path}")
    if vertex_count is None:
        raise ValueError(f"Missing vertex count in {path}")

    dtype = np.dtype(
        [(name, np.dtype(PLY_TYPE_TO_DTYPE[prop_type])) for name, prop_type in vertex_props]
    )
    if fmt == "binary_little_endian":
        return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))

    data = np.empty(vertex_count, dtype=dtype)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        for idx in range(vertex_count):
            values = handle.readline().split()
            if len(values) < len(vertex_props):
                raise ValueError(f"Unexpected short ASCII PLY row {idx} in {path}")
            for prop_idx, (name, _) in enumerate(vertex_props):
                data[name][idx] = values[prop_idx]
    return data


def read_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    vertex = read_ply_vertex_data(path)
    names = vertex.dtype.names or ()
    if "packed_position" in names:
        raise ValueError(f"Compressed Gaussian PLY is not supported by this cache builder: {path}")
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"Missing vertex property {required!r} in {path}")

    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    if "opacity" in names:
        opacity = np_sigmoid(vertex["opacity"].astype(np.float32)).astype(np.float32)
    else:
        opacity = np.ones((coord.shape[0],), dtype=np.float32)

    scale_cols = sorted_prefixed_names(names, "scale_")
    if len(scale_cols) >= 3:
        scale = np.stack(
            [np.exp(vertex[name].astype(np.float32)) for name in scale_cols[:3]],
            axis=-1,
        ).astype(np.float32)
    else:
        scale = np.ones((coord.shape[0], 3), dtype=np.float32)

    rot_cols = sorted_prefixed_names(names, "rot_")
    if len(rot_cols) >= 4:
        quat = np.stack([vertex[name] for name in rot_cols[:4]], axis=-1).astype(np.float32)
        quat = normalize_quat(quat)
    else:
        quat = np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (coord.shape[0], 1),
        )

    dc_cols = sorted_prefixed_names(names, "f_dc_")
    if len(dc_cols) >= 3:
        fdc = np.stack([vertex[name] for name in dc_cols[:3]], axis=-1).astype(np.float32)
        color = np.clip(fdc * SH_C0 + 0.5, 0.0, 1.0) * 255.0
        color = color.astype(np.uint8)
    elif all(name in names for name in ("red", "green", "blue")):
        color = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1).astype(
            np.uint8
        )
    else:
        color = np.full((coord.shape[0], 3), 128, dtype=np.uint8)

    return dict(coord=coord, color=color, opacity=opacity, scale=scale, quat=quat)


def bridge_grid(coord: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((coord - origin.reshape(1, 3)) / voxel_size).astype(np.int64)


def neighborhood_offsets(radius: int) -> np.ndarray:
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]
    offsets.sort(
        key=lambda row: (
            row[0] * row[0] + row[1] * row[1] + row[2] * row[2],
            max(abs(v) for v in row),
            row,
        )
    )
    return np.asarray(offsets, dtype=np.int64)


def encode_grid_rows(rows: np.ndarray, min_key: np.ndarray, dims: np.ndarray) -> np.ndarray:
    shifted = rows.astype(np.int64, copy=False) - min_key.reshape(1, 3)
    return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]


def assign_native_sonata_grid(
    raw_coord: np.ndarray,
    sonata_grid_raw_bridge: np.ndarray,
    sonata_grid: np.ndarray,
    sonata_centers_raw: np.ndarray,
    voxel_size: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_origin = sonata_centers_raw.min(axis=0).astype(np.float32)
    raw_grid = bridge_grid(raw_coord, raw_origin, voxel_size)
    sonata_raw = np.asarray(sonata_grid_raw_bridge, dtype=np.int64)
    sonata_grid = np.asarray(sonata_grid, dtype=np.int64)
    if sonata_raw.shape[0] != sonata_grid.shape[0]:
        raise ValueError(
            "sonata_grid_raw_bridge and sonata_grid row counts differ: "
            f"{sonata_raw.shape[0]} != {sonata_grid.shape[0]}"
        )

    offsets = neighborhood_offsets(max(int(radius), 0))
    candidate_grid = (sonata_raw[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    offset_sq = np.tile(np.sum(offsets * offsets, axis=1), sonata_raw.shape[0])
    offset_linf = np.tile(np.max(np.abs(offsets), axis=1), sonata_raw.shape[0])
    candidate_labels = np.repeat(sonata_grid, offsets.shape[0], axis=0)

    min_key = np.minimum(raw_grid.min(axis=0), candidate_grid.min(axis=0))
    max_key = np.maximum(raw_grid.max(axis=0), candidate_grid.max(axis=0))
    dims = np.maximum(max_key - min_key + 1, 1).astype(np.int64)
    candidate_keys = encode_grid_rows(candidate_grid, min_key, dims)
    raw_keys = encode_grid_rows(raw_grid, min_key, dims)

    order = np.lexsort(
        (
            candidate_labels[:, 2],
            candidate_labels[:, 1],
            candidate_labels[:, 0],
            offset_linf,
            offset_sq,
            candidate_keys,
        )
    )
    sorted_keys = candidate_keys[order]
    first_for_key = np.concatenate(
        [np.asarray([True]), sorted_keys[1:] != sorted_keys[:-1]]
    )
    lookup_keys = sorted_keys[first_for_key]
    lookup_labels = candidate_labels[order][first_for_key]

    positions = np.searchsorted(lookup_keys, raw_keys)
    valid = positions < lookup_keys.shape[0]
    valid[valid] = lookup_keys[positions[valid]] == raw_keys[valid]
    assigned_labels = np.full(
        (raw_coord.shape[0], 3),
        INVALID_SONATA_GRID,
        dtype=np.int64,
    )
    assigned_labels[valid] = lookup_labels[positions[valid]]
    raw_indices = np.arange(raw_coord.shape[0], dtype=np.int64)
    return valid, assigned_labels, raw_indices


def make_crop_mask(
    raw_coord: np.ndarray,
    sonata_grid_raw_bridge: np.ndarray,
    sonata_centers_raw: np.ndarray,
    voxel_size: float,
    crop_mode: str,
    crop_margin_meters: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_count = int(raw_coord.shape[0])
    if crop_mode == "none":
        return (
            np.ones((raw_count,), dtype=bool),
            {
                "crop_mode": "none",
                "crop_margin_meters": 0.0,
                "crop_bounds_min": None,
                "crop_bounds_max": None,
            },
        )

    if crop_mode != "bbox":
        if crop_mode == "radius":
            return make_radius_crop_mask(
                raw_coord,
                sonata_grid_raw_bridge,
                sonata_centers_raw,
                voxel_size,
                crop_margin_meters,
            )
        raise ValueError(f"Unsupported crop mode: {crop_mode}")
    margin = max(float(crop_margin_meters), 0.0)
    bounds_min = sonata_centers_raw.min(axis=0).astype(np.float32) - margin
    bounds_max = sonata_centers_raw.max(axis=0).astype(np.float32) + margin
    mask = np.all(
        (raw_coord >= bounds_min.reshape(1, 3))
        & (raw_coord <= bounds_max.reshape(1, 3)),
        axis=1,
    )
    return (
        mask,
        {
            "crop_mode": "bbox",
            "crop_margin_meters": margin,
            "crop_bounds_min": bounds_min.tolist(),
            "crop_bounds_max": bounds_max.tolist(),
        },
    )


def make_radius_crop_mask(
    raw_coord: np.ndarray,
    sonata_grid_raw_bridge: np.ndarray,
    sonata_centers_raw: np.ndarray,
    voxel_size: float,
    crop_margin_meters: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    margin = max(float(crop_margin_meters), 0.0)
    radius_voxels = int(np.ceil(margin / max(float(voxel_size), EPS)))
    raw_origin = sonata_centers_raw.min(axis=0).astype(np.float32)
    raw_grid = bridge_grid(raw_coord, raw_origin, voxel_size)
    unique_raw_grid, inverse = np.unique(raw_grid, axis=0, return_inverse=True)
    sonata_raw = np.unique(np.asarray(sonata_grid_raw_bridge, dtype=np.int64), axis=0)

    bucket_size = max(radius_voxels + 1, 1)
    sonata_buckets = np.floor_divide(sonata_raw, bucket_size)
    bucket_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for idx, bucket in enumerate(sonata_buckets.tolist()):
        bucket_to_indices.setdefault(tuple(int(value) for value in bucket), []).append(idx)

    neighbor_offsets = neighborhood_offsets(1)
    radius_sq = radius_voxels * radius_voxels
    keep_unique = np.zeros((unique_raw_grid.shape[0],), dtype=bool)
    for idx, row in enumerate(unique_raw_grid):
        bucket = np.floor_divide(row, bucket_size)
        candidate_indices: list[int] = []
        for offset in neighbor_offsets:
            key = tuple(int(value) for value in (bucket + offset))
            matches = bucket_to_indices.get(key)
            if matches:
                candidate_indices.extend(matches)
        if not candidate_indices:
            continue
        candidates = sonata_raw[np.asarray(candidate_indices, dtype=np.int64)]
        diff = candidates - row.reshape(1, 3)
        if np.any(np.sum(diff * diff, axis=1) <= radius_sq):
            keep_unique[idx] = True

    mask = keep_unique[inverse]
    return (
        mask,
        {
            "crop_mode": "radius",
            "crop_margin_meters": margin,
            "crop_margin_voxels": radius_voxels,
            "crop_metric": "euclidean_grid_distance",
            "crop_bounds_min": None,
            "crop_bounds_max": None,
        },
    )


def is_compatible_existing_summary(
    summary: dict[str, Any],
    crop_mode: str,
    crop_margin_meters: float,
    native_match_radius: int,
) -> bool:
    if summary.get("invalid_label_value") != list(INVALID_SONATA_GRID):
        return False
    existing_crop_mode = str(summary.get("crop_mode", "none"))
    if existing_crop_mode != crop_mode:
        return False
    if (
        abs(
            float(summary.get("crop_margin_meters", 0.0))
            - max(float(crop_margin_meters), 0.0)
        )
        > 1e-6
    ):
        return False
    if native_match_radius >= 0 and int(
        summary.get("native_match_radius", native_match_radius)
    ) != int(native_match_radius):
        return False
    return True


def prepare_scene(
    split: str,
    scene_dir: Path,
    output_root: Path,
    native_match_radius: int,
    crop_mode: str,
    crop_margin_meters: float,
    overwrite: bool,
) -> dict[str, Any]:
    scene_id = scene_dir.name
    out_dir = output_root / split / scene_id
    done_path = out_dir / "summary.json"
    replace_existing = False
    if done_path.exists() and (out_dir / "coord.npy").exists() and not overwrite:
        existing_summary = load_json(done_path)
        if is_compatible_existing_summary(
            existing_summary,
            crop_mode=crop_mode,
            crop_margin_meters=crop_margin_meters,
            native_match_radius=native_match_radius,
        ):
            return dict(
                scene_id=scene_id,
                split=split,
                status="skipped",
                raw_splats=int(existing_summary.get("raw_splats", 0)),
                cached_splats=int(existing_summary.get("cached_splats", 0)),
                kept_splats=int(
                    existing_summary.get(
                        "kept_splats",
                        existing_summary.get("cached_splats", 0),
                    )
                ),
                kept_fraction=float(existing_summary.get("kept_fraction", 0.0)),
                valid_labeled_splats=int(existing_summary.get("valid_labeled_splats", 0)),
                valid_labeled_splats_before_crop=int(
                    existing_summary.get(
                        "valid_labeled_splats_before_crop",
                        existing_summary.get("valid_labeled_splats", 0),
                    )
                ),
                valid_label_fraction=float(existing_summary.get("valid_label_fraction", 0.0)),
                valid_label_fraction_raw=float(
                    existing_summary.get(
                        "valid_label_fraction_raw",
                        existing_summary.get("valid_label_fraction", 0.0),
                    )
                ),
                invalid_context_splats=int(existing_summary.get("invalid_context_splats", 0)),
                invalid_context_fraction=float(existing_summary.get("invalid_context_fraction", 0.0)),
                crop_mode=existing_summary.get("crop_mode", "none"),
                crop_margin_meters=float(existing_summary.get("crop_margin_meters", 0.0)),
                radius=int(existing_summary.get("native_match_radius", native_match_radius)),
            )
        return dict(
            scene_id=scene_id,
            split=split,
            status="exists_mismatch",
            existing_crop_mode=existing_summary.get("crop_mode", "none"),
            requested_crop_mode=crop_mode,
            output_dir=str(out_dir),
        )

    summary_path = scene_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    summary = load_json(summary_path)
    chorus_ply = Path(summary["chorus_ply"]).expanduser()

    raw_data = read_gaussian_ply(chorus_ply)
    sonata_grid = np.load(scene_dir / "sonata_grid.npy").astype(np.int64, copy=False)
    sonata_raw = np.load(scene_dir / "sonata_grid_raw_bridge.npy").astype(np.int64, copy=False)
    sonata_centers_raw = np.load(scene_dir / "sonata_centers_raw.npy").astype(
        np.float32,
        copy=False,
    )
    voxel_size = float(summary.get("voxel_size", 0.025))
    radius = native_match_radius
    if radius < 0:
        radius = int(summary.get("match_radius_voxels", 2))

    valid_mask, assigned_sonata_grid, raw_indices = assign_native_sonata_grid(
        raw_data["coord"],
        sonata_raw,
        sonata_grid,
        sonata_centers_raw,
        voxel_size,
        radius,
    )
    valid_labeled = int(valid_mask.sum())
    raw_count = int(raw_data["coord"].shape[0])
    if valid_labeled == 0:
        raise RuntimeError(f"No raw splats matched {split}/{scene_id} with radius={radius}")

    crop_mask, crop_summary = make_crop_mask(
        raw_data["coord"],
        sonata_raw,
        sonata_centers_raw,
        voxel_size,
        crop_mode,
        crop_margin_meters,
    )
    crop_mask = np.asarray(crop_mask, dtype=bool)
    crop_mask |= valid_mask
    crop_kept = int(crop_mask.sum())
    if crop_kept == 0:
        raise RuntimeError(f"Crop kept zero raw splats for {split}/{scene_id}")
    crop_valid_labeled = int((valid_mask & crop_mask).sum())
    if crop_valid_labeled == 0:
        raise RuntimeError(
            f"Crop kept no labeled splats for {split}/{scene_id}; "
            f"crop_mode={crop_mode} margin={crop_margin_meters}"
        )
    raw_indices = raw_indices[crop_mask]
    assigned_sonata_grid = assigned_sonata_grid[crop_mask]
    cached_data = {name: value[crop_mask] for name, value in raw_data.items()}

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{scene_id}.tmp.", dir=str(parent)))
    try:
        np.save(temp_dir / "coord.npy", cached_data["coord"])
        np.save(temp_dir / "color.npy", cached_data["color"])
        np.save(temp_dir / "opacity.npy", cached_data["opacity"])
        np.save(temp_dir / "scale.npy", cached_data["scale"])
        np.save(temp_dir / "quat.npy", cached_data["quat"])
        np.save(temp_dir / "sonata_grid.npy", assigned_sonata_grid.astype(np.int64, copy=False))
        np.save(temp_dir / "raw_splat_indices.npy", raw_indices.astype(np.int64, copy=False))
        if (scene_dir / "sonata_origin.npy").exists():
            shutil.copy2(scene_dir / "sonata_origin.npy", temp_dir / "sonata_origin.npy")
        shutil.copy2(scene_dir / "sonata_centers_raw.npy", temp_dir / "sonata_centers_raw.npy")

        out_summary = dict(summary)
        out_summary.update(
            native_chorus_cache=True,
            source_raw_bridge_dir=str(scene_dir),
            cache_output_dir=str(out_dir),
            native_match_radius=int(radius),
            raw_splats=raw_count,
            cached_splats=crop_kept,
            kept_splats=crop_kept,
            kept_fraction=crop_kept / max(raw_count, 1),
            valid_labeled_splats=crop_valid_labeled,
            valid_labeled_splats_before_crop=valid_labeled,
            valid_label_fraction=crop_valid_labeled / max(crop_kept, 1),
            valid_label_fraction_raw=valid_labeled / max(raw_count, 1),
            invalid_context_splats=crop_kept - crop_valid_labeled,
            invalid_context_fraction=(crop_kept - crop_valid_labeled) / max(crop_kept, 1),
            invalid_label_value=list(INVALID_SONATA_GRID),
            emitted_arrays=["coord", "color", "opacity", "scale", "quat", "sonata_grid"],
            **crop_summary,
            matching_note=(
                "Native raw 3DGS splats are labeled before optional cropping. Splats "
                "inside the raw-bridge neighborhood of a Sonata voxel are labeled with "
                "the nearest Sonata grid among candidate voxels; other kept splats use "
                "the invalid label and remain as native context."
            ),
        )
        write_json(temp_dir / "summary.json", out_summary)

        if out_dir.exists():
            if overwrite or replace_existing:
                shutil.rmtree(out_dir)
            else:
                shutil.rmtree(temp_dir)
                return dict(scene_id=scene_id, split=split, status="skipped")
        temp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return dict(
        scene_id=scene_id,
        split=split,
        status="ok",
        raw_splats=raw_count,
        cached_splats=crop_kept,
        kept_splats=crop_kept,
        kept_fraction=crop_kept / max(raw_count, 1),
        valid_labeled_splats=crop_valid_labeled,
        valid_labeled_splats_before_crop=valid_labeled,
        valid_label_fraction=crop_valid_labeled / max(crop_kept, 1),
        valid_label_fraction_raw=valid_labeled / max(raw_count, 1),
        invalid_context_splats=crop_kept - crop_valid_labeled,
        invalid_context_fraction=(crop_kept - crop_valid_labeled) / max(crop_kept, 1),
        crop_mode=crop_summary["crop_mode"],
        crop_margin_meters=crop_summary["crop_margin_meters"],
        radius=int(radius),
    )


def collect_jobs(
    raw_bridge_root: Path,
    splits: list[str],
    scene_ids: set[str] | None,
) -> list[tuple[str, Path]]:
    jobs = []
    for split in splits:
        split_dir = raw_bridge_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")
        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if scene_ids is not None and scene_dir.name not in scene_ids:
                continue
            jobs.append((split, scene_dir))
    return jobs


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    scene_ids = set(args.scene_id) if args.scene_id else None
    jobs = collect_jobs(args.raw_bridge_root, splits, scene_ids)
    if not jobs:
        raise SystemExit("No scenes to process.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(jobs)} native Chorus cache scenes into {args.output_root}", flush=True)

    results = []
    workers = max(int(args.workers), 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                prepare_scene,
                split,
                scene_dir,
                args.output_root,
                args.native_match_radius,
                args.crop_mode,
                args.crop_margin_meters,
                args.overwrite,
            )
            for split, scene_dir in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    ok = sum(result["status"] == "ok" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    exists_mismatch = sum(result["status"] == "exists_mismatch" for result in results)
    if exists_mismatch:
        raise RuntimeError(
            f"{exists_mismatch} output scene dirs already exist with incompatible settings. "
            "Use --overwrite or choose a new --output-root."
        )
    kept = sum(int(result.get("kept_splats", 0)) for result in results)
    valid_labeled = sum(int(result.get("valid_labeled_splats", 0)) for result in results)
    valid_labeled_before_crop = sum(
        int(result.get("valid_labeled_splats_before_crop", result.get("valid_labeled_splats", 0)))
        for result in results
    )
    raw = sum(int(result.get("raw_splats", 0)) for result in results)
    summary = dict(
        raw_bridge_root=str(args.raw_bridge_root),
        output_root=str(args.output_root),
        splits=splits,
        scenes=len(results),
        ok=ok,
        skipped=skipped,
        exists_mismatch=exists_mismatch,
        raw_splats=raw,
        cached_splats=kept,
        kept_splats=kept,
        kept_fraction=kept / max(raw, 1),
        valid_labeled_splats=valid_labeled,
        valid_labeled_splats_before_crop=valid_labeled_before_crop,
        valid_label_fraction=valid_labeled / max(kept, 1),
        valid_label_fraction_raw=valid_labeled_before_crop / max(raw, 1),
        invalid_context_splats=kept - valid_labeled,
        invalid_context_fraction=(kept - valid_labeled) / max(kept, 1),
        invalid_label_value=list(INVALID_SONATA_GRID),
        native_match_radius=args.native_match_radius,
        crop_mode=args.crop_mode,
        crop_margin_meters=max(float(args.crop_margin_meters), 0.0) if args.crop_mode != "none" else 0.0,
    )
    write_json(args.output_root / "summary.json", summary)
    print("DONE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
