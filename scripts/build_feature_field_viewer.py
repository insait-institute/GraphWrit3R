from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        required=True,
        help="Path prefix without suffix, e.g. /work/.../scene_shared_pca3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <prefix>_viewer.html",
    )
    parser.add_argument(
        "--max_room_points",
        type=int,
        default=30000,
        help="Optional subsampling limit for room points.",
    )
    parser.add_argument(
        "--max_gaussian_points",
        type=int,
        default=30000,
        help="Optional subsampling limit for Gaussian centers.",
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=6000,
        help="Optional subsampling limit for cube-rendered input or encoded voxels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stable subsampling.",
    )
    parser.add_argument(
        "--export_voxel_svg",
        action="store_true",
        help="Export one or more standalone rotated cube SVGs colored from selected voxels.",
    )
    parser.add_argument(
        "--voxel_svg_output_dir",
        type=Path,
        default=None,
        help="Directory for voxel SVG exports. Defaults to <prefix>_voxel_svgs.",
    )
    parser.add_argument(
        "--voxel_svg_geometry",
        choices=["encoded_voxels", "input_voxels"],
        default="encoded_voxels",
        help="Voxel geometry to sample for SVG export.",
    )
    parser.add_argument(
        "--voxel_svg_modality",
        choices=["sonata", "chorus", "fused"],
        default="fused",
        help="Feature-color modality used for SVG cube color.",
    )
    parser.add_argument(
        "--voxel_svg_count",
        type=int,
        default=1,
        help="Number of voxel SVGs to export when indices/regions are not supplied.",
    )
    parser.add_argument(
        "--voxel_svg_indices",
        default=None,
        help="Comma-separated voxel row indices to export, e.g. 12,430,901.",
    )
    parser.add_argument(
        "--voxel_svg_region",
        action="append",
        default=[],
        help=(
            "Region box to sample as xmin,ymin,zmin,xmax,ymax,zmax. Can be repeated; "
            "the voxel nearest the region center is exported."
        ),
    )
    parser.add_argument(
        "--voxel_svg_size",
        type=int,
        default=256,
        help="SVG canvas size in pixels.",
    )
    return parser.parse_args()


def colors_to_hex(colors: np.ndarray) -> list[str]:
    colors = np.asarray(colors, dtype=np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors.tolist()]


def ensure_color_rows(colors: np.ndarray | None, count: int, fill: int = 180) -> np.ndarray:
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if colors is None:
        return np.full((count, 3), fill, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] < 3:
        return np.full((count, 3), fill, dtype=np.uint8)
    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating) and float(np.nanmax(colors)) <= 1.1:
        colors = colors * 255.0
    return np.clip(np.nan_to_num(colors), 0, 255).astype(np.uint8)


def subsample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False))


def transfer_colors(token_centers: np.ndarray, token_colors: np.ndarray, points: np.ndarray, k_neighbors: int) -> np.ndarray:
    tree = cKDTree(token_centers)
    k = max(int(k_neighbors), 1)
    distances, indices = tree.query(points, k=k)
    if k == 1:
        return token_colors[np.asarray(indices, dtype=np.int64)]
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
    mixed = np.sum(token_colors[indices].astype(np.float32) * weights[..., None], axis=1)
    return np.round(mixed).astype(np.uint8)


def transfer_scalar(token_centers: np.ndarray, token_values: np.ndarray, points: np.ndarray, k_neighbors: int) -> np.ndarray:
    tree = cKDTree(token_centers)
    k = max(int(k_neighbors), 1)
    distances, indices = tree.query(points, k=k)
    if k == 1:
        return token_values[np.asarray(indices, dtype=np.int64)]
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
    return np.sum(token_values[indices] * weights, axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    denom = np.maximum(a_norm * b_norm, 1e-6)
    return np.clip(np.abs(np.sum(a * b, axis=1, keepdims=True) / denom), 0.0, 1.0)


def fill_missing_rows(rows: np.ndarray, fallback_rows: np.ndarray) -> np.ndarray:
    mask = np.isnan(rows).any(axis=1)
    if mask.any():
        rows = rows.copy()
        rows[mask] = fallback_rows[mask]
    return rows


def build_geometry_payload(
    points: np.ndarray,
    token_centers: np.ndarray,
    token_colors: dict[str, np.ndarray],
    token_diffs: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    k_neighbors: int,
) -> dict[str, Any]:
    sampled_points = points[sample_indices]
    geometry = {
        "x": sampled_points[:, 0].astype(np.float32).tolist(),
        "y": sampled_points[:, 1].astype(np.float32).tolist(),
        "z": sampled_points[:, 2].astype(np.float32).tolist(),
        "count": int(sampled_points.shape[0]),
        "colors": {},
        "diffs": {},
    }
    for name, colors in token_colors.items():
        transferred = transfer_colors(token_centers, colors, sampled_points, k_neighbors)
        geometry["colors"][name] = colors_to_hex(transferred)
    for name, values in token_diffs.items():
        transferred = transfer_scalar(token_centers, values, sampled_points, k_neighbors).reshape(-1)
        geometry["diffs"][name] = {
            "values": transferred.astype(np.float32).tolist(),
            "cmin": 0.0,
            "cmax": 1.0,
        }
    return geometry


def build_original_point_payload(
    points: np.ndarray,
    colors: np.ndarray,
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    sampled_points = points[sample_indices]
    sampled_colors = colors[sample_indices]
    return {
        "kind": "points",
        "x": sampled_points[:, 0].astype(np.float32).tolist(),
        "y": sampled_points[:, 1].astype(np.float32).tolist(),
        "z": sampled_points[:, 2].astype(np.float32).tolist(),
        "count": int(sampled_points.shape[0]),
        "colors": colors_to_hex(sampled_colors),
    }


def cube_template() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def parse_index_list(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def parse_region_box(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 6:
        raise ValueError(f"Region must have six comma-separated floats: {value!r}")
    lo = np.minimum(parts[:3], parts[3:])
    hi = np.maximum(parts[:3], parts[3:])
    return np.asarray([*lo, *hi], dtype=np.float32)


def select_region_voxel_indices(voxel_centers: np.ndarray, region_values: list[str]) -> list[int]:
    selected = []
    for value in region_values:
        box = parse_region_box(value)
        lo = box[:3]
        hi = box[3:]
        center = (lo + hi) * 0.5
        inside = np.all((voxel_centers >= lo[None, :]) & (voxel_centers <= hi[None, :]), axis=1)
        candidates = np.flatnonzero(inside)
        if candidates.size == 0:
            distances = np.linalg.norm(voxel_centers - center[None, :], axis=1)
            selected.append(int(np.argmin(distances)))
            continue
        distances = np.linalg.norm(voxel_centers[candidates] - center[None, :], axis=1)
        selected.append(int(candidates[np.argmin(distances)]))
    return selected


def select_diverse_voxel_indices(voxel_centers: np.ndarray, voxel_colors: np.ndarray, count: int) -> list[int]:
    count = max(int(count), 0)
    if count == 0 or voxel_centers.shape[0] == 0:
        return []
    if count >= voxel_centers.shape[0]:
        return list(range(voxel_centers.shape[0]))

    centers = np.asarray(voxel_centers, dtype=np.float32)
    colors = np.asarray(voxel_colors, dtype=np.float32) / 255.0
    center_span = np.maximum(np.ptp(centers, axis=0), 1e-6)
    center_features = (centers - centers.min(axis=0, keepdims=True)) / center_span[None, :]
    features = np.concatenate([center_features, colors], axis=1)
    first = int(np.argmax(np.linalg.norm(features - features.mean(axis=0, keepdims=True), axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(features - features[first][None, :], axis=1)
    while len(selected) < count:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        next_dist = np.linalg.norm(features - features[next_idx][None, :], axis=1)
        min_dist = np.minimum(min_dist, next_dist)
    return selected


def shade_color(rgb: np.ndarray, factor: float) -> tuple[int, int, int]:
    color = np.asarray(rgb, dtype=np.float32)
    if factor >= 1.0:
        shaded = color + (255.0 - color) * (factor - 1.0)
    else:
        shaded = color * factor
    shaded = np.clip(np.round(shaded), 0, 255).astype(np.uint8)
    return tuple(int(value) for value in shaded.tolist())


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def write_voxel_cube_svg(path: Path, rgb: np.ndarray, size: int, title: str) -> None:
    size = max(int(size), 64)
    scale = size / 256.0
    points = {
        "top": [(128, 24), (218, 76), (128, 128), (38, 76)],
        "left": [(38, 76), (128, 128), (128, 232), (38, 180)],
        "right": [(128, 128), (218, 76), (218, 180), (128, 232)],
    }
    scaled = {
        name: " ".join(f"{x * scale:.1f},{y * scale:.1f}" for x, y in coords)
        for name, coords in points.items()
    }
    base = np.asarray(rgb, dtype=np.uint8).reshape(3)
    top = rgb_to_hex(shade_color(base, 1.22))
    left = rgb_to_hex(shade_color(base, 0.88))
    right = rgb_to_hex(shade_color(base, 0.68))
    stroke = rgb_to_hex(shade_color(base, 0.45))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-label="{title}">
  <title>{title}</title>
  <polygon points="{scaled['top']}" fill="{top}" stroke="{stroke}" stroke-width="{1.8 * scale:.2f}" stroke-linejoin="round"/>
  <polygon points="{scaled['left']}" fill="{left}" stroke="{stroke}" stroke-width="{1.8 * scale:.2f}" stroke-linejoin="round"/>
  <polygon points="{scaled['right']}" fill="{right}" stroke="{stroke}" stroke-width="{1.8 * scale:.2f}" stroke-linejoin="round"/>
  <path d="M {128 * scale:.1f} {128 * scale:.1f} L {128 * scale:.1f} {232 * scale:.1f}" stroke="{stroke}" stroke-width="{1.4 * scale:.2f}" opacity="0.42"/>
</svg>
""",
        encoding="utf-8",
    )


def export_voxel_svgs(
    output_dir: Path,
    voxel_centers: np.ndarray,
    voxel_colors: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    indices = parse_index_list(args.voxel_svg_indices)
    indices.extend(select_region_voxel_indices(voxel_centers, args.voxel_svg_region))
    if not indices:
        indices = select_diverse_voxel_indices(voxel_centers, voxel_colors, args.voxel_svg_count)

    seen = set()
    exports = []
    for ordinal, index in enumerate(indices, start=1):
        if index in seen:
            continue
        seen.add(index)
        if index < 0 or index >= voxel_centers.shape[0]:
            raise IndexError(f"Voxel index {index} is outside 0..{voxel_centers.shape[0] - 1}")
        color = np.asarray(voxel_colors[index], dtype=np.uint8)
        path = output_dir / f"voxel_{ordinal:02d}_{args.voxel_svg_geometry}_{args.voxel_svg_modality}_idx{index}.svg"
        title = f"{args.voxel_svg_geometry} {args.voxel_svg_modality} voxel {index}"
        write_voxel_cube_svg(path, color, args.voxel_svg_size, title)
        exports.append(
            {
                "path": str(path),
                "index": int(index),
                "center": voxel_centers[index].astype(float).tolist(),
                "color": color.astype(int).tolist(),
            }
        )
    return exports


def build_voxel_payload(
    voxel_centers: np.ndarray,
    voxel_size: float,
    token_centers: np.ndarray,
    token_colors: dict[str, np.ndarray],
    token_diffs: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    k_neighbors: int,
    direct_colors: bool = False,
) -> dict[str, Any]:
    sampled_centers = voxel_centers[sample_indices]
    unit_vertices, unit_faces = cube_template()
    cube_offsets = unit_vertices * float(voxel_size)
    vertex_count = sampled_centers.shape[0] * unit_vertices.shape[0]
    face_count = sampled_centers.shape[0] * unit_faces.shape[0]
    vertices = np.empty((vertex_count, 3), dtype=np.float32)
    faces = np.empty((face_count, 3), dtype=np.int64)
    for idx, center in enumerate(sampled_centers):
        start_v = idx * unit_vertices.shape[0]
        end_v = start_v + unit_vertices.shape[0]
        start_f = idx * unit_faces.shape[0]
        end_f = start_f + unit_faces.shape[0]
        vertices[start_v:end_v] = cube_offsets + center[None, :]
        faces[start_f:end_f] = unit_faces + start_v

    geometry = {
        "kind": "mesh",
        "x": vertices[:, 0].astype(np.float32).tolist(),
        "y": vertices[:, 1].astype(np.float32).tolist(),
        "z": vertices[:, 2].astype(np.float32).tolist(),
        "center_x": np.repeat(sampled_centers[:, 0], unit_vertices.shape[0]).astype(np.float32).tolist(),
        "center_y": np.repeat(sampled_centers[:, 1], unit_vertices.shape[0]).astype(np.float32).tolist(),
        "center_z": np.repeat(sampled_centers[:, 2], unit_vertices.shape[0]).astype(np.float32).tolist(),
        "offset_x": np.tile(cube_offsets[:, 0], sampled_centers.shape[0]).astype(np.float32).tolist(),
        "offset_y": np.tile(cube_offsets[:, 1], sampled_centers.shape[0]).astype(np.float32).tolist(),
        "offset_z": np.tile(cube_offsets[:, 2], sampled_centers.shape[0]).astype(np.float32).tolist(),
        "i": faces[:, 0].astype(np.int64).tolist(),
        "j": faces[:, 1].astype(np.int64).tolist(),
        "k": faces[:, 2].astype(np.int64).tolist(),
        "count": int(sampled_centers.shape[0]),
        "voxel_size": float(voxel_size),
        "colors": {},
        "diffs": {},
    }
    for name, colors in token_colors.items():
        if direct_colors:
            transferred = colors[sample_indices]
        else:
            transferred = transfer_colors(token_centers, colors, sampled_centers, k_neighbors)
        vertex_colors = np.repeat(transferred, unit_vertices.shape[0], axis=0)
        geometry["colors"][name] = colors_to_hex(vertex_colors)
    for name, values in token_diffs.items():
        if direct_colors:
            transferred = values[sample_indices].reshape(-1)
        else:
            transferred = transfer_scalar(token_centers, values, sampled_centers, k_neighbors).reshape(-1)
        vertex_values = np.repeat(transferred.astype(np.float32), unit_vertices.shape[0])
        geometry["diffs"][name] = {
            "values": vertex_values.astype(np.float32).tolist(),
            "cmin": 0.0,
            "cmax": 1.0,
        }
    return geometry


def default_camera(points: np.ndarray) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = ((mins + maxs) / 2.0).tolist()
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    eye = {"x": 1.7 * span, "y": 1.7 * span, "z": 1.1 * span}
    return {
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "eye": eye,
        "target_center": center,
    }


def build_html(data: dict[str, Any]) -> str:
    serialized = json.dumps(data)
    plotly_script_tag = build_plotly_script_tag()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Feature Field Viewer</title>
  {plotly_script_tag}
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #f6f7f9;
      --text: #111827;
      --muted: #5b6472;
      --accent: #0f7490;
      --border: #d9dee7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .shell {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 18px;
    }}
    .header {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      align-items: end;
      justify-content: space-between;
      margin-bottom: 14px;
    }}
    .title h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.2;
    }}
    .title p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 180px;
    }}
    .control label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .control select, .control input {{
      width: 100%;
    }}
    select, input[type="range"] {{
      background: #ffffff;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    button {{
      align-self: end;
      background: var(--text);
      color: #ffffff;
      border: 1px solid var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{
      background: #293241;
    }}
    .export-control {{
      min-width: 260px;
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .button-row button {{
      align-self: auto;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .stat .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .stat .v {{
      font-size: 18px;
      font-weight: 600;
    }}
    #plot {{
      height: 82vh;
      min-height: 720px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    #scenePlot {{
      height: 64vh;
      min-height: 560px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    .subheader {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
      margin: 22px 0 10px;
    }}
    .subheader h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
    }}
    .subheader p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .error {{
      margin-top: 14px;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      color: #9f1239;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 13px;
      display: none;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title">
        <h1>{data["scene_id"]}</h1>
        <p>{data["projection"]} comparison viewer. Drag any panel to rotate; camera sync is enabled.</p>
      </div>
      <div class="controls">
        <div class="control">
          <label for="geometry">Geometry</label>
          <select id="geometry">
            <option value="encoded_voxels">Encoded Voxels</option>
            <option value="room_pcd">Room Points</option>
            <option value="input_voxels">Input Voxels</option>
            <option value="gaussian_centers">Gaussian Centers</option>
          </select>
        </div>
        <div class="control">
          <label for="diffMode">Overlap Panel</label>
          <select id="diffMode">
            <option value="fused_vs_sonata">Fused vs Sonata</option>
            <option value="chorus_vs_sonata">Chorus vs Sonata</option>
            <option value="fused_vs_chorus">Fused vs Chorus</option>
          </select>
        </div>
        <div class="control">
          <label for="pointSize">Point Size</label>
          <input id="pointSize" type="range" min="1" max="8" step="0.5" value="2.5">
        </div>
        <div class="control" id="voxelScaleControl">
          <label for="voxelScale">Voxel Size <span id="voxelScaleValue">1.0x</span></label>
          <input id="voxelScale" type="range" min="0.5" max="8" step="0.25" value="1">
        </div>
        <div class="control">
          <label for="opacity">Opacity</label>
          <input id="opacity" type="range" min="0.2" max="1" step="0.05" value="0.95">
        </div>
        <div class="control export-control">
          <label>Export</label>
          <div class="button-row">
            <button id="exportSvg" type="button">SVG 2x2</button>
            <button id="exportPanelSvgs" type="button">SVG Panels</button>
            <button id="exportScaleSvg" type="button">SVG Scale</button>
            <button id="exportPng" type="button">PNG 2x2</button>
          </div>
        </div>
      </div>
    </div>

    <div class="stats">
      <div class="stat"><div class="k">Matched Tokens</div><div class="v">{data["matched_count"]}</div></div>
      <div class="stat"><div class="k">Token Count</div><div class="v">{data["token_count"]}</div></div>
      <div class="stat"><div class="k">Gate Avg</div><div class="v">{data["gate_avg"]}</div></div>
      <div class="stat"><div class="k">Room Points</div><div class="v">{data["room_point_count"]}</div></div>
      <div class="stat"><div class="k">Encoded Voxels</div><div class="v">{data["encoded_voxel_count"]}</div></div>
      <div class="stat"><div class="k">Input Voxels</div><div class="v">{data["input_voxel_count"]}</div></div>
      <div class="stat"><div class="k">Gaussian Centers</div><div class="v">{data["gaussian_point_count"]}</div></div>
    </div>

    <div id="plot"></div>
    <div class="note">Tip: the fourth panel uses the same heatmap palette but plots cosine similarity/overlap: hotter red means closer features.</div>
    <div id="originalSceneSection">
      <div class="subheader">
        <div>
          <h2>Original Scene</h2>
          <p>Raw RGB room point cloud and aligned Gaussian centers, without feature coloring.</p>
        </div>
        <div class="button-row">
          <button id="exportSceneSvg" type="button">SVG Original</button>
          <button id="exportScenePng" type="button">PNG Original</button>
        </div>
      </div>
      <div id="scenePlot"></div>
    </div>
    <div id="error" class="error"></div>
  </div>

  <script>
    const DATA = {serialized};
    const plot = document.getElementById('plot');
    const scenePlot = document.getElementById('scenePlot');
    const originalSceneSection = document.getElementById('originalSceneSection');
    const errorBox = document.getElementById('error');
    const geometrySelect = document.getElementById('geometry');
    const diffSelect = document.getElementById('diffMode');
    const pointSizeInput = document.getElementById('pointSize');
    const voxelScaleControl = document.getElementById('voxelScaleControl');
    const voxelScaleInput = document.getElementById('voxelScale');
    const voxelScaleValue = document.getElementById('voxelScaleValue');
    const opacityInput = document.getElementById('opacity');
    const exportSvgButton = document.getElementById('exportSvg');
    const exportPanelSvgsButton = document.getElementById('exportPanelSvgs');
    const exportScaleSvgButton = document.getElementById('exportScaleSvg');
    const exportPngButton = document.getElementById('exportPng');
    const exportSceneSvgButton = document.getElementById('exportSceneSvg');
    const exportScenePngButton = document.getElementById('exportScenePng');
    let syncing = false;
    let sceneSyncing = false;
    let relayoutListenerAttached = false;
    let sceneRelayoutListenerAttached = false;

    function showError(message) {{
      errorBox.textContent = message;
      errorBox.style.display = 'block';
      errorBox.style.background = '#fff1f2';
      errorBox.style.borderColor = '#fecdd3';
      errorBox.style.color = '#9f1239';
    }}

    function showNote(message) {{
      errorBox.textContent = message;
      errorBox.style.display = 'block';
      errorBox.style.background = '#eff6ff';
      errorBox.style.borderColor = '#bfdbfe';
      errorBox.style.color = '#1d4ed8';
    }}

    if (typeof Plotly === 'undefined') {{
      showError('Plotly failed to load. Rebuild the viewer in an environment with the Python plotly package installed, or open this file with internet access if it was built with the CDN fallback.');
    }}

    Array.from(geometrySelect.options).forEach((option) => {{
      if (!DATA.geometries[option.value]) {{
        option.disabled = true;
        option.hidden = true;
      }}
    }});
    if (!DATA.geometries[geometrySelect.value]) {{
      geometrySelect.value = Object.keys(DATA.geometries)[0];
    }}

    function makeSceneLayout(camera) {{
      return {{
        xaxis: {{visible: false, backgroundcolor: '#ffffff', gridcolor: '#ffffff', zerolinecolor: '#ffffff'}},
        yaxis: {{visible: false, backgroundcolor: '#ffffff', gridcolor: '#ffffff', zerolinecolor: '#ffffff'}},
        zaxis: {{visible: false, backgroundcolor: '#ffffff', gridcolor: '#ffffff', zerolinecolor: '#ffffff'}},
        bgcolor: '#ffffff',
        aspectmode: 'data',
        camera: {{
          up: camera.up,
          center: camera.center,
          eye: camera.eye
        }}
      }};
    }}

    function scaledMeshCoords(geom, voxelScale) {{
      if (geom.kind !== 'mesh' || !geom.center_x || voxelScale === 1) {{
        return {{x: geom.x, y: geom.y, z: geom.z}};
      }}
      const x = geom.center_x.map((center, idx) => center + geom.offset_x[idx] * voxelScale);
      const y = geom.center_y.map((center, idx) => center + geom.offset_y[idx] * voxelScale);
      const z = geom.center_z.map((center, idx) => center + geom.offset_z[idx] * voxelScale);
      return {{x, y, z}};
    }}

    function updateControlVisibility() {{
      const isVoxelMesh = geometrySelect.value === 'input_voxels' || geometrySelect.value === 'encoded_voxels';
      voxelScaleControl.style.display = isVoxelMesh ? 'flex' : 'none';
      pointSizeInput.closest('.control').style.display = isVoxelMesh ? 'none' : 'flex';
      voxelScaleValue.textContent = parseFloat(voxelScaleInput.value).toFixed(2).replace(/\\.00$/, '.0') + 'x';
    }}

    function makeRgbTrace(geom, modality, sceneName, pointSize, opacity, voxelScale) {{
      if (geom.kind === 'mesh') {{
        const coords = scaledMeshCoords(geom, voxelScale);
        return {{
          type: 'mesh3d',
          scene: sceneName,
          x: coords.x,
          y: coords.y,
          z: coords.z,
          i: geom.i,
          j: geom.j,
          k: geom.k,
          vertexcolor: geom.colors[modality],
          flatshading: true,
          opacity: opacity,
          lighting: {{ambient: 0.72, diffuse: 0.78, roughness: 0.95, specular: 0.05}},
          name: modality.charAt(0).toUpperCase() + modality.slice(1),
          hoverinfo: 'skip',
          showlegend: false
        }};
      }}
        return {{
            type: 'scatter3d',
            mode: 'markers',
            scene: sceneName,
            x: geom.x,
        y: geom.y,
        z: geom.z,
        marker: {{
          size: pointSize,
          opacity: opacity,
          color: geom.colors[modality]
        }},
        name: modality.charAt(0).toUpperCase() + modality.slice(1),
        hovertemplate: modality + '<extra></extra>',
        showlegend: false
      }};
    }}

    function makeDiffTrace(geom, diffMode, sceneName, pointSize, opacity, voxelScale) {{
      const diff = geom.diffs[diffMode];
      if (geom.kind === 'mesh') {{
        const coords = scaledMeshCoords(geom, voxelScale);
        return {{
          type: 'mesh3d',
          scene: sceneName,
          x: coords.x,
          y: coords.y,
          z: coords.z,
          i: geom.i,
          j: geom.j,
          k: geom.k,
          intensity: diff.values,
          colorscale: similarityColorscale(),
          cauto: false,
          cmin: 0,
          cmax: 1,
          flatshading: true,
          opacity: opacity,
          lighting: {{ambient: 0.72, diffuse: 0.78, roughness: 0.95, specular: 0.05}},
          colorbar: similarityColorbar(diffMode),
          name: 'Overlap',
          hoverinfo: 'skip',
          showlegend: false
        }};
      }}
      return {{
        type: 'scatter3d',
        mode: 'markers',
        scene: sceneName,
        x: geom.x,
        y: geom.y,
        z: geom.z,
        marker: {{
          size: pointSize,
          opacity: opacity,
          color: diff.values,
          colorscale: similarityColorscale(),
          cauto: false,
          cmin: 0,
          cmax: 1,
          colorbar: similarityColorbar(diffMode)
        }},
        name: 'Overlap',
        hovertemplate: diffMode + ' cosine similarity: %{{marker.color:.4f}}<extra></extra>',
        showlegend: false
      }};
    }}

    function makeOriginalTrace(geom, sceneName, title, pointSize, opacity) {{
      return {{
        type: 'scatter3d',
        mode: 'markers',
        scene: sceneName,
        x: geom.x,
        y: geom.y,
        z: geom.z,
        marker: {{
          size: pointSize,
          opacity: opacity,
          color: geom.colors
        }},
        name: title,
        hovertemplate: title + '<extra></extra>',
        showlegend: false
      }};
    }}

    function renderOriginalScene() {{
      if (typeof Plotly === 'undefined' || !DATA.original_scene || !DATA.original_scene.room_pcd) {{
        originalSceneSection.style.display = 'none';
        return;
      }}
      originalSceneSection.style.display = 'block';
      const pointSize = parseFloat(pointSizeInput.value);
      const opacity = parseFloat(opacityInput.value);
      const traces = [
        makeOriginalTrace(DATA.original_scene.room_pcd, 'scene', 'Room PCD', pointSize, opacity)
      ];
      const hasGaussians = !!DATA.original_scene.gaussian_centers;
      if (!hasGaussians) {{
        showNote('Original scene viewer has no Gaussian centers in this NPZ. Rerun visualize_3d_feature_fields.py with the updated exporter; it should include gaussian_centers when coord.npy exists in the aligned cache.');
      }}
      const layout = {{
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        margin: {{l: 0, r: 0, t: 42, b: 0}},
        font: {{color: '#111827'}},
        grid: {{rows: 1, columns: hasGaussians ? 2 : 1, pattern: 'independent'}},
        annotations: [
          {{text: 'Room PCD', x: hasGaussians ? 0.23 : 0.5, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}}
        ],
        scene: {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: hasGaussians ? [0.00, 0.48] : [0.00, 1.00], y: [0.00, 1.00]}}
        }}
      }};
      if (hasGaussians) {{
        traces.push(makeOriginalTrace(DATA.original_scene.gaussian_centers, 'scene2', 'Gaussian Centers', pointSize, opacity));
        layout.annotations.push(
          {{text: 'Gaussian Centers', x: 0.77, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}}
        );
        layout.scene2 = {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: [0.52, 1.00], y: [0.00, 1.00]}}
        }};
      }}
      Plotly.react(scenePlot, traces, layout, {{responsive: true, displaylogo: false}})
        .then(() => {{
          if (!sceneRelayoutListenerAttached && typeof scenePlot.on === 'function') {{
            scenePlot.on('plotly_relayout', (eventData) => {{
              if (sceneSyncing) return;
              const keys = Object.keys(eventData || {{}});
              const cameraKey = keys.find((key) => key.endsWith('.camera'));
              if (!cameraKey) return;
              const camera = eventData[cameraKey];
              sceneSyncing = true;
              const update = {{'scene.camera': camera}};
              if (DATA.original_scene.gaussian_centers) {{
                update['scene2.camera'] = camera;
              }}
              Plotly.relayout(scenePlot, update).then(() => {{
                sceneSyncing = false;
              }}).catch((err) => {{
                sceneSyncing = false;
                console.error(err);
              }});
            }});
            sceneRelayoutListenerAttached = true;
          }}
        }})
        .catch((err) => {{
          console.error(err);
          showError('Original scene rendering failed: ' + (err && err.message ? err.message : String(err)));
        }});
    }}

    function safeFilename(value) {{
      return String(value).replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
    }}

    function exportBaseName() {{
      return safeFilename(DATA.scene_id + '_' + geometrySelect.value + '_' + diffSelect.value);
    }}

    function similarityColorscale() {{
      return [
        [0.0, '#2563eb'],
        [0.5, '#f8fafc'],
        [1.0, '#dc2626']
      ];
    }}

    function similarityColorbar(diffMode) {{
      return {{
        len: 0.55,
        thickness: 16,
        x: 0.985,
        y: 0.5,
        tickmode: 'array',
        tickvals: [0, 0.5, 1],
        ticktext: ['0', '0.5', '1'],
        title: diffMode.replaceAll('_', ' ') + '<br>cos sim'
      }};
    }}

    function exportScaleBarSvg() {{
      const diffMode = diffSelect.value;
      const title = diffMode.replaceAll('_', ' ') + ' cosine similarity';
      const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="360" height="94" viewBox="0 0 360 94" role="img" aria-label="${{title}} scale 0 to 1">
  <defs>
    <linearGradient id="simScale" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="50%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
  </defs>
  <rect x="22" y="34" width="316" height="22" rx="3" fill="url(#simScale)" stroke="#111827" stroke-width="1"/>
  <text x="180" y="20" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#111827">${{title}}</text>
  <text x="22" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0</text>
  <text x="180" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0.5</text>
  <text x="338" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">1</text>
  <line x1="22" y1="58" x2="22" y2="64" stroke="#111827" stroke-width="1"/>
  <line x1="180" y1="58" x2="180" y2="64" stroke="#111827" stroke-width="1"/>
  <line x1="338" y1="58" x2="338" y2="64" stroke="#111827" stroke-width="1"/>
</svg>`;
      const dataUrl = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
      downloadDataUrl(dataUrl, exportBaseName() + '_similarity_scale.svg');
    }}

    function originalBaseName() {{
      return safeFilename(DATA.scene_id + '_original_scene');
    }}

    function currentCamera() {{
      const fullScene = plot._fullLayout && plot._fullLayout.scene;
      const layoutScene = plot.layout && plot.layout.scene;
      return (fullScene && fullScene.camera) || (layoutScene && layoutScene.camera) || DATA.camera;
    }}

    function downloadDataUrl(dataUrl, filename) {{
      const anchor = document.createElement('a');
      anchor.href = dataUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    }}

    function makePanelLayout(title, camera) {{
      return {{
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        margin: {{l: 0, r: 120, t: 42, b: 0}},
        font: {{color: '#111827'}},
        title: {{text: title, x: 0.5}},
        scene: {{
          ...makeSceneLayout(camera),
          domain: {{x: [0.00, 1.00], y: [0.00, 1.00]}}
        }}
      }};
    }}

    async function exportPanelSvgs() {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      const geometryKey = geometrySelect.value;
      const geom = DATA.geometries[geometryKey];
      if (!geom) {{
        showError('No geometry payload found for ' + geometryKey + '.');
        return;
      }}
      const pointSize = parseFloat(pointSizeInput.value);
      const voxelScale = parseFloat(voxelScaleInput.value);
      const opacity = parseFloat(opacityInput.value);
      const diffMode = diffSelect.value;
      const camera = currentCamera();
      const panels = [
        {{key: 'sonata', title: 'Sonata', trace: makeRgbTrace(geom, 'sonata', 'scene', pointSize, opacity, voxelScale)}},
        {{key: 'chorus', title: 'Chorus', trace: makeRgbTrace(geom, 'chorus', 'scene', pointSize, opacity, voxelScale)}},
        {{key: 'fused', title: 'Fused', trace: makeRgbTrace(geom, 'fused', 'scene', pointSize, opacity, voxelScale)}},
        {{key: diffMode, title: diffMode.replaceAll('_', ' ') + ' cosine similarity', trace: makeDiffTrace(geom, diffMode, 'scene', pointSize, opacity, voxelScale)}}
      ];
      const exportDiv = document.createElement('div');
      exportDiv.style.position = 'fixed';
      exportDiv.style.left = '-10000px';
      exportDiv.style.top = '0';
      exportDiv.style.width = '1200px';
      exportDiv.style.height = '1000px';
      document.body.appendChild(exportDiv);
      try {{
        for (const panel of panels) {{
          await Plotly.newPlot(
            exportDiv,
            [panel.trace],
            makePanelLayout(panel.title, camera),
            {{staticPlot: true, displayModeBar: false, displaylogo: false}}
          );
          const dataUrl = await Plotly.toImage(exportDiv, {{
            format: 'svg',
            width: 1200,
            height: 1000,
            scale: 1
          }});
          downloadDataUrl(dataUrl, exportBaseName() + '_' + safeFilename(panel.key) + '.svg');
        }}
      }} catch (err) {{
        console.error(err);
        showError('Panel SVG export failed: ' + (err && err.message ? err.message : String(err)));
      }} finally {{
        Plotly.purge(exportDiv);
        exportDiv.remove();
      }}
    }}

    function render() {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      const geometryKey = geometrySelect.value;
      const diffMode = diffSelect.value;
      const pointSize = parseFloat(pointSizeInput.value);
      const voxelScale = parseFloat(voxelScaleInput.value);
      const opacity = parseFloat(opacityInput.value);
      const geom = DATA.geometries[geometryKey];
      if (!geom) {{
        showError('No geometry payload found for ' + geometryKey + '. Rebuild the NPZ with the updated exporter.');
        return;
      }}
      updateControlVisibility();
      const scenes = ['scene', 'scene2', 'scene3', 'scene4'];
      const traces = [
        makeRgbTrace(geom, 'sonata', scenes[0], pointSize, opacity, voxelScale),
        makeRgbTrace(geom, 'chorus', scenes[1], pointSize, opacity, voxelScale),
        makeRgbTrace(geom, 'fused', scenes[2], pointSize, opacity, voxelScale),
        makeDiffTrace(geom, diffMode, scenes[3], pointSize, opacity, voxelScale)
      ];
      const layout = {{
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        margin: {{l: 0, r: 120, t: 50, b: 0}},
        font: {{color: '#111827'}},
        title: {{
          text: DATA.geometry_titles[geometryKey] || geometryKey,
          x: 0.5
        }},
        grid: {{rows: 2, columns: 2, pattern: 'independent'}},
        annotations: [
          {{text: 'Sonata', x: 0.20, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}},
          {{text: 'Chorus', x: 0.80, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}},
          {{text: 'Fused', x: 0.20, y: 0.47, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}},
          {{text: diffMode.replaceAll('_', ' ') + ' cosine similarity', x: 0.80, y: 0.47, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 16}}}}
        ],
        scene: {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: [0.00, 0.48], y: [0.52, 1.00]}}
        }},
        scene2: {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: [0.52, 1.00], y: [0.52, 1.00]}}
        }},
        scene3: {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: [0.00, 0.48], y: [0.00, 0.48]}}
        }},
        scene4: {{
          ...makeSceneLayout(DATA.camera),
          domain: {{x: [0.52, 1.00], y: [0.00, 0.48]}}
        }}
      }};
      Plotly.react(plot, traces, layout, {{responsive: true, displaylogo: false}})
        .then(() => {{
          if (!relayoutListenerAttached && typeof plot.on === 'function') {{
            plot.on('plotly_relayout', (eventData) => {{
              if (syncing) return;
              const keys = Object.keys(eventData || {{}});
              const cameraKey = keys.find((key) => key.endsWith('.camera'));
              if (!cameraKey) return;
              const camera = eventData[cameraKey];
              syncing = true;
              Plotly.relayout(plot, {{
                'scene.camera': camera,
                'scene2.camera': camera,
                'scene3.camera': camera,
                'scene4.camera': camera
              }}).then(() => {{
                syncing = false;
              }}).catch((err) => {{
                syncing = false;
                console.error(err);
              }});
            }});
            relayoutListenerAttached = true;
          }}
        }})
        .catch((err) => {{
          console.error(err);
          showError('Plot rendering failed: ' + (err && err.message ? err.message : String(err)));
        }});
    }}

    geometrySelect.addEventListener('change', render);
    diffSelect.addEventListener('change', render);
    pointSizeInput.addEventListener('input', () => {{
      render();
      renderOriginalScene();
    }});
    voxelScaleInput.addEventListener('input', render);
    opacityInput.addEventListener('input', () => {{
      render();
      renderOriginalScene();
    }});
    exportSvgButton.addEventListener('click', () => {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      Plotly.downloadImage(plot, {{
        format: 'svg',
        width: 2400,
        height: 1800,
        scale: 1,
        filename: exportBaseName()
      }}).catch((err) => {{
        console.error(err);
        showError('SVG export failed: ' + (err && err.message ? err.message : String(err)));
      }});
    }});
    exportPanelSvgsButton.addEventListener('click', () => {{
      exportPanelSvgs();
    }});
    exportScaleSvgButton.addEventListener('click', () => {{
      exportScaleBarSvg();
    }});
    exportPngButton.addEventListener('click', () => {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      Plotly.downloadImage(plot, {{
        format: 'png',
        width: 2400,
        height: 1800,
        scale: 2,
        filename: exportBaseName()
      }}).catch((err) => {{
        console.error(err);
        showError('PNG export failed: ' + (err && err.message ? err.message : String(err)));
      }});
    }});
    exportSceneSvgButton.addEventListener('click', () => {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      Plotly.downloadImage(scenePlot, {{
        format: 'svg',
        width: 2200,
        height: 1000,
        scale: 1,
        filename: originalBaseName()
      }}).catch((err) => {{
        console.error(err);
        showError('Original SVG export failed: ' + (err && err.message ? err.message : String(err)));
      }});
    }});
    exportScenePngButton.addEventListener('click', () => {{
      if (typeof Plotly === 'undefined') {{
        return;
      }}
      Plotly.downloadImage(scenePlot, {{
        format: 'png',
        width: 2200,
        height: 1000,
        scale: 2,
        filename: originalBaseName()
      }}).catch((err) => {{
        console.error(err);
        showError('Original PNG export failed: ' + (err && err.message ? err.message : String(err)));
      }});
    }});
    render();
    renderOriginalScene();
  </script>
</body>
</html>
"""


def build_plotly_script_tag() -> str:
    try:
        from plotly.offline.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    except Exception:
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if args.output is None:
        output_path = prefix.with_name(f"{prefix.name}_viewer.html")
    else:
        output_path = args.output

    data = np.load(npz_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    token_centers = np.asarray(data["sonata_centers_raw"], dtype=np.float32)
    room_points = np.asarray(data["room_points"], dtype=np.float32)
    room_colors = ensure_color_rows(
        np.asarray(data["room_colors"]) if "room_colors" in data.files else None,
        room_points.shape[0],
    )
    gaussian_centers = (
        np.asarray(data["gaussian_centers"], dtype=np.float32)
        if "gaussian_centers" in data.files
        else np.empty((0, 3), dtype=np.float32)
    )
    gaussian_colors = ensure_color_rows(
        np.asarray(data["gaussian_colors"]) if "gaussian_colors" in data.files else None,
        gaussian_centers.shape[0],
    )
    voxel_centers = (
        np.asarray(data["input_voxel_centers"], dtype=np.float32)
        if "input_voxel_centers" in data.files
        else None
    )
    encoded_voxel_centers = (
        np.asarray(data["encoded_voxel_centers"], dtype=np.float32)
        if "encoded_voxel_centers" in data.files
        else None
    )
    voxel_size = (
        float(np.asarray(data["voxel_size"], dtype=np.float32).item())
        if "voxel_size" in data.files
        else None
    )
    encoded_voxel_size = (
        float(np.asarray(data["encoded_voxel_size"], dtype=np.float32).item())
        if "encoded_voxel_size" in data.files
        else None
    )
    token_colors = {
        "sonata": np.asarray(data["sonata_colors"], dtype=np.uint8),
        "chorus": np.asarray(data["chorus_colors"], dtype=np.uint8),
        "fused": np.asarray(data["fused_colors"], dtype=np.uint8),
    }
    token_features = {
        "sonata": np.asarray(data["sonata_features"], dtype=np.float32),
        "chorus": np.asarray(data["chorus_features"], dtype=np.float32),
        "fused": np.asarray(data["fused_features"], dtype=np.float32),
    }
    token_features["chorus"] = fill_missing_rows(token_features["chorus"], token_features["sonata"])
    token_diffs = {
        "fused_vs_sonata": cosine_similarity(token_features["fused"], token_features["sonata"]),
        "chorus_vs_sonata": cosine_similarity(token_features["chorus"], token_features["sonata"]),
        "fused_vs_chorus": cosine_similarity(token_features["fused"], token_features["chorus"]),
    }

    room_k = 1
    room_transfer = meta.get("transfer", {}).get("room_pcd", {})
    if isinstance(room_transfer, dict):
        for value in room_transfer.values():
            if isinstance(value, dict) and "k_neighbors" in value:
                room_k = int(value["k_neighbors"])
                break

    svg_exports = []
    if args.export_voxel_svg:
        if args.voxel_svg_geometry == "encoded_voxels":
            if encoded_voxel_centers is None:
                raise ValueError("Cannot export encoded voxel SVGs because encoded_voxel_centers is missing.")
            voxel_svg_centers = encoded_voxel_centers
            voxel_svg_colors = token_colors[args.voxel_svg_modality]
        else:
            if voxel_centers is None:
                raise ValueError("Cannot export input voxel SVGs because input_voxel_centers is missing.")
            voxel_svg_centers = voxel_centers
            voxel_svg_colors = transfer_colors(
                token_centers,
                token_colors[args.voxel_svg_modality],
                voxel_centers,
                room_k,
            )
        svg_output_dir = (
            args.voxel_svg_output_dir
            if args.voxel_svg_output_dir is not None
            else prefix.with_name(f"{prefix.name}_voxel_svgs")
        )
        svg_exports = export_voxel_svgs(svg_output_dir, voxel_svg_centers, voxel_svg_colors, args)

    room_indices = subsample_indices(room_points.shape[0], args.max_room_points, args.seed)
    gaussian_indices = (
        subsample_indices(gaussian_centers.shape[0], args.max_gaussian_points, args.seed + 1)
        if gaussian_centers.shape[0] > 0
        else None
    )
    voxel_indices = (
        subsample_indices(voxel_centers.shape[0], args.max_voxels, args.seed + 2)
        if voxel_centers is not None
        else None
    )
    encoded_voxel_indices = (
        subsample_indices(encoded_voxel_centers.shape[0], args.max_voxels, args.seed + 3)
        if encoded_voxel_centers is not None
        else None
    )

    geometries = {
        "room_pcd": build_geometry_payload(
            room_points,
            token_centers,
            token_colors,
            token_diffs,
            room_indices,
            room_k,
        ),
    }
    if gaussian_indices is not None:
        geometries["gaussian_centers"] = build_geometry_payload(
            gaussian_centers,
            token_centers,
            token_colors,
            token_diffs,
            gaussian_indices,
            1,
        )
    if encoded_voxel_centers is not None and encoded_voxel_size is not None and encoded_voxel_indices is not None:
        geometries["encoded_voxels"] = build_voxel_payload(
            encoded_voxel_centers,
            encoded_voxel_size,
            token_centers,
            token_colors,
            token_diffs,
            encoded_voxel_indices,
            1,
            direct_colors=True,
        )
    if voxel_centers is not None and voxel_size is not None and voxel_indices is not None:
        geometries["input_voxels"] = build_voxel_payload(
            voxel_centers,
            voxel_size,
            token_centers,
            token_colors,
            token_diffs,
            voxel_indices,
            room_k,
        )

    original_scene = {
        "room_pcd": build_original_point_payload(room_points, room_colors, room_indices),
    }
    if gaussian_indices is not None:
        original_scene["gaussian_centers"] = build_original_point_payload(
            gaussian_centers,
            gaussian_colors,
            gaussian_indices,
        )

    viewer_data = {
        "scene_id": meta["scene_id"],
        "projection": meta["projection"],
        "matched_count": meta.get("matched_count"),
        "token_count": meta.get("token_count"),
        "gate_avg": meta.get("gate_avg"),
        "room_point_count": int(room_points.shape[0]),
        "encoded_voxel_count": int(encoded_voxel_centers.shape[0]) if encoded_voxel_centers is not None else meta.get("encoded_voxel_count"),
        "input_voxel_count": int(voxel_centers.shape[0]) if voxel_centers is not None else meta.get("input_voxel_count"),
        "gaussian_point_count": int(gaussian_centers.shape[0]),
        "camera": default_camera(room_points),
        "geometries": geometries,
        "original_scene": original_scene,
        "geometry_titles": {
            "encoded_voxels": "Encoded Voxels",
            "room_pcd": "Room Point Cloud",
            "input_voxels": "Input Voxels",
            "gaussian_centers": "Gaussian Centers",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(viewer_data), encoding="utf-8")
    result = {"status": "ok", "viewer": str(output_path)}
    if svg_exports:
        result["voxel_svgs"] = svg_exports
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
