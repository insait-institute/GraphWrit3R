from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_DATASET_DIR = Path("/work/luka_milivojevic/3rscan_subset_scene_graph_data_rio10_chorus_fusion_raw_bridge")
DEFAULT_ALIGNED_ROOT = Path("/work/luka_milivojevic/3rscan_subset_chorus_sonata_native_cache_rio10_crop_radius_m0p25")

PLY_DTYPE_MAP = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_dir", type=Path, default=None, help="Scene cache dir containing coord/scale/quat/opacity/color .npy files.")
    parser.add_argument("--aligned_root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--scene_id", default=None, help="Scene split id, used with --aligned_root.")
    parser.add_argument("--split", choices=["train", "val", "auto"], default="auto")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--pcd_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path. Defaults to <aligned_dir>/<scene_id>_3dgs_viewer.html.")
    parser.add_argument("--max_gaussians", type=int, default=0, help="Use 0 for all splats; positive values subsample for lighter HTML.")
    parser.add_argument("--max_room_points", type=int, default=100000, help="Use 0 for all room PCD points.")
    parser.add_argument("--opacity_min", type=float, default=0.0)
    parser.add_argument("--min_scale", type=float, default=0.001)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--quat_order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--frame", choices=["auto", "stored", "sceneverse"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_aligned_dir(args: argparse.Namespace) -> Path:
    if args.aligned_dir is not None:
        return args.aligned_dir
    if not args.scene_id:
        raise ValueError("Provide either --aligned_dir or --scene_id.")
    candidates = []
    if args.split == "auto":
        candidates = [
            args.aligned_root / "train" / args.scene_id,
            args.aligned_root / "val" / args.scene_id,
            args.aligned_root / args.scene_id,
        ]
    else:
        candidates = [args.aligned_root / args.split / args.scene_id]
    for candidate in candidates:
        if (candidate / "coord.npy").exists():
            return candidate
    raise FileNotFoundError(f"Could not find aligned cache for scene_id={args.scene_id} under {args.aligned_root}.")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path: Path) -> Optional[dict[str, Any]]:
    return load_json(path) if path.exists() else None


def resolve_pcd_path(args: argparse.Namespace, aligned_dir: Path) -> Optional[Path]:
    if args.pcd_path is not None:
        return args.pcd_path
    scene_id = args.scene_id or aligned_dir.name
    candidate = args.dataset_dir / "pcd" / f"{scene_id}.ply"
    if candidate.exists():
        return candidate
    summary = load_optional_json(aligned_dir / "summary.json")
    if summary is not None:
        scene_pcd = summary.get("scene_pcd")
        if scene_pcd and Path(scene_pcd).exists():
            return Path(scene_pcd)
    return None


def _read_ply_header(handle) -> tuple[list[str], int]:
    lines = []
    while True:
        raw = handle.readline()
        if raw == b"":
            raise ValueError("Unexpected EOF while reading PLY header.")
        line = raw.decode("ascii", errors="replace").strip()
        lines.append(line)
        if line == "end_header":
            break
    return lines, handle.tell()


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        header, data_offset = _read_ply_header(handle)
        fmt = next((line.split()[1] for line in header if line.startswith("format ")), None)
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and len(parts) == 3 and parts[0] == "property":
                properties.append((parts[2], parts[1]))
            elif in_vertex and len(parts) >= 5 and parts[0] == "property" and parts[1] == "list":
                raise ValueError(f"List vertex properties are not supported in {path}.")
        names = [name for name, _ in properties]
        if vertex_count <= 0 or not {"x", "y", "z"}.issubset(names):
            raise ValueError(f"No vertex xyz data found in {path}.")
        rgb_names = None
        for candidate in (("red", "green", "blue"), ("r", "g", "b")):
            if set(candidate).issubset(names):
                rgb_names = candidate
                break

        if fmt == "ascii":
            handle.seek(data_offset)
            rows = np.loadtxt(handle, max_rows=vertex_count)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
            xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
            if rgb_names is None:
                rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
            else:
                rgb = rows[:, [names.index(name) for name in rgb_names]]
            return xyz, colors_to_uint8(rgb, xyz.shape[0])

        endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else None
        if endian is None:
            raise ValueError(f"Unsupported PLY format in {path}: {fmt}")
        dtype_fields = []
        for name, type_name in properties:
            dtype_code = PLY_DTYPE_MAP.get(type_name)
            if dtype_code is None:
                raise ValueError(f"Unsupported PLY property type {type_name!r} in {path}.")
            dtype_fields.append((name, endian + dtype_code))
        dtype = np.dtype(dtype_fields)
        handle.seek(data_offset)
        vertex = np.frombuffer(handle.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        if rgb_names is None:
            rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
        else:
            rgb = np.stack([vertex[name] for name in rgb_names], axis=1)
        return xyz, colors_to_uint8(rgb, xyz.shape[0])


def colors_to_uint8(colors: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] < 3:
        return np.full((count, 3), 180, dtype=np.uint8)
    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating) and float(np.nanmax(colors)) <= 1.1:
        colors = colors * 255.0
    return np.clip(np.nan_to_num(colors), 0, 255).astype(np.uint8)


def quat_to_rotmat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = quat_xyzw.astype(np.float32, copy=False)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    quat = np.empty((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        quat[positive, 3] = 0.25 * s
        quat[positive, 0] = (rot[positive, 2, 1] - rot[positive, 1, 2]) / s
        quat[positive, 1] = (rot[positive, 0, 2] - rot[positive, 2, 0]) / s
        quat[positive, 2] = (rot[positive, 1, 0] - rot[positive, 0, 1]) / s
    for axis in range(3):
        mask = ~positive
        for other in range(3):
            if other != axis:
                mask &= rot[:, axis, axis] >= rot[:, other, other]
        if not np.any(mask):
            continue
        if axis == 0:
            s = np.sqrt(1.0 + rot[mask, 0, 0] - rot[mask, 1, 1] - rot[mask, 2, 2]) * 2.0
            quat[mask, 3] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
            quat[mask, 0] = 0.25 * s
            quat[mask, 1] = (rot[mask, 0, 1] + rot[mask, 1, 0]) / s
            quat[mask, 2] = (rot[mask, 0, 2] + rot[mask, 2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + rot[mask, 1, 1] - rot[mask, 0, 0] - rot[mask, 2, 2]) * 2.0
            quat[mask, 3] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
            quat[mask, 0] = (rot[mask, 0, 1] + rot[mask, 1, 0]) / s
            quat[mask, 1] = 0.25 * s
            quat[mask, 2] = (rot[mask, 1, 2] + rot[mask, 2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[mask, 2, 2] - rot[mask, 0, 0] - rot[mask, 1, 1]) * 2.0
            quat[mask, 3] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s
            quat[mask, 0] = (rot[mask, 0, 2] + rot[mask, 2, 0]) / s
            quat[mask, 1] = (rot[mask, 1, 2] + rot[mask, 2, 1]) / s
            quat[mask, 2] = 0.25 * s
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    return quat.astype(np.float32)


def to_xyzw(quat: np.ndarray, order: str) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    if order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    return quat.astype(np.float32)


def find_sceneverse_transform(aligned_dir: Path) -> Optional[dict[str, Any]]:
    for path in (aligned_dir / "match_diagnostics.json",):
        diagnostics = load_optional_json(path)
        if diagnostics and isinstance(diagnostics.get("sceneverse_to_raw"), dict):
            return diagnostics["sceneverse_to_raw"]
    summary = load_optional_json(aligned_dir / "summary.json")
    if summary is not None and summary.get("source_raw_bridge_dir"):
        diagnostics = load_optional_json(Path(summary["source_raw_bridge_dir"]) / "match_diagnostics.json")
        if diagnostics and isinstance(diagnostics.get("sceneverse_to_raw"), dict):
            return diagnostics["sceneverse_to_raw"]
    return None


def apply_sceneverse_frame(
    coord: np.ndarray,
    quat_xyzw: np.ndarray,
    transform: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(transform["rotation_row_major"], dtype=np.float32)
    center = np.asarray(transform["center_after_rotation"], dtype=np.float32)
    coord_scene = coord.astype(np.float32, copy=False) @ rot - center.reshape(1, 3)
    raw_rot = quat_to_rotmat_xyzw(quat_xyzw)
    scene_rot = np.einsum("ij,njk->nik", rot.T, raw_rot)
    return coord_scene.astype(np.float32), rotmat_to_quat_xyzw(scene_rot)


def sample_indices(count: int, max_count: int, seed: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        if float(weights.sum()) > 0.0:
            weights = weights / weights.sum()
        else:
            weights = None
    return np.sort(rng.choice(count, size=max_count, replace=False, p=weights))


def b64_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return base64.b64encode(array.tobytes()).decode("ascii")


def prepare_payload(args: argparse.Namespace, aligned_dir: Path) -> dict[str, Any]:
    coord = np.load(aligned_dir / "coord.npy").astype(np.float32)
    scale = np.load(aligned_dir / "scale.npy").astype(np.float32)
    quat = to_xyzw(np.load(aligned_dir / "quat.npy"), args.quat_order)
    opacity = np.load(aligned_dir / "opacity.npy").astype(np.float32).reshape(-1)
    color = colors_to_uint8(np.load(aligned_dir / "color.npy"), coord.shape[0])
    if coord.shape[0] != scale.shape[0] or coord.shape[0] != quat.shape[0] or coord.shape[0] != opacity.shape[0]:
        raise ValueError("Gaussian arrays have inconsistent row counts.")

    frame = args.frame
    transform = find_sceneverse_transform(aligned_dir)
    if frame == "auto":
        frame = "sceneverse" if transform is not None else "stored"
    if frame == "sceneverse":
        if transform is None:
            raise FileNotFoundError("Requested --frame sceneverse, but no match_diagnostics.json transform was found.")
        coord, quat = apply_sceneverse_frame(coord, quat, transform)

    keep = np.ones((coord.shape[0],), dtype=bool)
    if args.opacity_min > 0:
        keep &= opacity >= args.opacity_min
    coord = coord[keep]
    scale = scale[keep]
    quat = quat[keep]
    opacity = opacity[keep]
    color = color[keep]

    scale = np.maximum(scale * float(args.scale_multiplier), float(args.min_scale)).astype(np.float32)
    weights = opacity * np.maximum(np.linalg.norm(scale, axis=1), 1e-6)
    indices = sample_indices(coord.shape[0], args.max_gaussians, args.seed, weights=weights)
    coord = coord[indices]
    scale = scale[indices]
    quat = quat[indices]
    opacity = opacity[indices]
    color = color[indices]
    rgba = np.concatenate(
        [color, np.round(np.clip(opacity, 0.0, 1.0)[:, None] * 255.0).astype(np.uint8)],
        axis=1,
    )

    pcd_path = resolve_pcd_path(args, aligned_dir)
    room_points = np.empty((0, 3), dtype=np.float32)
    room_rgb = np.empty((0, 3), dtype=np.uint8)
    if pcd_path is not None and pcd_path.exists():
        room_points, room_rgb = read_ply_xyz_rgb(pcd_path)
        room_indices = sample_indices(room_points.shape[0], args.max_room_points, args.seed + 17)
        room_points = room_points[room_indices]
        room_rgb = room_rgb[room_indices]

    all_points = coord if room_points.shape[0] == 0 else np.concatenate([coord, room_points], axis=0)
    bounds_min = all_points.min(axis=0).astype(np.float32)
    bounds_max = all_points.max(axis=0).astype(np.float32)
    center = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    span = float(np.max(bounds_max - bounds_min))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0

    metadata = {
        "scene_id": args.scene_id or aligned_dir.name,
        "aligned_dir": str(aligned_dir),
        "pcd_path": None if pcd_path is None else str(pcd_path),
        "frame": frame,
        "gaussians_source": int(np.load(aligned_dir / "coord.npy", mmap_mode="r").shape[0]),
        "gaussians_exported": int(coord.shape[0]),
        "room_points_exported": int(room_points.shape[0]),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "center": center.tolist(),
        "span": span,
    }
    return {
        "meta": metadata,
        "gaussian_count": int(coord.shape[0]),
        "room_count": int(room_points.shape[0]),
        "gaussian_position": b64_array(coord.astype("<f4", copy=False)),
        "gaussian_scale": b64_array(scale.astype("<f4", copy=False)),
        "gaussian_quat": b64_array(quat.astype("<f4", copy=False)),
        "gaussian_rgba": b64_array(rgba.astype(np.uint8, copy=False)),
        "room_position": b64_array(room_points.astype("<f4", copy=False)),
        "room_rgb": b64_array(room_rgb.astype(np.uint8, copy=False)),
    }


def build_html(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DGS Scene Viewer</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #0b0f14; color: #e5edf5; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #gl {{ width: 100vw; height: 100vh; display: block; }}
    .panel {{ position: fixed; left: 16px; top: 16px; max-width: 380px; background: rgba(11, 15, 20, 0.84); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 10px; padding: 12px 14px; backdrop-filter: blur(8px); }}
    .panel h1 {{ margin: 0 0 6px; font-size: 15px; line-height: 1.25; }}
    .meta {{ color: #aab6c5; font-size: 12px; line-height: 1.45; margin-bottom: 10px; }}
    .controls {{ display: grid; gap: 8px; }}
    label {{ display: grid; grid-template-columns: 120px 1fr 46px; align-items: center; gap: 8px; font-size: 12px; color: #cbd5e1; }}
    label.check {{ display: flex; gap: 8px; }}
    input[type="range"] {{ width: 100%; }}
    button {{ background: #e5edf5; color: #111827; border: 0; border-radius: 8px; padding: 7px 10px; font-weight: 700; cursor: pointer; }}
    button:hover {{ background: #ffffff; }}
    .hint {{ position: fixed; right: 16px; bottom: 14px; color: #94a3b8; font-size: 12px; background: rgba(11, 15, 20, 0.68); padding: 8px 10px; border-radius: 8px; }}
    .error {{ position: fixed; inset: 20px; display: none; background: #450a0a; color: #fee2e2; border: 1px solid #ef4444; border-radius: 12px; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <canvas id="gl"></canvas>
  <div class="panel">
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
    <div class="controls">
      <label class="check"><input id="showSplats" type="checkbox" checked> 3DGS splats</label>
      <label class="check"><input id="showRoom" type="checkbox" checked> room PCD</label>
      <label>Splat radius <input id="radius" type="range" min="0.5" max="4.0" step="0.1" value="2.0"><span id="radiusV">2.0</span></label>
      <label>Splat scale <input id="scale" type="range" min="0.1" max="5.0" step="0.1" value="1.0"><span id="scaleV">1.0</span></label>
      <label>Opacity <input id="opacity" type="range" min="0.05" max="2.0" step="0.05" value="1.0"><span id="opacityV">1.0</span></label>
      <label>Point size <input id="pointSize" type="range" min="1" max="8" step="0.5" value="2.0"><span id="pointSizeV">2.0</span></label>
      <button id="exportPng" type="button">Export PNG</button>
    </div>
  </div>
  <div class="hint">Drag to orbit. Wheel to zoom.</div>
  <div class="error" id="error"></div>
  <script>
    const PAYLOAD = {serialized};
    const canvas = document.getElementById('gl');
    const errorBox = document.getElementById('error');
    const controls = {{
      showSplats: document.getElementById('showSplats'),
      showRoom: document.getElementById('showRoom'),
      radius: document.getElementById('radius'),
      scale: document.getElementById('scale'),
      opacity: document.getElementById('opacity'),
      pointSize: document.getElementById('pointSize')
    }};

    function showError(message) {{
      errorBox.textContent = message;
      errorBox.style.display = 'block';
    }}

    function decodeArray(b64, Ctor) {{
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return new Ctor(bytes.buffer);
    }}

    const meta = PAYLOAD.meta;
    document.getElementById('title').textContent = meta.scene_id;
    document.getElementById('meta').innerHTML =
      `${{meta.gaussians_exported.toLocaleString()}} / ${{meta.gaussians_source.toLocaleString()}} splats<br>` +
      `${{meta.room_points_exported.toLocaleString()}} room points<br>` +
      `frame: ${{meta.frame}}`;

    const gl = canvas.getContext('webgl2', {{antialias: true, alpha: false, preserveDrawingBuffer: true}});
    if (!gl) showError('WebGL2 is required for the 3DGS viewer.');

    const gaussianPosition = decodeArray(PAYLOAD.gaussian_position, Float32Array);
    const gaussianScale = decodeArray(PAYLOAD.gaussian_scale, Float32Array);
    const gaussianQuat = decodeArray(PAYLOAD.gaussian_quat, Float32Array);
    const gaussianRgba = decodeArray(PAYLOAD.gaussian_rgba, Uint8Array);
    const roomPosition = decodeArray(PAYLOAD.room_position, Float32Array);
    const roomRgb = decodeArray(PAYLOAD.room_rgb, Uint8Array);

    function compileShader(type, source) {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        throw new Error(gl.getShaderInfoLog(shader));
      }}
      return shader;
    }}

    function makeProgram(vs, fs) {{
      const program = gl.createProgram();
      gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vs));
      gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
        throw new Error(gl.getProgramInfoLog(program));
      }}
      return program;
    }}

    const splatVs = `#version 300 es
    precision highp float;
    layout(location=0) in vec2 aCorner;
    layout(location=1) in vec3 aCenter;
    layout(location=2) in vec3 aScale;
    layout(location=3) in vec4 aQuat;
    layout(location=4) in vec4 aColor;
    uniform mat4 uView;
    uniform mat4 uProj;
    uniform mat4 uViewProj;
    uniform float uRadius;
    uniform float uScaleMul;
    uniform float uMaxNdcRadius;
    out vec2 vCorner;
    out vec4 vColor;
    mat3 quatToMat(vec4 q) {{
      q = normalize(q);
      float x=q.x, y=q.y, z=q.z, w=q.w;
      return mat3(
        1.0-2.0*(y*y+z*z), 2.0*(x*y+z*w), 2.0*(x*z-y*w),
        2.0*(x*y-z*w), 1.0-2.0*(x*x+z*z), 2.0*(y*z+x*w),
        2.0*(x*z+y*w), 2.0*(y*z-x*w), 1.0-2.0*(x*x+y*y)
      );
    }}
    void main() {{
      vec4 viewCenter4 = uView * vec4(aCenter, 1.0);
      float d = max(0.02, -viewCenter4.z);
      mat3 viewRot = mat3(uView);
      mat3 R = viewRot * quatToMat(aQuat);
      vec3 s = max(aScale * uScaleMul, vec3(0.0001));
      mat3 S = mat3(
        s.x*s.x, 0.0, 0.0,
        0.0, s.y*s.y, 0.0,
        0.0, 0.0, s.z*s.z
      );
      mat3 cov3 = R * S * transpose(R);
      float fx = uProj[0][0];
      float fy = uProj[1][1];
      vec3 jx = vec3(fx / d, 0.0, fx * viewCenter4.x / (d*d));
      vec3 jy = vec3(0.0, fy / d, fy * viewCenter4.y / (d*d));
      float a = dot(jx, cov3 * jx);
      float b = dot(jx, cov3 * jy);
      float c = dot(jy, cov3 * jy);
      float mid = 0.5 * (a + c);
      float rad = sqrt(max(0.0, 0.25 * (a - c) * (a - c) + b * b));
      float l1 = max(mid + rad, 1e-10);
      float l2 = max(mid - rad, 1e-10);
      vec2 e1 = normalize(abs(b) > 1e-8 ? vec2(l1 - c, b) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)));
      vec2 e2 = vec2(-e1.y, e1.x);
      vec2 ndcOffset = (e1 * aCorner.x * sqrt(l1) + e2 * aCorner.y * sqrt(l2)) * uRadius;
      float ndcLen = length(ndcOffset);
      if (ndcLen > uMaxNdcRadius) ndcOffset *= uMaxNdcRadius / ndcLen;
      vec4 clip = uProj * viewCenter4;
      clip.xy += ndcOffset * clip.w;
      gl_Position = clip;
      vCorner = aCorner * uRadius;
      vColor = aColor;
    }}`;

    const splatFs = `#version 300 es
    precision highp float;
    in vec2 vCorner;
    in vec4 vColor;
    uniform float uOpacityMul;
    out vec4 outColor;
    void main() {{
      float alpha = vColor.a * uOpacityMul * exp(-0.5 * dot(vCorner, vCorner));
      if (alpha < 0.01) discard;
      outColor = vec4(vColor.rgb, clamp(alpha, 0.0, 1.0));
    }}`;

    const pointVs = `#version 300 es
    precision highp float;
    layout(location=0) in vec3 aPosition;
    layout(location=1) in vec3 aColor;
    uniform mat4 uViewProj;
    uniform float uPointSize;
    out vec3 vColor;
    void main() {{
      gl_Position = uViewProj * vec4(aPosition, 1.0);
      gl_PointSize = uPointSize;
      vColor = aColor;
    }}`;

    const pointFs = `#version 300 es
    precision highp float;
    in vec3 vColor;
    out vec4 outColor;
    void main() {{
      vec2 p = gl_PointCoord * 2.0 - 1.0;
      if (dot(p, p) > 1.0) discard;
      outColor = vec4(vColor, 1.0);
    }}`;

    let splatProgram, pointProgram;
    try {{
      splatProgram = makeProgram(splatVs, splatFs);
      pointProgram = makeProgram(pointVs, pointFs);
    }} catch (err) {{
      showError('Shader compile/link failed:\\n' + err.message);
    }}

    function makeBuffer(data, target=gl.ARRAY_BUFFER) {{
      const buffer = gl.createBuffer();
      gl.bindBuffer(target, buffer);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return buffer;
    }}

    const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const splatVao = gl.createVertexArray();
    gl.bindVertexArray(splatVao);
    makeBuffer(quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    makeBuffer(gaussianPosition);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    makeBuffer(gaussianScale);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(2, 1);
    makeBuffer(gaussianQuat);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(3, 1);
    makeBuffer(gaussianRgba);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.vertexAttribDivisor(4, 1);

    const pointVao = gl.createVertexArray();
    gl.bindVertexArray(pointVao);
    makeBuffer(roomPosition);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    makeBuffer(roomRgb);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindVertexArray(null);

    function mat4Perspective(fovy, aspect, near, far) {{
      const f = 1.0 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return new Float32Array([
        f/aspect,0,0,0,
        0,f,0,0,
        0,0,(far+near)*nf,-1,
        0,0,(2*far*near)*nf,0
      ]);
    }}
    function normalize(v) {{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }}
    function cross(a,b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
    function dot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}
    function mat4LookAt(eye, center, up) {{
      const z = normalize([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
      ]);
    }}
    function mat4Mul(a,b) {{
      const out = new Float32Array(16);
      for (let c=0;c<4;c++) for (let r=0;r<4;r++) {{
        out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
      }}
      return out;
    }}

    const center = meta.center;
    let radius = meta.span * 1.7;
    let theta = Math.PI * 0.22;
    let phi = Math.PI * 0.34;
    let dragging = false;
    let lastX = 0, lastY = 0;
    canvas.addEventListener('mousedown', (e) => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
    window.addEventListener('mouseup', () => {{ dragging = false; }});
    window.addEventListener('mousemove', (e) => {{
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      theta -= dx * 0.005;
      phi = Math.max(0.08, Math.min(Math.PI - 0.08, phi + dy * 0.005));
    }});
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      radius *= Math.exp(e.deltaY * 0.001);
      radius = Math.max(meta.span * 0.08, Math.min(meta.span * 20.0, radius));
    }}, {{passive:false}});

    function resize() {{
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w; canvas.height = h;
      }}
      gl.viewport(0, 0, canvas.width, canvas.height);
    }}
    window.addEventListener('resize', resize);

    function setUniformMat(program, name, value) {{ gl.uniformMatrix4fv(gl.getUniformLocation(program, name), false, value); }}
    function setUniform1f(program, name, value) {{ gl.uniform1f(gl.getUniformLocation(program, name), value); }}
    function updateLabels() {{
      for (const key of ['radius','scale','opacity','pointSize']) {{
        document.getElementById(key + 'V').textContent = parseFloat(controls[key].value).toFixed(1);
      }}
    }}
    for (const input of Object.values(controls)) input.addEventListener('input', updateLabels);
    updateLabels();

    function render() {{
      if (!gl || !splatProgram || !pointProgram) return;
      resize();
      const eye = [
        center[0] + radius * Math.sin(phi) * Math.cos(theta),
        center[1] + radius * Math.sin(phi) * Math.sin(theta),
        center[2] + radius * Math.cos(phi)
      ];
      const view = mat4LookAt(eye, center, [0,0,1]);
      const proj = mat4Perspective(Math.PI / 4, canvas.width / canvas.height, Math.max(0.001, meta.span * 0.001), meta.span * 50.0);
      const viewProj = mat4Mul(proj, view);
      gl.clearColor(0.96, 0.97, 0.98, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      if (controls.showRoom.checked && PAYLOAD.room_count > 0) {{
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.useProgram(pointProgram);
        setUniformMat(pointProgram, 'uViewProj', viewProj);
        setUniform1f(pointProgram, 'uPointSize', parseFloat(controls.pointSize.value));
        gl.bindVertexArray(pointVao);
        gl.drawArrays(gl.POINTS, 0, PAYLOAD.room_count);
      }}

      if (controls.showSplats.checked && PAYLOAD.gaussian_count > 0) {{
        gl.depthMask(false);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(splatProgram);
        setUniformMat(splatProgram, 'uView', view);
        setUniformMat(splatProgram, 'uProj', proj);
        setUniformMat(splatProgram, 'uViewProj', viewProj);
        setUniform1f(splatProgram, 'uRadius', parseFloat(controls.radius.value));
        setUniform1f(splatProgram, 'uScaleMul', parseFloat(controls.scale.value));
        setUniform1f(splatProgram, 'uOpacityMul', parseFloat(controls.opacity.value));
        setUniform1f(splatProgram, 'uMaxNdcRadius', 0.25);
        gl.bindVertexArray(splatVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, PAYLOAD.gaussian_count);
        gl.depthMask(true);
      }}
      requestAnimationFrame(render);
    }}

    document.getElementById('exportPng').addEventListener('click', () => {{
      const a = document.createElement('a');
      a.download = `${{meta.scene_id}}_3dgs.png`;
      a.href = canvas.toDataURL('image/png');
      a.click();
    }});
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    aligned_dir = resolve_aligned_dir(args)
    output = args.output
    if output is None:
        scene_name = args.scene_id or aligned_dir.name
        output = aligned_dir / f"{scene_name}_3dgs_viewer.html"
    payload = prepare_payload(args, aligned_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(payload), encoding="utf-8")
    print(json.dumps({"status": "ok", "viewer": str(output), **payload["meta"]}, indent=2))


if __name__ == "__main__":
    main()
