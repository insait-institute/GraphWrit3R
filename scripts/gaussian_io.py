from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

_EPS = 1e-9
_CHUNK_SIZE = 256
_SH_C0 = 0.28209479177387814
_UINT11_MASK = (1 << 11) - 1
_UINT10_MASK = (1 << 10) - 1
_UINT8_MASK = (1 << 8) - 1
_DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999)

DEFAULT_OUTLIER_FILTER = dict(
    enabled=True,
    method="mad",
    scale_mad_k=5.0,
    scale_threshold_factor=0.5,
    scale_max_real=10.0,
    scale_max_log=None,
    floater_candidate_quantile=0.995,
    floater_min_neighbors=3,
    floater_radius_mode="sqrt_scale",
    floater_radius_alpha=0.25,
    floater_radius_max=2.0,
    workers=1,
    verbose=True,
)


@dataclass
class GaussianReadResult:
    data: Dict[str, np.ndarray]
    input_path: str
    scene_name: str
    source_type: str
    raw_count: int
    kept_count: int
    outlier_filter_enabled: bool
    kept_indices: Optional[np.ndarray] = None
    filter_config: Optional[Dict[str, Any]] = None
    filter_report: Optional[Dict[str, Any]] = None

    def to_summary(self) -> Dict[str, Any]:
        summary = dict(
            input_path=self.input_path,
            scene_name=self.scene_name,
            source_type=self.source_type,
            raw_count=self.raw_count,
            kept_count=self.kept_count,
            outlier_filter=dict(
                enabled=self.outlier_filter_enabled,
                config=self.filter_config,
                report=self.filter_report,
            ),
        )
        if self.kept_indices is not None:
            summary["kept_index_shape"] = list(self.kept_indices.shape)
        return summary


def _log(
    logger: Optional[logging.Logger], level: int, message: str, *args: Any
) -> None:
    if logger is None:
        if args:
            message = message % args
        print(message)
    else:
        logger.log(level, message, *args)


def infer_scene_name(input_path: str | Path, explicit_name: Optional[str] = None) -> str:
    if explicit_name:
        return explicit_name
    path = Path(input_path)
    if path.is_dir():
        return path.name
    if path.suffix:
        return path.stem
    return path.name


def get_default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sorted_prefixed_names(names: Iterable[str], prefix: str) -> list[str]:
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda name: int(name.split("_")[-1]),
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + _EPS)
    sign = np.sign(quat[:, 0])
    sign[sign == 0] = 1
    return quat * sign[:, None]


def _read_standard_gaussian_ply(ply: PlyData) -> Dict[str, np.ndarray]:
    vertex = ply["vertex"]
    dtype_names = vertex.data.dtype.names or ()
    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    if "opacity" in dtype_names:
        opacity = _np_sigmoid(vertex["opacity"].astype(np.float32))
    else:
        opacity = np.ones(coord.shape[0], dtype=np.float32)

    scale_cols = _sorted_prefixed_names(dtype_names, "scale_")
    if scale_cols:
        scale = np.stack([np.exp(vertex[name]) for name in scale_cols], axis=-1).astype(
            np.float32
        )
    else:
        scale = np.ones((coord.shape[0], 3), dtype=np.float32)

    rot_cols = _sorted_prefixed_names(dtype_names, "rot_")
    if rot_cols:
        quat = np.stack([vertex[name] for name in rot_cols], axis=-1).astype(np.float32)
        quat = _normalize_quat(quat)
    else:
        quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (coord.shape[0], 1))

    dc_cols = _sorted_prefixed_names(dtype_names, "f_dc_")
    if len(dc_cols) >= 3:
        fdc = np.stack([vertex[name] for name in dc_cols[:3]], axis=-1).astype(np.float32)
        color = np.clip(fdc * _SH_C0 + 0.5, 0.0, 1.0) * 255.0
        color = color.astype(np.uint8)
    else:
        color = np.full((coord.shape[0], 3), 128, dtype=np.uint8)

    return dict(coord=coord, color=color, opacity=opacity, scale=scale, quat=quat)


def _read_compressed_gaussian_ply(ply: PlyData) -> Dict[str, np.ndarray]:
    chunk = ply["chunk"].data
    vertex = ply["vertex"].data
    num = vertex.shape[0]

    chunk_indices = np.arange(num, dtype=np.int64) // _CHUNK_SIZE
    chunk_indices = np.minimum(chunk_indices, len(chunk) - 1)

    min_x = chunk["min_x"][chunk_indices]
    min_y = chunk["min_y"][chunk_indices]
    min_z = chunk["min_z"][chunk_indices]
    max_x = chunk["max_x"][chunk_indices]
    max_y = chunk["max_y"][chunk_indices]
    max_z = chunk["max_z"][chunk_indices]

    min_scale_x = chunk["min_scale_x"][chunk_indices]
    min_scale_y = chunk["min_scale_y"][chunk_indices]
    min_scale_z = chunk["min_scale_z"][chunk_indices]
    max_scale_x = chunk["max_scale_x"][chunk_indices]
    max_scale_y = chunk["max_scale_y"][chunk_indices]
    max_scale_z = chunk["max_scale_z"][chunk_indices]

    min_r = chunk["min_r"][chunk_indices]
    min_g = chunk["min_g"][chunk_indices]
    min_b = chunk["min_b"][chunk_indices]
    max_r = chunk["max_r"][chunk_indices]
    max_g = chunk["max_g"][chunk_indices]
    max_b = chunk["max_b"][chunk_indices]

    packed_position = vertex["packed_position"].astype(np.uint32)
    packed_scale = vertex["packed_scale"].astype(np.uint32)
    packed_rotation = vertex["packed_rotation"].astype(np.uint32)
    packed_color = vertex["packed_color"].astype(np.uint32)

    px = ((packed_position >> 21) & _UINT11_MASK).astype(np.float32) / _UINT11_MASK
    py = ((packed_position >> 11) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    pz = (packed_position & _UINT11_MASK).astype(np.float32) / _UINT11_MASK

    coord = np.empty((num, 3), dtype=np.float32)
    coord[:, 0] = min_x * (1.0 - px) + max_x * px
    coord[:, 1] = min_y * (1.0 - py) + max_y * py
    coord[:, 2] = min_z * (1.0 - pz) + max_z * pz

    sx = ((packed_scale >> 21) & _UINT11_MASK).astype(np.float32) / _UINT11_MASK
    sy = ((packed_scale >> 11) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    sz = (packed_scale & _UINT11_MASK).astype(np.float32) / _UINT11_MASK

    scale_log = np.empty((num, 3), dtype=np.float32)
    scale_log[:, 0] = min_scale_x * (1.0 - sx) + max_scale_x * sx
    scale_log[:, 1] = min_scale_y * (1.0 - sy) + max_scale_y * sy
    scale_log[:, 2] = min_scale_z * (1.0 - sz) + max_scale_z * sz
    scale = np.exp(scale_log)

    norm = np.float32(1.0 / (np.sqrt(2.0) * 0.5))
    a = ((packed_rotation >> 20) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    b = ((packed_rotation >> 10) & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    c = (packed_rotation & _UINT10_MASK).astype(np.float32) / _UINT10_MASK
    a = (a - 0.5) * norm
    b = (b - 0.5) * norm
    c = (c - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
    which = packed_rotation >> 30

    quat = np.empty((num, 4), dtype=np.float32)
    mask = which == 0
    quat[mask, 0] = m[mask]
    quat[mask, 1] = a[mask]
    quat[mask, 2] = b[mask]
    quat[mask, 3] = c[mask]
    mask = which == 1
    quat[mask, 0] = a[mask]
    quat[mask, 1] = m[mask]
    quat[mask, 2] = b[mask]
    quat[mask, 3] = c[mask]
    mask = which == 2
    quat[mask, 0] = a[mask]
    quat[mask, 1] = b[mask]
    quat[mask, 2] = m[mask]
    quat[mask, 3] = c[mask]
    mask = which == 3
    quat[mask, 0] = a[mask]
    quat[mask, 1] = b[mask]
    quat[mask, 2] = c[mask]
    quat[mask, 3] = m[mask]
    quat = _normalize_quat(quat)

    cr = ((packed_color >> 24) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cg = ((packed_color >> 16) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cb = ((packed_color >> 8) & _UINT8_MASK).astype(np.float32) / _UINT8_MASK
    cw = (packed_color & _UINT8_MASK).astype(np.float32) / _UINT8_MASK

    r = min_r * (1.0 - cr) + max_r * cr
    g = min_g * (1.0 - cg) + max_g * cg
    b = min_b * (1.0 - cb) + max_b * cb
    fdc = np.stack([(r - 0.5) / _SH_C0, (g - 0.5) / _SH_C0, (b - 0.5) / _SH_C0], axis=-1)
    color = np.clip(fdc * _SH_C0 + 0.5, 0.0, 1.0) * 255.0
    color = color.astype(np.uint8)

    opacity = np.clip(cw, 0.0, 1.0).astype(np.float32)

    return dict(coord=coord, color=color, opacity=opacity, scale=scale, quat=quat)


def _read_gaussian_ply(path: Path) -> tuple[Dict[str, np.ndarray], str]:
    from plyfile import PlyData

    with path.open("rb") as handle:
        ply = PlyData.read(handle)
    vertex_names = ply["vertex"].data.dtype.names or ()
    if "packed_position" in vertex_names:
        return _read_compressed_gaussian_ply(ply), "compressed_ply"
    return _read_standard_gaussian_ply(ply), "standard_ply"


def _compute_quantiles(
    values: np.ndarray, quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {f"q{q:g}": float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))}


def _robust_upper_threshold(values: np.ndarray, k: float) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < _EPS:
        mad = float(max(np.std(values), _EPS))
    threshold = median + k * mad
    return threshold, median, mad


def _compute_scale_remove_mask(
    max_log_scale: np.ndarray,
    method: str,
    mad_k: float,
    threshold_factor: float,
    max_real_cap: float | None,
    max_log_cap: float | None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if method != "mad":
        raise ValueError(f"Unsupported scale method for Chorus release: {method}")

    base_threshold_log, median, mad = _robust_upper_threshold(max_log_scale, mad_k)
    base_threshold_real = float(np.exp(base_threshold_log))
    adjusted_threshold_real = base_threshold_real * threshold_factor
    adjusted_threshold_log = float(np.log(max(adjusted_threshold_real, _EPS)))

    adaptive_remove_mask = max_log_scale > adjusted_threshold_log
    real_cap_remove_mask = np.zeros_like(adaptive_remove_mask)
    log_cap_remove_mask = np.zeros_like(adaptive_remove_mask)

    if max_real_cap is not None:
        real_cap_remove_mask = np.exp(max_log_scale) > max_real_cap
    if max_log_cap is not None:
        log_cap_remove_mask = max_log_scale > max_log_cap

    remove_mask = adaptive_remove_mask | real_cap_remove_mask | log_cap_remove_mask
    stats = dict(
        method=method,
        median_log_scale=median,
        mad_log_scale=mad,
        mad_k=mad_k,
        base_threshold_log_scale=base_threshold_log,
        base_threshold_real_scale=base_threshold_real,
        threshold_factor=threshold_factor,
        threshold_log_scale=adjusted_threshold_log,
        threshold_real_scale=adjusted_threshold_real,
        max_real_cap=max_real_cap,
        max_log_cap=max_log_cap,
        removed_by_adaptive_threshold=int(adaptive_remove_mask.sum()),
        removed_by_real_cap=int(real_cap_remove_mask.sum()),
        removed_by_log_cap=int(log_cap_remove_mask.sum()),
        removed_total=int(remove_mask.sum()),
    )
    return remove_mask, stats


def _build_floater_radii(
    candidate_max_real_scale: np.ndarray,
    mode: str,
    alpha: float,
    max_radius: float,
) -> np.ndarray:
    if mode != "sqrt_scale":
        raise ValueError(f"Unsupported floater radius mode for Chorus release: {mode}")
    radii = alpha * np.sqrt(candidate_max_real_scale)
    return np.clip(radii, 0.0, max_radius)


def _compute_floaters(
    coords: np.ndarray,
    max_log_scale: np.ndarray,
    candidate_quantile: float,
    min_neighbors: int,
    radius_mode: str,
    radius_alpha: float,
    radius_max: float,
    workers: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    from scipy.spatial import cKDTree

    num_points = coords.shape[0]
    if num_points == 0:
        return np.zeros((0,), dtype=bool), dict(
            enabled=True,
            candidate_count=0,
            removed_count=0,
            candidate_quantile=candidate_quantile,
            candidate_threshold_log_scale=None,
            candidate_threshold_real_scale=None,
            radius_mode=radius_mode,
            radius_alpha=radius_alpha,
            radius_max=radius_max,
            min_neighbors=min_neighbors,
        )

    candidate_threshold = float(np.quantile(max_log_scale, candidate_quantile))
    candidate_mask = max_log_scale >= candidate_threshold
    candidate_idx = np.flatnonzero(candidate_mask)
    if candidate_idx.size == 0:
        return np.zeros((num_points,), dtype=bool), dict(
            enabled=True,
            candidate_count=0,
            removed_count=0,
            candidate_quantile=candidate_quantile,
            candidate_threshold_log_scale=candidate_threshold,
            candidate_threshold_real_scale=float(np.exp(candidate_threshold)),
            radius_mode=radius_mode,
            radius_alpha=radius_alpha,
            radius_max=radius_max,
            min_neighbors=min_neighbors,
        )

    max_real_scale = np.exp(max_log_scale[candidate_idx])
    radii = _build_floater_radii(max_real_scale, radius_mode, radius_alpha, radius_max)
    tree = cKDTree(coords)
    neighbor_counts = np.asarray(
        tree.query_ball_point(coords[candidate_idx], radii, return_length=True, workers=workers),
        dtype=np.int64,
    ) - 1

    remove_mask = np.zeros((num_points,), dtype=bool)
    remove_mask[candidate_idx] = neighbor_counts < min_neighbors
    stats = dict(
        enabled=True,
        candidate_count=int(candidate_idx.size),
        removed_count=int(remove_mask.sum()),
        candidate_quantile=candidate_quantile,
        candidate_threshold_log_scale=candidate_threshold,
        candidate_threshold_real_scale=float(np.exp(candidate_threshold)),
        radius_mode=radius_mode,
        radius_alpha=radius_alpha,
        radius_max=radius_max,
        min_neighbors=min_neighbors,
        radius_quantiles=_compute_quantiles(radii),
        neighbor_count_quantiles=_compute_quantiles(
            neighbor_counts.astype(np.float64, copy=False)
        ),
        neighbor_count_min=int(neighbor_counts.min()) if neighbor_counts.size else 0,
        neighbor_count_max=int(neighbor_counts.max()) if neighbor_counts.size else 0,
    )
    return remove_mask, stats


def _normalize_filter_cfg(
    input_reader_cfg: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    merged = dict(DEFAULT_OUTLIER_FILTER)
    if input_reader_cfg:
        outlier_cfg = input_reader_cfg.get("outlier_filter", input_reader_cfg)
        if outlier_cfg:
            merged.update(dict(outlier_cfg))
    return merged


def _apply_outlier_filter(
    data: Dict[str, np.ndarray],
    filter_cfg: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    coord = np.asarray(data["coord"], dtype=np.float32)
    scale = np.asarray(data["scale"], dtype=np.float32)
    total_points = int(coord.shape[0])
    max_real_scale = np.max(np.clip(scale, _EPS, None), axis=1)
    max_log_scale = np.log(max_real_scale)

    if filter_cfg.get("verbose", True):
        _log(logger, logging.INFO, "Raw 3DGS outlier filter settings:")
        for key in (
            "method",
            "scale_mad_k",
            "scale_threshold_factor",
            "scale_max_real",
            "scale_max_log",
            "floater_candidate_quantile",
            "floater_min_neighbors",
            "floater_radius_mode",
            "floater_radius_alpha",
            "floater_radius_max",
            "workers",
        ):
            _log(logger, logging.INFO, "  %s=%s", key, filter_cfg.get(key))

    scale_remove_mask, scale_stats = _compute_scale_remove_mask(
        max_log_scale=max_log_scale,
        method=str(filter_cfg.get("method", "mad")),
        mad_k=float(filter_cfg.get("scale_mad_k", 5.0)),
        threshold_factor=float(filter_cfg.get("scale_threshold_factor", 0.5)),
        max_real_cap=filter_cfg.get("scale_max_real"),
        max_log_cap=filter_cfg.get("scale_max_log"),
    )
    survivor_mask_after_scale = ~scale_remove_mask
    surviving_indices = np.flatnonzero(survivor_mask_after_scale)

    floater_remove_mask = np.zeros((total_points,), dtype=bool)
    if surviving_indices.size > 0:
        local_floater_mask, floater_stats = _compute_floaters(
            coords=coord[surviving_indices],
            max_log_scale=max_log_scale[surviving_indices],
            candidate_quantile=float(filter_cfg.get("floater_candidate_quantile", 0.995)),
            min_neighbors=int(filter_cfg.get("floater_min_neighbors", 3)),
            radius_mode=str(filter_cfg.get("floater_radius_mode", "sqrt_scale")),
            radius_alpha=float(filter_cfg.get("floater_radius_alpha", 0.25)),
            radius_max=float(filter_cfg.get("floater_radius_max", 2.0)),
            workers=int(filter_cfg.get("workers", 1)),
        )
        floater_remove_mask[surviving_indices] = local_floater_mask
    else:
        floater_stats = dict(
            enabled=True,
            candidate_count=0,
            removed_count=0,
            candidate_quantile=float(filter_cfg.get("floater_candidate_quantile", 0.995)),
            candidate_threshold_log_scale=None,
            candidate_threshold_real_scale=None,
            radius_mode=str(filter_cfg.get("floater_radius_mode", "sqrt_scale")),
            radius_alpha=float(filter_cfg.get("floater_radius_alpha", 0.25)),
            radius_max=float(filter_cfg.get("floater_radius_max", 2.0)),
            min_neighbors=int(filter_cfg.get("floater_min_neighbors", 3)),
        )

    keep_mask = ~(scale_remove_mask | floater_remove_mask)
    kept_indices = np.flatnonzero(keep_mask).astype(np.int64)
    if kept_indices.size == 0:
        raise RuntimeError("Outlier filter removed all splats from the input scene.")

    filtered = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == total_points:
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value

    report = dict(
        total_points=total_points,
        removed_by_scale=int(scale_remove_mask.sum()),
        removed_by_floater=int(floater_remove_mask.sum()),
        kept_points=int(kept_indices.size),
        scale_filter=scale_stats,
        floater_filter=floater_stats,
        distributions=dict(
            original_max_log_scale_quantiles=_compute_quantiles(max_log_scale),
            original_max_real_scale_quantiles=_compute_quantiles(max_real_scale),
            final_max_log_scale_quantiles=_compute_quantiles(max_log_scale[keep_mask]),
            final_max_real_scale_quantiles=_compute_quantiles(max_real_scale[keep_mask]),
        ),
    )

    if filter_cfg.get("verbose", True):
        _log(
            logger,
            logging.INFO,
            "Raw 3DGS outlier filter kept %d/%d splats (removed scale=%d, floater=%d).",
            kept_indices.size,
            total_points,
            report["removed_by_scale"],
            report["removed_by_floater"],
        )

    return filtered, kept_indices, report


def _load_scene_directory(
    input_dir: Path,
    required_keys: Iterable[str],
    optional_keys: Iterable[str] = (),
) -> Dict[str, np.ndarray]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    keys_to_try = {"coord", *required_keys, *optional_keys}
    data: Dict[str, np.ndarray] = {}
    for key in keys_to_try:
        path = input_dir / f"{key}.npy"
        if path.exists():
            data[key] = np.load(path)

    missing = [key for key in required_keys if key not in data]
    if missing:
        missing_str = ", ".join(f"{key}.npy" for key in missing)
        raise FileNotFoundError(f"Missing required input arrays in {input_dir}: {missing_str}")
    if "coord" not in data:
        raise FileNotFoundError(f"Missing coord.npy in {input_dir}")
    return data


def load_gaussian_input(
    input_path: str | Path,
    required_keys: Iterable[str],
    *,
    optional_keys: Iterable[str] = (),
    scene_name: Optional[str] = None,
    input_reader_cfg: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> GaussianReadResult:
    path = Path(input_path)
    resolved_scene_name = infer_scene_name(path, explicit_name=scene_name)

    if path.is_dir():
        data = _load_scene_directory(path, required_keys, optional_keys=optional_keys)
        data["name"] = resolved_scene_name
        raw_count = int(np.asarray(data["coord"]).shape[0])
        return GaussianReadResult(
            data=data,
            input_path=str(path),
            scene_name=resolved_scene_name,
            source_type="npy_dir",
            raw_count=raw_count,
            kept_count=raw_count,
            outlier_filter_enabled=False,
        )

    if not path.is_file() or path.suffix.lower() != ".ply":
        raise FileNotFoundError(f"Input path is not a supported scene directory or .ply file: {path}")

    data, source_type = _read_gaussian_ply(path)
    missing = [key for key in required_keys if key != "coord" and key not in data]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Raw PLY input does not provide required keys for this inference config: {missing_str}. "
            "For the point-params Chorus checkpoint, use a preprocessed scene folder with normal.npy."
        )

    filter_cfg = _normalize_filter_cfg(input_reader_cfg)
    raw_count = int(data["coord"].shape[0])
    kept_indices: Optional[np.ndarray] = None
    filter_report: Optional[Dict[str, Any]] = None
    outlier_filter_enabled = bool(filter_cfg.get("enabled", True))
    if outlier_filter_enabled:
        data, full_kept_indices, filter_report = _apply_outlier_filter(
            data,
            filter_cfg,
            logger=logger,
        )
        if full_kept_indices.shape[0] != raw_count:
            kept_indices = full_kept_indices
    data["name"] = resolved_scene_name
    kept_count = int(data["coord"].shape[0])

    return GaussianReadResult(
        data=data,
        input_path=str(path),
        scene_name=resolved_scene_name,
        source_type=source_type,
        raw_count=raw_count,
        kept_count=kept_count,
        outlier_filter_enabled=outlier_filter_enabled,
        kept_indices=kept_indices,
        filter_config=filter_cfg if outlier_filter_enabled else None,
        filter_report=filter_report,
    )
