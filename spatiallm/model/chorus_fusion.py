from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spatiallm.model import mm_debug
from spatiallm.model.sonata_encoder import Point, fourier_encode_vector


CONFIG_ALIASES = {
    "chorus_3dgs": "configs/inference/lang-enc-pretrain-chorus-3dgs.py",
    "chorus_pts": "configs/inference/lang-enc-pretrain-chorus-from-pts-params.py",
}

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
INVALID_MATCH_GRID = (-1_000_000_000, -1_000_000_000, -1_000_000_000)
SONATA_PHYSICAL_GRID_KEY = "sonata_physical_grid"


def _add_repo_to_path(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser()
    if not root.is_absolute():
        repo_relative_root = Path(__file__).resolve().parents[2] / root
        root = repo_relative_root if repo_relative_root.exists() else Path.cwd() / root
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Chorus repo root does not exist: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _resolve_chorus_config(repo_root: Path, config: str) -> Path:
    if config in CONFIG_ALIASES:
        return repo_root / CONFIG_ALIASES[config]
    return Path(config).expanduser().resolve()


def _load_scene_arrays(input_dir: Path, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    data = {}
    for key in {
        "coord",
        *required_keys,
        "sonata_grid",
        "sonata_grid_raw_chorus",
        "sonata_grid_raw_bridge",
        SONATA_PHYSICAL_GRID_KEY,
    }:
        path = input_dir / f"{key}.npy"
        if path.exists():
            data[key] = np.load(path)
    missing = [key for key in ("coord", *required_keys) if key not in data]
    if missing:
        raise FileNotFoundError(
            f"Missing required aligned Chorus arrays in {input_dir}: {', '.join(missing)}"
        )
    return data


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _np_sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _sorted_prefixed_names(names: tuple[str, ...], prefix: str) -> list[str]:
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


def _normalize_quat_np(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + EPS)
    sign = np.sign(quat[:, :1])
    sign[sign == 0] = 1.0
    return quat * sign


def _read_ply_vertex_data(path: Path) -> np.ndarray:
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
    if not vertex_props:
        raise ValueError(f"Missing vertex properties in {path}")

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


def _read_gaussian_ply_minimal(path: Path) -> dict[str, np.ndarray]:
    vertex = _read_ply_vertex_data(path)
    names = vertex.dtype.names or ()
    if "packed_position" in names:
        raise ValueError(
            f"Compressed Gaussian PLY is not supported by online native Chorus loading: {path}"
        )
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"Missing vertex property {required!r} in {path}")

    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    if "opacity" in names:
        opacity = _np_sigmoid(vertex["opacity"].astype(np.float32)).astype(np.float32)
    else:
        opacity = np.ones((coord.shape[0],), dtype=np.float32)

    scale_cols = _sorted_prefixed_names(names, "scale_")
    if len(scale_cols) >= 3:
        scale = np.stack(
            [np.exp(vertex[name].astype(np.float32)) for name in scale_cols[:3]],
            axis=-1,
        ).astype(np.float32)
    else:
        scale = np.ones((coord.shape[0], 3), dtype=np.float32)

    rot_cols = _sorted_prefixed_names(names, "rot_")
    if len(rot_cols) >= 4:
        quat = np.stack([vertex[name] for name in rot_cols[:4]], axis=-1).astype(np.float32)
        quat = _normalize_quat_np(quat)
    else:
        quat = np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (coord.shape[0], 1),
        )

    dc_cols = _sorted_prefixed_names(names, "f_dc_")
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

    return {
        "coord": coord,
        "color": color,
        "opacity": opacity,
        "scale": scale,
        "quat": quat,
    }


def _bridge_grid(coord: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((coord - origin.reshape(1, 3)) / voxel_size).astype(np.int64)


def _fit_rigid_rows(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit target ~= source @ rotation + translation for row-vector coordinates."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid.reshape(1, 3)
    target_centered = target - target_centroid.reshape(1, 3)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    translation = target_centroid - source_centroid @ rotation
    return rotation.astype(np.float32), translation.astype(np.float32)


def _neighborhood_offsets(radius: int) -> np.ndarray:
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


def _encode_grid_rows(rows: np.ndarray, min_key: np.ndarray, dims: np.ndarray) -> np.ndarray:
    shifted = rows.astype(np.int64, copy=False) - min_key.reshape(1, 3)
    return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]


def _assign_native_sonata_grid(
    raw_coord: np.ndarray,
    sonata_grid_raw_bridge: np.ndarray,
    sonata_grid: np.ndarray,
    sonata_centers_raw: np.ndarray,
    voxel_size: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_origin = sonata_centers_raw.min(axis=0).astype(np.float32)
    raw_grid = _bridge_grid(raw_coord, raw_origin, voxel_size)
    sonata_raw = np.asarray(sonata_grid_raw_bridge, dtype=np.int64)
    sonata_grid = np.asarray(sonata_grid, dtype=np.int64)
    if sonata_raw.shape[0] != sonata_grid.shape[0]:
        raise ValueError(
            "sonata_grid_raw_bridge and sonata_grid row counts differ: "
            f"{sonata_raw.shape[0]} != {sonata_grid.shape[0]}"
        )
    if sonata_raw.shape[0] == 0:
        raise ValueError("No Sonata raw bridge grid rows available for native Chorus matching.")

    offsets = _neighborhood_offsets(max(int(radius), 0))
    candidate_grid = (sonata_raw[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    offset_sq = np.tile(np.sum(offsets * offsets, axis=1), sonata_raw.shape[0])
    offset_linf = np.tile(np.max(np.abs(offsets), axis=1), sonata_raw.shape[0])
    candidate_labels = np.repeat(sonata_grid, offsets.shape[0], axis=0)

    min_key = np.minimum(raw_grid.min(axis=0), candidate_grid.min(axis=0))
    max_key = np.maximum(raw_grid.max(axis=0), candidate_grid.max(axis=0))
    dims = np.maximum(max_key - min_key + 1, 1).astype(np.int64)
    candidate_keys = _encode_grid_rows(candidate_grid, min_key, dims)
    raw_keys = _encode_grid_rows(raw_grid, min_key, dims)

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
    return valid, assigned_labels


def _find_aligned_dir(root: Path, scene_id: str) -> Optional[Path]:
    candidates = (
        root / "train" / scene_id,
        root / "val" / scene_id,
        root / scene_id,
    )
    for candidate in candidates:
        if (candidate / "coord.npy").exists() or (candidate / "summary.json").exists():
            return candidate
    return None


def _disable_drop_path(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "drop_prob"):
            child.drop_prob = 0.0


def _match_grid_indices(source_grid: torch.Tensor, target_grid: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    source = [tuple(int(v) for v in row) for row in source_grid.detach().cpu().tolist()]
    target = [tuple(int(v) for v in row) for row in target_grid.detach().cpu().tolist()]
    source_map = {key: idx for idx, key in enumerate(source)}
    target_map = {key: idx for idx, key in enumerate(target)}
    common = sorted(set(source_map).intersection(target_map))
    source_idx = torch.as_tensor([source_map[key] for key in common], device=device, dtype=torch.long)
    target_idx = torch.as_tensor([target_map[key] for key in common], device=device, dtype=torch.long)
    return source_idx, target_idx


def _stride_reduction_bits(stride) -> int:
    product = 1
    try:
        values = list(stride)
    except TypeError:
        values = [stride]
    for value in values:
        product *= int(value)
    if product <= 1:
        return 0
    if product & (product - 1):
        return 0
    return product.bit_length() - 1


def _logit_clamped(prob: float, eps: float = 1e-4) -> float:
    prob = min(max(float(prob), eps), 1.0 - eps)
    return math.log(prob / (1.0 - prob))


def _append_transform_keys(compose, keys: tuple[str, ...]) -> None:
    transforms = getattr(compose, "transforms", None)
    if transforms is None:
        transforms = [compose]
    for transform in transforms:
        if not hasattr(transform, "keys"):
            continue
        current = transform.keys
        if isinstance(current, str):
            current_keys = (current,)
        else:
            current_keys = tuple(current)
        additions = tuple(key for key in keys if key not in current_keys)
        if additions:
            transform.keys = current_keys + additions


class SonataFourierHead(nn.Module):
    def __init__(self, reduced_grid_size: int, use_layer_norm: bool = False):
        super().__init__()
        self.reduced_grid_size = int(reduced_grid_size)
        self.proj = nn.Linear(512 + 63, 512)
        self.norm = nn.LayerNorm(512) if use_layer_norm else nn.Identity()

    def forward(self, feat: torch.Tensor, grid_coord: torch.Tensor) -> torch.Tensor:
        coords_normalised = grid_coord.float() / (self.reduced_grid_size - 1)
        encoded_coords = fourier_encode_vector(coords_normalised).to(dtype=feat.dtype)
        return self.norm(self.proj(torch.cat([feat, encoded_coords], dim=-1)))


class MatchedPairFusionLayer(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query_norm = self.query_norm(query)
        memory_norm = self.memory_norm(memory)
        attn_out, _ = self.cross_attn(
            query_norm,
            memory_norm,
            memory_norm,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class MatchedPairFusionTransformer(nn.Module):
    """Pair-local cross-attention merger for matched Sonata and Chorus tokens."""

    def __init__(
        self,
        token_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_layers = max(int(num_layers), 1)
        self.num_heads = max(int(num_heads), 1)
        self.ffn_dim = max(int(ffn_dim), self.token_dim)
        self.dropout = min(max(float(dropout), 0.0), 1.0)
        if self.token_dim % self.num_heads != 0:
            raise ValueError(
                "MatchedPairFusionTransformer requires token_dim to be divisible by num_heads: "
                f"token_dim={self.token_dim}, num_heads={self.num_heads}"
            )

        self.query_modality_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.memory_modality_embed = nn.Parameter(torch.zeros(1, 2, self.token_dim))
        self.layers = nn.ModuleList(
            [
                MatchedPairFusionLayer(
                    token_dim=self.token_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        nn.init.normal_(self.query_modality_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_modality_embed, mean=0.0, std=0.02)

    def forward(
        self,
        sonata_tokens: torch.Tensor,
        chorus_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if sonata_tokens.shape != chorus_tokens.shape:
            raise ValueError(
                "MatchedPairFusionTransformer expects equal Sonata/Chorus shapes, got "
                f"{tuple(sonata_tokens.shape)} and {tuple(chorus_tokens.shape)}."
            )
        if sonata_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "MatchedPairFusionTransformer token dim mismatch: "
                f"expected {self.token_dim}, got {sonata_tokens.shape[-1]}."
            )
        if sonata_tokens.shape[0] == 0:
            return sonata_tokens.new_empty((0, self.token_dim))

        output_dtype = sonata_tokens.dtype
        compute_param = next(self.parameters(), None)
        compute_dtype = compute_param.dtype if compute_param is not None else sonata_tokens.dtype
        sonata_tokens = sonata_tokens.to(dtype=compute_dtype)
        chorus_tokens = chorus_tokens.to(device=sonata_tokens.device, dtype=compute_dtype)

        query = sonata_tokens.unsqueeze(1) + self.query_modality_embed.to(
            device=sonata_tokens.device,
            dtype=compute_dtype,
        )
        memory = torch.stack((sonata_tokens, chorus_tokens), dim=1)
        memory = memory + self.memory_modality_embed.to(
            device=sonata_tokens.device,
            dtype=compute_dtype,
        )
        for layer in self.layers:
            query = layer(query, memory)
        return query.squeeze(1).to(dtype=output_dtype)


class ChorusFusionPointEncoder(nn.Module):
    """Return Sonata-space 3D tokens from PCD, aligned Chorus splats, or their fusion."""

    def __init__(
        self,
        sonata_backbone: nn.Module,
        chorus_repo_root: str,
        chorus_config: str,
        chorus_checkpoint: Optional[str],
        aligned_root: str,
        input_mode: str = "prepared",
        native_match_radius: int = -1,
        use_sonata_lattice_for_coord_matched: bool = False,
        min_valid_label_fraction: float = 0.5,
        trainable_name_patterns: str = "backbone.enc.enc4",
        mirror_sonata_trainability: bool = False,
        fusion_mode: str = "avg",
        missing_policy: str = "pcd",
        modality_dropout_rate: float = 0.0,
        drop_pcd_probability: float = 0.5,
        match_grid_radius: int = 0,
        exact_match_first: bool = False,
        discard_unmatched_tokens: bool = False,
        fusion_chorus_weight: float = 0.5,
        fusion_gate_mode: str = "fixed",
        fusion_gate_hidden_dim: int = 256,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 8,
        fusion_transformer_ffn_dim: int = 1024,
        fusion_transformer_dropout: float = 0.0,
        contrastive_loss_weight: float = 0.0,
        contrastive_exact_only: bool = False,
        contrastive_backprop_sonata: bool = False,
        contrastive_batch_wide_enabled: bool = False,
        contrastive_cosine_enabled: bool = True,
        contrastive_mse_enabled: bool = True,
        contrastive_info_nce_enabled: bool = True,
        contrastive_cosine_weight: float = 1.0,
        contrastive_mse_weight: float = 0.25,
        contrastive_info_nce_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        contrastive_min_matched_tokens: int = 32,
        use_layer_norm: bool = False,
        init_fourier_head_from_sonata: bool = True,
        share_fourier_head_with_sonata: bool = False,
        disable_drop_path: bool = True,
    ):
        super().__init__()
        object.__setattr__(self, "sonata_backbone", sonata_backbone)
        self.aligned_root = Path(aligned_root).expanduser().resolve()
        self.input_mode = str(input_mode)
        if self.input_mode not in {"prepared", "native"}:
            raise ValueError(
                f"Unsupported Chorus input mode: {self.input_mode}. "
                "Expected one of: prepared, native."
            )
        self.native_match_radius = int(native_match_radius)
        if self.native_match_radius < -1:
            raise ValueError("native_match_radius must be -1 or non-negative.")
        self.use_sonata_lattice_for_coord_matched = bool(use_sonata_lattice_for_coord_matched)
        self._sonata_lattice_transform_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self.min_valid_label_fraction = float(min_valid_label_fraction)
        if not 0.0 <= self.min_valid_label_fraction <= 1.0:
            raise ValueError("min_valid_label_fraction must be between 0 and 1.")
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {
            "avg",
            "append",
            "pcd",
            "chorus",
            "chorus_matched",
            "chorus_coord_matched",
            "transformer",
        }:
            raise ValueError(
                f"Unsupported Chorus fusion mode: {self.fusion_mode}. "
                "Expected one of: avg, append, pcd, chorus, chorus_matched, "
                "chorus_coord_matched, transformer."
            )
        self.missing_policy = missing_policy
        self.modality_dropout_rate = float(modality_dropout_rate)
        self.drop_pcd_probability = float(drop_pcd_probability)
        self.match_grid_radius = max(int(match_grid_radius), 0)
        self.exact_match_first = bool(exact_match_first)
        self.discard_unmatched_tokens = bool(discard_unmatched_tokens)
        self.fusion_chorus_weight = min(max(float(fusion_chorus_weight), 0.0), 1.0)
        self.fusion_sonata_weight = 1.0 - self.fusion_chorus_weight
        self.fusion_gate_mode = str(fusion_gate_mode)
        if self.fusion_gate_mode not in {"fixed", "global", "token"}:
            raise ValueError(
                f"Unsupported fusion gate mode: {self.fusion_gate_mode}. "
                "Expected one of: fixed, global, token."
            )
        self.fusion_gate_hidden_dim = max(int(fusion_gate_hidden_dim), 1)
        self.fusion_transformer_layers = max(int(fusion_transformer_layers), 1)
        self.fusion_transformer_heads = max(int(fusion_transformer_heads), 1)
        self.fusion_transformer_ffn_dim = max(int(fusion_transformer_ffn_dim), 1)
        self.fusion_transformer_dropout = min(
            max(float(fusion_transformer_dropout), 0.0),
            1.0,
        )
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.contrastive_exact_only = bool(contrastive_exact_only)
        self.contrastive_backprop_sonata = bool(contrastive_backprop_sonata)
        self.contrastive_batch_wide_enabled = bool(contrastive_batch_wide_enabled)
        self.contrastive_cosine_enabled = bool(contrastive_cosine_enabled)
        self.contrastive_mse_enabled = bool(contrastive_mse_enabled)
        self.contrastive_info_nce_enabled = bool(contrastive_info_nce_enabled)
        self.contrastive_cosine_weight = float(contrastive_cosine_weight)
        self.contrastive_mse_weight = float(contrastive_mse_weight)
        self.contrastive_info_nce_weight = float(contrastive_info_nce_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_min_matched_tokens = int(contrastive_min_matched_tokens)
        self.share_fourier_head_with_sonata = bool(share_fourier_head_with_sonata)
        self.mirror_sonata_trainability = bool(mirror_sonata_trainability)
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        self.last_contrastive_pair = None
        self.last_modality_usage = {}
        self.last_native_input_stats = {}

        chorus_root = _add_repo_to_path(chorus_repo_root)
        from pointcept.inference import LangPretrainerInference
        from pointcept.utils.config import Config

        chorus_cfg = Config.fromfile(str(_resolve_chorus_config(chorus_root, chorus_config)))
        self.feat_keys = tuple(chorus_cfg.get("feat_keys", ()))
        inferencer_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inferencer = LangPretrainerInference(
            chorus_cfg,
            chorus_checkpoint,
            device=inferencer_device,
        )
        sidecar_keys = (
            "sonata_grid",
            "sonata_grid_raw_chorus",
            "sonata_grid_raw_bridge",
            SONATA_PHYSICAL_GRID_KEY,
        )
        _append_transform_keys(self.inferencer.transform, sidecar_keys)
        if self.inferencer.test_voxelize is not None:
            _append_transform_keys(self.inferencer.test_voxelize, sidecar_keys)
        _append_transform_keys(self.inferencer.post_transform, sidecar_keys)
        self.chorus_model = self.inferencer.model
        self.chorus_cfg = chorus_cfg
        try:
            chorus_stride = chorus_cfg.model.backbone.get("stride", (2, 2, 2, 2))
        except AttributeError:
            chorus_stride = (2, 2, 2, 2)
        self.chorus_match_reduction_bits = _stride_reduction_bits(chorus_stride)
        self.last_sidecar_match_stats = {}

        if disable_drop_path:
            _disable_drop_path(self.chorus_model)

        reduced_grid_size = int(getattr(self.sonata_backbone, "reduced_grid_size", 80))
        self.fourier_head = SonataFourierHead(reduced_grid_size, use_layer_norm=use_layer_norm)
        gate_init_logit = _logit_clamped(self.fusion_chorus_weight)
        if self.fusion_gate_mode == "global":
            self.fusion_gate_logit = nn.Parameter(torch.tensor(gate_init_logit, dtype=torch.float32))
            self.fusion_gate_mlp = None
        elif self.fusion_gate_mode == "token":
            self.fusion_gate_logit = None
            self.fusion_gate_mlp = nn.Sequential(
                nn.Linear(512 * 3, self.fusion_gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.fusion_gate_hidden_dim, 1),
            )
            final_linear = self.fusion_gate_mlp[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, gate_init_logit)
        else:
            self.fusion_gate_logit = None
            self.fusion_gate_mlp = None
        if self.fusion_mode == "transformer":
            self.pair_fusion_transformer = MatchedPairFusionTransformer(
                token_dim=512,
                num_layers=self.fusion_transformer_layers,
                num_heads=self.fusion_transformer_heads,
                ffn_dim=self.fusion_transformer_ffn_dim,
                dropout=self.fusion_transformer_dropout,
            )
        else:
            self.pair_fusion_transformer = None
        if self.share_fourier_head_with_sonata:
            self.tie_fourier_head_to_sonata(copy_from_sonata=True)
        elif (
            init_fourier_head_from_sonata
            and getattr(self.sonata_backbone, "input_proj", None) is not None
        ):
            self.fourier_head.proj.load_state_dict(self.sonata_backbone.input_proj.state_dict())

        if self.mirror_sonata_trainability:
            self.sync_trainability_from_sonata()
        else:
            self._set_chorus_trainability(
                [p.strip() for p in trainable_name_patterns.split(",") if p.strip()]
            )

    def _align_chorus_to_sonata(
        self,
        chorus_context: torch.Tensor,
        chorus_grid: torch.Tensor,
        sonata_grid: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        exact_chorus_idx, exact_sonata_idx = _match_grid_indices(chorus_grid, sonata_grid, device)
        if self.match_grid_radius <= 0:
            return (
                chorus_context[exact_chorus_idx],
                exact_sonata_idx,
                int(exact_chorus_idx.numel()),
                exact_chorus_idx,
                exact_sonata_idx,
            )

        chorus_rows = chorus_grid.detach().cpu().tolist()
        sonata_rows = sonata_grid.detach().cpu().tolist()
        chorus_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, row in enumerate(chorus_rows):
            chorus_by_key[tuple(int(v) for v in row)].append(idx)

        exact_map = {
            int(sonata_idx): int(chorus_idx)
            for chorus_idx, sonata_idx in zip(exact_chorus_idx.tolist(), exact_sonata_idx.tolist())
        }
        aligned_by_sonata: dict[int, torch.Tensor] = {}
        used_chorus_indices: set[int] = set()
        radius = self.match_grid_radius
        for sonata_idx, row in enumerate(sonata_rows):
            exact_idx = exact_map.get(sonata_idx)
            if self.exact_match_first and exact_idx is not None:
                aligned_by_sonata[sonata_idx] = chorus_context[exact_idx]
                used_chorus_indices.add(exact_idx)
                continue
            x, y, z = (int(v) for v in row)
            neighbor_indices: list[int] = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        key = (x + dx, y + dy, z + dz)
                        matches = chorus_by_key.get(key)
                        if matches:
                            neighbor_indices.extend(matches)
            if not neighbor_indices:
                continue
            unique_neighbor_indices = sorted(set(neighbor_indices))
            neighbor_idx_tensor = torch.as_tensor(
                unique_neighbor_indices,
                device=device,
                dtype=torch.long,
            )
            aligned_by_sonata[sonata_idx] = chorus_context[neighbor_idx_tensor].mean(dim=0)
            used_chorus_indices.update(unique_neighbor_indices)

        if not aligned_by_sonata:
            empty_feat = chorus_context.new_empty((0, chorus_context.shape[-1]))
            empty_idx = torch.empty((0,), device=device, dtype=torch.long)
            return empty_feat, empty_idx, 0, exact_chorus_idx, exact_sonata_idx

        matched_sonata_idx = sorted(aligned_by_sonata)
        return (
            torch.stack([aligned_by_sonata[idx] for idx in matched_sonata_idx], dim=0),
            torch.as_tensor(matched_sonata_idx, device=device, dtype=torch.long),
            len(used_chorus_indices),
            exact_chorus_idx,
            exact_sonata_idx,
        )

    def _pool_chorus_by_match_grid(
        self,
        chorus_context: torch.Tensor,
        match_grid: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if chorus_context.shape[0] != match_grid.shape[0]:
            raise ValueError(
                "chorus_context and match_grid must have the same first dimension: "
                f"{chorus_context.shape[0]} != {match_grid.shape[0]}"
            )
        if chorus_context.shape[0] == 0:
            empty_feat = chorus_context.new_empty((0, chorus_context.shape[-1]))
            empty_grid = torch.empty((0, 3), device=device, dtype=torch.long)
            return empty_feat, empty_grid, 0

        match_grid = match_grid.to(device=device, dtype=torch.long)
        valid_mask = torch.all(match_grid >= 0, dim=1)
        valid_context = chorus_context[valid_mask]
        valid_grid = match_grid[valid_mask]
        used_chorus_tokens = int(valid_context.shape[0])
        if used_chorus_tokens == 0:
            empty_feat = chorus_context.new_empty((0, chorus_context.shape[-1]))
            empty_grid = torch.empty((0, 3), device=device, dtype=torch.long)
            return empty_feat, empty_grid, 0

        unique_grid, inverse = torch.unique(
            valid_grid,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        pooled = valid_context.new_zeros((unique_grid.shape[0], valid_context.shape[-1]))
        pooled.index_add_(0, inverse, valid_context)
        counts = torch.bincount(inverse, minlength=unique_grid.shape[0]).to(
            device=device,
            dtype=pooled.dtype,
        )
        pooled = pooled / counts.clamp_min(1).unsqueeze(1)
        return pooled, unique_grid, used_chorus_tokens

    def _filter_chorus_by_coordinate_intersection(
        self,
        chorus_context: torch.Tensor,
        chorus_grid: torch.Tensor,
        sonata_grid: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if chorus_context.shape[0] != chorus_grid.shape[0]:
            raise ValueError(
                "chorus_context and chorus_grid must have the same first dimension: "
                f"{chorus_context.shape[0]} != {chorus_grid.shape[0]}"
            )
        if chorus_context.shape[0] == 0 or sonata_grid.shape[0] == 0:
            empty_feat = chorus_context.new_empty((0, chorus_context.shape[-1]))
            empty_grid = torch.empty((0, 3), device=device, dtype=torch.long)
            empty_idx = torch.empty((0,), device=device, dtype=torch.long)
            return empty_feat, empty_grid, empty_idx, 0

        chorus_grid = chorus_grid.to(device=device, dtype=torch.long)
        sonata_grid = sonata_grid.to(device=device, dtype=torch.long)
        sonata_rows = [tuple(int(v) for v in row) for row in sonata_grid.detach().cpu().tolist()]
        sonata_index_by_coord: dict[tuple[int, int, int], int] = {}
        for idx, row in enumerate(sonata_rows):
            sonata_index_by_coord.setdefault(row, idx)

        groups: dict[tuple[int, int, int], tuple[list[int], int]] = {}
        chorus_rows = [tuple(int(v) for v in row) for row in chorus_grid.detach().cpu().tolist()]
        for chorus_idx, row in enumerate(chorus_rows):
            sonata_idx = sonata_index_by_coord.get(row)
            if sonata_idx is None:
                continue
            if row in groups:
                groups[row][0].append(chorus_idx)
            else:
                groups[row] = ([chorus_idx], sonata_idx)

        used_chorus_tokens = sum(len(indices) for indices, _ in groups.values())
        if not groups:
            empty_feat = chorus_context.new_empty((0, chorus_context.shape[-1]))
            empty_grid = torch.empty((0, 3), device=device, dtype=torch.long)
            empty_idx = torch.empty((0,), device=device, dtype=torch.long)
            return empty_feat, empty_grid, empty_idx, 0

        matched_context = []
        matched_grid = []
        matched_sonata_idx = []
        for row, (indices, sonata_idx) in groups.items():
            index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
            matched_context.append(chorus_context[index_tensor].mean(dim=0))
            matched_grid.append(row)
            matched_sonata_idx.append(sonata_idx)

        return (
            torch.stack(matched_context, dim=0),
            torch.as_tensor(matched_grid, device=device, dtype=torch.long),
            torch.as_tensor(matched_sonata_idx, device=device, dtype=torch.long),
            used_chorus_tokens,
        )

    def _set_chorus_trainability(self, trainable_patterns: list[str]) -> None:
        for param in self.chorus_model.parameters():
            param.requires_grad = False
        for name, param in self.chorus_model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                param.requires_grad = True

    def tie_fourier_head_to_sonata(self, copy_from_sonata: bool = True) -> None:
        input_proj = getattr(self.sonata_backbone, "input_proj", None)
        if input_proj is None:
            raise RuntimeError(
                "Cannot share Chorus Fourier projection with Sonata because "
                "`sonata_backbone.input_proj` is missing."
            )
        if self.fourier_head.proj is input_proj:
            return
        if copy_from_sonata:
            self.fourier_head.proj.load_state_dict(input_proj.state_dict())
        self.fourier_head.proj = input_proj

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.share_fourier_head_with_sonata:
            root_prefix = (
                prefix[: -len("chorus_fusion_encoder.")]
                if prefix.endswith("chorus_fusion_encoder.")
                else ""
            )
            for leaf in ("weight", "bias"):
                sonata_key = root_prefix + "point_backbone.input_proj." + leaf
                chorus_key = prefix + "fourier_head.proj." + leaf
                if sonata_key in state_dict:
                    state_dict[chorus_key] = state_dict[sonata_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if self.share_fourier_head_with_sonata:
            self.tie_fourier_head_to_sonata(copy_from_sonata=False)

    def sync_trainability_from_sonata(self) -> dict[str, int]:
        sonata_params = dict(self.sonata_backbone.named_parameters())
        matched = 0
        trainable = 0
        for param in self.chorus_model.parameters():
            param.requires_grad = False
        for name, param in self.chorus_model.backbone.named_parameters():
            sonata_param = sonata_params.get(name)
            if sonata_param is None:
                continue
            matched += 1
            param.requires_grad = sonata_param.requires_grad
            if param.requires_grad:
                trainable += 1

        input_proj = getattr(self.sonata_backbone, "input_proj", None)
        shared_proj = input_proj is not None and self.fourier_head.proj is input_proj
        for name, param in self.fourier_head.named_parameters():
            if shared_proj and name.startswith("proj."):
                continue
            param.requires_grad = False
        if input_proj is not None and hasattr(self.fourier_head, "proj"):
            input_proj_params = dict(input_proj.named_parameters())
            for name, param in self.fourier_head.proj.named_parameters():
                sonata_param = input_proj_params.get(name)
                if sonata_param is None:
                    continue
                matched += 1
                param.requires_grad = sonata_param.requires_grad
                if param.requires_grad:
                    trainable += 1

        return {"matched": matched, "trainable": trainable}

    def _encode_sonata(self, point_cloud: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        input_dict = {
            "coord": feats[:, :3].to(device),
            "grid_coord": coords.to(device),
            "feat": feats.to(device),
            "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
        }
        backbone = self.sonata_backbone
        point = Point(input_dict)
        point = backbone.embedding(point)
        point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
        point.sparsify()
        point = backbone.enc(point)
        context = point["sparse_conv_feat"].features
        grid = point["grid_coord"].long()
        if backbone.enable_fourier_encode:
            coords_normalised = grid / (backbone.reduced_grid_size - 1)
            encoded_coords = fourier_encode_vector(coords_normalised)
            context = torch.cat([context, encoded_coords], dim=-1)
            context = backbone.input_proj(context)
        return context, grid

    def _prepare_chorus_input(self, aligned_dir: Path, scene_id: str, device: torch.device) -> dict[str, torch.Tensor]:
        from pointcept.datasets.utils import collate_fn

        if self.input_mode == "native":
            data = self._load_native_chorus_arrays(aligned_dir, scene_id)
        else:
            data = _load_scene_arrays(aligned_dir, self.feat_keys)
        if self._should_use_sonata_lattice():
            data = self._add_sonata_physical_grid(data, aligned_dir, scene_id)
        prepared = self._prepare_chorus_input_dict(data, scene_id)
        fragments = prepared["fragment_list"]
        if len(fragments) != 1:
            raise RuntimeError(f"Expected one Chorus fragment for {scene_id}, got {len(fragments)}.")
        input_dict = collate_fn([fragments[0]])
        for key, value in list(input_dict.items()):
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device, non_blocking=True)
        return input_dict

    def _should_use_sonata_lattice(self) -> bool:
        return (
            self.use_sonata_lattice_for_coord_matched
            and self.fusion_mode in {"chorus", "chorus_coord_matched"}
        )

    def _get_sonata_lattice_transform(
        self,
        aligned_dir: Path,
        scene_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        summary_path = aligned_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Cannot compute Sonata lattice for {scene_id}: missing {summary_path}"
            )
        summary = _load_json(summary_path)
        source_dir = Path(summary.get("source_raw_bridge_dir", aligned_dir)).expanduser()
        if not source_dir.is_absolute():
            source_dir = (aligned_dir / source_dir).resolve()
        source_dir = source_dir.resolve()
        cached = self._sonata_lattice_transform_cache.get(source_dir)
        if cached is not None:
            return cached

        sonata_grid_path = source_dir / "sonata_grid.npy"
        sonata_centers_raw_path = source_dir / "sonata_centers_raw.npy"
        sonata_origin_path = source_dir / "sonata_origin.npy"
        missing = [
            str(path)
            for path in (sonata_grid_path, sonata_centers_raw_path, sonata_origin_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot compute Sonata lattice for {scene_id}; missing: {', '.join(missing)}"
            )

        voxel_size = float(summary.get("voxel_size", 0.025))
        sonata_grid = np.load(sonata_grid_path).astype(np.int64, copy=False)
        sonata_centers_raw = np.load(sonata_centers_raw_path).astype(np.float32, copy=False)
        sonata_origin = np.load(sonata_origin_path).astype(np.float32, copy=False)
        if sonata_grid.shape[0] != sonata_centers_raw.shape[0]:
            raise ValueError(
                f"sonata_grid and sonata_centers_raw row counts differ for {scene_id}: "
                f"{sonata_grid.shape[0]} != {sonata_centers_raw.shape[0]}"
            )

        sonata_centers_sv = sonata_origin.reshape(1, 3) + (
            sonata_grid.astype(np.float32) + 0.5
        ) * np.float32(voxel_size)
        rotation, translation = _fit_rigid_rows(sonata_centers_raw, sonata_centers_sv)
        cached = (rotation, translation, sonata_origin.astype(np.float32), voxel_size)
        self._sonata_lattice_transform_cache[source_dir] = cached
        return cached

    def _add_sonata_physical_grid(
        self,
        data: dict[str, np.ndarray],
        aligned_dir: Path,
        scene_id: str,
    ) -> dict[str, np.ndarray]:
        if "coord" not in data:
            raise KeyError(f"Cannot compute Sonata lattice for {scene_id}: data has no coord array.")
        rotation, translation, sonata_origin, voxel_size = self._get_sonata_lattice_transform(
            aligned_dir,
            scene_id,
        )
        coord = np.asarray(data["coord"], dtype=np.float32)
        coord_sv = coord @ rotation + translation.reshape(1, 3)
        sonata_grid = np.floor(
            (coord_sv - sonata_origin.reshape(1, 3)) / float(voxel_size)
        ).astype(np.int64)
        keep = np.isfinite(coord_sv).all(axis=1) & np.all(sonata_grid >= 0, axis=1)
        if not np.any(keep):
            raise RuntimeError(
                f"Sonata-lattice override removed every Chorus splat for scene_id={scene_id}."
            )
        if not np.all(keep):
            data = {
                key: value[keep] if isinstance(value, np.ndarray) and value.shape[:1] == keep.shape else value
                for key, value in data.items()
            }
            sonata_grid = sonata_grid[keep]
        data[SONATA_PHYSICAL_GRID_KEY] = sonata_grid.astype(np.int64, copy=False)
        self.last_native_input_stats = {
            **dict(self.last_native_input_stats),
            "sonata_lattice_override": True,
            "sonata_lattice_rows": int(sonata_grid.shape[0]),
            "sonata_lattice_kept_fraction": float(np.mean(keep)),
        }
        return data

    def _load_native_chorus_arrays(self, aligned_dir: Path, scene_id: str) -> dict[str, np.ndarray]:
        summary_path = aligned_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing raw bridge summary for native Chorus input: {summary_path}")
        summary = _load_json(summary_path)
        chorus_ply_value = summary.get("chorus_ply")
        if not chorus_ply_value:
            raise KeyError(f"summary.json for {scene_id} has no 'chorus_ply' field.")
        chorus_ply = Path(chorus_ply_value).expanduser()
        if not chorus_ply.exists():
            raise FileNotFoundError(f"Native Chorus PLY does not exist for {scene_id}: {chorus_ply}")

        sonata_grid_path = aligned_dir / "sonata_grid.npy"
        sonata_raw_path = aligned_dir / "sonata_grid_raw_bridge.npy"
        centers_raw_path = aligned_dir / "sonata_centers_raw.npy"
        missing_sidecars = [
            str(path.name)
            for path in (sonata_grid_path, sonata_raw_path, centers_raw_path)
            if not path.exists()
        ]
        if missing_sidecars:
            raise FileNotFoundError(
                f"Missing native Chorus matching sidecars for {scene_id}: "
                + ", ".join(missing_sidecars)
            )

        raw_data = _read_gaussian_ply_minimal(chorus_ply)
        missing_features = [key for key in self.feat_keys if key not in raw_data]
        if missing_features:
            raise KeyError(
                f"Native Chorus PLY for {scene_id} is missing features: "
                + ", ".join(sorted(missing_features))
            )

        sonata_grid = np.load(sonata_grid_path).astype(np.int64, copy=False)
        sonata_raw = np.load(sonata_raw_path).astype(np.int64, copy=False)
        sonata_centers_raw = np.load(centers_raw_path).astype(np.float32, copy=False)
        if sonata_centers_raw.shape[0] < sonata_raw.shape[0]:
            raise ValueError(
                f"sonata_centers_raw has too few rows for {scene_id}: "
                f"{sonata_centers_raw.shape[0]} < {sonata_raw.shape[0]}"
            )

        voxel_size = float(summary.get("voxel_size", 0.025))
        radius = self.native_match_radius
        if radius < 0:
            radius = int(summary.get("match_radius_voxels", 2))

        valid_mask, assigned_sonata_grid = _assign_native_sonata_grid(
            raw_data["coord"],
            sonata_raw,
            sonata_grid,
            sonata_centers_raw,
            voxel_size,
            radius,
        )
        kept = int(valid_mask.sum())
        raw_count = int(raw_data["coord"].shape[0])
        if kept == 0:
            raise RuntimeError(
                f"No raw 3DGS splats matched Sonata bridge voxels for {scene_id} "
                f"(radius={radius}, raw_splats={raw_count})."
            )

        data = {
            "coord": raw_data["coord"],
            "sonata_grid": assigned_sonata_grid,
        }
        for key in self.feat_keys:
            data[key] = raw_data[key]
        self.last_native_input_stats = {
            "enabled": True,
            "scene_id": scene_id,
            "input_mode": self.input_mode,
            "ply": str(chorus_ply),
            "raw_splats": raw_count,
            "cached_splats": raw_count,
            "valid_labeled_splats": kept,
            "valid_label_fraction": kept / max(raw_count, 1),
            "radius": int(radius),
            "voxel_size": voxel_size,
        }
        return data

    def _prepare_chorus_input_dict(
        self,
        data: dict[str, np.ndarray],
        scene_id: str,
    ) -> dict:
        base_dict = self.inferencer._format_numpy_inputs(data, scene_id)
        point_count = base_dict["coord"].shape[0]
        for key in (
            "sonata_grid",
            "sonata_grid_raw_chorus",
            "sonata_grid_raw_bridge",
            SONATA_PHYSICAL_GRID_KEY,
        ):
            if key not in data:
                continue
            array = np.asarray(data[key], dtype=np.int64)
            if array.shape[0] != point_count:
                raise ValueError(
                    f"{key} shape mismatch for {scene_id}: "
                    f"expected {point_count} rows, got {array.shape[0]}."
                )
            base_dict[key] = array

        data_dict = self.inferencer.transform(base_dict)
        result = dict(
            segment=data_dict.pop("segment", None),
            name=data_dict.pop("name", scene_id or self.inferencer.default_scene_name),
        )
        for key in (
            "coord",
            "pc_coord",
            "pc_segment",
            "origin_coord",
            "origin_feat_mask",
            "origin_instance",
            "origin_dino_feat",
            "origin_segment",
            "inverse",
        ):
            if key in data_dict:
                result[key] = data_dict[key]

        fragments = [aug(deepcopy(data_dict)) for aug in self.inferencer.aug_transform]
        fragment_list = []
        for fragment in fragments:
            if self.inferencer.test_voxelize is not None:
                data_slices = self.inferencer.test_voxelize(fragment)
            else:
                fragment["index"] = np.arange(fragment["coord"].shape[0])
                data_slices = [fragment]
            for slice_data in data_slices:
                if self.inferencer.test_crop is not None:
                    crop_list = self.inferencer.test_crop(slice_data)
                else:
                    crop_list = [slice_data]
                fragment_list.extend(crop_list)

        fragment_list = [self.inferencer.post_transform(frag) for frag in fragment_list]
        result["fragment_list"] = fragment_list
        return result

    def _runtime_sidecar_match_grid(
        self,
        input_dict: dict[str, torch.Tensor],
        chorus_grid: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        input_grid = input_dict.get("grid_coord")
        sonata_grid = input_dict.get("sonata_grid")
        if input_grid is None or sonata_grid is None:
            return None
        if input_grid.shape[0] != sonata_grid.shape[0]:
            self.last_sidecar_match_stats = {
                "enabled": False,
                "reason": "runtime_sidecar_length_mismatch",
                "input_rows": int(input_grid.shape[0]),
                "sonata_rows": int(sonata_grid.shape[0]),
            }
            return None

        bits = int(self.chorus_match_reduction_bits)
        input_rows = input_grid.detach().cpu().long().numpy()
        sonata_rows = sonata_grid.detach().cpu().long().numpy()
        if bits > 0:
            input_rows = input_rows >> bits
            sonata_rows = sonata_rows >> bits

        buckets: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
        totals: Counter = Counter()
        for input_row, sonata_row in zip(input_rows.tolist(), sonata_rows.tolist()):
            input_key = tuple(int(v) for v in input_row)
            sonata_key = tuple(int(v) for v in sonata_row)
            totals[input_key] += 1
            buckets[input_key][sonata_key] += 1

        mapping = {}
        invalid_source_keys = 0
        valid_label_rows = 0
        for input_key, counts in buckets.items():
            total = totals[input_key]
            valid_counts = Counter(
                {
                    label: count
                    for label, count in counts.items()
                    if all(value >= 0 for value in label)
                }
            )
            valid_label_rows += sum(valid_counts.values())
            if not valid_counts:
                invalid_source_keys += 1
                continue
            label, count = valid_counts.most_common(1)[0]
            if count / max(total, 1) < self.min_valid_label_fraction:
                invalid_source_keys += 1
                continue
            mapping[input_key] = label

        mapped_rows = []
        missing = 0
        for row in chorus_grid.detach().cpu().tolist():
            input_key = tuple(int(v) for v in row)
            sonata_key = mapping.get(input_key)
            if sonata_key is None:
                missing += 1
                mapped_rows.append(INVALID_MATCH_GRID)
            else:
                mapped_rows.append(sonata_key)

        self.last_sidecar_match_stats = {
            "enabled": True,
            "sidecar": "runtime_sonata_grid",
            "input_mode": self.input_mode,
            "reduction_bits": bits,
            "source_keys": len(mapping),
            "invalid_source_keys": int(invalid_source_keys),
            "chorus_tokens": int(chorus_grid.shape[0]),
            "unmapped_chorus_tokens": int(missing),
            "input_rows": int(input_grid.shape[0]),
            "valid_label_rows": int(valid_label_rows),
            "valid_label_fraction": valid_label_rows / max(int(input_grid.shape[0]), 1),
            "min_valid_label_fraction": self.min_valid_label_fraction,
            "native_input": dict(self.last_native_input_stats),
        }
        return torch.as_tensor(mapped_rows, device=device, dtype=torch.long)

    def _sidecar_match_grid(
        self,
        aligned_dir: Path,
        chorus_grid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Map raw-frame Chorus output grids onto Sonata grids for fusion matching.

        Raw-bridge prepared scenes keep Chorus input coordinates in raw 3RScan
        frame. The sidecar grids record the row-wise correspondence between
        raw Chorus input voxels and Sonata input voxels. After Chorus pooling,
        we reduce those sidecar grids by the same total stride and use them only
        as the post-encoder matching grid.
        """
        raw_grid_path = aligned_dir / "sonata_grid_raw_chorus.npy"
        if not raw_grid_path.exists():
            raw_grid_path = aligned_dir / "sonata_grid_raw_bridge.npy"
        sonata_grid_path = aligned_dir / "sonata_grid.npy"
        if not raw_grid_path.exists() or not sonata_grid_path.exists():
            self.last_sidecar_match_stats = {"enabled": False, "reason": "missing_sidecar"}
            return chorus_grid.long()

        raw_grid = np.load(raw_grid_path).astype(np.int64, copy=False)
        sonata_grid = np.load(sonata_grid_path).astype(np.int64, copy=False)
        if raw_grid.shape[0] != sonata_grid.shape[0]:
            self.last_sidecar_match_stats = {
                "enabled": False,
                "reason": "sidecar_length_mismatch",
                "raw_rows": int(raw_grid.shape[0]),
                "sonata_rows": int(sonata_grid.shape[0]),
            }
            return chorus_grid.long()

        bits = int(self.chorus_match_reduction_bits)
        if bits > 0:
            raw_grid = raw_grid >> bits
            sonata_grid = sonata_grid >> bits

        buckets: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
        for raw_row, sonata_row in zip(raw_grid.tolist(), sonata_grid.tolist()):
            raw_key = tuple(int(v) for v in raw_row)
            sonata_key = tuple(int(v) for v in sonata_row)
            buckets[raw_key][sonata_key] += 1
        mapping = {
            raw_key: counts.most_common(1)[0][0]
            for raw_key, counts in buckets.items()
        }

        mapped_rows = []
        missing = 0
        for row in chorus_grid.detach().cpu().tolist():
            raw_key = tuple(int(v) for v in row)
            sonata_key = mapping.get(raw_key)
            if sonata_key is None:
                missing += 1
                mapped_rows.append(raw_key)
            else:
                mapped_rows.append(sonata_key)

        self.last_sidecar_match_stats = {
            "enabled": True,
            "sidecar": raw_grid_path.name,
            "reduction_bits": bits,
            "source_keys": len(mapping),
            "chorus_tokens": int(chorus_grid.shape[0]),
            "unmapped_chorus_tokens": int(missing),
        }
        return torch.as_tensor(mapped_rows, device=device, dtype=torch.long)

    def _encode_chorus(
        self,
        scene_id: str,
        device: torch.device,
        include_match_context: bool = True,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not scene_id:
            return None
        aligned_dir = _find_aligned_dir(self.aligned_root, scene_id)
        if aligned_dir is None:
            if self.missing_policy == "error":
                raise FileNotFoundError(f"No aligned Chorus split found for scene_id={scene_id} under {self.aligned_root}")
            return None
        input_dict = self._prepare_chorus_input(aligned_dir, scene_id, device)
        if self._should_use_sonata_lattice():
            sonata_physical_grid = input_dict.get(SONATA_PHYSICAL_GRID_KEY)
            if sonata_physical_grid is None:
                raise RuntimeError(
                    f"Sonata-lattice override requested for {scene_id}, but "
                    f"{SONATA_PHYSICAL_GRID_KEY} was not propagated through Chorus preprocessing."
                )
            if input_dict.get("grid_coord") is None:
                raise RuntimeError(f"Chorus input for {scene_id} has no grid_coord.")
            if sonata_physical_grid.shape[0] != input_dict["grid_coord"].shape[0]:
                raise RuntimeError(
                    f"Sonata-lattice override length mismatch for {scene_id}: "
                    f"{sonata_physical_grid.shape[0]} != {input_dict['grid_coord'].shape[0]}"
                )
            input_dict["grid_coord"] = sonata_physical_grid.long()
        # spconv kernels used by the Chorus backbone do not support bf16 biases.
        # Keep this branch in fp32 even when the surrounding SpatialLM is loaded
        # in bf16/fp16 for language-model inference.
        self.chorus_model.to(device=device, dtype=torch.float32)
        point = self.chorus_model.backbone(input_dict)
        feat = point.feat if hasattr(point, "feat") else point["feat"]
        grid = point.grid_coord if hasattr(point, "grid_coord") else point["grid_coord"]
        if include_match_context:
            match_grid = self._runtime_sidecar_match_grid(input_dict, grid.long(), device)
            if match_grid is None:
                match_grid = self._sidecar_match_grid(aligned_dir, grid.long(), device)
        try:
            head_dtype = next(self.fourier_head.parameters()).dtype
        except StopIteration:
            head_dtype = feat.dtype
        feat = feat.to(dtype=head_dtype)
        encoded = {
            "native_context": self.fourier_head(feat, grid.long()),
            "native_grid": grid.long(),
        }
        if include_match_context:
            encoded["match_context"] = self.fourier_head(feat, match_grid)
            encoded["match_grid"] = match_grid
        return encoded

    def _choose_modes(
        self,
        has_chorus: bool,
        device: torch.device,
        forced_mode: Optional[str] = None,
    ) -> tuple[bool, bool]:
        mode = self.fusion_mode
        use_pcd = mode in {"avg", "append", "pcd", "chorus_coord_matched", "transformer"}
        use_chorus = has_chorus and mode in {
            "avg",
            "append",
            "chorus",
            "chorus_matched",
            "chorus_coord_matched",
            "transformer",
        }
        if forced_mode:
            forced_mode = forced_mode.strip().lower()
            if forced_mode in {"pcd", "sonata"}:
                return True, False
            if forced_mode in {"chorus", "3dgs"}:
                return False, has_chorus
            if forced_mode in {"both", "fusion", "chorus_coord_matched"}:
                return use_pcd, use_chorus
            raise ValueError(
                "`forced_mode` must be one of pcd, sonata, chorus, 3dgs, both, "
                f"fusion, or chorus_coord_matched; got {forced_mode!r}."
            )
        if (
            not self.training
            or not has_chorus
            or self.modality_dropout_rate <= 0.0
            or mode not in {"avg", "chorus_coord_matched", "transformer"}
        ):
            return use_pcd, use_chorus
        if torch.rand((), device=device) >= self.modality_dropout_rate:
            return use_pcd, use_chorus
        if torch.rand((), device=device) < self.drop_pcd_probability:
            return False, True
        return True, False

    def _prepare_contrastive_pair(
        self,
        chorus_context: torch.Tensor,
        sonata_context: torch.Tensor,
        chorus_idx: torch.Tensor,
        sonata_idx: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.contrastive_loss_weight <= 0.0:
            return None
        if not (
            self.contrastive_cosine_enabled
            or self.contrastive_mse_enabled
            or self.contrastive_info_nce_enabled
        ):
            return None
        if chorus_idx.numel() == 0:
            return None

        pred = chorus_context[chorus_idx].float()
        target = sonata_context[sonata_idx]
        if not self.contrastive_backprop_sonata:
            target = target.detach()
        target = target.float()
        return pred, target

    def _compute_contrastive_loss_from_pairs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if pred.shape[0] < self.contrastive_min_matched_tokens:
            return None

        loss_terms = []
        metrics = {
            "matched": torch.as_tensor(float(pred.shape[0]), device=pred.device),
        }
        pred_norm = None
        target_norm = None
        if self.contrastive_cosine_enabled or self.contrastive_info_nce_enabled:
            pred_norm = F.normalize(pred, p=2, dim=1)
            target_norm = F.normalize(target, p=2, dim=1)

        if self.contrastive_cosine_enabled:
            cosine = 1.0 - (pred_norm * target_norm).sum(dim=1).mean()
            loss_terms.append(self.contrastive_cosine_weight * cosine)
            metrics["cosine"] = cosine.detach()

        if self.contrastive_mse_enabled:
            mse = F.mse_loss(pred, target)
            loss_terms.append(self.contrastive_mse_weight * mse)
            metrics["mse"] = mse.detach()

        if self.contrastive_info_nce_enabled:
            logits = pred_norm @ target_norm.T / self.contrastive_temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            info_nce = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            top1 = (logits.argmax(dim=1) == labels).float().mean()
            loss_terms.append(self.contrastive_info_nce_weight * info_nce)
            metrics["info_nce"] = info_nce.detach()
            metrics["top1"] = top1.detach()

        self.last_aux_metrics = metrics
        return sum(loss_terms)

    def _compute_contrastive_loss(
        self,
        chorus_context: torch.Tensor,
        sonata_context: torch.Tensor,
        chorus_idx: torch.Tensor,
        sonata_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pair = self._prepare_contrastive_pair(
            chorus_context,
            sonata_context,
            chorus_idx,
            sonata_idx,
        )
        if pair is None:
            return None
        return self._compute_contrastive_loss_from_pairs(*pair)

    def _collect_or_compute_contrastive_loss(
        self,
        chorus_context: torch.Tensor,
        sonata_context: torch.Tensor,
        chorus_idx: torch.Tensor,
        sonata_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.contrastive_batch_wide_enabled:
            return self._compute_contrastive_loss(
                chorus_context,
                sonata_context,
                chorus_idx,
                sonata_idx,
            )

        self.last_contrastive_pair = self._prepare_contrastive_pair(
            chorus_context,
            sonata_context,
            chorus_idx,
            sonata_idx,
        )
        return None

    def reset_batch_contrastive_loss(self) -> None:
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        self.last_contrastive_pair = None

    def finalize_batch_contrastive_loss(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        self.last_contrastive_pair = None
        if not pairs:
            return None

        pred = torch.cat([pair[0] for pair in pairs], dim=0)
        target = torch.cat([pair[1] for pair in pairs], dim=0)
        aux_loss = self._compute_contrastive_loss_from_pairs(pred, target)
        if aux_loss is not None:
            self.last_aux_loss = aux_loss
        return aux_loss

    def _compute_fusion_gate(
        self,
        sonata_tokens: torch.Tensor,
        chorus_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        num_tokens = sonata_tokens.shape[0]
        if num_tokens == 0:
            empty = sonata_tokens.new_empty((0, 1))
            return empty, 0.0

        if self.fusion_gate_mode == "fixed":
            alpha = sonata_tokens.new_full((num_tokens, 1), self.fusion_chorus_weight)
        elif self.fusion_gate_mode == "global":
            gate_value = torch.sigmoid(
                self.fusion_gate_logit.to(device=sonata_tokens.device, dtype=sonata_tokens.dtype)
            )
            alpha = gate_value.expand(num_tokens, 1)
        else:
            gate_input = torch.cat(
                [sonata_tokens, chorus_tokens, torch.abs(sonata_tokens - chorus_tokens)],
                dim=-1,
            )
            gate_param = next(self.fusion_gate_mlp.parameters(), None)
            gate_dtype = gate_param.dtype if gate_param is not None else gate_input.dtype
            alpha = torch.sigmoid(self.fusion_gate_mlp(gate_input.to(dtype=gate_dtype)))
            alpha = alpha.to(dtype=sonata_tokens.dtype)

        return alpha, float(alpha.mean().detach().item())

    def _fuse_matched_tokens(
        self,
        sonata_tokens: torch.Tensor,
        chorus_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[float]]:
        if self.fusion_mode == "avg":
            fusion_alpha, gate_avg = self._compute_fusion_gate(
                sonata_tokens,
                chorus_tokens,
            )
            matched_fused = (1.0 - fusion_alpha) * sonata_tokens + fusion_alpha * chorus_tokens
            return matched_fused, gate_avg

        if self.fusion_mode == "transformer":
            if self.pair_fusion_transformer is None:
                raise RuntimeError(
                    "chorus_fusion_mode=transformer requires a pair_fusion_transformer module. "
                    "Build/load the model with chorus_fusion_mode set to transformer."
                )
            return self.pair_fusion_transformer(sonata_tokens, chorus_tokens), None

        raise RuntimeError(
            f"_fuse_matched_tokens called for unsupported fusion_mode={self.fusion_mode}."
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
        scene_id: str,
        device: torch.device,
        forced_mode: Optional[str] = None,
    ) -> torch.Tensor:
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        self.last_contrastive_pair = None
        self.last_modality_usage = {}
        self.last_sidecar_match_stats = {}
        self.last_native_input_stats = {}
        if self.fusion_mode == "pcd":
            sonata_context, _ = self._encode_sonata(point_cloud, device)
            self.last_modality_usage = {
                "mode": "pcd",
                "reason": "forced_pcd",
                "has_chorus": False,
                "used_pcd": True,
                "used_chorus": False,
                "matched_tokens": 0,
                "sonata_tokens": int(sonata_context.shape[0]),
                "chorus_tokens": 0,
            }
            mm_debug.log("chorus_fusion", f"scene_id={scene_id} mode=pcd tokens={sonata_context.shape[0]}")
            return sonata_context

        needs_match_context = self.fusion_mode in {"avg", "chorus_matched", "transformer"}
        chorus_encoded = self._encode_chorus(
            scene_id,
            device,
            include_match_context=needs_match_context,
        )
        has_chorus = chorus_encoded is not None
        use_pcd, use_chorus = self._choose_modes(has_chorus, device, forced_mode=forced_mode)

        if not has_chorus:
            if self.missing_policy == "error":
                raise RuntimeError(f"No Chorus features available for scene_id={scene_id}")
            sonata_context, _ = self._encode_sonata(point_cloud, device)
            reason = "missing_scene_id" if not scene_id else "missing_chorus"
            self.last_modality_usage = {
                "mode": "pcd",
                "reason": reason,
                "has_chorus": False,
                "used_pcd": True,
                "used_chorus": False,
                "matched_tokens": 0,
                "returned_tokens": int(sonata_context.shape[0]),
                "sonata_tokens": int(sonata_context.shape[0]),
                "chorus_tokens": 0,
            }
            mm_debug.log("chorus_fusion", f"scene_id={scene_id} mode=pcd tokens={sonata_context.shape[0]}")
            return sonata_context

        if use_chorus and not use_pcd:
            if self.fusion_mode == "chorus_matched":
                chorus_context = chorus_encoded["match_context"]
                chorus_grid = chorus_encoded["match_grid"]
                matched_context, matched_grid, used_chorus_tokens = (
                    self._pool_chorus_by_match_grid(chorus_context, chorus_grid, device)
                )
                matched_tokens = int(matched_context.shape[0])
                if matched_tokens == 0:
                    if self.missing_policy == "error":
                        raise RuntimeError(
                            f"No valid Sonata-matched Chorus tokens for scene_id={scene_id}"
                        )
                    sonata_context, _ = self._encode_sonata(point_cloud, device)
                    self.last_modality_usage = {
                        "mode": "pcd",
                        "reason": "no_match",
                        "has_chorus": True,
                        "used_pcd": True,
                        "used_chorus": False,
                        "matched_tokens": 0,
                        "returned_tokens": int(sonata_context.shape[0]),
                        "used_chorus_tokens": 0,
                        "sonata_tokens": int(sonata_context.shape[0]),
                        "chorus_tokens": int(chorus_context.shape[0]),
                        "sidecar_match": dict(self.last_sidecar_match_stats),
                    }
                    return sonata_context

                self.last_modality_usage = {
                    "mode": "chorus_matched",
                    "reason": "matched_chorus",
                    "has_chorus": True,
                    "used_pcd": False,
                    "used_chorus": True,
                    "matched_tokens": matched_tokens,
                    "returned_tokens": matched_tokens,
                    "used_chorus_tokens": used_chorus_tokens,
                    "sonata_tokens": 0,
                    "chorus_tokens": int(chorus_context.shape[0]),
                    "matched_grid_tokens": int(matched_grid.shape[0]),
                    "sidecar_match": dict(self.last_sidecar_match_stats),
                }
                mm_debug.log(
                    "chorus_fusion",
                    (
                        f"scene_id={scene_id} mode=chorus_matched "
                        f"tokens={matched_tokens} chorus={chorus_context.shape[0]} "
                        f"used_chorus={used_chorus_tokens} standalone=true"
                    ),
                )
                return matched_context

            chorus_context = chorus_encoded["native_context"]
            reason = (
                "forced_chorus"
                if self.fusion_mode == "chorus" or forced_mode in {"chorus", "3dgs"}
                else "drop_pcd"
            )
            self.last_modality_usage = {
                "mode": "chorus",
                "reason": reason,
                "has_chorus": True,
                "used_pcd": False,
                "used_chorus": True,
                "matched_tokens": 0,
                "returned_tokens": int(chorus_context.shape[0]),
                "used_chorus_tokens": int(chorus_context.shape[0]),
                "sonata_tokens": 0,
                "chorus_tokens": int(chorus_context.shape[0]),
                "sidecar_match": dict(self.last_sidecar_match_stats),
            }
            mm_debug.log(
                "chorus_fusion",
                (
                    f"scene_id={scene_id} mode=chorus tokens={chorus_context.shape[0]} "
                    "standalone=true"
                ),
            )
            return chorus_context

        if use_pcd and not use_chorus:
            sonata_context, _ = self._encode_sonata(point_cloud, device)
            reason = "missing_scene_id" if not scene_id else "missing_chorus"
            if has_chorus and use_pcd and not use_chorus:
                reason = "forced_pcd" if forced_mode in {"pcd", "sonata"} else "drop_chorus"
            self.last_modality_usage = {
                "mode": "pcd",
                "reason": reason,
                "has_chorus": bool(has_chorus),
                "used_pcd": True,
                "used_chorus": False,
                "matched_tokens": 0,
                "returned_tokens": int(sonata_context.shape[0]),
                "sonata_tokens": int(sonata_context.shape[0]),
                "chorus_tokens": 0,
            }
            mm_debug.log("chorus_fusion", f"scene_id={scene_id} mode=pcd tokens={sonata_context.shape[0]}")
            return sonata_context

        if self.fusion_mode == "append":
            sonata_context, _ = self._encode_sonata(point_cloud, device)
            chorus_context = chorus_encoded["native_context"].to(
                device=sonata_context.device,
                dtype=sonata_context.dtype,
            )
            appended = torch.cat((sonata_context, chorus_context), dim=0)
            self.last_modality_usage = {
                "mode": "append",
                "reason": "both",
                "has_chorus": True,
                "used_pcd": True,
                "used_chorus": True,
                "matched_tokens": 0,
                "returned_tokens": int(appended.shape[0]),
                "used_chorus_tokens": int(chorus_context.shape[0]),
                "sonata_tokens": int(sonata_context.shape[0]),
                "chorus_tokens": int(chorus_context.shape[0]),
                "appended_chorus_tokens": int(chorus_context.shape[0]),
            }
            mm_debug.log(
                "chorus_fusion",
                (
                    f"scene_id={scene_id} mode=append tokens={appended.shape[0]} "
                    f"sonata={sonata_context.shape[0]} chorus={chorus_context.shape[0]}"
                ),
            )
            return appended

        if self.fusion_mode == "chorus_coord_matched":
            if not (use_pcd and use_chorus):
                raise RuntimeError(
                    "Internal error: chorus_coord_matched fusion reached without both "
                    f"modalities enabled for scene_id={scene_id} "
                    f"(use_pcd={use_pcd}, use_chorus={use_chorus})."
                )
            sonata_context, sonata_grid = self._encode_sonata(point_cloud, device)
            chorus_context = chorus_encoded["native_context"]
            chorus_grid = chorus_encoded["native_grid"]
            matched_chorus_context, matched_grid, sonata_idx, used_chorus_tokens = (
                self._filter_chorus_by_coordinate_intersection(
                    chorus_context,
                    chorus_grid,
                    sonata_grid,
                    device,
                )
            )
            matched_tokens = int(matched_chorus_context.shape[0])
            chorus_tokens = int(chorus_context.shape[0])
            sonata_tokens = int(sonata_context.shape[0])
            match_ratio = used_chorus_tokens / max(chorus_tokens, 1)
            coverage_ratio = matched_tokens / max(sonata_tokens, 1)
            if matched_tokens == 0:
                raise RuntimeError(
                    "No exact Sonata/Chorus coordinate intersections for "
                    f"scene_id={scene_id}. This usually indicates a coordinate frame, "
                    "voxel origin, quantization, or encoder stride mismatch. "
                    f"sonata_tokens={sonata_tokens} chorus_tokens={chorus_tokens}"
                )

            aux_loss = self._collect_or_compute_contrastive_loss(
                matched_chorus_context,
                sonata_context,
                torch.arange(matched_tokens, device=device, dtype=torch.long),
                sonata_idx,
            )
            if aux_loss is not None:
                self.last_aux_loss = aux_loss

            matched_sonata_context = sonata_context[sonata_idx].to(
                device=matched_chorus_context.device,
                dtype=matched_chorus_context.dtype,
            )
            matched_context = 0.5 * (matched_sonata_context + matched_chorus_context)
            if self.discard_unmatched_tokens:
                fused = matched_context
            else:
                fused = sonata_context.clone()
                fused[sonata_idx] = matched_context.to(
                    device=fused.device,
                    dtype=fused.dtype,
                )

            self.last_modality_usage = {
                "mode": "chorus_coord_matched",
                "reason": "averaged_coordinate_intersection",
                "has_chorus": True,
                "used_pcd": True,
                "used_chorus": True,
                "discard_unmatched_tokens": bool(self.discard_unmatched_tokens),
                "matched_tokens": matched_tokens,
                "returned_tokens": int(fused.shape[0]),
                "used_chorus_tokens": used_chorus_tokens,
                "sonata_tokens": sonata_tokens,
                "chorus_tokens": chorus_tokens,
                "matched_grid_tokens": int(matched_grid.shape[0]),
                "coord_match_ratio": float(match_ratio),
                "coord_coverage_ratio": float(coverage_ratio),
            }
            mm_debug.log(
                "chorus_fusion",
                (
                    f"scene_id={scene_id} mode=chorus_coord_matched "
                    f"fusion=avg tokens={fused.shape[0]} matched={matched_tokens} sonata={sonata_tokens} "
                    f"chorus={chorus_tokens} used_chorus={used_chorus_tokens} "
                    f"match_ratio={match_ratio:.4f} coverage={coverage_ratio:.4f}"
                    f" discard_unmatched={self.discard_unmatched_tokens}"
                ),
            )
            return fused

        sonata_context, sonata_grid = self._encode_sonata(point_cloud, device)
        chorus_context = chorus_encoded["match_context"]
        chorus_grid = chorus_encoded["match_grid"]
        (
            aligned_chorus_context,
            sonata_idx,
            used_chorus_tokens,
            exact_chorus_idx,
            exact_sonata_idx,
        ) = self._align_chorus_to_sonata(
            chorus_context,
            chorus_grid,
            sonata_grid,
            device,
        )
        matched_tokens = int(sonata_idx.numel())
        if matched_tokens == 0:
            if self.missing_policy == "error":
                raise RuntimeError(f"No matching Sonata/Chorus token grids for scene_id={scene_id}")
            self.last_modality_usage = {
                "mode": "pcd",
                "reason": "no_match",
                "has_chorus": True,
                "used_pcd": True,
                "used_chorus": False,
                "matched_tokens": 0,
                "returned_tokens": int(sonata_context.shape[0]),
                "used_chorus_tokens": 0,
                "sonata_tokens": int(sonata_context.shape[0]),
                "chorus_tokens": int(chorus_context.shape[0]),
                "sidecar_match": dict(self.last_sidecar_match_stats),
            }
            return sonata_context
        contrastive_chorus_context = aligned_chorus_context
        contrastive_chorus_idx = torch.arange(matched_tokens, device=device, dtype=torch.long)
        contrastive_sonata_idx = sonata_idx
        if self.contrastive_exact_only:
            contrastive_chorus_context = chorus_context
            contrastive_chorus_idx = exact_chorus_idx
            contrastive_sonata_idx = exact_sonata_idx

        aux_loss = self._collect_or_compute_contrastive_loss(
            contrastive_chorus_context,
            sonata_context,
            contrastive_chorus_idx,
            contrastive_sonata_idx,
        )
        if aux_loss is not None:
            self.last_aux_loss = aux_loss

        matched_fused, gate_avg = self._fuse_matched_tokens(
            sonata_context[sonata_idx],
            aligned_chorus_context,
        )
        if self.discard_unmatched_tokens:
            fused = matched_fused
        else:
            fused = sonata_context.clone()
            fused[sonata_idx] = matched_fused
        self.last_modality_usage = {
            "mode": self.fusion_mode,
            "reason": "both",
            "has_chorus": True,
            "used_pcd": True,
            "used_chorus": True,
            "discard_unmatched_tokens": bool(self.discard_unmatched_tokens),
            "matched_tokens": matched_tokens,
            "returned_tokens": int(fused.shape[0]),
            "used_chorus_tokens": used_chorus_tokens,
            "sonata_tokens": int(sonata_context.shape[0]),
            "chorus_tokens": int(chorus_context.shape[0]),
            "sidecar_match": dict(self.last_sidecar_match_stats),
        }
        if gate_avg is not None:
            self.last_modality_usage["chorus_gate_avg"] = gate_avg
            self.last_modality_usage["chorus_gate_count"] = matched_tokens
        gate_avg_text = f"{gate_avg:.4f}" if gate_avg is not None else "n/a"
        mm_debug.log(
            "chorus_fusion",
            (
                f"scene_id={scene_id} mode={self.fusion_mode} tokens={fused.shape[0]} "
                f"matched={matched_tokens} sonata={sonata_context.shape[0]} "
                f"chorus={chorus_context.shape[0]} radius={self.match_grid_radius} "
                f"used_chorus={used_chorus_tokens} gate_mode={self.fusion_gate_mode} "
                f"gate_avg={gate_avg_text} "
                f"discard_unmatched={self.discard_unmatched_tokens}"
            ),
        )
        return fused
