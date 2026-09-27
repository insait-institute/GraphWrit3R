from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatiallm.layout.scene_graph_layout import SceneGraphLayout
from spatiallm.model.chorus_fusion import _find_aligned_dir
from spatiallm.model.sonata_encoder import Point, fourier_encode_vector
from spatiallm.pcd import Compose, cleanup_pcd, get_points_and_colors, load_o3d_pcd


DEFAULT_CONFIG = "/home/luka_milivojevic/SpatialLM-SG/configs/spatiallm_scene_graph_sft_chorus_fusion_contrastive_rio10.yaml"
DEFAULT_DATASET_DIR = "/work/luka_milivojevic/3rscan_subset_scene_graph_data_rio10_chorus_fusion_raw_bridge"
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
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint/model path to load.")
    parser.add_argument(
        "--architecture_config",
        type=str,
        default=None,
        help=(
            "Optional checkpoint/model directory to read config.json from while loading weights "
            "from --checkpoint. Use this for initialization baselines with the trained run's exact architecture."
        ),
    )
    parser.add_argument("--scene_id", required=True, help="Scene split id such as <scan>_split1.")
    parser.add_argument("--split", choices=["train", "val", "auto"], default="auto")
    parser.add_argument("--dataset_dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--aligned_root",
        type=Path,
        default=None,
        help="Optional override for the checkpoint's chorus_aligned_root.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        choices=["encoded_voxels", "room_pcd", "input_voxels", "gaussian_centers", "both", "all"],
        default="encoded_voxels",
        help=(
            "Geometry support to color. 'encoded_voxels' writes direct final-layer Sonata-token cubes; "
            "'both' keeps the historical room+Gaussian transfer export; 'all' exports every support."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=["shared_pca3", "delta_pca3", "delta_shared_scale_pca3", "distance_heatmap"],
        default="shared_pca3",
    )
    parser.add_argument("--distance_reference", choices=["sonata", "chorus", "fused"], default="sonata")
    parser.add_argument("--distance_metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--k_neighbors", type=int, default=1, help="Neighbors for room-point color transfer.")
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export_npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--export_gaussian_ellipsoids",
        action="store_true",
        help="Also export oriented ellipsoid mesh PLYs for full Gaussian extents.",
    )
    parser.add_argument("--ellipsoid_stacks", type=int, default=6)
    parser.add_argument("--ellipsoid_slices", type=int, default=12)
    parser.add_argument("--ellipsoid_scale_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--ellipsoid_max_count",
        type=int,
        default=0,
        help="Limit Gaussian ellipsoid mesh export count. Use 0 for all Gaussians.",
    )
    parser.add_argument(
        "--quat_order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion component order in the aligned Gaussian cache.",
    )
    parser.add_argument("--inference_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def preprocess_point_cloud(points: np.ndarray, colors: np.ndarray, grid_size: float, num_bins: int) -> torch.Tensor:
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform({"name": "pcd", "coord": points.copy(), "color": colors.copy()})
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    return torch.as_tensor(np.stack([np.concatenate([coord, xyz, rgb], axis=1)], axis=0))


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must parse to a mapping: {path}")
    return data


def _set_if_missing(config: Any, key: str, value: Any) -> None:
    if value is None:
        return
    if not hasattr(config, key) or getattr(config, key) is None:
        setattr(config, key, value)


def checkpoint_config_path(config_source: str) -> Path:
    return Path(config_source).expanduser() / "config.json"


def has_local_config(config_source: str) -> bool:
    return checkpoint_config_path(config_source).is_file()


def merge_config_into_hf_config(hf_config: Any, yaml_config: dict[str, Any], args: argparse.Namespace) -> Any:
    config_source = args.architecture_config or args.checkpoint
    checkpoint_has_own_config = has_local_config(config_source)
    # Local checkpoints should define the model architecture themselves. The
    # YAML fallback is only for base/HF checkpoints that predate the local
    # Chorus-fusion fields.
    if not checkpoint_has_own_config:
        _set_if_missing(hf_config, "chorus_fusion_enabled", yaml_config.get("chorus_fusion_enabled", True))
        _set_if_missing(hf_config, "chorus_repo_root", yaml_config.get("chorus_repo_root"))
        _set_if_missing(hf_config, "chorus_config", yaml_config.get("chorus_config"))
        _set_if_missing(hf_config, "chorus_checkpoint", yaml_config.get("chorus_checkpoint"))
        _set_if_missing(hf_config, "chorus_input_mode", yaml_config.get("chorus_input_mode"))
        _set_if_missing(hf_config, "chorus_native_match_radius", yaml_config.get("chorus_native_match_radius"))
        _set_if_missing(hf_config, "chorus_min_valid_label_fraction", yaml_config.get("chorus_min_valid_label_fraction"))
        _set_if_missing(hf_config, "chorus_missing_policy", yaml_config.get("chorus_missing_policy"))
        _set_if_missing(hf_config, "chorus_trainable_name_patterns", yaml_config.get("chorus_trainable_name_patterns"))
        _set_if_missing(hf_config, "chorus_mirror_sonata_trainability", yaml_config.get("chorus_mirror_sonata_trainability"))
        _set_if_missing(hf_config, "chorus_fusion_mode", yaml_config.get("chorus_fusion_mode"))
        _set_if_missing(hf_config, "chorus_drop_pcd_probability", yaml_config.get("chorus_drop_pcd_probability"))
        _set_if_missing(hf_config, "chorus_match_grid_radius", yaml_config.get("chorus_match_grid_radius"))
        _set_if_missing(hf_config, "chorus_exact_match_first", yaml_config.get("chorus_exact_match_first"))
        _set_if_missing(hf_config, "chorus_discard_unmatched_tokens", yaml_config.get("chorus_discard_unmatched_tokens"))
        _set_if_missing(hf_config, "chorus_fusion_chorus_weight", yaml_config.get("chorus_fusion_chorus_weight"))
        _set_if_missing(hf_config, "chorus_fusion_gate_mode", yaml_config.get("chorus_fusion_gate_mode"))
        _set_if_missing(hf_config, "chorus_fusion_gate_hidden_dim", yaml_config.get("chorus_fusion_gate_hidden_dim"))
        _set_if_missing(hf_config, "chorus_fusion_transformer_layers", yaml_config.get("chorus_fusion_transformer_layers"))
        _set_if_missing(hf_config, "chorus_fusion_transformer_heads", yaml_config.get("chorus_fusion_transformer_heads"))
        _set_if_missing(hf_config, "chorus_fusion_transformer_ffn_dim", yaml_config.get("chorus_fusion_transformer_ffn_dim"))
        _set_if_missing(hf_config, "chorus_fusion_transformer_dropout", yaml_config.get("chorus_fusion_transformer_dropout"))
        _set_if_missing(hf_config, "chorus_contrastive_loss_weight", yaml_config.get("chorus_contrastive_loss_weight"))
        _set_if_missing(hf_config, "chorus_contrastive_loss_final_weight", yaml_config.get("chorus_contrastive_loss_final_weight"))
        _set_if_missing(hf_config, "chorus_contrastive_warmup_ratio", yaml_config.get("chorus_contrastive_warmup_ratio"))
        _set_if_missing(hf_config, "chorus_contrastive_decay_start_ratio", yaml_config.get("chorus_contrastive_decay_start_ratio"))
        _set_if_missing(hf_config, "chorus_contrastive_exact_only", yaml_config.get("chorus_contrastive_exact_only"))
        _set_if_missing(hf_config, "chorus_contrastive_backprop_sonata", yaml_config.get("chorus_contrastive_backprop_sonata"))
        _set_if_missing(hf_config, "chorus_contrastive_batch_wide_enabled", yaml_config.get("chorus_contrastive_batch_wide_enabled"))
        _set_if_missing(hf_config, "chorus_contrastive_cosine_enabled", yaml_config.get("chorus_contrastive_cosine_enabled"))
        _set_if_missing(hf_config, "chorus_contrastive_mse_enabled", yaml_config.get("chorus_contrastive_mse_enabled"))
        _set_if_missing(hf_config, "chorus_contrastive_info_nce_enabled", yaml_config.get("chorus_contrastive_info_nce_enabled"))
        _set_if_missing(hf_config, "chorus_contrastive_cosine_weight", yaml_config.get("chorus_contrastive_cosine_weight"))
        _set_if_missing(hf_config, "chorus_contrastive_mse_weight", yaml_config.get("chorus_contrastive_mse_weight"))
        _set_if_missing(hf_config, "chorus_contrastive_info_nce_weight", yaml_config.get("chorus_contrastive_info_nce_weight"))
        _set_if_missing(hf_config, "chorus_contrastive_temperature", yaml_config.get("chorus_contrastive_temperature"))
        _set_if_missing(hf_config, "chorus_contrastive_min_matched_tokens", yaml_config.get("chorus_contrastive_min_matched_tokens"))
        _set_if_missing(hf_config, "chorus_fourier_head_layer_norm", yaml_config.get("chorus_fourier_head_layer_norm"))
        _set_if_missing(hf_config, "chorus_init_fourier_head_from_sonata", yaml_config.get("chorus_init_fourier_head_from_sonata"))
        _set_if_missing(hf_config, "chorus_disable_drop_path", yaml_config.get("chorus_disable_drop_path"))

    # Allow the aligned root to be redirected at inference because local caches
    # can legitimately differ from the training machine.
    if args.aligned_root is not None:
        hf_config.chorus_aligned_root = str(args.aligned_root)

    # Disable stochastic modality dropping for deterministic feature export.
    hf_config.chorus_modality_dropout_rate = 0.0
    hf_config.use_3d_tokens = True
    return hf_config


def resync_missing_chorus_fourier_head(model: Any, loading_info: dict[str, Any] | None) -> bool:
    """Initialize an absent Chorus Fourier head from the restored Sonata head.

    ChorusFusionPointEncoder copies Sonata's input projection during module
    construction. In a `from_pretrained(base_spatiallm, config=finetuned_config)`
    baseline, construction happens before the base checkpoint weights are
    restored, so the copied head would otherwise keep random initialization.
    Fine-tuned checkpoints contain explicit Chorus Fourier head tensors and are
    intentionally left untouched.
    """
    if loading_info is None:
        return False
    missing_keys = set(loading_info.get("missing_keys") or [])
    head_prefix = "chorus_fusion_encoder.fourier_head.proj."
    if not any(key.startswith(head_prefix) for key in missing_keys):
        return False

    encoder = getattr(model, "chorus_fusion_encoder", None)
    if encoder is None:
        return False
    if not bool(getattr(model.config, "chorus_init_fourier_head_from_sonata", True)):
        return False
    fourier_head = getattr(encoder, "fourier_head", None)
    sonata_backbone = getattr(encoder, "sonata_backbone", None)
    sonata_input_proj = getattr(sonata_backbone, "input_proj", None)
    if fourier_head is None or sonata_input_proj is None or not hasattr(fourier_head, "proj"):
        return False

    fourier_head.proj.load_state_dict(sonata_input_proj.state_dict())
    return True


def unpack_chorus_encoding(chorus_encoded: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Support both legacy tuple and current dict Chorus encoder outputs."""
    if isinstance(chorus_encoded, dict):
        if "match_context" in chorus_encoded and "match_grid" in chorus_encoded:
            return chorus_encoded["match_context"], chorus_encoded["match_grid"]
        if "native_context" in chorus_encoded and "native_grid" in chorus_encoded:
            return chorus_encoded["native_context"], chorus_encoded["native_grid"]
        raise KeyError(
            "Chorus encoding dict must contain either match_context/match_grid "
            "or native_context/native_grid."
        )
    return chorus_encoded


def load_model(args: argparse.Namespace, yaml_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    config_source = args.architecture_config or args.checkpoint
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    config = merge_config_into_hf_config(config, yaml_config, args)
    dtype = getattr(torch, args.inference_dtype)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        ignore_mismatched_sizes=args.architecture_config is not None,
        low_cpu_mem_usage=not getattr(config, "chorus_fusion_enabled", False),
        output_loading_info=True,
    )
    resynced_fourier_head = resync_missing_chorus_fourier_head(model, loading_info)
    model._feature_field_loading_info = {
        "config_source": str(config_source),
        "checkpoint_config_path": str(checkpoint_config_path(config_source)),
        "checkpoint_has_own_config": has_local_config(config_source),
        "architecture_config": args.architecture_config,
        "resynced_missing_chorus_fourier_head_from_sonata": bool(resynced_fourier_head),
    }
    model.to(args.device)
    if hasattr(model, "set_point_backbone_dtype"):
        model.set_point_backbone_dtype(torch.float32)
    model.eval()
    if getattr(model, "chorus_fusion_encoder", None) is None:
        raise RuntimeError("Loaded model does not have a chorus fusion encoder attached.")
    return tokenizer, model


def resolve_aligned_dir(root: Path, scene_id: str, split: str) -> Path:
    if split == "auto":
        resolved = _find_aligned_dir(root, scene_id)
    else:
        candidate = root / split / scene_id
        resolved = candidate if (candidate / "coord.npy").exists() or (candidate / "summary.json").exists() else None
    if resolved is None:
        raise FileNotFoundError(f"Could not find aligned directory for {scene_id} under {root}.")
    return resolved


def raw_to_sceneverse(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rot = np.asarray(transform["rotation_row_major"], dtype=np.float32)
    center = np.asarray(transform["center_after_rotation"], dtype=np.float32)
    return points.astype(np.float32, copy=False) @ rot - center.reshape(1, 3)


def _candidate_frame_score(points: np.ndarray, token_centers: np.ndarray) -> dict[str, float]:
    if points.shape[0] == 0 or token_centers.shape[0] == 0:
        return {"nearest_median": float("inf"), "nearest_p95": float("inf")}
    step = max(int(np.ceil(points.shape[0] / 5000)), 1)
    sampled = points[::step]
    distances, _ = cKDTree(token_centers).query(sampled, k=1)
    distances = np.asarray(distances, dtype=np.float32)
    return {
        "nearest_median": float(np.median(distances)),
        "nearest_p95": float(np.percentile(distances, 95.0)),
    }


def _load_sceneverse_transform(aligned_dir: Path) -> Optional[dict[str, Any]]:
    diagnostics_path = aligned_dir / "match_diagnostics.json"
    if not diagnostics_path.exists():
        return None
    with open(diagnostics_path, "r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    transform = diagnostics.get("sceneverse_to_raw")
    if isinstance(transform, dict) and "rotation_row_major" in transform and "center_after_rotation" in transform:
        return transform
    return None


def load_optional_npy(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def colors_to_uint8(colors: Optional[np.ndarray], count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if colors is None:
        return np.full((count, 3), 180, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] < 3:
        return np.full((count, 3), 180, dtype=np.uint8)
    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating) and float(np.nanmax(colors)) <= 1.1:
        colors = colors * 255.0
    return np.clip(np.nan_to_num(colors), 0, 255).astype(np.uint8)


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


def read_ply_xyz_rgb(path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Minimal PLY vertex reader for original-scene RGB fallback."""
    with open(path, "rb") as handle:
        header, data_offset = _read_ply_header(handle)
        if not header or header[0] != "ply":
            return None
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
                return None
        if vertex_count <= 0:
            return None

        names = [name for name, _ in properties]
        if not {"x", "y", "z"}.issubset(names):
            return None
        rgb_names = None
        for candidate in (("red", "green", "blue"), ("r", "g", "b")):
            if set(candidate).issubset(names):
                rgb_names = candidate
                break
        if rgb_names is None:
            return None

        if fmt == "ascii":
            handle.seek(data_offset)
            rows = np.loadtxt(handle, max_rows=vertex_count)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
            xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
            rgb = rows[:, [names.index(name) for name in rgb_names]]
            return xyz, colors_to_uint8(rgb, xyz.shape[0])

        endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else None
        if endian is None:
            return None
        dtype_fields = []
        for name, type_name in properties:
            dtype_code = PLY_DTYPE_MAP.get(type_name)
            if dtype_code is None:
                return None
            dtype_fields.append((name, endian + dtype_code))
        dtype = np.dtype(dtype_fields)
        handle.seek(data_offset)
        vertex = np.frombuffer(handle.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        rgb = np.stack([vertex[name] for name in rgb_names], axis=1)
        return xyz, colors_to_uint8(rgb, xyz.shape[0])


def room_colors_from_ply(path: Path, room_points: np.ndarray, open3d_colors: np.ndarray) -> np.ndarray:
    colors = colors_to_uint8(open3d_colors, room_points.shape[0])
    if colors.shape[0] == room_points.shape[0] and np.any(colors != 0):
        return colors
    fallback = read_ply_xyz_rgb(path)
    if fallback is None:
        return colors
    ply_points, ply_colors = fallback
    if ply_points.shape[0] != room_points.shape[0]:
        return colors
    return ply_colors


def choose_visual_frame(
    centers: np.ndarray,
    token_centers: np.ndarray,
    aligned_dir: Path,
    raw_origin: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Put aligned-cache coordinates into the room/Sonata visualization frame."""
    candidates: list[tuple[str, np.ndarray]] = [
        ("as_stored", centers.astype(np.float32, copy=False)),
        ("positive_shift_plus_room_origin", centers.astype(np.float32, copy=False) + raw_origin.reshape(1, 3)),
    ]
    transform = _load_sceneverse_transform(aligned_dir)
    if transform is not None:
        candidates.append(("raw_3rscan_to_sceneverse", raw_to_sceneverse(centers, transform)))

    scored = []
    for name, candidate in candidates:
        score = _candidate_frame_score(candidate, token_centers)
        scored.append((score["nearest_median"], score["nearest_p95"], name, candidate, score))
    scored.sort(key=lambda row: (row[0], row[1]))
    _, _, best_name, best_centers, best_score = scored[0]
    return best_centers.astype(np.float32, copy=False), {
        "selected": best_name,
        "scores": {
            name: score
            for _, _, name, _, score in scored
        },
    }


def encode_sonata_with_raw_centers(
    encoder,
    point_cloud: torch.Tensor,
    device: torch.device,
    raw_origin: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
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
    backbone = encoder.sonata_backbone
    point = Point(input_dict)
    point = backbone.embedding(point)
    point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
    point.sparsify()
    point = backbone.enc(point)
    pooled_coords_shifted = point["coord"].detach().float().cpu().numpy()
    context = point["sparse_conv_feat"].features
    grid = point["grid_coord"].long()
    if backbone.enable_fourier_encode:
        coords_normalised = grid / (backbone.reduced_grid_size - 1)
        encoded_coords = fourier_encode_vector(coords_normalised)
        context = torch.cat([context, encoded_coords], dim=-1)
        context = backbone.input_proj(context)
    pooled_coords_raw = pooled_coords_shifted + raw_origin[None, :]
    return context, grid, pooled_coords_raw


def input_voxel_centers_from_point_cloud(point_cloud: torch.Tensor, raw_origin: np.ndarray) -> np.ndarray:
    """Return the raw-frame centers of the GridSample voxels fed to Sonata."""
    nan_mask = torch.isnan(point_cloud).any(dim=1)
    valid = point_cloud[~nan_mask]
    shifted_xyz = valid[:, 3:6].detach().float().cpu().numpy()
    return shifted_xyz + raw_origin.reshape(1, 3)


def encoded_voxel_centers_from_grid(
    encoded_grid: torch.Tensor,
    raw_origin: np.ndarray,
    encoded_voxel_size: float,
) -> np.ndarray:
    """Return raw-frame centers of the final Sonata encoder voxels."""
    grid = encoded_grid.detach().float().cpu().numpy()
    return raw_origin.reshape(1, 3) + (grid + 0.5) * float(encoded_voxel_size)


def encoded_voxel_stride_from_backbone(backbone: Any, num_bins: int) -> float:
    pooling_strides = []
    for module in getattr(backbone, "enc", torch.nn.Module()).modules():
        if module.__class__.__name__ != "GridPooling" or not hasattr(module, "stride"):
            continue
        stride = getattr(module, "stride")
        if isinstance(stride, (tuple, list)):
            stride_values = np.asarray(stride, dtype=np.float32).reshape(-1)
            if not np.allclose(stride_values, stride_values[0]):
                raise ValueError(f"Anisotropic GridPooling stride is not supported for cube export: {stride}")
            pooling_strides.append(float(stride_values[0]))
        else:
            pooling_strides.append(float(stride))
    if pooling_strides:
        return float(np.prod(pooling_strides))

    reduced_grid_size = getattr(backbone, "reduced_grid_size", None)
    if reduced_grid_size is None or int(reduced_grid_size) <= 0:
        raise ValueError("Sonata backbone does not expose a valid reduced_grid_size.")
    return float(num_bins) / float(reduced_grid_size)


def extract_scene_features(
    model,
    input_pcd: torch.Tensor,
    scene_id: str,
    aligned_dir: Path,
    device: torch.device,
    raw_origin: np.ndarray,
    encoded_voxel_size: float,
    encoded_voxel_stride: float,
    include_gaussian_support: bool,
) -> dict[str, Any]:
    encoder = model.chorus_fusion_encoder
    point_cloud = input_pcd[0]
    with torch.inference_mode():
        sonata_context, sonata_grid, sonata_token_centers_raw = encode_sonata_with_raw_centers(
            encoder,
            point_cloud,
            device,
            raw_origin,
        )
        input_voxel_centers_raw = input_voxel_centers_from_point_cloud(point_cloud, raw_origin)
        encoded_voxel_centers_raw = encoded_voxel_centers_from_grid(
            sonata_grid,
            raw_origin,
            encoded_voxel_size,
        )
        chorus_encoded = encoder._encode_chorus(scene_id, device)
        if chorus_encoded is None:
            raise RuntimeError(f"Chorus encoding unavailable for scene {scene_id}")
        chorus_context, chorus_grid = unpack_chorus_encoding(chorus_encoded)
        (
            aligned_chorus_context,
            sonata_idx,
            used_chorus_tokens,
            exact_chorus_idx,
            exact_sonata_idx,
        ) = encoder._align_chorus_to_sonata(
            chorus_context,
            chorus_grid,
            sonata_grid,
            device,
        )

        fused_context = sonata_context.clone()
        gate_avg = None
        if sonata_idx.numel() > 0:
            fusion_alpha, gate_avg = encoder._compute_fusion_gate(
                sonata_context[sonata_idx],
                aligned_chorus_context,
            )
            fused_context[sonata_idx] = (
                (1.0 - fusion_alpha) * sonata_context[sonata_idx]
                + fusion_alpha * aligned_chorus_context
            )

    aligned_centers_path = aligned_dir / "sonata_centers_raw.npy"
    sonata_centers_raw = load_optional_npy(aligned_centers_path)
    raw_sonata_grid = load_optional_npy(aligned_dir / "sonata_grid.npy")
    representative_path = aligned_dir / "representative_chorus_indices.npy"
    representative_chorus_indices = load_optional_npy(representative_path)
    representative_source = str(representative_path)
    if representative_chorus_indices is None:
        raw_splat_path = aligned_dir / "raw_splat_indices.npy"
        representative_chorus_indices = load_optional_npy(raw_splat_path)
        representative_source = str(raw_splat_path) if representative_chorus_indices is not None else None

    gaussian_centers_stored = None
    gaussian_centers = None
    gaussian_frame = None
    scale = None
    quat = None
    opacity = None
    gaussian_color = None
    gaussian_path = aligned_dir / "coord.npy"
    if gaussian_path.exists():
        gaussian_centers_stored = np.load(gaussian_path)
        gaussian_centers, gaussian_frame = choose_visual_frame(
            gaussian_centers_stored,
            sonata_token_centers_raw,
            aligned_dir,
            raw_origin,
        )
        gaussian_color = colors_to_uint8(load_optional_npy(aligned_dir / "color.npy"), gaussian_centers.shape[0])
        scale = load_optional_npy(aligned_dir / "scale.npy")
        quat = load_optional_npy(aligned_dir / "quat.npy")
        opacity = load_optional_npy(aligned_dir / "opacity.npy")
    elif include_gaussian_support:
        raise FileNotFoundError(
            f"Gaussian support export requested, but aligned cache has no coord.npy: {gaussian_path}"
        )

    chorus_aligned_full = sonata_context.detach().float().new_full(sonata_context.shape, float("nan"))
    if sonata_idx.numel() > 0:
        chorus_aligned_full[sonata_idx] = aligned_chorus_context.detach().float()

    exact_mask = np.zeros((int(sonata_context.shape[0]),), dtype=bool)
    exact_mask[exact_sonata_idx.detach().cpu().numpy()] = True
    matched_mask = np.zeros((int(sonata_context.shape[0]),), dtype=bool)
    matched_mask[sonata_idx.detach().cpu().numpy()] = True

    return {
        "sonata_features": sonata_context.detach().float().cpu().numpy(),
        "chorus_features": chorus_aligned_full.detach().float().cpu().numpy(),
        "fused_features": fused_context.detach().float().cpu().numpy(),
        "sonata_grid": sonata_grid.detach().cpu().numpy(),
        "chorus_grid": chorus_grid.detach().cpu().numpy(),
        "sonata_centers_raw": sonata_token_centers_raw,
        "encoded_voxel_centers_raw": encoded_voxel_centers_raw.astype(np.float32, copy=False),
        "encoded_voxel_size": float(encoded_voxel_size),
        "encoded_voxel_stride": float(encoded_voxel_stride),
        "input_voxel_centers_raw": input_voxel_centers_raw.astype(np.float32, copy=False),
        "aligned_sonata_centers_raw": sonata_centers_raw if sonata_centers_raw is not None else sonata_token_centers_raw,
        "aligned_sonata_centers_raw_source": str(aligned_centers_path if sonata_centers_raw is not None else "runtime_sonata_encoder"),
        "gaussian_centers_stored": gaussian_centers_stored,
        "raw_sonata_grid": raw_sonata_grid,
        "gaussian_centers": gaussian_centers,
        "gaussian_color": gaussian_color,
        "gaussian_visual_frame": gaussian_frame,
        "representative_chorus_indices": representative_chorus_indices,
        "representative_chorus_indices_source": representative_source,
        "gaussian_scale": scale,
        "gaussian_quat": quat,
        "gaussian_opacity": opacity,
        "matched_mask": matched_mask,
        "exact_mask": exact_mask,
        "matched_count": int(sonata_idx.numel()),
        "exact_match_count": int(exact_sonata_idx.numel()),
        "used_chorus_tokens": int(used_chorus_tokens),
        "gate_avg": None if gate_avg is None else float(gate_avg),
        "sidecar_match_stats": dict(getattr(encoder, "last_sidecar_match_stats", {})),
    }


def robust_rgb_from_projection(projected: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = np.maximum(maxs - mins, 1e-6)
    rgb = np.clip((projected - mins) / denom, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def fill_missing_rows(rows: np.ndarray, fallback_rows: np.ndarray) -> np.ndarray:
    mask = np.isnan(rows).any(axis=1)
    if mask.any():
        rows = rows.copy()
        rows[mask] = fallback_rows[mask]
    return rows


def compute_projection_payload(features: dict[str, np.ndarray], projection: str, distance_reference: str, distance_metric: str) -> dict[str, Any]:
    sonata = features["sonata_features"]
    chorus = fill_missing_rows(features["chorus_features"], sonata)
    fused = features["fused_features"]
    modality_arrays = {"sonata": sonata, "chorus": chorus, "fused": fused}

    if projection == "shared_pca3":
        pooled = np.concatenate([sonata, chorus, fused], axis=0)
        pca = PCA(n_components=3)
        pooled_proj = pca.fit_transform(pooled)
        start = 0
        projected = {}
        for name, arr in modality_arrays.items():
            end = start + arr.shape[0]
            projected[name] = pooled_proj[start:end]
            start = end
        mins = np.percentile(pooled_proj, 1.0, axis=0)
        maxs = np.percentile(pooled_proj, 99.0, axis=0)
        colors = {name: robust_rgb_from_projection(arr, mins, maxs) for name, arr in projected.items()}
        return {
            "mode": projection,
            "colors": colors,
            "projection_stats": {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "components": pca.components_.tolist(),
                "mean": pca.mean_.tolist(),
                "percentile_min": mins.tolist(),
                "percentile_max": maxs.tolist(),
            },
        }

    if projection == "delta_pca3":
        chorus_delta = chorus - sonata
        fused_delta = fused - sonata
        delta_pooled = np.concatenate([chorus_delta, fused_delta], axis=0)
        delta_pca = PCA(n_components=3)
        delta_proj = delta_pca.fit_transform(delta_pooled)
        sonata_pca = PCA(n_components=3)
        sonata_proj = sonata_pca.fit_transform(sonata)
        projected = {
            "sonata": sonata_proj,
            "chorus": delta_proj[: chorus_delta.shape[0]],
            "fused": delta_proj[chorus_delta.shape[0] :],
        }
        delta_mins = np.percentile(delta_proj, 1.0, axis=0)
        delta_maxs = np.percentile(delta_proj, 99.0, axis=0)
        sonata_mins = np.percentile(sonata_proj, 1.0, axis=0)
        sonata_maxs = np.percentile(sonata_proj, 99.0, axis=0)
        colors = {
            "sonata": robust_rgb_from_projection(projected["sonata"], sonata_mins, sonata_maxs),
            "chorus": robust_rgb_from_projection(projected["chorus"], delta_mins, delta_maxs),
            "fused": robust_rgb_from_projection(projected["fused"], delta_mins, delta_maxs),
        }
        return {
            "mode": projection,
            "colors": colors,
            "projection_stats": {
                "basis": "sonata_feature_pca_and_full_dimensional_delta_pca",
                "sonata": "PCA(sonata_features)",
                "chorus_delta": "chorus_features - sonata_features",
                "fused_delta": "fused_features - sonata_features",
                "delta_explained_variance_ratio": delta_pca.explained_variance_ratio_.tolist(),
                "delta_components": delta_pca.components_.tolist(),
                "delta_mean": delta_pca.mean_.tolist(),
                "delta_percentile_min": delta_mins.tolist(),
                "delta_percentile_max": delta_maxs.tolist(),
                "sonata_explained_variance_ratio": sonata_pca.explained_variance_ratio_.tolist(),
                "sonata_components": sonata_pca.components_.tolist(),
                "sonata_mean": sonata_pca.mean_.tolist(),
                "sonata_percentile_min": sonata_mins.tolist(),
                "sonata_percentile_max": sonata_maxs.tolist(),
            },
        }

    if projection == "delta_shared_scale_pca3":
        pooled = np.concatenate([sonata, chorus, fused], axis=0)
        shared_pca = PCA(n_components=3)
        pooled_proj = shared_pca.fit_transform(pooled)
        shared_mins = np.percentile(pooled_proj, 1.0, axis=0)
        shared_maxs = np.percentile(pooled_proj, 99.0, axis=0)

        chorus_delta = chorus - sonata
        fused_delta = fused - sonata
        sonata_proj = shared_pca.transform(sonata)
        chorus_delta_proj = chorus_delta @ shared_pca.components_.T
        fused_delta_proj = fused_delta @ shared_pca.components_.T
        colors = {
            "sonata": robust_rgb_from_projection(sonata_proj, shared_mins, shared_maxs),
            "chorus": robust_rgb_from_projection(chorus_delta_proj, shared_mins, shared_maxs),
            "fused": robust_rgb_from_projection(fused_delta_proj, shared_mins, shared_maxs),
        }
        delta_proj = np.concatenate([chorus_delta_proj, fused_delta_proj], axis=0)
        return {
            "mode": projection,
            "colors": colors,
            "projection_stats": {
                "basis": "shared_pca3_feature_basis_and_color_scale",
                "sonata": "PCA(sonata_features) using the same PCA/range as shared_pca3",
                "chorus_delta": "(chorus_features - sonata_features) projected onto shared_pca3 components",
                "fused_delta": "(fused_features - sonata_features) projected onto shared_pca3 components",
                "shared_explained_variance_ratio": shared_pca.explained_variance_ratio_.tolist(),
                "shared_components": shared_pca.components_.tolist(),
                "shared_mean": shared_pca.mean_.tolist(),
                "shared_percentile_min": shared_mins.tolist(),
                "shared_percentile_max": shared_maxs.tolist(),
                "delta_projected_percentile_min": np.percentile(delta_proj, 1.0, axis=0).tolist(),
                "delta_projected_percentile_max": np.percentile(delta_proj, 99.0, axis=0).tolist(),
            },
        }

    reference = modality_arrays[distance_reference]
    scalar_maps = {}
    for name, arr in modality_arrays.items():
        if distance_metric == "cosine":
            ref_norm = np.linalg.norm(reference, axis=1, keepdims=True)
            arr_norm = np.linalg.norm(arr, axis=1, keepdims=True)
            denom = np.maximum(ref_norm * arr_norm, 1e-6)
            scalar = 1.0 - np.sum(reference * arr, axis=1, keepdims=True) / denom
        else:
            scalar = np.linalg.norm(reference - arr, axis=1, keepdims=True)
        scalar_maps[name] = scalar

    pooled_scalar = np.concatenate(list(scalar_maps.values()), axis=0)
    smin = float(np.percentile(pooled_scalar, 1.0))
    smax = float(np.percentile(pooled_scalar, 99.0))
    denom = max(smax - smin, 1e-6)
    colors = {}
    for name, scalar in scalar_maps.items():
        norm = np.clip((scalar - smin) / denom, 0.0, 1.0)
        rgb = np.concatenate([norm, np.zeros_like(norm), 1.0 - norm], axis=1)
        colors[name] = np.round(rgb * 255.0).astype(np.uint8)

    return {
        "mode": projection,
        "colors": colors,
        "projection_stats": {
            "distance_reference": distance_reference,
            "distance_metric": distance_metric,
            "percentile_min": smin,
            "percentile_max": smax,
        },
    }


def feature_delta_stats(features: dict[str, np.ndarray]) -> dict[str, Any]:
    sonata = features["sonata_features"].astype(np.float32)
    chorus = features["chorus_features"].astype(np.float32)
    fused = features["fused_features"].astype(np.float32)
    fused_delta = np.linalg.norm(fused - sonata, axis=1)
    chorus_delta = np.linalg.norm(chorus - sonata, axis=1)
    unmatched_mask = np.isnan(chorus).any(axis=1)
    unchanged_mask = fused_delta < 1e-7
    matched_mask = np.asarray(features["matched_mask"], dtype=bool)
    return {
        "fused_eq_sonata_count": int(np.sum(unchanged_mask)),
        "fused_eq_sonata_fraction": float(np.mean(unchanged_mask)),
        "unmatched_count": int(np.sum(unmatched_mask)),
        "unmatched_fraction": float(np.mean(unmatched_mask)),
        "matched_count_from_mask": int(np.sum(matched_mask)),
        "fused_eq_sonata_unmatched_count": int(np.sum(unchanged_mask & unmatched_mask)),
        "fused_eq_sonata_matched_count": int(np.sum(unchanged_mask & matched_mask)),
        "fused_sonata_l2_mean": float(np.mean(fused_delta)),
        "fused_sonata_l2_max": float(np.max(fused_delta)),
        "fused_sonata_l2_mean_matched": float(np.mean(fused_delta[matched_mask])) if np.any(matched_mask) else None,
        "chorus_sonata_l2_mean_matched": float(np.nanmean(chorus_delta[matched_mask])) if np.any(matched_mask) else None,
        "chorus_nan_rows": int(np.sum(unmatched_mask)),
    }


def transfer_token_colors_to_points(token_centers: np.ndarray, token_colors: np.ndarray, points: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, dict[str, Any]]:
    tree = cKDTree(token_centers)
    k = max(int(k_neighbors), 1)
    distances, indices = tree.query(points, k=k)
    if k == 1:
        colors = token_colors[indices]
        dists = np.asarray(distances, dtype=np.float32)
    else:
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        weights = 1.0 / np.maximum(distances, 1e-6)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
        colors = np.sum(token_colors[indices].astype(np.float32) * weights[..., None], axis=1)
        colors = np.round(colors).astype(np.uint8)
        dists = distances[:, 0]
    return colors, {
        "k_neighbors": k,
        "nearest_distance_mean": float(np.mean(dists)),
        "nearest_distance_max": float(np.max(dists)),
    }


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, extra_fields: Optional[list[tuple[str, np.ndarray]]] = None) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz/rgb row count mismatch for PLY export")
    extra_fields = extra_fields or []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        for name, values in extra_fields:
            values = np.asarray(values)
            if values.ndim == 1:
                handle.write(f"property float {name}\n")
            else:
                for idx in range(values.shape[1]):
                    handle.write(f"property float {name}_{idx}\n")
        handle.write("end_header\n")
        for row_idx in range(xyz.shape[0]):
            values = [
                f"{float(xyz[row_idx, 0]):.6f}",
                f"{float(xyz[row_idx, 1]):.6f}",
                f"{float(xyz[row_idx, 2]):.6f}",
                str(int(rgb[row_idx, 0])),
                str(int(rgb[row_idx, 1])),
                str(int(rgb[row_idx, 2])),
            ]
            for _, extra in extra_fields:
                extra = np.asarray(extra)
                if extra.ndim == 1:
                    values.append(f"{float(extra[row_idx]):.6f}")
                else:
                    values.extend(f"{float(value):.6f}" for value in extra[row_idx])
            handle.write(" ".join(values) + "\n")


def write_ascii_mesh_ply(path: Path, vertices: np.ndarray, rgb: np.ndarray, faces: np.ndarray) -> None:
    if vertices.shape[0] != rgb.shape[0]:
        raise ValueError("vertices/rgb row count mismatch for mesh PLY export")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for point, color in zip(vertices, rgb):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


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
        dtype=np.int32,
    )
    return vertices, faces


def build_voxel_cube_mesh(
    centers: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    centers = centers.astype(np.float32, copy=False)
    colors = colors.astype(np.uint8, copy=False)
    unit_vertices, unit_faces = cube_template()
    cube_vertices = unit_vertices * float(voxel_size)
    per_cube_vertices = cube_vertices.shape[0]
    per_cube_faces = unit_faces.shape[0]
    count = int(centers.shape[0])
    all_vertices = np.empty((count * per_cube_vertices, 3), dtype=np.float32)
    all_colors = np.empty((count * per_cube_vertices, 3), dtype=np.uint8)
    all_faces = np.empty((count * per_cube_faces, 3), dtype=np.int32)
    for idx in range(count):
        start_v = idx * per_cube_vertices
        end_v = start_v + per_cube_vertices
        start_f = idx * per_cube_faces
        end_f = start_f + per_cube_faces
        all_vertices[start_v:end_v] = cube_vertices + centers[idx][None, :]
        all_colors[start_v:end_v] = colors[idx][None, :]
        all_faces[start_f:end_f] = unit_faces + start_v
    metadata = {
        "voxel_count": count,
        "vertices": int(all_vertices.shape[0]),
        "faces": int(all_faces.shape[0]),
        "voxel_size": float(voxel_size),
    }
    return all_vertices, all_colors, all_faces, metadata


def sphere_template(stacks: int, slices: int) -> tuple[np.ndarray, np.ndarray]:
    stacks = max(int(stacks), 3)
    slices = max(int(slices), 6)
    vertices = [(0.0, 0.0, 1.0)]
    for stack in range(1, stacks):
        phi = np.pi * stack / stacks
        z = np.cos(phi)
        radius = np.sin(phi)
        for slc in range(slices):
            theta = 2.0 * np.pi * slc / slices
            vertices.append((radius * np.cos(theta), radius * np.sin(theta), z))
    vertices.append((0.0, 0.0, -1.0))

    north = 0
    south = len(vertices) - 1
    faces = []
    first_ring = 1
    for slc in range(slices):
        faces.append((north, first_ring + slc, first_ring + (slc + 1) % slices))
    for stack in range(stacks - 2):
        ring_a = 1 + stack * slices
        ring_b = ring_a + slices
        for slc in range(slices):
            a0 = ring_a + slc
            a1 = ring_a + (slc + 1) % slices
            b0 = ring_b + slc
            b1 = ring_b + (slc + 1) % slices
            faces.append((a0, b0, a1))
            faces.append((a1, b0, b1))
    last_ring = 1 + (stacks - 2) * slices
    for slc in range(slices):
        faces.append((last_ring + slc, south, last_ring + (slc + 1) % slices))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def quat_to_rotmat(quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-8)
    if order == "wxyz":
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    else:
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


def build_gaussian_ellipsoid_mesh(
    centers: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    colors: np.ndarray,
    stacks: int,
    slices: int,
    scale_multiplier: float,
    max_count: int,
    quat_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    count = int(centers.shape[0]) if max_count <= 0 else min(int(max_count), int(centers.shape[0]))
    centers = centers[:count].astype(np.float32, copy=False)
    scale = np.maximum(scale[:count].astype(np.float32, copy=False), 1e-6) * float(scale_multiplier)
    quat = quat[:count].astype(np.float32, copy=False)
    colors = colors[:count].astype(np.uint8, copy=False)

    unit_vertices, unit_faces = sphere_template(stacks, slices)
    rotations = quat_to_rotmat(quat, quat_order)
    per_gaussian_vertices = unit_vertices.shape[0]
    per_gaussian_faces = unit_faces.shape[0]
    all_vertices = np.empty((count * per_gaussian_vertices, 3), dtype=np.float32)
    all_colors = np.empty((count * per_gaussian_vertices, 3), dtype=np.uint8)
    all_faces = np.empty((count * per_gaussian_faces, 3), dtype=np.int32)

    for idx in range(count):
        start_v = idx * per_gaussian_vertices
        end_v = start_v + per_gaussian_vertices
        start_f = idx * per_gaussian_faces
        end_f = start_f + per_gaussian_faces
        local = unit_vertices * scale[idx][None, :]
        all_vertices[start_v:end_v] = local @ rotations[idx].T + centers[idx][None, :]
        all_colors[start_v:end_v] = colors[idx][None, :]
        all_faces[start_f:end_f] = unit_faces + start_v

    metadata = {
        "gaussians_exported": count,
        "vertices": int(all_vertices.shape[0]),
        "faces": int(all_faces.shape[0]),
        "stacks": int(max(stacks, 3)),
        "slices": int(max(slices, 6)),
        "scale_multiplier": float(scale_multiplier),
        "quat_order": quat_order,
    }
    return all_vertices, all_colors, all_faces, metadata


def export_room_geometry(
    output_prefix: Path,
    room_points: np.ndarray,
    token_centers: np.ndarray,
    modality_colors: dict[str, np.ndarray],
    k_neighbors: int,
    export_ply: bool,
) -> dict[str, Any]:
    metadata = {}
    for modality, token_colors in modality_colors.items():
        room_colors, transfer_meta = transfer_token_colors_to_points(token_centers, token_colors, room_points, k_neighbors)
        metadata[modality] = transfer_meta
        if export_ply:
            write_ascii_ply(output_prefix.with_name(f"{output_prefix.name}_{modality}_room.ply"), room_points, room_colors)
    return metadata


def export_gaussian_geometry(
    output_prefix: Path,
    gaussian_centers: np.ndarray,
    token_centers: np.ndarray,
    modality_colors: dict[str, np.ndarray],
    scale: Optional[np.ndarray],
    quat: Optional[np.ndarray],
    opacity: Optional[np.ndarray],
    export_ply: bool,
    export_ellipsoids: bool,
    ellipsoid_stacks: int,
    ellipsoid_slices: int,
    ellipsoid_scale_multiplier: float,
    ellipsoid_max_count: int,
    quat_order: str,
) -> dict[str, Any]:
    extra_fields = []
    if scale is not None:
        extra_fields.append(("scale", scale))
    if quat is not None:
        extra_fields.append(("quat", quat))
    if opacity is not None:
        extra_fields.append(("opacity", opacity))
    metadata = {"point_count": int(gaussian_centers.shape[0])}
    if export_ellipsoids and (scale is None or quat is None):
        metadata["ellipsoid_export"] = {
            "enabled": False,
            "reason": "missing_scale_or_quat",
        }
    if export_ply:
        for modality, token_colors in modality_colors.items():
            colors, transfer_meta = transfer_token_colors_to_points(
                token_centers,
                token_colors,
                gaussian_centers,
                1,
            )
            metadata[f"{modality}_transfer"] = transfer_meta
            write_ascii_ply(
                output_prefix.with_name(f"{output_prefix.name}_{modality}_gaussians.ply"),
                gaussian_centers,
                colors,
                extra_fields=extra_fields,
            )
            if export_ellipsoids and scale is not None and quat is not None:
                vertices, vertex_colors, faces, ellipsoid_meta = build_gaussian_ellipsoid_mesh(
                    gaussian_centers,
                    scale,
                    quat,
                    colors,
                    ellipsoid_stacks,
                    ellipsoid_slices,
                    ellipsoid_scale_multiplier,
                    ellipsoid_max_count,
                    quat_order,
                )
                mesh_path = output_prefix.with_name(f"{output_prefix.name}_{modality}_gaussian_ellipsoids.ply")
                write_ascii_mesh_ply(mesh_path, vertices, vertex_colors, faces)
                metadata[f"{modality}_ellipsoids"] = {
                    **ellipsoid_meta,
                    "path": str(mesh_path),
                }
    return metadata


def export_input_voxel_geometry(
    output_prefix: Path,
    voxel_centers: np.ndarray,
    token_centers: np.ndarray,
    modality_colors: dict[str, np.ndarray],
    voxel_size: float,
    k_neighbors: int,
    export_ply: bool,
) -> dict[str, Any]:
    metadata = {
        "voxel_count": int(voxel_centers.shape[0]),
        "voxel_size": float(voxel_size),
    }
    if export_ply:
        for modality, token_colors in modality_colors.items():
            colors, transfer_meta = transfer_token_colors_to_points(
                token_centers,
                token_colors,
                voxel_centers,
                k_neighbors,
            )
            vertices, vertex_colors, faces, mesh_meta = build_voxel_cube_mesh(
                voxel_centers,
                colors,
                voxel_size,
            )
            mesh_path = output_prefix.with_name(f"{output_prefix.name}_{modality}_input_voxels.ply")
            write_ascii_mesh_ply(mesh_path, vertices, vertex_colors, faces)
            metadata[modality] = {
                **transfer_meta,
                **mesh_meta,
                "path": str(mesh_path),
            }
    return metadata


def export_encoded_voxel_geometry(
    output_prefix: Path,
    voxel_centers: np.ndarray,
    modality_colors: dict[str, np.ndarray],
    feature_payload: dict[str, Any],
    voxel_size: float,
    export_ply: bool,
) -> dict[str, Any]:
    metadata = {
        "voxel_count": int(voxel_centers.shape[0]),
        "voxel_size": float(voxel_size),
        "color_transfer": False,
        "support": "final_encoder_voxels",
    }
    chorus_valid = ~np.isnan(feature_payload["chorus_features"]).any(axis=1)
    for modality, token_colors in modality_colors.items():
        row_mask = np.ones((voxel_centers.shape[0],), dtype=bool)
        if modality == "chorus":
            row_mask = chorus_valid
        exported_centers = voxel_centers[row_mask]
        exported_colors = token_colors[row_mask]
        modality_meta = {
            "rows_exported": int(exported_centers.shape[0]),
            "rows_available": int(voxel_centers.shape[0]),
            "unmatched_rows_skipped": int(np.sum(~row_mask)),
        }
        if export_ply and exported_centers.shape[0] > 0:
            vertices, vertex_colors, faces, mesh_meta = build_voxel_cube_mesh(
                exported_centers,
                exported_colors,
                voxel_size,
            )
            mesh_path = output_prefix.with_name(f"{output_prefix.name}_{modality}_encoded_voxels.ply")
            write_ascii_mesh_ply(mesh_path, vertices, vertex_colors, faces)
            modality_meta.update(
                {
                    **mesh_meta,
                    "path": str(mesh_path),
                }
            )
        metadata[modality] = modality_meta
    return metadata


def export_npz_payload(
    output_prefix: Path,
    feature_payload: dict[str, Any],
    projection_payload: dict[str, Any],
    room_points: np.ndarray,
    room_colors: np.ndarray,
    voxel_size: float,
) -> None:
    empty_xyz = np.empty((0, 3), dtype=np.float32)
    empty_rgb = np.empty((0, 3), dtype=np.uint8)
    gaussian_centers = feature_payload["gaussian_centers"]
    gaussian_centers_stored = feature_payload["gaussian_centers_stored"]
    gaussian_color = feature_payload["gaussian_color"]
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        room_points=room_points,
        room_colors=colors_to_uint8(room_colors, room_points.shape[0]),
        sonata_centers_raw=feature_payload["sonata_centers_raw"],
        encoded_voxel_centers=feature_payload["encoded_voxel_centers_raw"],
        encoded_voxel_size=np.asarray(feature_payload["encoded_voxel_size"], dtype=np.float32),
        encoded_voxel_stride=np.asarray(feature_payload["encoded_voxel_stride"], dtype=np.float32),
        input_voxel_centers=feature_payload["input_voxel_centers_raw"],
        voxel_size=np.asarray(voxel_size, dtype=np.float32),
        aligned_sonata_centers_raw=feature_payload["aligned_sonata_centers_raw"],
        aligned_sonata_centers_raw_source=np.asarray(
            feature_payload["aligned_sonata_centers_raw_source"],
            dtype="U",
        ),
        gaussian_centers=empty_xyz if gaussian_centers is None else gaussian_centers,
        gaussian_centers_stored=empty_xyz if gaussian_centers_stored is None else gaussian_centers_stored,
        gaussian_colors=empty_rgb if gaussian_color is None else gaussian_color,
        sonata_features=feature_payload["sonata_features"],
        chorus_features=feature_payload["chorus_features"],
        fused_features=feature_payload["fused_features"],
        sonata_colors=projection_payload["colors"]["sonata"],
        chorus_colors=projection_payload["colors"]["chorus"],
        fused_colors=projection_payload["colors"]["fused"],
        matched_mask=feature_payload["matched_mask"],
        exact_mask=feature_payload["exact_mask"],
    )


def make_output_prefix(output_dir: Path, scene_id: str, projection: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{scene_id}_{projection}"


def main() -> None:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    _, model = load_model(args, yaml_config)

    effective_aligned_root = Path(getattr(model.config, "chorus_aligned_root"))
    aligned_dir = resolve_aligned_dir(effective_aligned_root, args.scene_id, args.split)
    pcd_path = args.dataset_dir / "pcd" / f"{args.scene_id}.ply"
    if not pcd_path.is_file():
        raise FileNotFoundError(f"Room point cloud not found: {pcd_path}")

    num_bins = int(getattr(model.config, "num_bins", 0) or model.config.point_config["num_bins"])
    grid_size = SceneGraphLayout.get_grid_size(num_bins)
    encoded_voxel_stride = encoded_voxel_stride_from_backbone(model.chorus_fusion_encoder.sonata_backbone, num_bins)
    encoded_voxel_size = float(grid_size) * encoded_voxel_stride
    raw_pcd = load_o3d_pcd(str(pcd_path))
    room_points, room_colors = get_points_and_colors(raw_pcd)
    room_colors = room_colors_from_ply(pcd_path, room_points, room_colors)

    model_pcd = raw_pcd
    if args.cleanup:
        model_pcd = cleanup_pcd(model_pcd, voxel_size=grid_size)
    model_points, model_colors = get_points_and_colors(model_pcd)
    input_pcd = preprocess_point_cloud(model_points, model_colors, grid_size, num_bins).to(args.device)
    include_gaussian_support = args.geometry in {"gaussian_centers", "both", "all"}

    feature_payload = extract_scene_features(
        model,
        input_pcd,
        args.scene_id,
        aligned_dir,
        torch.device(args.device),
        model_points.min(axis=0).astype(np.float32),
        encoded_voxel_size,
        encoded_voxel_stride,
        include_gaussian_support,
    )
    projection_payload = compute_projection_payload(
        feature_payload,
        args.projection,
        args.distance_reference,
        args.distance_metric,
    )
    delta_stats = feature_delta_stats(feature_payload)

    output_prefix = make_output_prefix(args.output_dir, args.scene_id, args.projection)
    geometry_meta: dict[str, Any] = {}
    if args.geometry in {"encoded_voxels", "all"}:
        geometry_meta["encoded_voxels"] = export_encoded_voxel_geometry(
            output_prefix,
            feature_payload["encoded_voxel_centers_raw"],
            projection_payload["colors"],
            feature_payload,
            feature_payload["encoded_voxel_size"],
            args.export_ply,
        )
    if args.geometry in {"room_pcd", "both", "all"}:
        geometry_meta["room_pcd"] = export_room_geometry(
            output_prefix,
            room_points,
            feature_payload["sonata_centers_raw"],
            projection_payload["colors"],
            args.k_neighbors,
            args.export_ply,
        )
    if args.geometry in {"input_voxels", "all"}:
        geometry_meta["input_voxels"] = export_input_voxel_geometry(
            output_prefix,
            feature_payload["input_voxel_centers_raw"],
            feature_payload["sonata_centers_raw"],
            projection_payload["colors"],
            grid_size,
            args.k_neighbors,
            args.export_ply,
        )
    if args.geometry in {"gaussian_centers", "both", "all"}:
        geometry_meta["gaussian_centers"] = export_gaussian_geometry(
            output_prefix,
            feature_payload["gaussian_centers"],
            feature_payload["sonata_centers_raw"],
            projection_payload["colors"],
            feature_payload["gaussian_scale"],
            feature_payload["gaussian_quat"],
            feature_payload["gaussian_opacity"],
            args.export_ply,
            args.export_gaussian_ellipsoids,
            args.ellipsoid_stacks,
            args.ellipsoid_slices,
            args.ellipsoid_scale_multiplier,
            args.ellipsoid_max_count,
            args.quat_order,
        )

    if args.export_npz:
        export_npz_payload(output_prefix, feature_payload, projection_payload, room_points, room_colors, grid_size)

    metadata = {
        "scene_id": args.scene_id,
        "checkpoint": args.checkpoint,
        "architecture_config": args.architecture_config,
        "model_loading": getattr(model, "_feature_field_loading_info", {}),
        "config": str(args.config),
        "dataset_dir": str(args.dataset_dir),
        "aligned_root": str(effective_aligned_root),
        "aligned_root_overridden": args.aligned_root is not None,
        "aligned_dir": str(aligned_dir),
        "aligned_sonata_centers_raw_source": feature_payload["aligned_sonata_centers_raw_source"],
        "gaussian_visual_frame": feature_payload["gaussian_visual_frame"],
        "pcd_path": str(pcd_path),
        "effective_chorus_config": {
            "chorus_checkpoint": getattr(model.config, "chorus_checkpoint", None),
            "chorus_config": getattr(model.config, "chorus_config", None),
            "input_mode": getattr(model.chorus_fusion_encoder, "input_mode", None),
            "aligned_root": str(getattr(model.chorus_fusion_encoder, "aligned_root", "")),
            "fusion_mode": getattr(model.chorus_fusion_encoder, "fusion_mode", None),
            "fusion_gate_mode": getattr(model.chorus_fusion_encoder, "fusion_gate_mode", None),
            "fusion_chorus_weight": getattr(model.chorus_fusion_encoder, "fusion_chorus_weight", None),
            "match_grid_radius": getattr(model.chorus_fusion_encoder, "match_grid_radius", None),
            "exact_match_first": getattr(model.chorus_fusion_encoder, "exact_match_first", None),
            "contrastive_exact_only": getattr(model.chorus_fusion_encoder, "contrastive_exact_only", None),
        },
        "projection": projection_payload["mode"],
        "projection_stats": projection_payload["projection_stats"],
        "geometry": args.geometry,
        "cleanup": bool(args.cleanup),
        "gaussian_ellipsoid_export": {
            "enabled": bool(args.export_gaussian_ellipsoids),
            "stacks": int(args.ellipsoid_stacks),
            "slices": int(args.ellipsoid_slices),
            "scale_multiplier": float(args.ellipsoid_scale_multiplier),
            "max_count": int(args.ellipsoid_max_count),
            "quat_order": args.quat_order,
        },
        "room_point_count": int(room_points.shape[0]),
        "input_voxel_count": int(feature_payload["input_voxel_centers_raw"].shape[0]),
        "input_voxel_size": float(grid_size),
        "encoded_voxel_count": int(feature_payload["encoded_voxel_centers_raw"].shape[0]),
        "encoded_voxel_size": float(feature_payload["encoded_voxel_size"]),
        "encoded_voxel_stride": float(feature_payload["encoded_voxel_stride"]),
        "model_input_point_count": int(model_points.shape[0]),
        "original_scene": {
            "room_has_color": bool(np.any(room_colors != 0)),
            "gaussian_count": 0 if feature_payload["gaussian_centers"] is None else int(feature_payload["gaussian_centers"].shape[0]),
            "gaussian_has_color": bool(
                feature_payload["gaussian_color"] is not None
                and np.any(feature_payload["gaussian_color"] != 0)
            ),
        },
        "matched_count": feature_payload["matched_count"],
        "exact_match_count": feature_payload["exact_match_count"],
        "used_chorus_tokens": feature_payload["used_chorus_tokens"],
        "token_count": int(feature_payload["sonata_features"].shape[0]),
        "feature_delta_stats": delta_stats,
        "gate_avg": feature_payload["gate_avg"],
        "sidecar_match_stats": feature_payload["sidecar_match_stats"],
        "transfer": geometry_meta,
        "outputs": {
            "prefix": str(output_prefix),
            "npz": str(output_prefix.with_suffix(".npz")) if args.export_npz else None,
        },
    }
    metadata_path = output_prefix.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({"status": "ok", "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
