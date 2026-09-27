from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SpatialLM dataset root for Chorus+PCD fusion training."
    )
    parser.add_argument(
        "--source-root",
        default="/work/luka_milivojevic/3rscan_subset_scene_graph_data_rio10",
        help="Original SpatialLM scene-graph dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="/work/luka_milivojevic/3rscan_subset_scene_graph_data_rio10_chorus_fusion",
        help="Dataset root to create for fusion SFT.",
    )
    parser.add_argument(
        "--aligned-root",
        default="/work/luka_milivojevic/3rscan_subset_chorus_sonata_aligned_rio10",
        help="Root with prepared aligned Chorus splits.",
    )
    parser.add_argument(
        "--train-name",
        default="scene_graph_train",
        help="Dataset-info key/file stem for train rows.",
    )
    parser.add_argument(
        "--val-name",
        default="scene_graph_val_3dgs",
        help="Dataset-info key/file stem for validation rows with aligned 3DGS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated JSON files.",
    )
    parser.add_argument(
        "--train-3dgs-only",
        action="store_true",
        help="Keep only train rows whose split has a prepared aligned Chorus/3DGS directory.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def scene_id_from_row(row: dict) -> str | None:
    point_clouds = row.get("point_clouds") or []
    if not point_clouds:
        return None
    return Path(point_clouds[0]).stem


def has_aligned_chorus(aligned_root: Path, split_name: str, scene_id: str) -> bool:
    candidates = (
        aligned_root / split_name / scene_id / "coord.npy",
        aligned_root / scene_id / "coord.npy",
    )
    return any(path.exists() for path in candidates)


def ensure_pcd_link(source_root: Path, output_root: Path) -> None:
    source_pcd = source_root / "pcd"
    output_pcd = output_root / "pcd"
    if output_pcd.exists():
        return
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_pcd, output_pcd, target_is_directory=True)
    except OSError:
        shutil.copytree(source_pcd, output_pcd)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    aligned_root = Path(args.aligned_root).expanduser().resolve()

    ensure_pcd_link(source_root, output_root)
    train_rows_source = load_json(source_root / "scene_graph_train.json")
    if args.train_3dgs_only:
        train_rows = [
            row
            for row in train_rows_source
            if (scene_id := scene_id_from_row(row)) is not None
            and has_aligned_chorus(aligned_root, "train", scene_id)
        ]
    else:
        train_rows = train_rows_source
    val_rows = load_json(source_root / "scene_graph_val.json")
    val_3dgs_rows = [
        row
        for row in val_rows
        if (scene_id := scene_id_from_row(row)) is not None
        and has_aligned_chorus(aligned_root, "val", scene_id)
    ]

    train_path = output_root / f"{args.train_name}.json"
    val_path = output_root / f"{args.val_name}.json"
    dataset_info_path = output_root / "dataset_info.json"
    for path in (train_path, val_path, dataset_info_path):
        if path.exists() and not args.overwrite:
            continue
    dump_json(train_path, train_rows)
    dump_json(val_path, val_3dgs_rows)
    dump_json(
        dataset_info_path,
        {
            args.train_name: {
                "file_name": f"{args.train_name}.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
            },
            args.val_name: {
                "file_name": f"{args.val_name}.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
            },
        },
    )
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "aligned_root": str(aligned_root),
                "train_3dgs_only": bool(args.train_3dgs_only),
                "train_rows_source": len(train_rows_source),
                "train_rows": len(train_rows),
                "val_rows_source": len(val_rows),
                "val_rows_with_aligned_3dgs": len(val_3dgs_rows),
                "train_json": str(train_path),
                "val_json": str(val_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
