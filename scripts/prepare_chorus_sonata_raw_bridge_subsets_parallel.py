from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Prepare raw-bridge Chorus/Sonata subset scenes in parallel."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chorus-root", required=True)
    parser.add_argument("--r3scan-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--cleanup-num-nb", type=int, default=15)
    parser.add_argument("--cleanup-std-ratio", type=float, default=2.0)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--match-radius-voxels", type=int, default=2)
    parser.add_argument("--missing-voxel-policy", choices=("nearest", "error", "drop"), default="nearest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def split_json_candidates(dataset_root: Path, split_name: str) -> list[Path]:
    candidates = [dataset_root / f"scene_graph_{split_name}.json"]
    if split_name == "val":
        candidates.append(dataset_root / "scene_graph_val_3dgs.json")
    return candidates


def resolve_split_json(dataset_root: Path, split_name: str) -> Path | None:
    for candidate in split_json_candidates(dataset_root, split_name):
        if candidate.exists():
            return candidate
    return None


def discover_scenes(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset_root = Path(args.dataset_root)
    chorus_root = Path(args.chorus_root)
    r3scan_root = Path(args.r3scan_root)
    scenes: list[dict[str, str]] = []

    for split_name in [part.strip() for part in args.splits.split(",") if part.strip()]:
        split_json = resolve_split_json(dataset_root, split_name)
        if split_json is None:
            continue
        with split_json.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            point_clouds = row.get("point_clouds") or []
            if not point_clouds:
                continue
            scene_pcd = dataset_root / point_clouds[0]
            scene_id = scene_pcd.stem
            scan_id = scene_id.rsplit("_split", 1)[0]
            chorus_ply = chorus_root / scan_id / "ckpts" / "point_cloud_30000.ply"
            scan_dir = r3scan_root / scan_id
            output_dir = Path(args.output_root) / split_name / scene_id
            if not scene_pcd.exists() or not chorus_ply.exists() or not scan_dir.exists():
                continue
            if not args.overwrite and (output_dir / "coord.npy").exists():
                continue
            scenes.append(
                {
                    "split": split_name,
                    "scene_id": scene_id,
                    "scan_id": scan_id,
                    "scene_pcd": str(scene_pcd),
                    "chorus_ply": str(chorus_ply),
                    "scan_dir": str(scan_dir),
                    "output": str(output_dir),
                }
            )
    return sorted(scenes, key=lambda item: (item["split"], item["scene_id"]))


def run_one(args: argparse.Namespace, item: dict[str, str]) -> tuple[str, int, str]:
    script = Path(__file__).resolve().parent / "prepare_chorus_scene_raw_bridge_from_sonata_voxels.py"
    cmd = [
        args.python,
        str(script),
        "--scene-pcd",
        item["scene_pcd"],
        "--chorus-ply",
        item["chorus_ply"],
        "--scan-dir",
        item["scan_dir"],
        "--output",
        item["output"],
        "--voxel-size",
        str(args.voxel_size),
        "--cleanup-num-nb",
        str(args.cleanup_num_nb),
        "--cleanup-std-ratio",
        str(args.cleanup_std_ratio),
        "--match-radius-voxels",
        str(args.match_radius_voxels),
        "--missing-voxel-policy",
        args.missing_voxel_policy,
    ]
    if args.no_cleanup:
        cmd.append("--no-cleanup")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    return item["scene_id"], completed.returncode, completed.stdout


def progress_iter(futures):
    try:
        from tqdm import tqdm

        return tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="prepare raw bridge scenes",
        )
    except ImportError:
        return concurrent.futures.as_completed(futures)


def main() -> None:
    args = parse_args()
    scenes = discover_scenes(args)
    split_breakdown: dict[str, int] = {}
    for item in scenes:
        split_breakdown[item["split"]] = split_breakdown.get(item["split"], 0) + 1

    print(
        json.dumps(
            {
                "scene_count": len(scenes),
                "split_breakdown": split_breakdown,
                "workers": args.workers,
                "overwrite": args.overwrite,
                "output_root": args.output_root,
            },
            indent=2,
        ),
        flush=True,
    )
    if not scenes:
        print("No scenes need preparation.", flush=True)
        return

    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, args, item) for item in scenes]
        for future in progress_iter(futures):
            scene_id, returncode, output = future.result()
            print(f"\n===== {scene_id} returncode={returncode} =====", flush=True)
            print(output[-4000:], flush=True)
            if returncode != 0:
                failures.append(scene_id)

    if failures:
        raise RuntimeError(f"Raw-bridge preparation failed for {len(failures)} scenes: {failures[:20]}")


if __name__ == "__main__":
    main()
