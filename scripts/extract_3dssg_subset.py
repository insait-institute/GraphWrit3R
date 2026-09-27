import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

POINT_CLOUD_PLACEHOLDER = "<point_cloud>"
LAYOUT_S_PLACEHOLDER = "<|layout_s|>"
LAYOUT_E_PLACEHOLDER = "<|layout_e|>"

CODE_TEMPLATE = (
    '{"num_objects": <int>, "objects": [{"id": <int>, "label": <str>, '
    '"position": [<int>, <int>, <int>], "size": [<int>, <int>, <int>], '
    '"angle_z": <int>}], "relationships": [[<subject_id>, <object_id>, <predicate>]]}\n'
)


def load_subset_entries(subset_dir):
    """Load train and val subset entries, tagged with dataset split name."""
    entries = []
    for split_name, filename in [("train", "relationships_train.json"),
                                  ("val", "relationships_validation.json")]:
        path = Path(subset_dir) / filename
        with open(path) as f:
            data = json.load(f)
        for scan in data["scans"]:
            entries.append({
                "scan_id": scan["scan"],
                "split_num": scan["split"],
                "objects": {int(k): v for k, v in scan["objects"].items()},
                "relationships": scan["relationships"],
                "dataset_split": split_name,
            })
    return entries


def filter_scene_graph(full_sg, subset_obj_ids, subset_rels):
    """Filter a full scene_graph.json to only subset objects and relationships."""
    # Filter objects
    objects_out = [o for o in full_sg["objects"] if o["id"] in subset_obj_ids]
    valid_ids = {o["id"] for o in objects_out}

    # Use subset relationships, but only keep those where both objects have AABBs
    relationships_out = []
    for rel in subset_rels:
        subj_id, obj_id, _pred_id, pred_name = rel
        if subj_id in valid_ids and obj_id in valid_ids:
            relationships_out.append([subj_id, obj_id, pred_name])

    return {
        "num_objects": len(objects_out),
        "objects": objects_out,
        "relationships": relationships_out,
    }


def scene_graph_to_language_string(sg_data):
    """Convert scene graph dict to compact JSON string (matching SceneGraphLayout output)."""

    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(sg_data, separators=(",", ":"), default=_default)


def filter_pointcloud(points, colors, instance_labels, subset_obj_ids):
    """Filter pointcloud to only include points belonging to subset objects."""
    mask = np.isin(instance_labels, list(subset_obj_ids))
    return points[mask], colors[mask]


def write_ply(path, points, colors):
    """Write a binary PLY file matching the existing format (double xyz, uchar rgb)."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by extract_3dssg_subset\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Colors are in [0, 255] range (per SceneVerse convention)
    colors_u8 = np.clip(np.round(colors), 0, 255).astype(np.uint8)
    points_f64 = np.asarray(points, dtype=np.float64)

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<ddd", points_f64[i, 0], points_f64[i, 1], points_f64[i, 2]))
            f.write(struct.pack("<BBB", colors_u8[i, 0], colors_u8[i, 1], colors_u8[i, 2]))


def make_conversation(language_string, scene_id):
    """Build a ShareGPT conversation entry for SpatialLM training."""
    task_prompt = (
        f"Detect objects and their relationships. "
        f"The reference code is as followed: {CODE_TEMPLATE}"
    )
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"{POINT_CLOUD_PLACEHOLDER}{task_prompt}",
            },
            {
                "from": "gpt",
                "value": f"{LAYOUT_S_PLACEHOLDER}{language_string}{LAYOUT_E_PLACEHOLDER}",
            },
        ],
        "point_clouds": [
            f"pcd/{scene_id}.ply",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3DSSG-subset-aligned pointclouds and scene graphs")
    parser.add_argument("--subset_dir", type=str,
                        default="/home/luka_milivojevic/3DSSG_subset/3DSSG_subset")
    parser.add_argument("--scene_graph_dir", type=str,
                        default="/work/luka_milivojevic/SceneVerse/preprocess/ssg/relationships/3RScan")
    parser.add_argument("--pcd_root", type=str,
                        default="/work/luka_milivojevic/sceneverse_3rscan/3RScan/scan_data")
    parser.add_argument("--output_dir", type=str,
                        default="/work/luka_milivojevic/3rscan_subset_scene_graph_data")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process single entry: 'scan_id:split_num' (e.g. abc123:1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pcd_out_dir = output_dir / "pcd"
    pcd_out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading 3DSSG subset entries...")
    entries = load_subset_entries(args.subset_dir)
    print(f"  Total entries: {len(entries)}")

    # Filter to single scene if requested
    if args.scene:
        parts = args.scene.split(":")
        scan_filter = parts[0]
        split_filter = int(parts[1]) if len(parts) > 1 else None
        entries = [e for e in entries
                   if e["scan_id"] == scan_filter
                   and (split_filter is None or e["split_num"] == split_filter)]
        print(f"  Filtered to {len(entries)} entries for --scene {args.scene}")

    # Group by scan_id for pointcloud caching
    entries_by_scan = defaultdict(list)
    for e in entries:
        entries_by_scan[e["scan_id"]].append(e)

    dataset = {"train": [], "val": []}
    stats = {"processed": 0, "skipped_no_sg": 0, "skipped_no_pcd": 0,
             "skipped_empty": 0, "total_objects": 0, "total_rels": 0}

    pcd_root = Path(args.pcd_root)
    sg_root = Path(args.scene_graph_dir)

    for scan_id in tqdm(sorted(entries_by_scan.keys()), desc="Processing scans"):
        # Load full scene graph
        sg_path = sg_root / scan_id / "scene_graph.json"
        if not sg_path.exists():
            stats["skipped_no_sg"] += len(entries_by_scan[scan_id])
            continue

        with open(sg_path) as f:
            full_sg = json.load(f)

        # Load pointcloud with instance labels
        pcd_path = pcd_root / "pcd_with_global_alignment" / f"{scan_id}.pth"
        if not pcd_path.exists():
            stats["skipped_no_pcd"] += len(entries_by_scan[scan_id])
            continue

        pcd_data = torch.load(pcd_path, weights_only=False)
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

        for entry in entries_by_scan[scan_id]:
            subset_obj_ids = set(entry["objects"].keys())
            scene_id = f"{scan_id}_split{entry['split_num']}"

            # Filter scene graph
            filtered_sg = filter_scene_graph(full_sg, subset_obj_ids, entry["relationships"])

            if len(filtered_sg["objects"]) == 0:
                stats["skipped_empty"] += 1
                continue

            # Filter and write pointcloud
            filt_points, filt_colors = filter_pointcloud(
                points, colors, instance_labels, subset_obj_ids)

            if len(filt_points) == 0:
                stats["skipped_empty"] += 1
                continue

            ply_path = pcd_out_dir / f"{scene_id}.ply"
            write_ply(ply_path, filt_points, filt_colors)

            # Build SpatialLM training entry
            language_string = scene_graph_to_language_string(filtered_sg)
            conversation = make_conversation(language_string, scene_id)
            dataset[entry["dataset_split"]].append(conversation)

            stats["processed"] += 1
            stats["total_objects"] += len(filtered_sg["objects"])
            stats["total_rels"] += len(filtered_sg["relationships"])

    # Write output files
    for split_name in ["train", "val"]:
        out_path = output_dir / f"scene_graph_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(dataset[split_name], f, indent=2)

    dataset_info = {
        "scene_graph_train": {
            "file_name": "scene_graph_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
        },
        "scene_graph_val": {
            "file_name": "scene_graph_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
        },
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDone. Processed: {stats['processed']}")
    print(f"  Skipped (no scene_graph.json): {stats['skipped_no_sg']}")
    print(f"  Skipped (no .pth pointcloud):  {stats['skipped_no_pcd']}")
    print(f"  Skipped (empty after filter):  {stats['skipped_empty']}")
    print(f"  Total objects: {stats['total_objects']}, Total rels: {stats['total_rels']}")
    print(f"  Train entries: {len(dataset['train'])}, Val entries: {len(dataset['val'])}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
