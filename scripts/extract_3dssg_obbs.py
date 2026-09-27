import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_3dssg_data(objects_path, relationships_path):
    """Load and index 3DSSG objects and relationships by scan ID."""
    with open(objects_path) as f:
        obj_data = json.load(f)
    with open(relationships_path) as f:
        rel_data = json.load(f)

    objects_by_scan = {}
    for entry in obj_data["scans"]:
        objects_by_scan[entry["scan"]] = entry["objects"]

    rels_by_scan = {}
    for entry in rel_data["scans"]:
        rels_by_scan[entry["scan"]] = entry["relationships"]

    return objects_by_scan, rels_by_scan


def convert_pc_to_box(obj_pc):
    """Compute an oriented bounding box from a point cloud.

    Args:
        obj_pc: (N, 3+) array – at least the first 3 columns are xyz.

    Returns:
        center   – [cx, cy, cz]
        box_size – [sx, sy, sz]
        angle_z  – yaw angle in radians
    """
    pts = obj_pc[:, :3]

    # Fallback to AABB behavior for very small/degenerate point sets.
    if pts.shape[0] < 3:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        box_size = maxs - mins
        return center.tolist(), box_size.tolist(), 0.0

    xy = pts[:, :2]
    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean
    cov = np.cov(xy_centered, rowvar=False)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        box_size = maxs - mins
        return center.tolist(), box_size.tolist(), 0.0

    order = np.argsort(eigvals)[::-1]
    axes_2d = eigvecs[:, order]
    if np.linalg.det(axes_2d) < 0:
        axes_2d[:, 1] = -axes_2d[:, 1]

    # Rotate into local XY frame; keep Z aligned with gravity.
    rot2d = axes_2d.T
    xy_local = (xy - xy_mean) @ rot2d.T
    z_vals = pts[:, 2]

    local_min = np.array([xy_local[:, 0].min(), xy_local[:, 1].min(), z_vals.min()])
    local_max = np.array([xy_local[:, 0].max(), xy_local[:, 1].max(), z_vals.max()])
    local_center = (local_min + local_max) / 2.0
    box_size = local_max - local_min

    world_xy_center = rot2d.T @ local_center[:2] + xy_mean
    center = [float(world_xy_center[0]), float(world_xy_center[1]), float(local_center[2])]
    angle_z = float(np.arctan2(rot2d[1, 0], rot2d[0, 0]))

    return center, box_size.tolist(), angle_z


def process_scan(scan_id, dssg_objects, dssg_rels, pcd_root, output_dir):
    """Compute OBBs from SceneVerse pointcloud, pair with 3DSSG labels/rels."""
    pcd_path = pcd_root / "pcd_with_global_alignment" / f"{scan_id}.pth"
    if not pcd_path.exists():
        return None

    pcd_data = torch.load(pcd_path, weights_only=False)
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    pcds = np.concatenate([points, colors], axis=1)

    # Get instance IDs present in the pointcloud
    pcd_inst_ids = set(int(i) for i in np.unique(instance_labels) if i >= 0)

    objects_out = []
    skipped = []
    for obj in dssg_objects:
        inst_id = int(obj["id"])
        label = obj["label"]

        if inst_id not in pcd_inst_ids:
            skipped.append((inst_id, label))
            continue

        mask = instance_labels == inst_id
        if np.sum(mask) == 0:
            skipped.append((inst_id, label))
            continue

        obj_pc = pcds[mask]
        center, box_size, angle_z = convert_pc_to_box(obj_pc)

        objects_out.append({
            "id": inst_id,
            "label": label,
            "position": [round(c, 4) for c in center],
            "size": [round(s, 4) for s in box_size],
            "angle_z": round(angle_z, 4),
        })

    valid_ids = {o["id"] for o in objects_out}
    relationships_out = []
    for rel in (dssg_rels or []):
        subj_id, obj_id, rel_type_id, rel_name = rel
        if subj_id in valid_ids and obj_id in valid_ids:
            relationships_out.append([subj_id, obj_id, rel_name])

    scene_graph = {
        "objects": objects_out,
        "relationships": relationships_out,
    }

    scan_dir = output_dir / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    with open(scan_dir / "scene_graph.json", "w") as f:
        json.dump(scene_graph, f, indent=2)

    if skipped:
        print(f"  {scan_id}: {len(skipped)}/{len(dssg_objects)} objects not in pcd: "
              f"{[s[1] for s in skipped[:5]]}{'...' if len(skipped) > 5 else ''}")

    return len(objects_out), len(relationships_out)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3DSSG scene graphs with OBBs from SceneVerse pointclouds")
    parser.add_argument("--objects_json", type=str,
                        default="/work/luka_milivojevic/3DSSG/objects.json")
    parser.add_argument("--relationships_json", type=str,
                        default="/work/luka_milivojevic/3DSSG/relationships.json")
    parser.add_argument("--pcd_root", type=str,
                        default="/work/luka_milivojevic/sceneverse_3rscan/3RScan/scan_data",
                        help="Root of SceneVerse scan_data (contains pcd_with_global_alignment/)")
    parser.add_argument("--output_dir", type=str,
                        default="/work/luka_milivojevic/SceneVerse/preprocess/ssg/relationships/3RScan")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process only this single scene (for testing)")
    args = parser.parse_args()

    print("Loading 3DSSG data...")
    objects_by_scan, rels_by_scan = load_3dssg_data(
        args.objects_json, args.relationships_json)

    pcd_root = Path(args.pcd_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find scans that have SceneVerse processed pointclouds
    available_scans = {p.stem for p in (pcd_root / "pcd_with_global_alignment").glob("*.pth")}

    if args.scene:
        scans_to_process = [args.scene]
    else:
        scans_to_process = sorted(set(objects_by_scan.keys()) & available_scans)

    print(f"3DSSG scans: {len(objects_by_scan)}, "
          f"SceneVerse pcd scans: {len(available_scans)}, "
          f"to process: {len(scans_to_process)}")

    stats = {"processed": 0, "skipped": 0, "total_objects": 0, "total_rels": 0}
    for scan_id in tqdm(scans_to_process, desc="Extracting OBBs"):
        dssg_objects = objects_by_scan.get(scan_id, [])
        dssg_rels = rels_by_scan.get(scan_id, [])
        result = process_scan(
            scan_id, dssg_objects, dssg_rels, pcd_root, output_dir)
        if result:
            stats["processed"] += 1
            stats["total_objects"] += result[0]
            stats["total_rels"] += result[1]
        else:
            stats["skipped"] += 1

    print(f"\nDone. Processed: {stats['processed']}, Skipped: {stats['skipped']}")
    print(f"Total objects: {stats['total_objects']}, "
          f"Total relationships: {stats['total_rels']}")


if __name__ == "__main__":
    main()
