import os
import argparse
import torch
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required. Install with: pip install open3d")


def pth_to_ply(pth_path: str, output_path: str):
    """Convert a SceneVerse .pth (points, colors, instance_labels) to a PLY file."""
    data = torch.load(pth_path, weights_only=False)
    points = data[0]  # (N, 3) float
    colors = data[1]  # (N, 3) uint8 or float

    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.numpy()

    points = points.astype(np.float64)

    # Normalize colors to [0, 1] if they seem to be in [0, 255]
    if colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0
    else:
        colors = colors.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd)


def main():
    parser = argparse.ArgumentParser(description="Convert .pth point clouds to PLY")
    parser.add_argument("--input_dir", required=True, help="Directory with .pth files")
    parser.add_argument("--output_dir", required=True, help="Output directory for .ply files")
    parser.add_argument("--scene_list", default=None,
                        help="Optional text file with scene IDs (one per line) to convert")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scene_list:
        with open(args.scene_list) as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        pth_files = [os.path.join(args.input_dir, f"{sid}.pth") for sid in scene_ids]
        pth_files = [p for p in pth_files if os.path.exists(p)]
    else:
        pth_files = sorted(
            [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".pth")]
        )

    print(f"Converting {len(pth_files)} point clouds...")
    for pth_path in pth_files:
        scene_id = os.path.splitext(os.path.basename(pth_path))[0]
        output_path = os.path.join(args.output_dir, f"{scene_id}.ply")
        try:
            pth_to_ply(pth_path, output_path)
        except Exception as e:
            print(f"Error converting {scene_id}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()
