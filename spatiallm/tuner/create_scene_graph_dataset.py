import os
import json
import random
import argparse
from glob import glob
from tqdm import tqdm

from spatiallm.tuner.data import (
    LAYOUT_S_PLACEHOLDER,
    LAYOUT_E_PLACEHOLDER,
    POINT_CLOUD_PLACEHOLDER,
)
from spatiallm.layout.scene_graph_layout import SceneGraphLayout


def main():
    parser = argparse.ArgumentParser("Create scene-graph SpatialLM training data")
    parser.add_argument("--dataset_dir", required=True,
                        help="Root dir that will contain pcd/ and the output JSONs")
    parser.add_argument("--scene_graph_dir", required=True,
                        help="Dir that contains <scene_id>/scene_graph.json")
    parser.add_argument("--code_template_file", default="scene_graph_code_template.txt")
    parser.add_argument("--dataset_name", default="scene_graph")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_split_file", type=str, default=None,
                        help="Path to a text file containing scene IDs for training (one per line)")
    parser.add_argument("--val_split_file", type=str, default=None,
                        help="Path to a text file containing scene IDs for validation (one per line)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load code template
    with open(args.code_template_file) as f:
        code_template = f.read()

    # Load explicit splits if provided
    train_scenes = None
    val_scenes = None
    if args.train_split_file:
        with open(args.train_split_file) as f:
            train_scenes = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(train_scenes)} training scenes from {args.train_split_file}")
    if args.val_split_file:
        with open(args.val_split_file) as f:
            val_scenes = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(val_scenes)} validation scenes from {args.val_split_file}")

    # Discover scenes
    pcd_dir = os.path.join(args.dataset_dir, "pcd")
    sg_dirs = sorted(glob(os.path.join(args.scene_graph_dir, "*/scene_graph.json")))
    print(f"Found {len(sg_dirs)} scene_graph.json files")

    dataset = {"train": [], "val": []}
    skipped = 0
    total_rels = 0
    max_rels = 0

    for sg_path in tqdm(sg_dirs):
        scene_id = os.path.basename(os.path.dirname(sg_path))
        ply_path = os.path.join(pcd_dir, f"{scene_id}.ply")
        if not os.path.exists(ply_path):
            skipped += 1
            continue

        with open(sg_path) as f:
            sg_data = json.load(f)

        # Build SceneGraphLayout (all relationships kept)
        layout = SceneGraphLayout.from_json(sg_data)

        if len(layout.objects) == 0:
            skipped += 1
            continue

        total_rels += len(layout.relationships)
        max_rels = max(max_rels, len(layout.relationships))

        # Serialize scene graph to JSON string (raw float coordinates)
        language_string = layout.to_language_string()

        task_prompt = (
            f"Detect objects and their relationships. "
            f"The reference code is as followed: {code_template}"
        )

        conversation = {
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
                os.path.join("pcd", f"{scene_id}.ply"),
            ],
        }

        # Split
        if train_scenes is not None and val_scenes is not None:
            if scene_id in val_scenes:
                dataset["val"].append(conversation)
            elif scene_id in train_scenes:
                dataset["train"].append(conversation)
            else:
                skipped += 1
        else:
            if random.random() < args.val_ratio:
                dataset["val"].append(conversation)
            else:
                dataset["train"].append(conversation)

    n_scenes = len(dataset['train']) + len(dataset['val'])
    print(f"\nCreated {len(dataset['train'])} train / {len(dataset['val'])} val samples")
    print(f"Skipped {skipped} scenes (missing PLY or empty)")
    print(f"Relationships: avg={total_rels / max(n_scenes, 1):.0f}, max={max_rels}")

    # ---- Write JSON ----
    os.makedirs(args.dataset_dir, exist_ok=True)
    train_file = os.path.join(args.dataset_dir, f"{args.dataset_name}_train.json")
    val_file = os.path.join(args.dataset_dir, f"{args.dataset_name}_val.json")

    with open(train_file, "w") as f:
        json.dump(dataset["train"], f, indent=2)
    with open(val_file, "w") as f:
        json.dump(dataset["val"], f, indent=2)

    # ---- Write dataset_info.json ----
    dataset_info = {
        f"{args.dataset_name}_train": {
            "file_name": f"{args.dataset_name}_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        },
        f"{args.dataset_name}_val": {
            "file_name": f"{args.dataset_name}_val.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        },
    }

    info_path = os.path.join(args.dataset_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            existing = json.load(f)
        existing.update(dataset_info)
        dataset_info = existing

    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved to {train_file}, {val_file}, {info_path}")


if __name__ == "__main__":
    main()
