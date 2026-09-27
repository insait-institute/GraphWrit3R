import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.extract_scannet_3dssg_subgraphs import (
    GenerationConfig,
    align_json_objects_to_instances,
    balanced_group_capacities,
    build_expanded_aabb_neighbors,
    filter_scene_graph,
    generate_subgraphs,
    select_floor_anchor,
    spatially_partition_objects,
)


def write_legacy_pth(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(value, protocol=2))
        archive.writestr("archive/version", b"3\n")


def dataset_row(scene_id: str, graph: dict) -> dict:
    return {
        "conversations": [
            {
                "from": "human",
                "value": "<point_cloud>Detect objects and their relationships.",
            },
            {
                "from": "gpt",
                "value": (
                    "<|layout_s|>"
                    + json.dumps(graph, separators=(",", ":"))
                    + "<|layout_e|>"
                ),
            },
        ],
        "point_clouds": [f"pcd/{scene_id}.ply"],
    }


class MappingAndGroupingTests(unittest.TestCase):
    def test_ordered_instance_alignment_skips_filtered_labels(self):
        source_objects = [
            {"id": 0, "label": "floor"},
            {"id": 1, "label": "chair"},
            {"id": 2, "label": "window"},
        ]
        instance_labels = {
            0: "floor",
            1: "object",
            2: "chair",
            3: "ceiling",
            4: "window",
        }

        mapping, skipped = align_json_objects_to_instances(
            source_objects, instance_labels
        )

        self.assertEqual(mapping, {0: 0, 1: 2, 2: 4})
        self.assertEqual(skipped, [1, 3])

    def test_ordered_instance_alignment_rejects_ambiguous_duplicate_label(self):
        source_objects = [
            {"id": 0, "label": "floor"},
            {"id": 1, "label": "chair"},
        ]
        instance_labels = {0: "floor", 1: "chair", 2: "chair"}

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            align_json_objects_to_instances(source_objects, instance_labels)

    def test_expanded_aabb_neighbors_are_symmetric(self):
        aabbs = {
            1: (np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.2])),
            2: (np.array([1.0, 0.0, 0.0]), np.array([1.2, 0.2, 0.2])),
            3: (np.array([3.0, 0.0, 0.0]), np.array([3.2, 0.2, 0.2])),
        }

        neighbors = build_expanded_aabb_neighbors(aabbs, margin=0.5)

        self.assertEqual(neighbors[1], {2})
        self.assertEqual(neighbors[2], {1})
        self.assertEqual(neighbors[3], set())

    def test_floor_anchor_uses_largest_floor_instance(self):
        objects = [
            {"id": 3, "label": "floor"},
            {"id": 7, "label": "floor"},
            {"id": 9, "label": "chair"},
        ]
        self.assertEqual(select_floor_anchor(objects, {3: 10, 7: 20, 9: 5}), 7)

    def test_balanced_capacities_match_pilot_expectations(self):
        self.assertEqual(
            balanced_group_capacities(
                31, max_objects=9, min_objects=5
            ),
            [8, 8, 8, 7],
        )
        self.assertEqual(
            balanced_group_capacities(
                28, max_objects=9, min_objects=5
            ),
            [7, 7, 7, 7],
        )

    def test_spatial_partition_is_deterministic_and_complete(self):
        object_ids = list(range(1, 11))
        centers = {
            object_id: np.array([float(object_id), 0.0, 0.0])
            for object_id in object_ids
        }
        neighbors = {
            object_id: {
                candidate
                for candidate in (object_id - 1, object_id + 1)
                if candidate in object_ids
            }
            for object_id in object_ids
        }
        capacities = [4, 3, 3]

        first = spatially_partition_objects(
            object_ids,
            centers=centers,
            neighbors=neighbors,
            capacities=capacities,
            seed=2020,
        )
        second = spatially_partition_objects(
            object_ids,
            centers=centers,
            neighbors=neighbors,
            capacities=capacities,
            seed=2020,
        )

        self.assertEqual(first, second)
        self.assertEqual([len(group) for group in first], capacities)
        self.assertEqual(
            sorted(object_id for group in first for object_id in group),
            object_ids,
        )

    def test_filter_scene_graph_keeps_exact_induced_relationships(self):
        graph = {
            "num_objects": 3,
            "objects": [
                {"id": 0, "label": "floor"},
                {"id": 1, "label": "chair"},
                {"id": 2, "label": "table"},
            ],
            "relationships": [
                [1, 0, "standing on"],
                [2, 0, "standing on"],
                [1, 2, "close by"],
            ],
        }

        filtered = filter_scene_graph(graph, [0, 2])

        self.assertEqual(filtered["num_objects"], 2)
        self.assertEqual([obj["id"] for obj in filtered["objects"]], [0, 2])
        self.assertEqual(filtered["relationships"], [[2, 0, "standing on"]])


class EndToEndGenerationTests(unittest.TestCase):
    def _create_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        scene_id = "scene_test_00"
        source_objects = []
        mapped_objects = []
        labels = ["floor", "chair", "table", "wall", "sink"]
        mapped_labels = ["floor", "chair", "table", "wall", "sink"]
        for object_id, (source_label, mapped_label) in enumerate(
            zip(labels, mapped_labels)
        ):
            geometry = {
                "id": object_id,
                "position": [float(object_id), 0.0, 0.5],
                "size": [0.5, 0.5, 1.0],
                "angle_z": 0.0,
            }
            source_objects.append({**geometry, "label": source_label})
            mapped_objects.append({**geometry, "label": mapped_label})

        source_graph = {
            "num_objects": 5,
            "objects": source_objects,
            "relationships": [
                [1, 0, "support"],
                [2, 0, "support"],
                [1, 2, "close to"],
                [4, 3, "to the left of"],
            ],
        }
        mapped_graph = {
            "num_objects": 5,
            "objects": mapped_objects,
            "relationships": [
                [1, 0, "standing on"],
                [2, 0, "standing on"],
                [1, 2, "close by"],
                [4, 3, "left"],
            ],
        }

        source_json = root / "source.json"
        mapped_json = root / "mapped.json"
        source_json.write_text(json.dumps([dataset_row(scene_id, source_graph)]))
        mapped_json.write_text(json.dumps([dataset_row(scene_id, mapped_graph)]))

        sceneverse_root = root / "scan_data"
        instance_to_label = {
            0: "floor",
            1: "object",
            2: "chair",
            3: "table",
            4: "wall",
            5: "sink",
        }
        write_legacy_pth(
            sceneverse_root / "instance_id_to_label" / f"{scene_id}.pth",
            instance_to_label,
        )

        points = []
        colors = []
        instance_ids = []
        for instance_id in [-100, 0, 1, 2, 3, 4, 5]:
            x = float(instance_id if instance_id >= 0 else -10)
            points.extend([[x, 0.0, 0.0], [x + 0.1, 0.1, 0.1]])
            colors.extend([[100.0, 110.0, 120.0], [120.0, 130.0, 140.0]])
            instance_ids.extend([instance_id, instance_id])
        point_cloud = (
            np.asarray(points, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
            np.zeros(len(points), dtype=np.int64),
            np.asarray(instance_ids, dtype=np.int64),
        )
        write_legacy_pth(
            sceneverse_root
            / "pcd_with_global_alignment"
            / f"{scene_id}.pth",
            point_cloud,
        )
        return mapped_json, source_json, sceneverse_root

    def test_generation_is_reproducible_and_object_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapped_json, source_json, sceneverse_root = self._create_inputs(root)
            manifests = []
            for output_name in ("output_a", "output_b"):
                config = GenerationConfig(
                    mapped_json=mapped_json,
                    source_json=source_json,
                    sceneverse_root=sceneverse_root,
                    output_dir=root / output_name,
                    scene_ids=("scene_test_00",),
                )
                manifests.append(generate_subgraphs(config))

            self.assertEqual(manifests[0], manifests[1])
            self.assertEqual(manifests[0]["summary"]["subgraph_count"], 1)
            split = manifests[0]["scenes"][0]["splits"][0]
            self.assertEqual(split["object_count"], 5)
            self.assertEqual(split["point_count"], 10)
            self.assertEqual(split["selected_instance_ids"], [0, 2, 3, 4, 5])

            output_root = root / "output_a"
            rows = json.loads((output_root / "scene_graph_val.json").read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["point_clouds"], ["pcd/scene_test_00_split1.ply"]
            )
            ply_bytes = (
                output_root / "pcd" / "scene_test_00_split1.ply"
            ).read_bytes()
            header = ply_bytes.split(b"end_header\n", 1)[0]
            self.assertIn(b"element vertex 10", header)


if __name__ == "__main__":
    unittest.main()
