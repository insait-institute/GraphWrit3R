import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from sg_eval.io_utils import load_predictions

MODULE_PATH = Path(__file__).parents[1] / "spatiallm" / "relationships_only.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "relationships_only_helpers",
    MODULE_PATH,
)
relationships_only_helpers = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(relationships_only_helpers)

build_relationships_only_prompt = (
    relationships_only_helpers.build_relationships_only_prompt
)
aggregate_relationship_diagnostics = (
    relationships_only_helpers.aggregate_relationship_diagnostics
)
extract_json_value = relationships_only_helpers.extract_json_value
load_given_objects_map = relationships_only_helpers.load_given_objects_map
normalize_relationships = relationships_only_helpers.normalize_relationships
renumber_given_objects = relationships_only_helpers.renumber_given_objects
relationships_summary_path = relationships_only_helpers.relationships_summary_path


def dataset_sample(scene_id: str, graph: dict) -> dict:
    return {
        "point_clouds": [f"pcd/{scene_id}.ply"],
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<point_cloud>Detect objects and their relationships. "
                    "The reference code is as followed: {...}"
                ),
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
    }


class RelationshipsOnlyTests(unittest.TestCase):
    def setUp(self):
        self.objects = {
            "num_objects": 3,
            "objects": [
                {
                    "id": 0,
                    "label": "floor",
                    "position": [0.0, 0.0, 0.0],
                    "size": [5.0, 5.0, 0.1],
                    "angle_z": 0.0,
                },
                {
                    "id": 3,
                    "label": "table",
                    "position": [1.0, 1.0, 1.0],
                    "size": [1.0, 1.0, 1.0],
                    "angle_z": 0.0,
                },
                {
                    "id": 7,
                    "label": "chair",
                    "position": [2.0, 1.0, 0.5],
                    "size": [0.5, 0.5, 1.0],
                    "angle_z": 0.0,
                },
            ],
        }

    def test_loads_objects_from_full_scene_graph_assistant_target(self):
        graph = {
            **self.objects,
            "relationships": [[3, 0, "standing on"]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.json"
            path.write_text(json.dumps([dataset_sample("scene_a", graph)]))

            without_targets = load_given_objects_map(str(path))
            with_targets = load_given_objects_map(
                str(path),
                include_assistant_targets=True,
            )

        self.assertEqual(without_targets, {})
        self.assertEqual(with_targets["scene_a"], self.objects)

    def test_prompt_exposes_only_immutable_ids_and_labels(self):
        renumbered, _ = renumber_given_objects(self.objects)
        prompt = build_relationships_only_prompt(renumbered)

        self.assertIn("<|point_start|><|point_pad|><|point_end|>", prompt)
        self.assertIn("complete and immutable", prompt)
        self.assertIn("do not output an object list", prompt)
        self.assertIn('"id": 1', prompt)
        self.assertIn('"label": "table"', prompt)
        self.assertNotIn('"id": 3', prompt)
        self.assertNotIn('"id": 7', prompt)
        self.assertNotIn('"position"', prompt)
        self.assertNotIn('"size"', prompt)
        self.assertNotIn("between the provided object IDs", prompt)

    def test_renumbers_a_copy_without_altering_dataset_objects(self):
        original_json = json.dumps(self.objects, sort_keys=True)

        renumbered, mapping = renumber_given_objects(self.objects)

        self.assertEqual([obj["id"] for obj in renumbered["objects"]], [0, 1, 2])
        self.assertEqual(renumbered["num_objects"], 3)
        self.assertEqual(
            mapping,
            [
                {"original_id": 0, "renumbered_id": 0},
                {"original_id": 3, "renumbered_id": 1},
                {"original_id": 7, "renumbered_id": 2},
            ],
        )
        self.assertEqual(renumbered["objects"][1]["position"], [1.0, 1.0, 1.0])
        self.assertEqual(json.dumps(self.objects, sort_keys=True), original_json)

    def test_extracts_list_from_wrapped_model_text(self):
        value = extract_json_value(
            "Here is the result:\n"
            '[{"subject_id": 3, "predicate": "close by", "object_id": 7}]'
        )

        self.assertEqual(value[0]["subject_id"], 3)

    def test_normalizes_dict_and_legacy_list_relationships(self):
        renumbered, _ = renumber_given_objects(self.objects)
        parsed = [
            {"subject_id": 1, "predicate": "Close By", "object_id": 2},
            [2, 0, "standing on"],
        ]

        relationships, diagnostics = normalize_relationships(parsed, renumbered)

        self.assertEqual(
            relationships,
            [[1, 2, "close by"], [2, 0, "standing on"]],
        )
        self.assertEqual(diagnostics["malformed_relationships"], 0)
        self.assertEqual(diagnostics["unknown_endpoint_relationships"], 0)

    def test_retains_unknown_ids_and_duplicates_and_counts_them(self):
        renumbered, _ = renumber_given_objects(self.objects)
        parsed = {
            "relationships": [
                {"subject_id": 1, "predicate": "close by", "object_id": 2},
                {"subject_id": 1, "predicate": "close by", "object_id": 2},
                {"subject_id": 99, "predicate": "close by", "object_id": 2},
                {"subject_id": 1, "predicate": "attached to", "object_id": 101},
                {"subject_id": 99, "predicate": "", "object_id": 101},
                {"unexpected": "shape"},
            ]
        }

        relationships, diagnostics = normalize_relationships(parsed, renumbered)

        self.assertEqual(
            relationships,
            [
                [1, 2, "close by"],
                [1, 2, "close by"],
                [99, 2, "close by"],
                [1, 101, "attached to"],
                [99, 101, ""],
            ],
        )
        self.assertEqual(diagnostics["saved_relationships"], 5)
        self.assertEqual(diagnostics["malformed_relationships"], 1)
        self.assertEqual(diagnostics["known_endpoint_relationships"], 2)
        self.assertEqual(diagnostics["unknown_endpoint_relationships"], 3)
        self.assertEqual(diagnostics["unknown_subject_relationships"], 2)
        self.assertEqual(diagnostics["unknown_object_relationships"], 2)
        self.assertEqual(diagnostics["unknown_both_relationships"], 1)
        self.assertEqual(diagnostics["unknown_subject_ids"], [99])
        self.assertEqual(diagnostics["unknown_object_ids"], [101])

    def test_unparseable_output_still_yields_empty_valid_relationships(self):
        parsed = extract_json_value("not JSON")
        relationships, diagnostics = normalize_relationships(parsed, self.objects)
        output = {
            **self.objects,
            "relationships": relationships,
        }

        self.assertIsNone(parsed)
        self.assertEqual(diagnostics["malformed_relationships"], 0)
        self.assertEqual(output["relationships"], [])
        self.assertEqual(json.loads(json.dumps(output)), output)

    def test_normalized_output_loads_in_existing_evaluator(self):
        renumbered, _ = renumber_given_objects(self.objects)
        relationships, diagnostics = normalize_relationships(
            [
                {"subject_id": 1, "predicate": "close by", "object_id": 2},
                {"subject_id": 99, "predicate": "attached to", "object_id": 2},
            ],
            renumbered,
        )
        output = {**renumbered, "relationships": relationships}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "scene_a_sg.json"
            output_path.write_text(json.dumps(output))
            predictions, file_count, failed_ids, _ = load_predictions(directory)

        self.assertEqual(diagnostics["malformed_relationships"], 0)
        self.assertEqual(file_count, 1)
        self.assertEqual(failed_ids, [])
        self.assertIn("scene_a", predictions)
        self.assertEqual(
            predictions["scene_a"].relationships[0].predicate,
            "close by",
        )
        self.assertEqual(
            predictions["scene_a"].relationships[1].subject_id,
            99,
        )

    def test_aggregate_report_is_a_sibling_of_prediction_directory(self):
        per_scene = {
            "scene_a": {
                "input_entries": 3,
                "saved_relationships": 3,
                "malformed_relationships": 0,
                "known_endpoint_relationships": 1,
                "unknown_endpoint_relationships": 2,
                "unknown_subject_relationships": 1,
                "unknown_object_relationships": 1,
                "unknown_both_relationships": 0,
                "unknown_subject_ids": [99],
                "unknown_object_ids": [101],
            },
            "scene_b": {
                "input_entries": 1,
                "saved_relationships": 1,
                "malformed_relationships": 0,
                "known_endpoint_relationships": 0,
                "unknown_endpoint_relationships": 1,
                "unknown_subject_relationships": 1,
                "unknown_object_relationships": 1,
                "unknown_both_relationships": 1,
                "unknown_subject_ids": [99],
                "unknown_object_ids": [102],
            },
        }

        summary = aggregate_relationship_diagnostics(
            "/tmp/predictions",
            per_scene,
            ["scene_c"],
        )

        self.assertEqual(
            relationships_summary_path("/tmp/predictions"),
            "/tmp/predictions_relationships_only_summary.json",
        )
        self.assertEqual(summary["totals"]["saved_relationships"], 4)
        self.assertEqual(summary["totals"]["unknown_endpoint_relationships"], 3)
        self.assertEqual(summary["unique_unknown_subject_ids"], [99])
        self.assertEqual(summary["unique_unknown_object_ids"], [101, 102])
        self.assertEqual(summary["skipped_existing_scenes"], ["scene_c"])


if __name__ == "__main__":
    unittest.main()
