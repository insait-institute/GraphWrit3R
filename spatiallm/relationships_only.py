"""Helpers for relationship-only scene-graph inference."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any


POINT_PLACEHOLDER = "<|point_start|><|point_pad|><|point_end|>"
DIAGNOSTIC_COUNT_KEYS = (
    "input_entries",
    "saved_relationships",
    "malformed_relationships",
    "known_endpoint_relationships",
    "unknown_endpoint_relationships",
    "unknown_subject_relationships",
    "unknown_object_relationships",
    "unknown_both_relationships",
)


def extract_json_value(text: str) -> Any:
    """Extract the first JSON object or list from model text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _objects_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    objects = data.get("objects")
    if not isinstance(objects, list):
        return None
    return {
        "num_objects": int(data.get("num_objects", len(objects))),
        "objects": objects,
    }


def extract_objects_from_prompt(prompt_text: str) -> dict[str, Any] | None:
    """Parse the ``Given objects JSON:`` payload from a human prompt."""
    marker = "Given objects JSON:"
    if marker not in prompt_text:
        return None
    return _objects_payload(extract_json_value(prompt_text.split(marker, 1)[1]))


def load_given_objects_map(
    dataset_json_path: str,
    *,
    include_assistant_targets: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load immutable objects for each scene in a ShareGPT-style dataset.

    Existing relationship-only datasets put the objects in the human prompt.
    Full scene-graph datasets instead put them in the assistant target; those
    targets are used only when explicitly requested so ``auto`` mode retains
    its previous behavior.
    """
    with open(dataset_json_path, "r", encoding="utf-8") as file:
        samples = json.load(file)

    mapping: dict[str, dict[str, Any]] = {}
    for sample in samples:
        point_clouds = sample.get("point_clouds", [])
        conversations = sample.get("conversations", [])
        if not point_clouds or not conversations:
            continue

        scene_id = os.path.basename(point_clouds[0]).removesuffix(".ply")
        human_text = next(
            (
                str(message.get("value", ""))
                for message in conversations
                if message.get("from") in {"human", "user"}
            ),
            "",
        )
        objects_payload = extract_objects_from_prompt(human_text)

        if objects_payload is None and include_assistant_targets:
            assistant_text = next(
                (
                    str(message.get("value", ""))
                    for message in conversations
                    if message.get("from") in {"gpt", "assistant"}
                ),
                "",
            )
            objects_payload = _objects_payload(extract_json_value(assistant_text))

        if objects_payload is not None:
            mapping[scene_id] = objects_payload

    return mapping


def build_relationships_only_prompt(given_objects: dict[str, Any]) -> str:
    """Build a relationship-only prompt with immutable object IDs and labels."""
    prompt_objects = [
        {"id": obj["id"], "label": obj["label"]}
        for obj in given_objects.get("objects", [])
        if isinstance(obj, dict) and "id" in obj and "label" in obj
    ]
    object_json = json.dumps(prompt_objects, indent=2, ensure_ascii=False)
    return (
        f"{POINT_PLACEHOLDER}"
        "You are given a 3D point cloud and a fixed list of ground-truth objects.\n\n"
        "The object list is complete and immutable.\n"
        "Do not add, remove, rename, merge, or duplicate objects.\n"
        "Predict only relationships; do not output an object list.\n\n"
        "Return only a JSON list of relationship triplets in this format:\n"
        "[\n"
        "  {\n"
        '    "subject_id": <provided object ID>,\n'
        '    "predicate": <relationship>,\n'
        '    "object_id": <provided object ID>\n'
        "  }\n"
        "]\n"
        "Return [] when no relationships are present.\n\n"
        f"Ground-truth objects:\n{object_json}"
    )


def renumber_given_objects(
    given_objects: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Copy GT objects and assign deterministic scene-local IDs ``0..N-1``."""
    objects = given_objects.get("objects", [])
    if not isinstance(objects, list):
        raise TypeError("Ground-truth objects must be a list.")

    renumbered_objects: list[dict[str, Any]] = []
    id_mapping: list[dict[str, int]] = []
    for new_id, source_object in enumerate(objects):
        if not isinstance(source_object, dict) or "id" not in source_object:
            raise ValueError(
                f"Ground-truth object at index {new_id} is missing an integer ID."
            )
        try:
            original_id = int(source_object["id"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Ground-truth object at index {new_id} has an invalid ID."
            ) from error

        renumbered_object = deepcopy(source_object)
        renumbered_object["id"] = new_id
        renumbered_objects.append(renumbered_object)
        id_mapping.append(
            {
                "original_id": original_id,
                "renumbered_id": new_id,
            }
        )

    return (
        {
            "num_objects": len(renumbered_objects),
            "objects": renumbered_objects,
        },
        id_mapping,
    )


def normalize_relationships(
    parsed: Any,
    given_objects: dict[str, Any],
) -> tuple[list[list[Any]], dict[str, Any]]:
    """Convert model output to triplets without filtering unknown endpoints."""
    if isinstance(parsed, dict):
        candidates = parsed.get("relationships", [])
    elif isinstance(parsed, list):
        candidates = parsed
    else:
        candidates = []

    diagnostics: dict[str, Any] = {
        "input_entries": 0,
        "saved_relationships": 0,
        "malformed_relationships": 0,
        "known_endpoint_relationships": 0,
        "unknown_endpoint_relationships": 0,
        "unknown_subject_relationships": 0,
        "unknown_object_relationships": 0,
        "unknown_both_relationships": 0,
        "unknown_subject_ids": [],
        "unknown_object_ids": [],
    }
    if not isinstance(candidates, list):
        diagnostics["malformed_relationships"] = 1
        return [], diagnostics
    diagnostics["input_entries"] = len(candidates)

    valid_ids = {
        int(obj["id"])
        for obj in given_objects.get("objects", [])
        if isinstance(obj, dict) and "id" in obj
    }
    normalized: list[list[Any]] = []
    unknown_subject_ids: set[int] = set()
    unknown_object_ids: set[int] = set()

    for relation in candidates:
        if isinstance(relation, dict):
            subject = relation.get("subject_id")
            object_id = relation.get("object_id")
            predicate = relation.get("predicate")
        elif isinstance(relation, list) and len(relation) >= 3:
            subject, object_id, predicate = relation[:3]
        else:
            diagnostics["malformed_relationships"] += 1
            continue

        try:
            subject = int(subject)
            object_id = int(object_id)
        except (TypeError, ValueError):
            diagnostics["malformed_relationships"] += 1
            continue
        if predicate is None:
            diagnostics["malformed_relationships"] += 1
            continue
        predicate = str(predicate).strip().lower()

        normalized.append([subject, object_id, predicate])

        unknown_subject = subject not in valid_ids
        unknown_object = object_id not in valid_ids
        if unknown_subject:
            unknown_subject_ids.add(subject)
            diagnostics["unknown_subject_relationships"] += 1
        if unknown_object:
            unknown_object_ids.add(object_id)
            diagnostics["unknown_object_relationships"] += 1
        if unknown_subject and unknown_object:
            diagnostics["unknown_both_relationships"] += 1
        if unknown_subject or unknown_object:
            diagnostics["unknown_endpoint_relationships"] += 1

    diagnostics["saved_relationships"] = len(normalized)
    diagnostics["known_endpoint_relationships"] = (
        len(normalized) - diagnostics["unknown_endpoint_relationships"]
    )
    diagnostics["unknown_subject_ids"] = sorted(unknown_subject_ids)
    diagnostics["unknown_object_ids"] = sorted(unknown_object_ids)
    return normalized, diagnostics


def relationships_summary_path(output_arg: str) -> str:
    """Return a sibling path that an evaluator will not scan as a prediction."""
    normalized = output_arg.rstrip(os.sep)
    if os.path.splitext(normalized)[-1]:
        stem, _ = os.path.splitext(normalized)
        return f"{stem}_relationships_only_summary.json"
    return f"{normalized}_relationships_only_summary.json"


def aggregate_relationship_diagnostics(
    output_arg: str,
    per_scene: dict[str, dict[str, Any]],
    skipped_existing_scenes: list[str],
) -> dict[str, Any]:
    totals = {
        key: sum(int(stats.get(key, 0)) for stats in per_scene.values())
        for key in DIAGNOSTIC_COUNT_KEYS
    }
    unknown_subject_ids = sorted(
        {
            object_id
            for stats in per_scene.values()
            for object_id in stats.get("unknown_subject_ids", [])
        }
    )
    unknown_object_ids = sorted(
        {
            object_id
            for stats in per_scene.values()
            for object_id in stats.get("unknown_object_ids", [])
        }
    )
    return {
        "mode": "relationships_only",
        "prediction_output": output_arg,
        "processed_scenes": len(per_scene),
        "skipped_existing_scenes": skipped_existing_scenes,
        "totals": totals,
        "unique_unknown_subject_ids": unknown_subject_ids,
        "unique_unknown_object_ids": unknown_object_ids,
        "per_scene": per_scene,
    }
