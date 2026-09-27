import os
import re
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .data_types import Object3D, SceneGraph

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def extract_json_from_model_output(text: str) -> Optional[str]:
    """Extract JSON from model output wrapped in <|layout_s|>...<|layout_e|> tags."""
    match = re.search(r"<\|layout_s\|>(.*?)<\|layout_e\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*"objects".*"relationships".*\}', text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None


def parse_scene_graph(text: str) -> Optional[SceneGraph]:
    json_str = extract_json_from_model_output(text)
    if json_str is None:
        json_str = text.strip()
    try:
        data = json.loads(json_str)
        return SceneGraph.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

def load_ground_truth(gt_path: str) -> Tuple[Dict[str, SceneGraph], Dict[str, str]]:
    with open(gt_path, "r") as f:
        data = json.load(f)

    gt_dict = {}
    pcd_paths = {}
    for sample in data:
        pcd_path = sample["point_clouds"][0]
        scene_id = os.path.basename(pcd_path).replace(".ply", "")
        gpt_response = sample["conversations"][1]["value"]
        sg = parse_scene_graph(gpt_response)
        if sg:
            gt_dict[scene_id] = sg
            pcd_paths[scene_id] = pcd_path
    return gt_dict, pcd_paths


def filter_gt_with_point_clouds(
    gt_dict: Dict[str, SceneGraph],
    pcd_paths: Dict[str, str],
    dataset_dir: str,
    num_points: int = 0, # ORIGINAL SPATIALLM DEFAULT IS 100
    margin: float = 0.15,
    verbose: bool = True,
) -> Dict[str, SceneGraph]:
    if not HAS_O3D:
        if verbose:
            print("Warning: open3d not installed, skipping GT filtering.")
        return gt_dict

    filtered = {}
    total_objs_before = 0
    total_objs_after = 0
    total_rels_before = 0
    total_rels_after = 0
    skipped = 0

    for scene_id, sg in gt_dict.items():
        rel_path = pcd_paths.get(scene_id)
        if rel_path is None:
            filtered[scene_id] = sg
            continue

        abs_path = os.path.join(dataset_dir, rel_path)
        if not os.path.exists(abs_path):
            skipped += 1
            filtered[scene_id] = sg
            continue

        pcd = o3d.io.read_point_cloud(abs_path)
        points = np.asarray(pcd.points)

        objs_before = len(sg.objects)
        rels_before = len(sg.relationships)

        surviving_ids: set = set()
        surviving_objects: List[Object3D] = []
        for obj in sg.objects:
            center = np.asarray(obj.position, dtype=np.float32)
            half_size = np.asarray(obj.size, dtype=np.float32) / 2.0 + margin

            # OBB containment: rotate points into object's local frame first.
            diff = points - center
            angle = float(obj.angle_z)
            if abs(angle) > 1e-6:
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                local_x = diff[:, 0] * cos_a - diff[:, 1] * sin_a
                local_y = diff[:, 0] * sin_a + diff[:, 1] * cos_a
                diff = np.column_stack([local_x, local_y, diff[:, 2]])

            mask = np.all(np.abs(diff) <= half_size, axis=1)
            if np.sum(mask) > num_points:
                surviving_ids.add(obj.id)
                surviving_objects.append(obj)

        surviving_rels = [
            r for r in sg.relationships
            if r.subject_id in surviving_ids and r.object_id in surviving_ids
        ]

        filtered[scene_id] = SceneGraph(objects=surviving_objects, relationships=surviving_rels)
        total_objs_before += objs_before
        total_objs_after += len(surviving_objects)
        total_rels_before += rels_before
        total_rels_after += len(surviving_rels)

    removed_objs = total_objs_before - total_objs_after
    removed_rels = total_rels_before - total_rels_after
    pct_objs = removed_objs / total_objs_before * 100 if total_objs_before else 0
    pct_rels = removed_rels / total_rels_before * 100 if total_rels_before else 0
    if verbose:
        print(f"  GT filtering: removed {removed_objs}/{total_objs_before} objects ({pct_objs:.1f}%), "
              f"{removed_rels}/{total_rels_before} relationships ({pct_rels:.1f}%)")
    if verbose and skipped:
        print(f"  Warning: {skipped} PLY files not found, those GT scenes kept unfiltered")

    return filtered


def load_vocab_from_txt(txt_path: str) -> Set[str]:
    """Load labels from a .txt file (one label per line)."""
    labels: Set[str] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip().lower()
            if label:
                labels.add(label)
    return labels


def _extract_predicates_from_scene_graph_dict(data: Dict[str, Any]) -> Set[str]:
    preds: Set[str] = set()
    if not isinstance(data, dict):
        return preds

    rels = data.get("relationships")
    if isinstance(rels, list):
        for rel in rels:
            if isinstance(rel, list) and len(rel) >= 3:
                pred = str(rel[2]).strip().lower()
            elif isinstance(rel, dict):
                pred = str(rel.get("predicate", "")).strip().lower()
            else:
                pred = ""
            if pred:
                preds.add(pred)

    scene_graph = data.get("scene_graph")
    if isinstance(scene_graph, dict):
        preds.update(_extract_predicates_from_scene_graph_dict(scene_graph))
    return preds


def _extract_object_labels_from_scene_graph_dict(data: Dict[str, Any]) -> Set[str]:
    labels: Set[str] = set()
    if not isinstance(data, dict):
        return labels

    objs = data.get("objects")
    if isinstance(objs, list):
        for obj in objs:
            label = ""
            if isinstance(obj, dict):
                for key in ("label", "class", "name"):
                    value = obj.get(key)
                    if value is not None:
                        label = str(value).strip().lower()
                        if label:
                            break
            elif isinstance(obj, list) and len(obj) >= 2:
                label = str(obj[1]).strip().lower()

            if label:
                labels.add(label)

    scene_graph = data.get("scene_graph")
    if isinstance(scene_graph, dict):
        labels.update(_extract_object_labels_from_scene_graph_dict(scene_graph))
    return labels


def extract_predicates_from_scene_graph_json(json_path: str) -> Set[str]:
    preds: Set[str] = set()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        preds.update(_extract_predicates_from_scene_graph_dict(data))
        for value in data.values():
            if isinstance(value, dict):
                preds.update(_extract_predicates_from_scene_graph_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        preds.update(_extract_predicates_from_scene_graph_dict(item))

    if isinstance(data, list):
        for sample in data:
            if not isinstance(sample, dict):
                continue
            preds.update(_extract_predicates_from_scene_graph_dict(sample))
            convs = sample.get("conversations")
            if isinstance(convs, list):
                for conv in convs:
                    if not isinstance(conv, dict):
                        continue
                    sg = parse_scene_graph(str(conv.get("value", "")))
                    if sg:
                        preds.update(rel.predicate for rel in sg.relationships)
    return preds


def extract_object_labels_from_scene_graph_json(json_path: str) -> Set[str]:
    labels: Set[str] = set()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        labels.update(_extract_object_labels_from_scene_graph_dict(data))
        for value in data.values():
            if isinstance(value, dict):
                labels.update(_extract_object_labels_from_scene_graph_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        labels.update(_extract_object_labels_from_scene_graph_dict(item))

    if isinstance(data, list):
        for sample in data:
            if not isinstance(sample, dict):
                continue
            labels.update(_extract_object_labels_from_scene_graph_dict(sample))
            convs = sample.get("conversations")
            if isinstance(convs, list):
                for conv in convs:
                    if not isinstance(conv, dict):
                        continue
                    sg = parse_scene_graph(str(conv.get("value", "")))
                    if sg:
                        labels.update(obj.label for obj in sg.objects)

    return labels


def _load_prediction_scene_graph(
    sg_data: Any,
) -> Optional[SceneGraph]:
    """Load a prediction scene graph."""
    if isinstance(sg_data, dict):
        candidates: List[Any] = []
        scene_graph_value = sg_data.get("scene_graph")
        if scene_graph_value is not None:
            candidates.append(scene_graph_value)
        candidates.append(sg_data)

        for candidate in candidates:
            try:
                if isinstance(candidate, dict) and "objects" in candidate:
                    return SceneGraph.from_dict(candidate)
                if candidate is not None:
                    sg = parse_scene_graph(str(candidate))
                    if sg is not None:
                        return sg
            except Exception:
                continue

        return None

    return parse_scene_graph(str(sg_data))


def load_predictions(pred_path: str) -> Tuple[Dict[str, SceneGraph], int, List[str], List[str]]:
    pred_dict = {}
    num_files = 0
    failed_ids: List[str] = []

    if os.path.isfile(pred_path):
        with open(pred_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                num_files += 1
                scene_id = item.get("scene_id", item.get("name", f"item_{num_files}"))
                sg = _load_prediction_scene_graph(item)
                if sg:
                    pred_dict[scene_id] = sg
                else:
                    failed_ids.append(scene_id)
        elif isinstance(data, dict):
            if any(key in data for key in ("objects", "relationships", "scene_graph")):
                num_files = 1
                scene_id = os.path.basename(pred_path).replace(".json", "").replace("_sg", "")
                sg = _load_prediction_scene_graph(data)
                if sg:
                    pred_dict[scene_id] = sg
                else:
                    failed_ids.append(scene_id)
            else:
                for scene_id, sg_data in data.items():
                    num_files += 1
                    sg = _load_prediction_scene_graph(sg_data)
                    if sg:
                        pred_dict[scene_id] = sg
                    else:
                        failed_ids.append(scene_id)
    else:
        for fname in sorted(os.listdir(pred_path)):
            if fname.endswith(".json"):
                if (
                    fname.startswith("parent_gt_")
                    or fname.endswith("_prompt.json")
                    or fname.endswith("_merge_meta.json")
                ):
                    continue
                num_files += 1
                fpath = os.path.join(pred_path, fname)
                scene_id = fname.replace(".json", "").replace("_sg", "")
                with open(fpath, "r") as f:
                    try:
                        data = json.load(f)
                        sg = _load_prediction_scene_graph(data)
                        if sg is None:
                            failed_ids.append(scene_id)
                            continue
                        pred_dict[scene_id] = sg
                    except Exception:
                        failed_ids.append(scene_id)
                        continue

    return pred_dict, num_files, failed_ids, []
