# Scene graph layout for SpatialLM finetuning.
#
# The LLM input/output format is JSON matching the user's scene_graph.json schema:
#
# {"num_objects":5,"objects":[{"id":0,"label":"bed","position":[432,312,56],"size":[198,254,105],"angle_z":128},
#              ...],
#  "relationships":[[0,1,"support"],[1,0,"beside"],...]}
#
# Positions, sizes, and angle_z are discretized integers during training/inference;
# floats in raw ground-truth data.

import json
import numpy as np
from spatiallm.layout.entity import NORMALIZATION_PRESET
from spatiallm.layout.scene_graph_entity import (
    SceneGraphObject,
    SceneGraphRelationship,
    is_distance_predicate,
    parse_distance_predicate,
)


def _json_default(obj):
    """Handle numpy types when serializing to JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    #if isinstance(obj, (np.floating, float)):
        #return round(float(obj), 1)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class SceneGraphLayout:
    """Scene graph as objects + relationships, serialized as JSON."""

    def __init__(
        self,
        s: str = None,
        *,
        distance_predicates_are_quantized: bool | None = None,
    ):
        self.objects: list[SceneGraphObject] = []
        self.relationships: list[SceneGraphRelationship] = []
        self.distance_predicates_are_quantized = distance_predicates_are_quantized
        if s:
            self.from_str(s, distance_predicates_are_quantized=distance_predicates_are_quantized)

    @staticmethod
    def get_grid_size(num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        return (world_max - world_min) / num_bins

    # ── parsing (from JSON string) ───────────────────────────────────

    def from_str(self, s: str, *, distance_predicates_are_quantized: bool | None = None):
        """Parse a JSON string into objects and relationships."""
        s = s.strip()
        data = json.loads(s)
        self._load_from_dict(
            data, distance_predicates_are_quantized=distance_predicates_are_quantized
        )

    @staticmethod
    def _objects_look_discretized(objects: list[dict]) -> bool:
        if not objects:
            return False
        required_keys = ("position", "size")
        saw_numeric = False
        for obj_data in objects:
            if not isinstance(obj_data, dict):
                return False
            for key in required_keys:
                values = obj_data.get(key)
                if not isinstance(values, list) or len(values) != 3:
                    return False
                for value in values:
                    if not isinstance(value, (int, float, np.integer, np.floating)):
                        return False
                    saw_numeric = True
                    if not float(value).is_integer():
                        return False
            angle_z = obj_data.get("angle_z", 0.0)
            if not isinstance(angle_z, (int, float, np.integer, np.floating)):
                return False
            saw_numeric = True
            if not float(angle_z).is_integer():
                return False
        return saw_numeric

    @classmethod
    def _infer_distance_quantized_state(
        cls, data: dict, explicit_value: bool | None
    ) -> bool:
        if explicit_value is not None:
            return explicit_value
        relationships = data.get("relationships", [])
        has_distance_predicates = any(
            isinstance(rel_data, list)
            and len(rel_data) >= 3
            and is_distance_predicate(str(rel_data[2]))
            for rel_data in relationships
        )
        if not has_distance_predicates:
            return False
        objects = data.get("objects", [])
        if isinstance(objects, list) and cls._objects_look_discretized(objects):
            return True
        return False

    def _load_from_dict(
        self,
        data: dict,
        *,
        distance_predicates_are_quantized: bool | None = None,
    ):
        self.objects = []
        self.relationships = []
        self.distance_predicates_are_quantized = self._infer_distance_quantized_state(
            data, distance_predicates_are_quantized
        )
        old_to_new = {}
        for new_id, obj_data in enumerate(data.get("objects", [])):
            old_id = obj_data["id"]
            old_to_new[old_id] = new_id
            pos = obj_data["position"]
            size = obj_data["size"]
            angle_z = obj_data.get("angle_z", 0.0)
            obj = SceneGraphObject(
                id=new_id,
                label=obj_data["label"],
                position_x=pos[0],
                position_y=pos[1],
                position_z=pos[2],
                size_x=size[0],
                size_y=size[1],
                size_z=size[2],
                angle_z=angle_z,
                source_id=old_id,
            )
            self.objects.append(obj)

        for rel_data in data.get("relationships", []):
            src, tgt, predicate = int(rel_data[0]), int(rel_data[1]), str(rel_data[2])
            # Remap IDs if needed (handles non-sequential original IDs)
            src_new = old_to_new.get(src, src)
            tgt_new = old_to_new.get(tgt, tgt)
            parsed_distance = parse_distance_predicate(predicate)
            rel = SceneGraphRelationship(
                subject_id=src_new,
                object_id=tgt_new,
                predicate=predicate,
                distance_value=parsed_distance,
                distance_is_quantized=(
                    self.distance_predicates_are_quantized and parsed_distance is not None
                ),
            )
            self.relationships.append(rel)

    # ── serialization (to JSON string) ───────────────────────────────

    def to_language_string(self) -> str:
        """Serialize to a compact JSON string (same schema as scene_graph.json)."""
        data = {"num_objects": len(self.objects), "objects": [], "relationships": []}
        for obj in self.objects:
            data["objects"].append({
                "id": int(obj.id),
                "label": obj.label,
                "position": [_json_default(obj.position_x),
                              _json_default(obj.position_y),
                              _json_default(obj.position_z)],
                "size": [_json_default(obj.size_x),
                          _json_default(obj.size_y),
                          _json_default(obj.size_z)],
                "angle_z": _json_default(obj.angle_z),
            })
        for rel in self.relationships:
            data["relationships"].append([
                int(rel.subject_id),
                int(rel.object_id),
                rel.to_serializable_predicate(),
            ])
        return json.dumps(data, separators=(',', ':'), default=_json_default)

    def to_json(self) -> dict:
        """Return a Python dict in the scene_graph.json format."""
        return json.loads(self.to_language_string())

    def source_object_ids(self) -> list[int]:
        """Return original source object ids in the current serialized order."""
        return [int(obj.source_id if obj.source_id is not None else obj.id) for obj in self.objects]

    # ── spatial transforms (objects only; rels are structural) ────────

    def translate(self, translation: np.ndarray):
        for obj in self.objects:
            obj.translate(translation)

    def scale(self, scaling: float):
        for obj in self.objects:
            obj.scale(scaling)

    def rotate(self, angle: float):
        for obj in self.objects:
            obj.rotate(angle)

    def normalize_and_discretize(self, num_bins):
        for rel in self.relationships:
            rel.refresh_distance_from_objects(self.objects)
        for obj in self.objects:
            obj.normalize_and_discretize(num_bins)
        for rel in self.relationships:
            rel.normalize_and_discretize(num_bins)
        self.distance_predicates_are_quantized = True

    def undiscretize_and_unnormalize(self, num_bins):
        for obj in self.objects:
            obj.undiscretize_and_unnormalize(num_bins)
        for rel in self.relationships:
            rel.undiscretize_and_unnormalize(num_bins)
        self.distance_predicates_are_quantized = False

    # ── compatibility with mm_plugin transform pipeline ──────────────

    def filter_empty_bboxes(self, points, num_points=100, margin=0.15):
        """Remove objects that have no points inside them, and prune dangling rels.

        Handles both AABBs (angle_z == 0) and OBBs (angle_z != 0) by rotating
        points into each object's local frame before the containment check.
        """
        surviving_ids = set()
        surviving_objects = []
        for obj in self.objects:
            center = np.array([obj.position_x, obj.position_y, obj.position_z])
            half_size = np.array([obj.size_x, obj.size_y, obj.size_z]) / 2.0 + margin
            diff = points - center
            # Rotate points into the object's local frame (inverse rotation)
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
        self.objects = surviving_objects

        # Remove relationships referencing deleted objects
        self.relationships = [
            r for r in self.relationships
            if r.subject_id in surviving_ids and r.object_id in surviving_ids
        ]

    def reorder_entities(self):
        """Sort objects by position, assign sequential IDs, remap relationships."""
        if not self.objects:
            return
        sort_keys = np.array([o.sort_key() for o in self.objects])
        sorted_idx = np.lexsort(sort_keys.T)

        old_to_new = {}
        sorted_objects = []
        for new_id, old_idx in enumerate(sorted_idx):
            obj = self.objects[old_idx]
            old_to_new[obj.id] = new_id
            obj.id = new_id
            sorted_objects.append(obj)
        self.objects = sorted_objects

        valid_rels = []
        for rel in self.relationships:
            if rel.subject_id in old_to_new and rel.object_id in old_to_new:
                rel.subject_id = old_to_new[rel.subject_id]
                rel.object_id = old_to_new[rel.object_id]
                valid_rels.append(rel)
        valid_rels.sort(key=lambda r: (r.subject_id, r.object_id))
        self.relationships = valid_rels

    # ── constructors ─────────────────────────────────────────────────

    @classmethod
    def from_json(cls, data: dict) -> "SceneGraphLayout":
        """Create from a scene_graph.json dict.  IDs are remapped to sequential."""
        layout = cls()
        layout._load_from_dict(data)
        return layout
