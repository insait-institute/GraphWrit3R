# Scene graph entity types for SpatialLM finetuning.
# Objects have label + position + size + angle_z (yaw around gravity axis).
# Relationships are triples: [subject_id, object_id, predicate].
# The output format is JSON matching the scene_graph.json schema.

import math
from dataclasses import dataclass, field

import numpy as np

from spatiallm.layout.entity import NORMALIZATION_PRESET


DISTANCE_PREDICATE_PREFIX = "distance:"
DISTANCE_DECIMALS = 4
DISTANCE_MIN = 0.0
_WORLD_MIN, _WORLD_MAX = NORMALIZATION_PRESET["world"]
DISTANCE_MAX = math.sqrt(3.0) * (_WORLD_MAX - _WORLD_MIN)


def parse_distance_predicate(predicate: str) -> float | None:
    """Return the numeric distance payload for `distance:<value>` predicates."""
    if not isinstance(predicate, str):
        return None
    lowered = predicate.lower().strip()
    if not lowered.startswith(DISTANCE_PREDICATE_PREFIX):
        return None
    payload = lowered[len(DISTANCE_PREDICATE_PREFIX) :].strip()
    if not payload:
        return None
    try:
        return float(payload)
    except ValueError:
        return None


def is_distance_predicate(predicate: str) -> bool:
    return parse_distance_predicate(predicate) is not None


def quantize_distance(distance_value: float, num_bins: int) -> int:
    clipped = float(np.clip(float(distance_value), DISTANCE_MIN, DISTANCE_MAX))
    normalized = (clipped - DISTANCE_MIN) / (DISTANCE_MAX - DISTANCE_MIN)
    return int(np.clip(int(normalized * num_bins), 0, num_bins - 1))


def dequantize_distance(distance_bin: float, num_bins: int) -> float:
    clipped_bin = float(np.clip(float(distance_bin), 0, num_bins - 1))
    normalized = clipped_bin / num_bins
    return normalized * (DISTANCE_MAX - DISTANCE_MIN) + DISTANCE_MIN


def format_distance_predicate(distance_value: float, *, quantized: bool) -> str:
    if quantized:
        return f"{DISTANCE_PREDICATE_PREFIX}{int(distance_value)}"
    return f"{DISTANCE_PREDICATE_PREFIX}{float(distance_value):.{DISTANCE_DECIMALS}f}"


@dataclass
class SceneGraphObject:
    """A scene graph object node with label, position, size, and Z-rotation."""

    id: int
    label: str
    position_x: float
    position_y: float
    position_z: float
    size_x: float
    size_y: float
    size_z: float
    angle_z: float = 0.0  # yaw angle in radians (rotation around Z / gravity axis)
    source_id: int | None = None

    def __post_init__(self):
        self.id = int(self.id)
        self.source_id = self.id if self.source_id is None else int(self.source_id)
        self.label = str(self.label)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.size_x = abs(float(self.size_x))
        self.size_y = abs(float(self.size_y))
        self.size_z = abs(float(self.size_z))
        self.angle_z = float(self.angle_z)

    def translate(self, translation: np.ndarray):
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        self.size_x *= scaling
        self.size_y *= scaling
        self.size_z *= scaling
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling
        # angle_z is not affected by uniform scaling

    def rotate(self, angle: float):
        from scipy.spatial.transform import Rotation as R

        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        center = np.array([self.position_x, self.position_y, self.position_z])
        new_center = rotmat @ center
        self.position_x = new_center[0]
        self.position_y = new_center[1]
        self.position_z = new_center[2]
        # Accumulate the rotation into angle_z
        self.angle_z = float(self.angle_z + angle)

    def normalize_and_discretize(self, num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]

        self.position_x = (self.position_x - world_min) / (world_max - world_min) * num_bins
        self.position_y = (self.position_y - world_min) / (world_max - world_min) * num_bins
        self.position_z = (self.position_z - world_min) / (world_max - world_min) * num_bins
        self.size_x = (self.size_x - scale_min) / (scale_max - scale_min) * num_bins
        self.size_y = (self.size_y - scale_min) / (scale_max - scale_min) * num_bins
        self.size_z = (self.size_z - scale_min) / (scale_max - scale_min) * num_bins
        self.angle_z = (self.angle_z - angle_min) / (angle_max - angle_min) * num_bins

        self.position_x = int(np.clip(int(self.position_x), 0, num_bins - 1))
        self.position_y = int(np.clip(int(self.position_y), 0, num_bins - 1))
        self.position_z = int(np.clip(int(self.position_z), 0, num_bins - 1))
        self.size_x = int(np.clip(int(self.size_x), 0, num_bins - 1))
        self.size_y = int(np.clip(int(self.size_y), 0, num_bins - 1))
        self.size_z = int(np.clip(int(self.size_z), 0, num_bins - 1))
        self.angle_z = int(np.clip(int(self.angle_z), 0, num_bins - 1))

    def undiscretize_and_unnormalize(self, num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]

        self.position_x = self.position_x / num_bins * (world_max - world_min) + world_min
        self.position_y = self.position_y / num_bins * (world_max - world_min) + world_min
        self.position_z = self.position_z / num_bins * (world_max - world_min) + world_min
        self.size_x = self.size_x / num_bins * (scale_max - scale_min) + scale_min
        self.size_y = self.size_y / num_bins * (scale_max - scale_min) + scale_min
        self.size_z = self.size_z / num_bins * (scale_max - scale_min) + scale_min
        self.angle_z = self.angle_z / num_bins * (angle_max - angle_min) + angle_min

    def sort_key(self):
        return np.array([self.position_x, self.position_y])


@dataclass
class SceneGraphRelationship:
    """A scene graph relationship: [subject_id, object_id, predicate]."""

    subject_id: int
    object_id: int
    predicate: str
    distance_value: float | None = field(default=None, repr=False)
    distance_is_quantized: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.subject_id = int(self.subject_id)
        self.object_id = int(self.object_id)
        self.predicate = str(self.predicate)
        if self.distance_value is not None:
            self.distance_value = float(self.distance_value)
        elif is_distance_predicate(self.predicate):
            self.distance_value = parse_distance_predicate(self.predicate)

    def is_distance(self) -> bool:
        return self.distance_value is not None or is_distance_predicate(self.predicate)

    def refresh_distance_from_objects(self, objects: list[SceneGraphObject]) -> None:
        """Recompute center-to-center distance from the current object geometry."""
        if not self.is_distance():
            return
        if self.subject_id >= len(objects) or self.object_id >= len(objects):
            raise IndexError(
                f"Relationship references missing objects: {self.subject_id=} {self.object_id=}."
            )
        subject = objects[self.subject_id]
        target = objects[self.object_id]
        subject_pos = np.array(
            [subject.position_x, subject.position_y, subject.position_z], dtype=np.float32
        )
        target_pos = np.array(
            [target.position_x, target.position_y, target.position_z], dtype=np.float32
        )
        self.distance_value = float(np.linalg.norm(subject_pos - target_pos))
        self.distance_is_quantized = False
        self.predicate = format_distance_predicate(self.distance_value, quantized=False)

    def normalize_and_discretize(self, num_bins: int) -> None:
        if not self.is_distance():
            return
        if self.distance_value is None:
            parsed = parse_distance_predicate(self.predicate)
            if parsed is None:
                raise ValueError(f"Invalid distance predicate: {self.predicate}")
            self.distance_value = parsed
        self.distance_value = float(quantize_distance(self.distance_value, num_bins))
        self.distance_is_quantized = True
        self.predicate = format_distance_predicate(self.distance_value, quantized=True)

    def undiscretize_and_unnormalize(self, num_bins: int) -> None:
        if not self.is_distance():
            return
        if self.distance_value is None:
            parsed = parse_distance_predicate(self.predicate)
            if parsed is None:
                raise ValueError(f"Invalid distance predicate: {self.predicate}")
            self.distance_value = parsed
        self.distance_value = float(dequantize_distance(self.distance_value, num_bins))
        self.distance_is_quantized = False
        self.predicate = format_distance_predicate(self.distance_value, quantized=False)

    def to_serializable_predicate(self) -> str:
        if self.is_distance():
            if self.distance_value is None:
                parsed = parse_distance_predicate(self.predicate)
                if parsed is None:
                    return self.predicate
                self.distance_value = parsed
            return format_distance_predicate(
                self.distance_value, quantized=self.distance_is_quantized
            )
        return self.predicate
